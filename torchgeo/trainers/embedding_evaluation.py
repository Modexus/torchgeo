"""Embedding classifciation task."""

from copy import deepcopy
from typing import Any, Sequence, cast

import torch
from deepspeed.ops.adam import FusedAdam
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.scheduler import CosineLRScheduler
from torch import Tensor, optim
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Identity,
    Linear,
    Module,
    Sequential,
)
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    FBetaScore,
    JaccardIndex,
    MetricCollection,
)

from ..utils import _to_tuple
from .byol import BYOLTask
from .cae import CAETask
from .mae import MAETask
from .msae import MSAETask
from .msn import MSNTask
from .tile2vec import Tile2VecTask
from .utils import LayerWiseDecayScheduler, generate_mask, param_groups_lrd
from .vicreg import VICRegTask

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        crop_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: tuple of integers defining the image size
            crop_size: tuple of integers defining the crop size
        """
        super().__init__()

        if crop_size is None:
            crop_size = image_size

        self.augmentation = {
            "train": Sequential(
                # K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size,
                    scale=(0.2, 1.0),
                    align_corners=False,
                    resample="BICUBIC",
                ),
                K.RandomHorizontalFlip(),
            ),
            "val": Sequential(
                # K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC")
            ),
            "test": Sequential(
                # K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC")
            ),
        }

    def forward(self, x: Tensor, stage: str | None = None) -> Tensor:
        """Applys augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        return cast(Tensor, self.augmentation[stage](x))


class EmbeddingEvaluator(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.channel_wise = self.hyperparams.get("channel_wise", False)
        self.in_channels = self.hyperparams.get("in_channels", 4)
        self.out_channels = self.hyperparams.get("out_channels", self.in_channels)
        self.mean_patches = self.hyperparams.get("mean_patches", False)
        self.patch_size = self.hyperparams.get("patch_size", 16)
        self.batch_size = self.hyperparams.get("batch_size", 256)
        self.linear_probing = self.hyperparams.get("linear_probing", False)
        self.multi_label = self.hyperparams.get("multi_label", False)
        self.layer_wise_decay = self.hyperparams.get("layer_wise_decay", True)
        self.num_batches = self.hyperparams.get("num_batches", 1536)

        image_size = _to_tuple(self.hyperparams["image_size"])
        self.crop_size = _to_tuple(self.hyperparams.get("crop_size", image_size))
        self.num_classes = self.hyperparams["num_classes"]

        self.num_patches = (self.crop_size[0] // self.patch_size) * (
            self.crop_size[1] // self.patch_size
        )

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="macro"
                ),
                "F1Score": FBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                ),
                "AveragePrecision": AveragePrecision(
                    self.hyperparams["num_classes"], average="macro"
                ),
            },
            prefix="train_",
        )
        if not self.multi_label:
            self.train_metrics.add_metrics(
                {
                    "JaccardIndex": JaccardIndex(
                        num_classes=self.hyperparams["num_classes"]
                    )
                }
            )
        else:
            self.train_metrics.add_metrics(
                {
                    "OverallPrecision": AveragePrecision(
                        self.hyperparams["num_classes"], average="micro"
                    )
                }
            )

        self.create_model()

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

        self.augment = self.hyperparams.get(
            "augment_fn", Augmentations(image_size, self.crop_size)
        )

    def create_model(self):
        """Creates the model."""
        self.projector: Module | None = None
        if self.hyperparams["task_name"] == "tile2vec":
            if "checkpoint_path" in self.hyperparams:
                task = Tile2VecTask.load_from_checkpoint(
                    checkpoint_path=self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = Tile2VecTask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            if "resnet" in self.hyperparams["encoder_name"]:
                self.encoder = task.model.encoder
            else:
                self.encoder = task.model
        elif self.hyperparams["task_name"] == "byol":
            if "checkpoint_path" in self.hyperparams:
                task = BYOLTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = BYOLTask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model.encoder.model
        elif self.hyperparams["task_name"] == "vicreg":
            if "checkpoint_path" in self.hyperparams:
                task = VICRegTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = VICRegTask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "mae":
            if "checkpoint_path" in self.hyperparams:
                task = MAETask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MAETask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "cae":
            if "checkpoint_path" in self.hyperparams:
                task = CAETask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = CAETask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "msn":
            if "checkpoint_path" in self.hyperparams:
                task = MSNTask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MSNTask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model
        elif self.hyperparams["task_name"] == "msae":
            if "checkpoint_path" in self.hyperparams:
                task = MSAETask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MSAETask(**self.hyperparams)
            if self.linear_probing:
                task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "identity":
            self.encoder = Identity()  # type: ignore[no-untyped-call]
        else:
            raise ValueError(
                f"Task type '{self.hyperparams['task_name']}' is not valid."
            )

        self.mask_fns = self.hyperparams.get("mask_fns", None)
        self.mask_kwargs = self.hyperparams.get("mask_kwargs", None)
        self.mask_num_channels = self.hyperparams.get(
            "mask_num_channels", self.in_channels if self.channel_wise else 1
        )

        mask = (
            generate_mask(
                self.mask_fns,
                self.mask_kwargs,
                self.num_patches,
                self.mask_num_channels,
                self.device,
            ).flatten()
            if self.mask_fns is not None
            else None
        )

        output = self.get_latent(
            torch.zeros(
                (2, self.in_channels, self.crop_size[0], self.crop_size[1]),
                device=self.device,
            ),
            mask,
        )
        if isinstance(output, Sequence):
            output = output[0]
        output = output.reshape(2, -1)
        if self.projector is not None:
            output = self.projector(output)
        out_dim = output.shape[1]

        if self.mean_patches:
            out_dim = output.view(2, self.num_patches * self.out_channels, -1).shape[-1]

        self.classifier = Linear(
            out_dim
            if not self.channel_wise or not self.mean_patches
            else out_dim * self.out_channels,
            self.num_classes,
        )
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        self.classifier_loss = (
            CrossEntropyLoss() if not self.multi_label else BCEWithLogitsLoss()
        )

    def configure_sharded_model(self):
        """Configures the model for sharded training."""
        pass
        # self.create_model()

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with Tile2Vec.

        Keyword Args:
            sensor: type of sensor
            bands: which bands of sensor
            encoder_name: either "resnet18" or "resnet50"
            imagenet_pretrained: bool indicating whether to use imagenet pretrained
                weights

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(dict[str, Any], self.hparams)

        self.config_task()

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.trainer is None:
            return {}

        optimizer_name = self.hyperparams.get("optimizer", "SGD")
        optimizer_class = (
            getattr(optim, optimizer_name) if optimizer_name != "ADAMW" else FusedAdam
        )
        lr = self.hyperparams.get("lr", 1e-3)
        effective_batch_size = (
            self.batch_size
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_devices
        )
        actual_lr = lr * effective_batch_size / 256
        optimizer_kwargs = self.hyperparams.get("optimizer_kwargs", {})

        if self.layer_wise_decay:
            params = param_groups_lrd(self.encoder.encoder)
            params += [
                {"params": p, "weight_decay": optimizer_kwargs["weight_decay"]}
                for p in self.classifier.parameters()
            ]
            optimizer = optimizer_class(params=params, lr=actual_lr, **optimizer_kwargs)
        else:
            optimizer = optimizer_class(
                params=self.trainer.model.parameters(), lr=actual_lr, **optimizer_kwargs
            )

        lr_min = self.hyperparams.get("lr_min", 1e-6)
        warmup_lr_init = self.hyperparams.get("warmup_lr_init", 1.5e-7)
        num_warmup = self.hyperparams.get("num_warmup", 10)

        if self.layer_wise_decay:
            scheduler = LayerWiseDecayScheduler(
                optimizer=optimizer,
                lr=actual_lr,
                min_lr=lr_min,
                num_warmup=num_warmup
                * self.num_batches
                // self.trainer.accumulate_grad_batches,
                max_epochs=self.trainer.max_epochs
                * self.num_batches
                // self.trainer.accumulate_grad_batches,
            )
        else:
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=self.trainer.max_epochs
                * self.num_batches
                // self.trainer.accumulate_grad_batches,
                lr_min=lr_min,
                cycle_mul=1.0,
                cycle_limit=1,
                warmup_t=num_warmup
                * self.num_batches
                // self.trainer.accumulate_grad_batches,
                warmup_lr_init=warmup_lr_init,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # Fix for timm scheduler: https://github.com/Lightning-AI/lightning/issues/5555
    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric: Any | None
    ) -> None:
        """Step the learning rate scheduler."""
        if metric is None:
            scheduler.step(epoch=self.global_step)
        else:
            scheduler.step(metric, epoch=self.global_step)

    def shared_step(
        self, stage: str = "train", *args: Any, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Perform a step of the model."""
        batch = args[0]
        x = batch["image"] if isinstance(batch, dict) else batch[0]
        x = x.permute(0, 3, 1, 2)  # Only ffcv, wrong order by mistake
        x = x[:, [3, 4, 5, 6, 7, 8, 9, 10, 12, 13]]
        # x = x[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]]
        y = batch["label"] if isinstance(batch, dict) else batch[1]
        y = y.squeeze()

        with torch.no_grad():
            x = (
                self.augment(x, stage)
                if x.dtype != torch.bfloat16
                else self.augment(x.half(), stage).bfloat16()
            )
            mask = (
                generate_mask(
                    self.mask_fns,
                    self.mask_kwargs,
                    self.num_patches,
                    self.mask_num_channels,
                    x.device,
                ).flatten()
                if self.mask_fns is not None
                else None
            )

        x = self.get_latent(x, mask)
        loss = self.evaluate_classification(x, y, stage)

        return {"loss": loss}

    def get_latent(self, x: Tensor, mask: Tensor) -> Tensor:
        """TODO: Docstring."""
        B, *_ = x.shape

        if self.channel_wise:
            x = x.flatten(0, 1).unsqueeze(1)  # Flatten per channel

        # TODO: Change
        item = {
            "input": x,
            "encoder_channels": torch.arange(
                self.in_channels, device=x.device
            ).tolist(),
            "mask": mask,
        }

        latent: Tensor = self.encoder(item)
        if isinstance(latent, Sequence):
            latent = latent[0]
        latent = latent.reshape(B, -1)

        if self.projector is not None:
            latent = self.projector(latent)

        return latent.squeeze()

    def classify(self, latent: Tensor) -> Tensor:
        """Classify the input tensor."""
        if not self.mean_patches:
            y_hat = self.classifier(latent)
            return cast(Tensor, y_hat)

        B, *_ = latent.shape
        if not self.channel_wise:
            latent = latent.view(B, self.num_patches, -1)
        else:
            latent = (
                latent.view(B, self.out_channels, self.num_patches, -1)
                .transpose(1, 2)
                .flatten(-2)
            )

        latent = latent.mean(dim=1)
        y_hat = self.classifier(latent)

        return cast(Tensor, y_hat)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a train step of the model."""
        item = self.shared_step("train", *args, **kwargs)

        return item["loss"]

    def training_epoch_end(self, outputs):
        self.metrics["train"].reset()

    def validation_epoch_end(self, outputs):
        self.metrics["val"].reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a validation step of the model."""
        _ = self.shared_step("val", *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a test step of the model."""
        _ = self.shared_step("test", *args, **kwargs)

        # metrics: dict[str, Tensor] = {}
        # if self.channel_wise:
        #     metrics = self.evaluate_dimensionality(item["latent"])

        # return metrics

    def evaluate_classification(
        self, latent: Tensor, y: Tensor, stage: str = "train"
    ) -> Tensor:
        """TODO: Docstring."""
        y_hat = self.classify(latent)

        if self.multi_label:
            loss = self.classifier_loss(y_hat, y.float())
            y_hat = y_hat.softmax(dim=-1)
        else:
            loss = self.classifier_loss(y_hat, y)

        metrics = self.metrics[stage](y_hat.detach(), y.detach())
        metrics |= {f"{stage}_loss": loss}

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        return cast(Tensor, loss)

    def evaluate_dimensionality(self, latent: Tensor) -> dict[str, Tensor]:
        """Evaluate the dimensionality of the embeddings using PCA."""
        B, *_ = latent.shape

        latent = latent.view(B, self.num_patches, -1).flatten(0, 1)
        latent_normalized = torch.nn.functional.normalize(latent, dim=-1)
        cov_latent = torch.cov(latent_normalized.T)
        svdvals_latent = torch.linalg.svdvals(cov_latent.float())
        svdvals_latent = svdvals_latent.log().sort(descending=True)[0]

        return {"svdvals_latent": svdvals_latent}

    # def test_epoch_end(
    #     self,
    #     outputs: list[Tensor | dict[str, Any]] | list[list[Tensor | dict[str, Any]]],
    # ) -> None:
    #     """TODO: Docstring."""
    #     if not (
    #         len(outputs)
    #         and isinstance(outputs[0], dict)
    #         and "svdvals_latent" in outputs[0].keys()
    #     ):
    #         return

    #     svdvals: list[Tensor] = []
    #     for output in outputs:
    #         svdvals_latent = output["svdvals_latent"]
    #         svdvals.append(svdvals_latent)

    #     svdvals_mean = torch.stack(svdvals).mean(0)

    #     data = [[x, y] for (x, y) in zip(range(len(svdvals_mean)), svdvals_mean)]
    #     table: wandb.data_types.Table = wandb.Table(data=data, columns=["Singular Value Rank Index", "Log of singular values"])  # type: ignore
    #     wandb.log(
    #         {
    #             "singular_values_latent": wandb.plot.line(
    #                 table,
    #                 "Singular Value Rank Index",
    #                 "Log of singular values",
    #                 title="Singular Values of Latent Embeddings",
    #             )
    #         }
    #     )
