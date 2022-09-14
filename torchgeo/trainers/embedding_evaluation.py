"""Embedding classifciation task."""

from os.path import isfile
from typing import Any, Sequence, cast

import torch
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Identity,
    Linear,
    Module,
    Sequential,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from .vicreg import VICRegTask


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
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size, align_corners=False, resample="BICUBIC"
                ),
                K.RandomHorizontalFlip(),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
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
        self.B = self.hyperparams.get("batch_size", 256)
        self.multi_label = self.hyperparams.get("multi_label", False)

        image_size = _to_tuple(self.hyperparams["image_size"])
        crop_size = _to_tuple(self.hyperparams.get("crop_size", image_size))
        num_classes = self.hyperparams["num_classes"]

        self.num_patches = (crop_size[0] // self.patch_size) * (
            crop_size[1] // self.patch_size
        )

        self.projector: Module | None = None
        if self.hyperparams["task_name"] == "tile2vec":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = Tile2VecTask.load_from_checkpoint(
                    checkpoint_path=self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = Tile2VecTask(**self.hyperparams)
            task.freeze()
            if "resnet" in self.hyperparams["encoder_name"]:
                self.encoder = task.model.encoder
            else:
                self.encoder = task.model
        elif self.hyperparams["task_name"] == "byol":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = BYOLTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = BYOLTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder.model
        elif self.hyperparams["task_name"] == "vicreg":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = VICRegTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = VICRegTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "mae":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = MAETask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MAETask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "cae":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = CAETask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = CAETask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder

        elif self.hyperparams["task_name"] == "msn":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = MSNTask.load_from_checkpoint(self.hyperparams["checkpoint_path"])
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MSNTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model
        elif self.hyperparams["task_name"] == "msae":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = MSAETask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = MSAETask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "identity":
            self.encoder = Identity()  # type: ignore[no-untyped-call]
        else:
            raise ValueError(
                f"Task type '{self.hyperparams['task_name']}' is not valid."
            )

        output = self.get_latent(
            torch.zeros((2, self.in_channels, crop_size[0], crop_size[1]))
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
            num_classes,
        )
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        self.classifier_loss = (
            CrossEntropyLoss() if not self.multi_label else BCEWithLogitsLoss()
        )

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    average="micro",
                    multiclass=False if self.multi_label else None,
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"],
                    average="macro",
                    multiclass=False if self.multi_label else None,
                ),
                "F1Score": FBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    multiclass=False if self.multi_label else None,
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

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }

        self.augment = self.hyperparams.get(
            "augment_fn", Augmentations(image_size, crop_size)
        )

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

        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "SGD"))
        lr = self.hyperparams.get("lr", 2e-2)
        actual_lr = lr * self.B / 256
        weight_decay = self.hyperparams.get("weight_decay", 1e-6)
        momentum = self.hyperparams.get("momentum", 0.9)
        optimizer = optimizer_class(
            self.parameters(),
            lr=actual_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def get_latent(self, x: Tensor) -> Tensor:
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
        # y_hat = y_hat.mean(dim=1)

        return cast(Tensor, y_hat)

    def shared_step(
        self, stage: str = "train", *args: Any, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Perform a step of the model."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        with torch.no_grad():
            aug = self.augment(x, "train")
            latent = self.get_latent(aug)

        loss = self.evaluate_classification(latent, y, stage)

        return {"loss": loss, "latent": latent}

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a train step of the model."""
        item = self.shared_step("train", *args, **kwargs)

        return item["loss"]

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a validation step of the model."""
        self.shared_step("val", *args, **kwargs)

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

        metrics = self.metrics[stage](y_hat, y)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=self.B,
        )
        self.log_dict(metrics, on_step=stage != "val", on_epoch=True, batch_size=self.B)

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
