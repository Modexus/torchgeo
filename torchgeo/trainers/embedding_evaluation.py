"""Embedding classifciation task."""

from os.path import isfile
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import torch
import wandb
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import CrossEntropyLoss, Identity, Linear, Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MetricCollection

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
        image_size: Tuple[int, int] = (256, 256),
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
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

    def forward(self, x: Tensor, stage: Optional[str]) -> Tensor:
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
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.channel_wise = self.hyperparams.get("channel_wise", False)
        self.in_channels = self.hyperparams.get("in_channels", 4)
        self.out_channels = self.hyperparams.get("out_channels", self.in_channels)
        self.mean_patches = self.hyperparams.get("mean_patches", False)
        self.patch_size = self.hyperparams.get("patch_size", 16)

        image_size = _to_tuple(self.hyperparams["image_size"])
        crop_size = _to_tuple(self.hyperparams.get("crop_size", image_size))
        num_classes = self.hyperparams["num_classes"]

        self.projector: Optional[Module] = None
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

        output = self.get_embeddings(
            torch.zeros((2, self.in_channels, crop_size[0], crop_size[1]))
        )
        if isinstance(output, Sequence):
            output = output[0]
        output = output.reshape(2, -1)

        if self.projector is not None:
            output = self.projector(output)

        out_dim = output.shape[1]

        if self.mean_patches:
            self.num_patches = (crop_size[0] // self.patch_size) * (
                crop_size[1] // self.patch_size
            )
            out_dim = output.view(2, self.num_patches * self.out_channels, -1).shape[-1]

        self.classifier = Linear(
            out_dim if not self.channel_wise else out_dim * self.out_channels,
            num_classes,
        )
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        self.classifier_loss = CrossEntropyLoss()

        self.metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="macro"
                ),
                "JaccardIndex": JaccardIndex(
                    num_classes=self.hyperparams["num_classes"]
                ),
                "F1Score": FBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                ),
            }
        )

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
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "SGD"))
        lr = self.hyperparams.get("lr", 2e-2)
        weight_decay = self.hyperparams.get("weight_decay", 1e-6)
        momentum = self.hyperparams.get("momentum", 0.9)
        optimizer = optimizer_class(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)  # type: ignore

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def get_embeddings(self, x: Tensor) -> Tensor:
        """TODO: Docstring."""
        B, *_ = x.shape

        if self.channel_wise:
            x = x.flatten(0, 1).unsqueeze(1)  # Reorder per channel

        # TODO: Change
        item = {"input": x, "encoder_channels": [0, 1, 2, 3]}

        embeddings: Tensor = self.encoder(item)
        if isinstance(embeddings, Sequence):
            embeddings = embeddings[0]
        embeddings = embeddings.reshape(B, -1)

        if self.projector is not None:
            embeddings = self.projector(embeddings)

        return embeddings.squeeze()

    def classify(self, embeddings: Tensor) -> Tensor:
        """Classify the input tensor."""
        if not self.mean_patches:
            y_hat = self.classifier(embeddings)
            return cast(Tensor, y_hat)

        B, *_ = embeddings.shape

        if not self.channel_wise:
            embeddings = embeddings.view(B, self.num_patches, -1)
        else:
            embeddings = (
                embeddings.view(B, self.out_channels, self.num_patches, -1)
                .transpose(1, 2)
                .flatten(-2)
            )

        y_hat = self.classifier(embeddings)
        y_hat = y_hat.mean(dim=1)

        return cast(Tensor, y_hat)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        with torch.no_grad():
            aug = self.augment(x, "train")
            embeddings = self.get_embeddings(aug)

        y_hat = self.classify(embeddings)

        loss = self.classifier_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        aug = self.augment(x, "val")
        embeddings = self.get_embeddings(aug)

        metrics = self.evaluate_classification(embeddings, y, "val")
        self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=x.shape[0])

        return metrics

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        aug = self.augment(x, "val")
        embeddings = self.get_embeddings(aug)

        metrics = self.evaluate_classification(embeddings, y, "test")
        self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=x.shape[0])

        if self.channel_wise:
            metrics |= self.evaluate_dimensionality(embeddings)

        return metrics

    def evaluate_classification(
        self, embeddings: Tensor, y: Tensor, stage: Optional[str] = None
    ) -> Dict[str, Tensor]:
        """TODO: Docstring."""
        y_hat = self.classify(embeddings)
        metrics = self.metrics(y_hat, y)

        if stage:
            metrics = {f"{stage}_{k}": v for k, v in metrics.items()}
            metrics[f"{stage}_loss"] = metrics[f"{stage}_OverallAccuracy"]

        return cast(Dict[str, Tensor], metrics)

    def evaluate_dimensionality(self, embeddings: Tensor) -> Dict[str, Tensor]:
        """Evaluate the dimensionality of the embeddings using PCA."""
        B, *_ = embeddings.shape

        embeddings = embeddings.view(B, self.num_patches, -1).flatten(0, 1)
        embeddings_normalized = torch.nn.functional.normalize(embeddings, dim=-1)
        cov_embeddings = torch.cov(embeddings_normalized.T)
        svdvals_embeddings = torch.linalg.svdvals(cov_embeddings.float())
        svdvals_embeddings = svdvals_embeddings.log().sort(descending=True)[0]

        return {"svdvals_embeddings": svdvals_embeddings}

    def test_epoch_end(
        self,
        outputs: list[Tensor | Dict[str, Any]] | list[list[Tensor | Dict[str, Any]]],
    ) -> None:
        """TODO: Docstring."""
        svdvals: list[Tensor] = []

        for output in cast(list[Dict[str, Tensor]], outputs):
            svdvals_embeddings = output["svdvals_embeddings"]
            svdvals.append(svdvals_embeddings)

        svdvals_mean = torch.stack(svdvals).mean(0)

        data = [[x, y] for (x, y) in zip(range(len(svdvals_mean)), svdvals_mean)]
        table: wandb.data_types.Table = wandb.Table(data=data, columns=["Singular Value Rank Index", "Log of singular values"])  # type: ignore
        wandb.log(
            {
                "singular_values_embeddings": wandb.plot.line(
                    table,
                    "Singular Value Rank Index",
                    "Log of singular values",
                    title="Singular Values of Embeddings",
                )
            }
        )
