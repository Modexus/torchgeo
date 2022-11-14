# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tile2Vec tasks."""

from typing import Any, cast

import torch
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import TripletMarginLoss
from torch.nn.modules import Linear, Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models import MaskedViT, resnet18, resnet50
from ..utils import _to_tuple

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def triplet_loss(
    anchor: Tensor,
    neighbor: Tensor,
    distant: Tensor,
    triplet_loss_fn: Module,
    l2: float = 0,
) -> Tensor:
    """Computes the triplet_loss between anchor, neighbor, and distant.

    Args:
        anchor: tensor anchor
        neighbor: tensor neighbor
        disant: tensor distant

    Returns:
        the normalized MSE between x and y
    """
    loss = triplet_loss_fn(anchor, neighbor, distant)

    if l2 > 0:
        anchor_norm = torch.linalg.norm(anchor, dim=1)
        neighbor_norm = torch.linalg.norm(neighbor, dim=1)
        distant_norm = torch.linalg.norm(distant, dim=1)

        norm = l2 * (anchor_norm + neighbor_norm + distant_norm).mean()

        loss += norm

    return cast(Tensor, loss)


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        crop_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
        """
        super().__init__()

        crop_size = crop_size or image_size
        rotations = [0.0, 90.0, 180.0, 270.0]

        self.augmentation = {
            "train": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size,
                    scale=(0.2, 1.0),
                    align_corners=False,
                    resample="BICUBIC",
                ),
                K.RandomHorizontalFlip(),
                K.ImageSequential(
                    *[
                        K.RandomRotation([rotation, rotation], p=1.0)
                        for rotation in rotations
                    ],
                    random_apply=1,
                ),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
            ),
        }

    def forward(self, x: Tensor, stage: str | None = None) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        return cast(Tensor, self.augmentation[stage](x))


class Tile2Vec(Module):
    """Tile2Vec implementation.

    See https://aaai.org/ojs/index.php/AAAI/article/view/4288 for more details
    (and please cite it if you use it in your own work).
    """

    def __init__(
        self,
        encoder: Module,
        projector: Module | None = None,
        mean_patches: bool = False,
        patch_size: int = 16,
    ) -> None:
        """Sets up a model for pre-training with tile2vec using projection heads.

        Args:
            encoder: the encoder to pretrain using tile2vec
            projector: the projector to project the patches
            mean_patches: whether to use patchwise projection
            patch_size: the size of the vit patches
        """
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.mean_patches = mean_patches
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        x = self.encoder(x).squeeze()

        if not self.mean_patches:
            return x

        if self.projector is not None:
            x = self.projector(x)

        x = x.mean(1)

        return x


class Tile2VecTask(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self, **kwargs: Any) -> None:
        """TODO: Docstring."""
        self.image_size: int = self.hyperparams.get("image_size", 256)
        self.crop_size: int = self.hyperparams.get("crop_size", 224)
        self.mean_patches = self.hyperparams.get("mean_patches", False)
        self.patch_size = self.hyperparams.get("patch_size", 16)

        pretrained = self.hyperparams.get("pretrained", False)
        imagenet_pretrained = self.hyperparams.get("imagenet_pretrained", False)
        sensor = self.hyperparams["sensor"]
        bands = self.hyperparams.get("bands", "all")

        encoder: Module
        projector: Module | None = None
        if self.hyperparams["encoder_name"] == "resnet18":
            encoder = resnet18(
                sensor=sensor,
                bands=bands,
                pretrained=pretrained,
                imagenet_pretrained=imagenet_pretrained,
            )
            encoder = Sequential(*(list(encoder.children())[:-1]))
        elif self.hyperparams["encoder_name"] == "resnet50":
            encoder = resnet50(
                sensor=sensor,
                bands=bands,
                pretrained=pretrained,
                imagenet_pretrained=imagenet_pretrained,
            )
            encoder = Sequential(*(list(encoder.children())[:-1]))
        elif self.hyperparams["encoder_name"] == "vit":
            embed_dim = self.hyperparams.get("embed_dim", 512)
            projection_dim = self.hyperparams.get("projection_dim", 2048)

            encoder = MaskedViT(
                sensor=sensor,
                bands=bands,
                image_size=self.crop_size,
                patch_size=self.patch_size,
                embed_dim=embed_dim,
                depth=self.hyperparams.get("depth", 24),
                num_heads=self.hyperparams.get("num_heads", 16),
                dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
                dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
            )

            projector = Linear(embed_dim, projection_dim)
            projector.weight.data.normal_(mean=0.0, std=0.01)
            projector.bias.data.zero_()
        else:
            raise ValueError(
                f"Encoder type '{self.hyperparams['encoder_name']}' is not valid."
            )

        self.model = Tile2Vec(encoder, projector, self.mean_patches, self.patch_size)
        self.triplet_loss_fn = TripletMarginLoss(
            margin=self.hyperparams.get("margin", 0.1)
        )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with Tile2Vec.

        Keyword Args:
            sensor: type of sensor
            bands: which bands of sensor
            encoder_name: either "resnet18" or "resnet50"
            imagenet_pretraining: bool indicating whether to use imagenet pretrained
                weights

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(dict[str, Any], self.hparams)

        self.config_task()

    def setup(self, stage: str | None = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        image_size = _to_tuple(self.image_size)
        crop_size = _to_tuple(self.crop_size)

        self.augment = self.hyperparams.get(
            "augment_fn", Augmentations(image_size, crop_size)
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "AdamW"))
        lr = self.hyperparams.get("lr", 1e-3)
        actual_lr = lr * self.hyperparams.get("batch_size", 64) / 256
        weight_decay = self.hyperparams.get("weight_decay", 0.05)
        betas = self.hyperparams.get("betas", (0.9, 0.95))
        optimizer = optimizer_class(
            self.parameters(), lr=actual_lr, weight_decay=weight_decay, betas=betas
        )

        if self.trainer is None:
            return {}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    self.trainer.max_epochs,
                    self.hyperparams.get("min_lr", 1e-6),
                ),
                "monitor": "val_loss",
            },
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def get_embeddings(self, x: Tensor) -> Tensor:
        """Get the embeddings."""
        B, *_ = x.shape

        x = self.model(x)
        x = x.view(B, -1)

        return x

    def shared_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]

        with torch.no_grad():
            anchor = self.augment(x[:, 0])
            neighbor = self.augment(x[:, 1])
            distant = self.augment(x[:, 2])

        anchor_embeddings = self.get_embeddings(anchor)
        neighbor_embeddings = self.get_embeddings(neighbor)
        distant_embeddings = self.get_embeddings(distant)

        loss = triplet_loss(
            anchor_embeddings,
            neighbor_embeddings,
            distant_embeddings,
            self.triplet_loss_fn,
            self.hyperparams.get("l2", 0),
        )

        return loss

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        loss = self.shared_step(*args, **kwargs)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        loss = self.shared_step(*args, **kwargs)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
