# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tile2Vec tasks."""

from typing import Any, Dict, Optional, Tuple, cast

import torch
from kornia import augmentation as K
from kornia.augmentation.container.image import ImageSequential
from kornia.geometry.transform import Rotate
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn.functional import relu
from torch.nn.modules import Module, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet

from torchgeo.models import resnet18, resnet50

from ..utils import _to_tuple

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def triplet_loss(
    anchor: Tensor,
    neighbor: Tensor,
    distant: Tensor,
    margin: float = 0.1,
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
    positive = torch.sqrt(((anchor - neighbor) ** 2).sum(dim=1))
    negative = torch.sqrt(((anchor - distant) ** 2).sum(dim=1))

    distance = positive - negative + margin
    loss = relu(distance, inplace=True).mean()

    if l2 > 0:
        anchor_norm = torch.linalg.norm(anchor, dim=1)
        neighbor_norm = torch.linalg.norm(neighbor, dim=1)
        distant_norm = torch.linalg.norm(distant, dim=1)

        norm = l2 * (anchor_norm + neighbor_norm + distant_norm).mean()

        loss += norm

    return loss


# TODO: This isn't _really_ applying the augmentations from SimCLR as we have
# multispectral imagery and thus can't naively apply color jittering or grayscale
# conversions. We should think more about what makes sense here.
class Augmentations(Module):
    """A module for applying SimCLR augmentations.

    SimCLR was one of the first papers to show the effectiveness of random data
    augmentation in self-supervised-learning setups. See
    https://arxiv.org/pdf/2002.05709.pdf for more details.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
        """
        super().__init__()
        self.size = image_size
        self.rotations = (torch.tensor([90.0]) * torch.tensor([0, 1, 2, 3])).to(device)

        self.augmentation = Sequential(
            K.Resize(size=image_size, align_corners=False),
            K.RandomHorizontalFlip(),
            ImageSequential(
                *[Rotate(rotation) for rotation in self.rotations],
                random_apply=1,
                same_on_batch=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        return cast(Tensor, self.augmentation(x))


class Tile2Vec(Module):
    """Tile2Vec implementation.

    See https://aaai.org/ojs/index.php/AAAI/article/view/4288 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(self, model: Module, **kwargs: Any) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            augment_fn: an instance of a module that performs data augmentation
        """
        super().__init__()

        self.encoder = model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.encoder(x).squeeze())


class Tile2VecTask(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        pretrained = self.hyperparams.get("pretrained", False)
        imagenet_pretraining = self.hyperparams.get("imagenet_pretraining", False)
        sensor = self.hyperparams["sensor"]
        bands = self.hyperparams.get("bands", "all")
        encoder = None

        if self.hyperparams["encoder_name"] == "resnet18":
            if imagenet_pretraining:
                encoder = _resnet("resnet18", BasicBlock, [2, 2, 2, 2], True, True)
            else:
                encoder = resnet18(
                    sensor=sensor,
                    bands=bands,
                    block=BasicBlock,
                    layers=[2, 2, 2, 2, 2],
                    pretrained=pretrained,
                )
        elif self.hyperparams["encoder_name"] == "resnet50":
            if imagenet_pretraining:
                encoder = _resnet("resnet50", BasicBlock, [3, 4, 6, 3], True, True)
            else:
                encoder = resnet50(
                    sensor=sensor, bands=bands, block=BasicBlock, pretrained=pretrained
                )
        else:
            raise ValueError(
                f"Encoder type '{self.hyperparams['encoder_name']}' is not valid."
            )

        encoder = Sequential(*(list(encoder.children())[:-1]))

        self.model = Tile2Vec(encoder)

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
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

    def setup(self, stage: Optional[str] = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/13108
        # current workaround
        if self.trainer is not None:
            device = self.trainer.strategy.root_device

        patch_size = self.hyperparams.get("patch_size", (256, 256))
        patch_size = _to_tuple(patch_size)

        self.augment = self.hyperparams.get(
            "augment_fn", Augmentations(patch_size, device)
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "Adam"))
        lr = self.hyperparams.get("lr", 1e-3)
        weight_decay = self.hyperparams.get("weight_decay", 0)
        betas = self.hyperparams.get("betas", (0.5, 0.999))
        optimizer = optimizer_class(
            self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams.get("learning_rate_schedule_patience", 0),
                ),
                "monitor": "train_loss",
            },
        }

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]

        with torch.no_grad():
            anchor = self.augment(x[:, 0])
            neighbor = self.augment(x[:, 1])
            distant = self.augment(x[:, 2])

        pred1, pred2, pred3 = (
            self.forward(anchor),
            self.forward(neighbor),
            self.forward(distant),
        )

        loss = triplet_loss(
            pred1,
            pred2,
            pred3,
            self.hyperparams.get("margin", 0.1),
            self.hyperparams.get("l2", 0),
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]

        anchor = self.augment(x[:, 0])
        neighbor = self.augment(x[:, 1])
        distant = self.augment(x[:, 2])

        pred1, pred2, pred3 = (
            self.forward(anchor),
            self.forward(neighbor),
            self.forward(distant),
        )

        loss = triplet_loss(
            pred1,
            pred2,
            pred3,
            self.hyperparams.get("margin", 0.1),
            self.hyperparams.get("l2", 0),
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
