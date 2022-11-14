# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""VICReg tasks."""

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from kornia import augmentation as K
from kornia import filters
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import init
from torch.nn.modules import BatchNorm1d, Conv2d, Linear, Module, ReLU, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchgeo.models import MaskedViT, resnet18, resnet50

from ..utils import _to_tuple
from .utils import unpatchify

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


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
                    size=crop_size,
                    scale=(0.2, 1.0),
                    align_corners=False,
                    resample="BICUBIC",
                ),
                K.RandomHorizontalFlip(),
                K.ImageSequential(
                    filters.GaussianBlur2d((3, 3), (1.5, 1.5)),
                    random_apply_weights=[0.1],
                ),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
                K.ImageSequential(
                    filters.GaussianBlur2d((3, 3), (1.5, 1.5)),
                    random_apply_weights=[0.1],
                ),
            ),
        }

    def forward(self, x: Tensor, stage: Optional[str]) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        return cast(Tensor, self.augmentation[stage](x))


def vic_loss(
    x: Tensor,
    y: Tensor,
    invar_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """TODO: Docstring."""
    invar_loss = invar_term(x, y)
    var_loss = var_term(x, y)
    cov_loss = cov_term(x, y)

    loss = invar_loss * invar_coeff + var_loss * var_coeff + cov_loss * cov_coeff

    return loss, invar_loss, var_loss, cov_loss


def invar_term(x: Tensor, y: Tensor) -> Tensor:
    """Invariance term for the VICReg loss."""
    return F.mse_loss(x, y)


def var_term(x: Tensor, y: Tensor) -> Tensor:
    """Variance term for the VICReg loss."""
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    return std_loss


def cov_term(x: Tensor, y: Tensor) -> Tensor:
    """Covariance term for the VICReg loss."""
    batch_size = x.shape[0]
    num_features = x.shape[1]

    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)

    cov_loss_x = off_diagonal(cov_x).pow_(2).sum() / num_features
    cov_loss_y = off_diagonal(cov_y).pow_(2).sum() / num_features

    return cov_loss_x + cov_loss_y


class VICReg(Module):
    """VICReg implementation."""

    def __init__(
        self,
        model: Module,
        projector: Module,
        mean_patches: bool = True,
        patch_size: int = 16,
    ) -> None:
        """Setup the VICReg model.

        Args:
            model: The encoder model
            projector: The projector model
            mean_patches: Whether to mean patches
            patch_size: The patch size
        """
        super().__init__()

        self.encoder = model
        self.projector = projector
        self.mean_patches = mean_patches
        self.patch_size = patch_size

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass of the encoder model through the projector model."""
        B, *_ = x.shape

        embeddings = self.encoder(x, *args, **kwargs)

        embeddings = embeddings.view(B, -1)
        embeddings = self.projector(embeddings)

        if self.mean_patches:
            embeddings = embeddings.mean(1)

        return cast(Tensor, embeddings)


class Projector(Module):
    """TODO:Docstring."""

    def __init__(
        self, num_layers: int, embedding_dim: int, projection_dim: int
    ) -> None:
        """TODO: Docstring."""
        super().__init__()

        layers: List[Module] = []

        layers.append(Linear(embedding_dim, projection_dim))
        layers.append(BatchNorm1d(projection_dim))  # type: ignore[no-untyped-call]
        layers.append(ReLU(True))

        for _ in range(num_layers - 2):
            layers.append(Linear(projection_dim, projection_dim))
            layers.append(BatchNorm1d(projection_dim))  # type: ignore[no-untyped-call]
            layers.append(ReLU(True))

        layers.append(Linear(projection_dim, projection_dim, bias=False))

        self.projector = Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize the weights."""
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """TODO: Docstring."""
        return cast(Tensor, self.projector(x))


class VICRegTask(LightningModule):
    """Class for pre-training any PyTorch model using VICReg."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        pretrained = self.hyperparams.get("pretrained", False)
        imagenet_pretrained = self.hyperparams.get("imagenet_pretrained", False)
        sensor = self.hyperparams["sensor"]
        bands = self.hyperparams.get("bands", "all")
        encoder = None

        mean_patches = self.hyperparams.get("mean_patches", False)
        patch_size = self.hyperparams.get("patch_size", 16)
        embed_dim = self.hyperparams.get("embed_dim", 512)
        projection_dim = self.hyperparams.get("projection_dim", 2048)

        self.image_size = self.hyperparams.get("image_size", (256, 256))
        self.crop_size = self.hyperparams.get("crop_size", (224, 224))

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
            encoder = MaskedViT(
                sensor=sensor,
                bands=bands,
                image_size=self.crop_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=self.hyperparams.get("depth", 24),
                num_heads=self.hyperparams.get("num_heads", 16),
                dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
                dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
            )
        else:
            raise ValueError(
                f"Encoder type '{self.hyperparams['encoder_name']}' is not valid."
            )

        projector = Projector(
            num_layers=self.hyperparams.get("projector_num_layers", 3),
            embedding_dim=embed_dim,
            projection_dim=projection_dim,
        )

        self.model = VICReg(encoder, projector, mean_patches, patch_size)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with BYOL.

        Keyword Args:
            in_channels: number of channels on the input imagery
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

    def setup(self, stage: Optional[str] = None) -> None:
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

    def shared_step(
        self, stage: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> Tensor:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]

        with torch.no_grad():
            aug1, aug2 = self.augment(x, stage), self.augment(x, stage)

        pred1, pred2 = self.forward(aug1), self.forward(aug2)

        loss, invar_loss, var_loss, cov_loss = vic_loss(
            pred1,
            pred2,
            invar_coeff=self.hyperparams.get("invar_coeff", 0.5),
            var_coeff=self.hyperparams.get("var_coeff", 0.5),
            cov_coeff=self.hyperparams.get("cov_coeff", 0.5),
        )

        if stage is not None:
            self.log_dict(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_invar_loss": invar_loss,
                    f"{stage}_var_loss": var_loss,
                    f"{stage}_cov_loss": cov_loss,
                },
                on_step=stage != "val",
                on_epoch=True,
            )

        return loss

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        loss = self.shared_step("train", *args, **kwargs)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        self.shared_step("val", *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""


def off_diagonal(x: Tensor) -> Tensor:
    """Indices of off-diagonal elements in a 2D tensor."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def exclude_bias_and_norm(p: Tensor) -> bool:
    """Exclude bias and norm layers from weight decay."""
    return p.ndim == 1


class LARS(optim.Optimizer):
    """Implements LARS (LARS: Least Angle Regression) as an optimizer."""

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ) -> None:
        """Initialize LARS."""
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)  # type: ignore[no-untyped-call]
                    update_norm = torch.norm(dp)  # type: ignore[no-untyped-call]
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

        return loss
