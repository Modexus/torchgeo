# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""VICReg tasks."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

import wandb

from ..models import MaskedAutoencoderViT
from ..utils import _to_tuple
from .utils import generate_mask, pad_imgs_dims, patchify, unpatchify

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def latent_loss(latent1: Tensor, latent2: Tensor, mean_patches: bool = True) -> Tensor:
    """Compute embedding loss."""
    if mean_patches:
        return F.mse_loss(latent1.mean(dim=-1), latent2.mean(dim=-1))

    return F.mse_loss(latent1, latent2)


def masked_reconstruction_loss(
    target: Tensor, pred: Tensor, mask: Tensor, patch_size: int
) -> Tensor:
    """Compute masked reconstruction loss."""
    B, *_ = pred.shape

    target = patchify(target, patch_size)
    target = target.view(pred.shape)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    num_hidden = mask.sum()
    if num_hidden == 0:
        return cast(Tensor, loss.mean())

    loss = (loss * mask).sum() / (B * num_hidden)  # mean loss on removed patches

    return cast(Tensor, loss)


def reconstruction_loss(target: Tensor, pred: Tensor, patch_size: int) -> Tensor:
    """Compute masked reconstruction loss."""
    B, *_ = pred.shape

    target = patchify(target, patch_size)
    target = target.view(pred.shape)

    loss = (pred - target) ** 2
    loss = loss.mean()  # [N, L], mean loss per patch

    return cast(Tensor, loss)


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = (256, 256),
        crop_size: int | tuple[int, int] | None = None,
        patch_size: int = 16,
        embed_dim: int | None = 512,
        scale: float | tuple[float, float] = (0.6, 1.0),
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
            patch_size: Integer defining the patch size
            scale: Float or tuple of floats defining the scale range
        """
        super().__init__()

        crop_size = crop_size or image_size
        self.patch_size = patch_size
        self.embed_patch_size = (
            min(int(embed_dim**0.5), self.patch_size)
            if embed_dim is not None
            else self.patch_size
        )
        scale = _to_tuple(scale)

        self.augmentation = {
            "train": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size, scale=scale, align_corners=False, resample="BICUBIC"
                ),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
            ),
        }

    def forward(
        self, x: Tensor, stage: str | None = None, mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        return cast(Tensor, self.augmentation[stage](x))


class MSAETask(LightningModule):
    """Class for pre-training any PyTorch model using Masked Siamese Network."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.image_size = self.hyperparams.get("image_size", 256)
        self.crop_size = self.hyperparams.get("crop_size", 224)
        self.patch_size = self.hyperparams.get("patch_size", 16)
        self.mean_patches = self.hyperparams.get("mean_patches", False)
        self.channel_wise = self.hyperparams.get("channel_wise", False)
        self.channel_shuffle = self.hyperparams.get("channel_shuffle", False)
        self.embed_dim = self.hyperparams.get("embed_dim", 1024)
        self.embed_token = self.hyperparams.get("embed_token", False)
        self.embed_token_reduction = self.hyperparams.get(
            "embed_token_reduction", False
        )
        self.num_in_channels = self.hyperparams.get("num_in_channels", 3)
        self.num_out_channels = self.hyperparams.get("num_out_channels", 3)
        self.B = self.hyperparams.get("batch_size", 64)
        self.num_patches = (self.crop_size // self.patch_size) ** 2

        self.model = MaskedAutoencoderViT(
            sensor=self.hyperparams["sensor"],
            bands=self.hyperparams.get("bands", "all"),
            image_size=self.crop_size,
            patch_size=self.patch_size,
            channel_wise=self.channel_wise,
            embed_dim=self.embed_dim,
            depth=self.hyperparams.get("depth", 24),
            num_heads=self.hyperparams.get("num_heads", 16),
            dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
            dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
        )

        self.mask_fns = self.hyperparams.get(
            "mask_fn", ["random_masking"]  # ["focal_masking", "random_masking"]
        )
        self.mask_kwargs = self.hyperparams.get(
            "mask_kwargs", {"random_mask_num_keep": 256, "random_mask_probability": 1.0}
        )

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
        self.hyperparams = cast(dict[str, Any], self.hparams)

        self.config_task()

    def setup(self, stage: str | None = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        image_size = _to_tuple(self.image_size)
        crop_size = _to_tuple(self.crop_size)

        self.augment = Augmentations(
            image_size, crop_size, self.patch_size, self.embed_dim
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.trainer is None:
            return {}

        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "AdamW"))
        lr = self.hyperparams.get("lr", 1e-3)
        actual_lr = (
            lr * self.hyperparams.get("batch_size", 64) / 256 * self.trainer.num_devices
        )
        weight_decay = self.hyperparams.get("weight_decay", 0.05)
        betas = self.hyperparams.get("betas", (0.9, 0.95))
        optimizer = optimizer_class(
            self.parameters(), lr=actual_lr, weight_decay=weight_decay, betas=betas
        )

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
        self, stage: str | None = None, *args: Any, **kwargs: Any
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """TODO: Docstring."""
        batch = args[0]
        x: Tensor = batch["image"]
        _, self.C, *_ = x.shape

        with torch.no_grad():
            target = self.augment(x, stage)

            item1 = self.get_item(target, "transform")
            item2 = self.get_item(target, "transform_prime")

        item1 = self.forward(item1)
        item2 = self.forward(item2)

        # Calculate losses
        losses = self.get_losses(item1, item2, stage)

        self.log_dict(
            losses, on_step=stage != "val", on_epoch=True, sync_dist=stage != "train"
        )

        return {"loss": losses[f"{stage}_loss"], "item1": item1, "item2": item2}

    def get_item(self, target: Tensor, transform_stage: str) -> dict[str, Tensor]:
        """Get the item for the given transform stage."""
        input = target.clone()
        mask = generate_mask(
            self.mask_fns, self.mask_kwargs, self.num_patches, self.C, input.device
        )

        encoder_channels = torch.randperm(self.C, device=target.device).tolist()[
            : self.num_in_channels
        ]
        decoder_channels = torch.randperm(self.C, device=target.device).tolist()[
            : self.num_out_channels
        ]

        input = input[:, encoder_channels]
        target = target[:, decoder_channels]
        mask = mask[encoder_channels].flatten()

        input = input.flatten(0, 1).unsqueeze(1)
        target = target.flatten(0, 1).unsqueeze(1)

        return {
            "input": input,
            "target": target,
            "mask": mask,
            "encoder_channels": encoder_channels,  # type: ignore
            "decoder_channels": decoder_channels,  # type: ignore
        }

    def get_losses(
        self, item1: dict[str, Tensor], item2: dict[str, Tensor], stage: str | None
    ) -> dict[str, Tensor]:
        """Calculate the losses for the given stage."""
        loss_rec1 = reconstruction_loss(item1["target"], item1["pred"], self.patch_size)
        loss_rec2 = reconstruction_loss(item2["target"], item2["pred"], self.patch_size)
        loss_embedding = latent_loss(
            item1["latent"], item2["latent"], self.mean_patches
        )
        loss = loss_embedding + (loss_rec1 + loss_rec2) / 2

        return {
            f"{stage}_loss": loss,
            f"{stage}_loss_rec1": loss_rec1,
            f"{stage}_loss_rec2": loss_rec2,
            f"{stage}_loss_embedding": loss_embedding,
        }

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        output = self.shared_step("train", *args, **kwargs)

        return output["loss"]

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        output = self.shared_step("val", *args, **kwargs)

        if args[1] > 0 or self.device.index != 0:
            return {}

        images1 = self.prepare_output(output["item1"])  # type: ignore
        images2 = self.prepare_output(output["item2"])  # type: ignore

        return {"images1": images1, "images2": images2}

    def prepare_output(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare the output for logging.

        Args:
            aug: tensor of data to run through the model
            aug_shuffled: tensor of data to run through the model
            pred: tensor of data to run through the model

        Returns:
            output from the model
        """
        input, target, pred, mask, latent = (
            item["input"],
            item["target"],
            item["pred"],
            item["mask"],
            item["latent"],
        )

        pred = pred.view(self.B * self.num_out_channels, self.num_patches, -1)
        mask = mask.view(self.num_in_channels, -1).repeat(self.B, 1)

        target_patch = patchify(input, self.patch_size)
        mask_expanded = (~mask.bool()).unsqueeze(-1).expand(target_patch.shape)
        masked_target = torch.zeros_like(target_patch, device=input.device)
        masked_target[mask_expanded] = target_patch[mask_expanded]

        pred = unpatchify(pred, self.patch_size)
        masked_target = unpatchify(masked_target, self.patch_size)
        latent = unpatchify(latent, int(self.embed_dim**0.5))[:, :3]

        latent /= latent.amax()
        pred /= pred.amax()

        target = target.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]
        input = input.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]
        masked_target = masked_target.view(self.B, -1, self.crop_size, self.crop_size)[
            :, :3
        ]
        pred = pred.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]

        images = pad_imgs_dims([input, masked_target, pred, target], 3)

        return {"images": images, "latent": latent}

    def validation_epoch_end(
        self,
        validation_step_outputs: list[Tensor | dict[str, Any]]
        | list[list[Tensor | dict[str, Any]]],
    ) -> None:
        """Log images."""
        if self.device.index != 0:
            return

        images1 = validation_step_outputs[0]["images1"]["images"]  # type: ignore
        latent1 = validation_step_outputs[0]["images1"]["latent"]  # type: ignore

        images2 = validation_step_outputs[0]["images2"]["images"]  # type: ignore
        latent2 = validation_step_outputs[0]["images2"]["latent"]  # type: ignore

        grid1 = make_grid(images1[:, :8].flatten(0, 1)[:, :3], nrow=8)
        images1 = wandb.Image(grid1, caption="Images")
        latent1 = latent1[:8].flatten(0, 1).unsqueeze(1)
        latent_grid1 = make_grid(latent1)
        latent_grid1 = wandb.Image(latent_grid1, caption="Latent")

        grid2 = make_grid(images2[:, :8].flatten(0, 1)[:, :3], nrow=8)
        images2 = wandb.Image(grid2, caption="Images")
        latent2 = latent2[:8].flatten(0, 1).unsqueeze(1)
        latent_grid2 = make_grid(latent2)
        latent_grid2 = wandb.Image(latent_grid2, caption="Latent")

        self.logger.experiment.log({"Images1": images1})
        self.logger.experiment.log({"Images2": images2})
        self.logger.experiment.log({"Latent1": latent_grid1})
        self.logger.experiment.log({"Latent2": latent_grid2})

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
