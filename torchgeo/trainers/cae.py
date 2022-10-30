# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# TODO: Attribution (Facebook)
"""MAE tasks."""

from typing import Any, cast

import torch
import wandb
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from timm.scheduler import CosineLRScheduler
from torch import Tensor, optim
from torch.nn.modules import Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

from torchgeo.models.utils import reduce_mask_token

from ..models import MaskedAutoencoderHiViT
from ..utils import _to_tuple
from .utils import generate_mask, pad_imgs_dims, patchify, unpatchify

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def masked_reconstruction_loss(
    target: Tensor,
    pred: Tensor,
    mask: Tensor,
    patch_size: int,
    norm_pix_loss: bool = False,
) -> Tensor:
    """Compute masked reconstruction loss."""
    B, *_ = pred.shape

    target = patchify(target, patch_size)
    _, PS, _ = target.shape
    target = target.view(pred.shape)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    # Only completely hidden patches contribute to the loss
    mask = ((~mask.view(-1, PS)).sum(0) == 0).repeat(len(mask) // PS)
    loss = (loss * mask).sum() / (B * mask.sum())  # mean loss on removed patches

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


class CAETask(LightningModule):
    """Class for pre-training any PyTorch model using VICReg."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.image_size = self.hyperparams.get("image_size", 256)
        self.crop_size = self.hyperparams.get("crop_size", 224)
        self.patch_size = self.hyperparams.get("patch_size", 16)
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
        self.norm_pix_loss = self.hyperparams.get("norm_pix_loss", False)

        self.model = MaskedAutoencoderHiViT(
            sensor=self.hyperparams["sensor"],
            bands=self.hyperparams.get("bands", "all"),
            image_size=self.crop_size,
            patch_size=self.patch_size,
            channel_wise=self.channel_wise,
            mlp_ratio=self.hyperparams.get("mlp_ratio", 4),
            embed_dim=self.embed_dim,
            depth=self.hyperparams.get("depth", 24),
            num_heads=self.hyperparams.get("num_heads", 16),
            dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
            dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
            decoder_embed_dim=self.hyperparams.get("decoder_embed_dim", 512),
            decoder_depth=self.hyperparams.get("expander_depth", 2),
            decoder_num_heads=self.hyperparams.get("expander_num_heads", 1),
            mask_tokens_encoder=self.hyperparams.get("mask_tokens_encoder", False),
            mask_tokens_decoder=self.hyperparams.get("mask_tokens_decoder", False),
            mask_tokens_reduction_encoder=self.hyperparams.get(
                "mask_tokens_reduction_encoder", False
            ),
            mask_tokens_reduction_decoder=self.hyperparams.get(
                "mask_tokens_reduction_decoder", False
            ),
            keep_unreduced_decoder=self.hyperparams.get(
                "keep_unreduced_decoder", False
            ),
        )

        self.mask_fns = self.hyperparams.get("mask_fns", ["random_masking"])
        self.mask_kwargs = self.hyperparams.get(
            "mask_kwargs",
            {
                "num_patches": self.num_patches,
                "random_mask_num_keep": 256,
                "random_mask_probability": 1.0,
            },
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

        self.augment = Augmentations(image_size, crop_size)

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
        actual_lr = lr * self.B / 256
        lr_min = self.hyperparams.get("min_lr", 1e-6)
        warmup_lr_init = self.hyperparams.get("warmup_lr_init", 1e-7)
        weight_decay = self.hyperparams.get("weight_decay", 0.05)
        num_warmup = self.hyperparams.get("num_warmup", 5)
        betas = self.hyperparams.get("betas", (0.9, 0.95))
        optimizer = optimizer_class(
            self.parameters(), lr=actual_lr, weight_decay=weight_decay, betas=betas
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineLRScheduler(
                    optimizer=optimizer,
                    t_initial=self.trainer.max_epochs
                    * self.trainer.limit_train_batches,
                    lr_min=lr_min,
                    cycle_mul=1.0,
                    cycle_decay=weight_decay,
                    cycle_limit=1,
                    warmup_t=num_warmup * self.trainer.limit_train_batches,
                    warmup_lr_init=warmup_lr_init,
                ),
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def shared_step(self, stage: str, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        _, self.C, *_ = x.shape

        with torch.no_grad():
            item = self.get_item(x, stage)

        item = self.forward(item)

        item["loss"] = masked_reconstruction_loss(
            item["target"],
            item["pred"],
            item["mask_target"],
            self.patch_size,
            self.norm_pix_loss,
        )

        self.log(
            f"{stage}_loss",
            item["loss"],
            on_step=stage == "train",
            on_epoch=True,
            batch_size=self.B,
            sync_dist=stage != "train",
        )

        return item

    def get_item(self, x: Tensor, stage: str) -> dict[str, Any]:
        """Preparation of the data for the forward pass."""
        target = self.augment(x, stage)
        mask_input = generate_mask(
            self.mask_fns, self.mask_kwargs, self.num_patches, self.C, x.device
        )
        # mask_input = (
        #     torch.nn.functional.interpolate(
        #         mask_input.unsqueeze(0).view(1, 4, 6, 6).float(),
        #         scale_factor=self.patch_size,
        #         mode="nearest-exact",
        #     )
        #     .squeeze()
        #     .bool()
        # )

        input = target.clone()
        mask_target = mask_input
        encoder_channels = decoder_channels = []

        if self.channel_wise:
            if self.channel_shuffle:
                encoder_channels = torch.randperm(self.C, device=x.device).tolist()[
                    : self.num_in_channels
                ]
                decoder_channels = torch.randperm(self.C, device=x.device).tolist()[
                    : self.num_out_channels
                ]

                input = input[:, encoder_channels]
                target = target[:, decoder_channels]
                mask_input = mask_input[encoder_channels]
                mask_target = mask_target[decoder_channels]
            else:
                encoder_channels = decoder_channels = torch.arange(
                    self.C, device=x.device
                ).tolist()

            input = input.flatten(0, 1).unsqueeze(1)
            target = target.flatten(0, 1).unsqueeze(1)
            mask_input = mask_input.flatten()
            mask_target = mask_target.flatten()

        return {
            "input": input,
            "target": target,
            "mask": mask_input,
            "mask_target": mask_target,
            "encoder_channels": encoder_channels,
            "decoder_channels": decoder_channels,
        }

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        item = self.shared_step("train", *args, **kwargs)

        return item["loss"]

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        item = self.shared_step("val", *args, **kwargs)

        # Only log images of first batch
        if (
            self.trainer is None
            or args[1] > 0
            or (self.trainer.num_devices > 1 and self.device.index != 1)
        ):
            return {}

        images = self.prepare_output(item)

        return images

    def prepare_output(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare the output for logging.

        Args:
            aug: tensor of data to run through the model
            aug_shuffled: tensor of data to run through the model
            pred: tensor of data to run through the model

        Returns:
            output from the model
        """
        input, target, pred, mask_input = (
            item["input"],
            item["target"],
            item["pred"],
            item["mask"],
        )

        if self.channel_wise:
            pred = pred.view(self.B * self.num_out_channels, self.num_patches, -1)

        input_patch = patchify(input, self.patch_size).view(
            self.B, self.num_in_channels * self.num_patches, -1
        )
        input_patch = input_patch[:, ~mask_input]
        *_, H = input_patch.shape

        masked_input = torch.zeros((self.B, self.num_patches, H), device=input.device)
        masked_input = reduce_mask_token(
            input_patch, mask_input, masked_input, self.num_patches
        )

        pred = unpatchify(pred, self.patch_size)
        masked_input = unpatchify(masked_input, self.patch_size).repeat(1, 3, 1, 1)

        if self.channel_wise:
            target = target.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]
            input = input.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]
            masked_input = masked_input.view(
                self.B, -1, self.crop_size, self.crop_size
            )[:, :3]
            pred = pred.view(self.B, -1, self.crop_size, self.crop_size)[:, :3]

        images = pad_imgs_dims([input, masked_input, pred, target], 3)

        return {"images": images}

    def validation_epoch_end(
        self,
        validation_step_outputs: list[Tensor | dict[str, Any]]
        | list[list[Tensor | dict[str, Any]]],
    ) -> None:
        """Log images."""
        if self.trainer.num_devices > 1 and self.device.index != 1:
            return

        images = validation_step_outputs[0]["images"]  # type: ignore

        grid = make_grid(images[:, :8].flatten(0, 1)[:, :3])
        images_grid = wandb.Image(grid, caption="Images")

        self.logger.experiment.log({"Images": images_grid})

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
