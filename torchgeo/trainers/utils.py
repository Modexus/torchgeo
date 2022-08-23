# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from typing import Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d, Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "nn.Module"
Conv2d.__module__ = "nn.Conv2d"


def extract_encoder(path: str) -> Tuple[str, "OrderedDict[str, Tensor]"]:
    """Extracts an encoder from a pytorch lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'classification_model' or 'encoder' not in
            checkpoint['hyper_parameters']
    """
    checkpoint = torch.load(  # type: ignore[no-untyped-call]
        path, map_location=torch.device("cpu")
    )

    if "classification_model" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["classification_model"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if "model." in k})
        state_dict = OrderedDict(
            {k.replace("model.", ""): v for k, v in state_dict.items()}
        )
    elif "encoder_name" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["encoder_name"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "model.encoder.model" in k}
        )
        state_dict = OrderedDict(
            {k.replace("model.encoder.model.", ""): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            "Unknown checkpoint task. Only encoder or classification_model"
            " extraction is supported"
        )

    return name, state_dict


def load_state_dict(model: Module, state_dict: "OrderedDict[str, Tensor]") -> Module:
    """Load pretrained resnet weights to a model.

    Args:
        model: model to load the pretrained weights to
        state_dict: dict containing tensor parameters

    Returns:
        the model with pretrained weights

    Warns:
        If input channels in model != pretrained model input channels
        If num output classes in model != pretrained model num classes
    """
    in_channels = cast(nn.Module, model.conv1).in_channels
    expected_in_channels = state_dict["conv1.weight"].shape[1]
    num_classes = cast(nn.Module, model.fc).out_features
    expected_num_classes = state_dict["fc.weight"].shape[0]

    if in_channels != expected_in_channels:
        warnings.warn(
            f"input channels {in_channels} != input channels in pretrained"
            f" model {expected_in_channels}. Overriding with new input channels"
        )
        del state_dict["conv1.weight"]

    if num_classes != expected_num_classes:
        warnings.warn(
            f"num classes {num_classes} != num classes in pretrained model"
            f" {expected_num_classes}. Overriding with new num classes"
        )
        del state_dict["fc.weight"], state_dict["fc.bias"]

    model.load_state_dict(state_dict, strict=False)

    return model


def reinit_initial_conv_layer(
    layer: Conv2d,
    new_in_channels: int,
    keep_rgb_weights: bool,
    new_stride: Optional[Union[int, Tuple[int, int]]] = None,
    new_padding: Optional[Union[str, Union[int, Tuple[int, int]]]] = None,
) -> Conv2d:
    """Clones a Conv2d layer while optionally retaining some of the original weights.

    When replacing the first convolutional layer in a model with one that operates over
    different number of input channels, we sometimes want to keep a subset of the kernel
    weights the same (e.g. the RGB weights of an ImageNet pretrained model). This is a
    convenience function that performs that function.

    Args:
        layer: the Conv2d layer to initialize
        new_in_channels: the new number of input channels
        keep_rgb_weights: flag indicating whether to re-initialize the first 3 channels
        new_stride: optionally, overwrites the ``layer``'s stride with this value
        new_padding: optionally, overwrites the ``layers``'s padding with this value

    Returns:
        a Conv2d layer with new kernel weights
    """
    use_bias = layer.bias is not None
    if keep_rgb_weights:
        w_old = layer.weight.data[:, :3, :, :].clone()
        if use_bias:
            b_old = cast(Tensor, layer.bias).data.clone()

    updated_stride = layer.stride if new_stride is None else new_stride
    updated_padding = layer.padding if new_padding is None else new_padding

    new_layer = Conv2d(
        new_in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,  # type: ignore[arg-type]
        stride=updated_stride,  # type: ignore[arg-type]
        padding=updated_padding,  # type: ignore[arg-type]
        dilation=layer.dilation,  # type: ignore[arg-type]
        groups=layer.groups,
        bias=use_bias,
        padding_mode=layer.padding_mode,
    )
    nn.init.kaiming_normal_(new_layer.weight, mode="fan_out", nonlinearity="relu")

    if keep_rgb_weights:
        new_layer.weight.data[:, :3, :, :] = w_old
        if use_bias:
            cast(Tensor, new_layer.bias).data = b_old

    return new_layer


def patchify(imgs: Tensor, patch_size: int) -> Tensor:
    """Patches a batch of images.

    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    B, C, H, W = imgs.shape

    assert H == W and H % patch_size == 0

    h = w = H // patch_size
    x = imgs.reshape(shape=(B, C, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)  # type: ignore[no-untyped-call]
    x = x.reshape(shape=(B, h * w, patch_size**2 * C))
    return x


def unpatchify(x: Tensor, patch_size: int, flat: bool = False) -> Tensor:
    """Unpatchify an image.

    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    B, P, _ = x.shape

    if not flat:
        h = w = int(P**0.5)
        assert h * w == P

        x = x.reshape(shape=(B, h, w, patch_size, patch_size, -1))
        x = torch.einsum("nhwpqc->nchpwq", x)  # type: ignore[no-untyped-call]
        imgs = x.reshape(shape=(B, -1, h * patch_size, h * patch_size))
        return imgs

    x = x.reshape(shape=(B, P, patch_size, patch_size, -1))
    x = torch.einsum("nhpqc->ncphq", x)  # type: ignore[no-untyped-call]

    imgs = x.reshape(shape=(B, -1, patch_size, P * patch_size))
    return imgs


def pad_imgs_dims(images: list[Tensor], pad_dim: int) -> Tensor:
    """Pad the image dimensions to match."""
    images = [pad_img_dims(image, pad_dim) for image in images]
    return torch.stack(images, dim=0)


def pad_img_dims(img: Tensor, pad_dim: int) -> Tensor:
    """Pad the image dimensions to match the original dimensions."""
    B, C, H, W = img.shape
    if C == pad_dim:
        return img

    img_padded = torch.zeros((B, pad_dim, H, W), device=img.device)
    img_padded[:, :C] = img

    return img_padded


def random_masking(
    mask: Tensor,
    random_mask_ratio: float,
    random_mask_probability: float,
    **kwargs: Any,
) -> Tensor:
    """Perform per-sample random masking by per-sample shuffling.

    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    if torch.rand(1) > random_mask_probability:
        return mask

    C, PS = mask.shape
    mask = mask.flatten()

    P = len(mask)
    num_removed = mask.sum()
    num_kept = P - num_removed
    len_remove = max(int(P * (1 - (1 - random_mask_ratio) / C)) - num_removed, 0)

    ids_kept = (~mask).flatten().argwhere().view(num_kept)
    noise = torch.rand(num_kept, device=mask.device)
    ids_shuffle = torch.argsort(noise, dim=0)
    ids_remove = ids_kept.gather(dim=0, index=ids_shuffle[:len_remove]).flatten()

    mask.flatten()[ids_remove] = True
    mask = mask.view(C, PS)

    return mask


# TODO: Rework
def focal_masking(
    masks: Tensor,
    focal_mask_ratio: float,
    focal_mask_probability: float,
    num_patches: int | None = None,
    **kwargs: Any,
) -> Tensor:
    """Perform focal masking."""
    if num_patches is not None:
        masks = []
        for mask_split in mask.split(num_patches, dim=1):  # type: ignore
            mask_split_focal = focal_masking(
                mask_split, focal_mask_ratio, focal_mask_probability
            )
            masks.append(mask_split_focal)

        return torch.cat(masks, dim=1)

    if torch.rand(1) > focal_mask_probability:
        return mask

    B, P = mask.shape
    focal_ratio = 1 - focal_mask_ratio
    side = int(P**0.5)

    # Generate focal mask
    center = side / 2
    half_scaled = side // 2 * focal_ratio
    low, high = round(center - half_scaled), round(center + half_scaled)

    focal_mask = torch.ones((B, side, side), device=mask.device, dtype=torch.bool)
    focal_mask[:, low:high, low:high] = False

    mask |= focal_mask.view(B, P)

    return mask
