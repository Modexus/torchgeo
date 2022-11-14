"""MaskedVit."""

from typing import cast

import deepspeed
import torch
from timm.models.layers import Mlp
from timm.models.vision_transformer import Block
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, ModuleList, Sequential

from .utils import (
    get_channel_encodings,
    get_mask_tokens,
    get_positional_encodings,
    init_weights,
    reduce_mask_token,
)

IN_CHANNELS = {"sentinel2": {"all": 10}, "naip": {"all": 4}, "bigearthnet": {"all": 14}}
NUM_CLASSES = {"sentinel2": 17, "naip": 0}


class TransformerEncoder(Module):
    """TransformerEncoder."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_checkpoints: int = 12,
    ) -> None:
        """Initialize a TransformerEncoder."""
        super().__init__()
        self.num_checkpoints = num_checkpoints

        self.blocks = Sequential(
            *(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(depth)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.num_checkpoints == 0:
            return cast(Tensor, self.blocks(x))

        for i, block in enumerate(self.blocks):
            if i >= self.num_checkpoints:
                x = block(x)
            else:
                x = deepspeed.checkpointing.checkpoint(block, x)

        return x


class EncoderEmbedding(Module):
    """Compute the 2d image patch embedding ready to pass to transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        channel_wise: bool = False,
        mask_tokens_encoder: bool = False,
    ) -> None:
        """Initialize the encoder embedding module."""
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.channel_wise = channel_wise
        self.mask_tokens_encoder = mask_tokens_encoder

        self.embedder = Conv2d(
            input_dim if not channel_wise else 1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.apply(init_weights)

    def forward(
        self, x: Tensor, channels: tuple[int, ...] = (), mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of the encoder embedding module.

        First embed the image to patches.
        Secondly, add the positional embeddings for each patch.
        Finally, add the channel embeddings if channels are passed.
        """
        x = self.embedder(x)

        B, H, PW, _ = x.shape
        x = x.view(B, H, PW**2).permute(0, 2, 1)  # BxCxPSxPS -> BxPxH
        *_, H = x.shape

        x += get_positional_encodings(H, self.num_patches, self.channel_wise, x.device)

        if self.channel_wise:
            x = x.reshape(-1, len(channels) * PW**2, H)
            x += get_channel_encodings(H, channels, self.num_patches, x.device)

        return x


class EncoderEmbeddingMultipleConv(Module):
    """Compute the 2d image patch embedding ready to pass to transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        channel_wise: bool = False,
        mask_tokens_encoder: bool = False,
    ) -> None:
        """Initialize the encoder embedding module."""
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.channel_wise = channel_wise
        self.mask_tokens_encoder = mask_tokens_encoder

        self.embedders = ModuleList(
            Linear(self.patch_size**2, embed_dim) for _ in range(input_dim)
        )

        self.apply(init_weights)

    def forward(
        self, x: Tensor, channels: tuple[int, ...] = (), mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of the encoder embedding module.

        First embed the image to patches.
        Secondly, add the positional embeddings for each patch.
        Finally, add the channel embeddings if channels are passed.
        """
        x = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).permute(0, 2, 1)
        x = x.view(-1, len(channels), self.num_patches, self.patch_size**2)

        xs = []
        for i, channel in enumerate(channels):
            xx = self.embedders[channel - 1](x[:, i : i + 1])
            xs.append(xx.squeeze())
        x = torch.stack(xs, dim=1).flatten(0, 1)

        *_, H = x.shape
        x += get_positional_encodings(H, self.num_patches, self.channel_wise, x.device)

        if self.channel_wise:
            x = x.reshape(-1, len(channels) * self.num_patches, H)
            x += get_channel_encodings(H, channels, self.num_patches, x.device)

        return x


class EncoderEmbeddingSatMAE(Module):
    """Compute the 2d image patch embedding ready to pass to transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        channel_wise: bool = False,
        mask_tokens_encoder: bool = False,
    ) -> None:
        """Initialize the encoder embedding module."""
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.channel_wise = channel_wise
        self.mask_tokens_encoder = mask_tokens_encoder

        self.embedders = ModuleList(
            (
                Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size),
                Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size),
                Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size),
            )
        )

        self.apply(init_weights)

    def forward(
        self, x: Tensor, channels: tuple[int, ...] = (), mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of the encoder embedding module.

        First embed the image to patches.
        Secondly, add the positional embeddings for each patch.
        Finally, add the channel embeddings if channels are passed.
        """
        *_, H, W = x.shape
        x = x.view(-1, len(channels), H, W)
        x = x[:, 2:]  # Remove SAR S1
        xs = [x[:, [0, 1, 2, 6]], x[:, [3, 4, 5, 7]], x[:, [8, 9]]]

        for i, xx in enumerate(xs):
            xx = self.embedders[i](xx)
            xx = xx.flatten(-2).permute(0, 2, 1)
            xs[i] = xx

        x = torch.cat(xs, dim=1)
        *_, H = x.shape

        x += get_positional_encodings(H, self.num_patches, False, x.device).repeat(3, 1)

        if self.channel_wise:
            x = x.reshape(-1, 3 * self.num_patches, H)
            x += get_channel_encodings(H, [1, 2, 3], self.num_patches, x.device)

        return x


class MaskedEncoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        multi_conv: bool = False,
        satmae: bool = False,
        num_checkpoints: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_tokens_encoder: bool = False,
        mask_tokens_decoder: bool = False,
        mask_tokens_reduction_encoder: bool = True,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.embed_dim = embed_dim
        self.channel_wise = channel_wise
        self.satmae = satmae
        self.use_mask_tokens_encoder = mask_tokens_encoder
        self.use_mask_tokens_decoder = mask_tokens_decoder
        self.mask_tokens_reduction = mask_tokens_reduction_encoder

        if multi_conv:
            if not satmae:
                self.embed_module = EncoderEmbeddingMultipleConv(
                    in_channels,
                    embed_dim,
                    patch_size,
                    image_size,
                    channel_wise,
                    mask_tokens_encoder or mask_tokens_decoder,
                )
            else:
                self.embed_module = EncoderEmbeddingSatMAE(
                    in_channels,
                    embed_dim,
                    patch_size,
                    image_size,
                    channel_wise,
                    mask_tokens_encoder or mask_tokens_decoder,
                )

        else:
            self.embed_module = EncoderEmbedding(
                in_channels,
                embed_dim,
                patch_size,
                image_size,
                channel_wise,
                mask_tokens_encoder or mask_tokens_decoder,
            )
        self.num_patches = self.embed_module.num_patches

        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_ratio, num_checkpoints
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(init_weights)

    def apply_mask_tokens(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply the mask tokens to the input."""
        B, *_ = x.shape
        mask_tokens = get_mask_tokens(
            self.embed_dim,
            self.num_patches,
            channel_wise=self.channel_wise,
            dtype=x.dtype,
            device=x.device,
        ).repeat(B, 1, 1)

        if self.mask_tokens_reduction:
            if mask is not None:
                x = reduce_mask_token(
                    x, mask, mask_tokens, self.num_patches, keep_unreduced=True
                )
        else:
            x = torch.cat([mask_tokens, x], dim=1)

        return x

    def mean_channels(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Reduce the channels by taking the mean."""
        B, _, H = x.shape

        if mask is None:
            return x.view(B, -1, self.num_patches, H).mean(dim=1), mask

        mask = mask.view(-1, self.num_patches)  # (C, P)

        visible_pos_indices = (~mask).nonzero()[:, 1]
        sorted_visible, indices = visible_pos_indices.sort(stable=True)
        _, counts = sorted_visible.unique_consecutive(return_counts=True)
        x = x[:, indices].split(counts.tolist(), dim=1)  # type: ignore[no-untyped-call]
        x = torch.cat([split.mean(dim=1).unsqueeze(1) for split in x], dim=1)
        mask = (~mask).sum(0) == 0

        return x, mask

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        x = cast(Tensor, item["input"])
        channels = cast(list[int], item.get("encoder_channels", []))
        channels = [channel + 1 for channel in channels]
        mask = cast(Tensor | None, item.get("mask", None))

        x = self.embed_module(x, channels, mask)

        if mask is not None:
            x = x[:, ~mask]

        if self.use_mask_tokens_encoder:
            x = self.apply_mask_tokens(x, mask)

        x = self.encoder(x)
        x = self.norm(x)

        if self.channel_wise and not self.satmae:
            x, item["mask_decoder"] = self.mean_channels(x, mask)
        else:
            item["mask_decoder"] = mask

        return x


class DecoderEmbedding(Module):
    """Decoder embedding module."""

    def __init__(self, embed_dim: int, decoder_embed_dim: int) -> None:
        """Initialize a new DecoderEmbedding module."""
        super().__init__()

        self.embedder = Linear(embed_dim, decoder_embed_dim)
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Embed the decoder input with channel encodings."""
        return cast(Tensor, self.embedder(x))


class MaskedDecoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        decoder_embed_dim: int,
        out_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        multi_predictors: bool = True,
        satmae: bool = False,
        num_checkpoints: int = 0,
        depth: int = 2,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        mask_tokens_decoder: bool = False,
        mask_tokens_reduction_decoder: bool = True,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.mask_tokens_decoder = mask_tokens_decoder
        self.mask_tokens_reduction_decoder = mask_tokens_reduction_decoder
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.channel_wise = channel_wise
        self.satmae = satmae
        self.multi_predictors = multi_predictors

        self.out_features = patch_size**2
        if not channel_wise:
            self.out_features *= out_channels

        self.embed_module = DecoderEmbedding(embed_dim, decoder_embed_dim)
        self.norm = LayerNorm(decoder_embed_dim)

        if self.mask_tokens_decoder:
            self.encoder = TransformerEncoder(
                decoder_embed_dim, depth, num_heads, mlp_ratio, num_checkpoints
            )

        if not self.satmae:
            self.predictors = ModuleList(
                Mlp(in_features=decoder_embed_dim, out_features=self.out_features)
                for _ in range(out_channels)
            )
        else:
            self.predictors = Mlp(
                in_features=decoder_embed_dim * 3, out_features=self.out_features * 12
            )

        # self.predictor = Mlp(
        #     in_features=decoder_embed_dim, out_features=self.out_features
        # )

        self.apply(init_weights)

    def apply_mask_tokens(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply mask tokens to the decoder input."""
        if mask is not None:
            B, *_ = x.shape
            mask_tokens = get_mask_tokens(
                self.decoder_embed_dim,
                self.num_patches,
                channel_wise=self.channel_wise,
                dtype=x.dtype,
                device=x.device,
            ).repeat(B, 1, 1)

            # Re-apply positional encoding
            x += get_positional_encodings(
                self.decoder_embed_dim, self.num_patches, self.channel_wise, x.device
            ).repeat(len(mask) // self.num_patches, 1)[~mask]

            if self.satmae:
                mask_tokens = mask_tokens.repeat(1, 3, 1)
                mask_tokens[:, ~mask] = x
                x = mask_tokens
            elif self.mask_tokens_reduction_decoder:
                x = reduce_mask_token(x, mask, mask_tokens, self.num_patches)
            else:
                x = torch.cat([mask_tokens, x], dim=1)

        return x

    def predict(self, x: Tensor, channels: tuple[int, ...]) -> Tensor:
        """Predict the pixel values of the image patch-by-patch."""
        if len(channels) == 0:
            return cast(Tensor, self.predictor(x))

        x = x.repeat(1, len(channels), 1)
        if len(channels):
            *_, H = x.shape
            x += get_channel_encodings(H, channels, self.num_patches, x.device)
        x = self.predictor(x)

        return x

    def predict_multiple_predictors(
        self, x: Tensor, channels: tuple[int, ...]
    ) -> Tensor:
        """Predict the pixel values of the image patch-by-patch."""
        if len(channels) == 0:
            return cast(Tensor, self.predictor(x))

        x = x.repeat(1, len(channels), 1)
        B, PS, H = x.shape

        x = x.view(B, len(channels), PS // len(channels), H)
        xs = []
        for i, channel in enumerate(channels):
            xs.append(self.predictors[channel - 1](x[:, i : i + 1]))
        x = torch.cat(xs, dim=1).flatten(1, 2)

        return x

    def predict_satmae_predictors(self, x: Tensor, channels: tuple[int, ...]) -> Tensor:
        """Predict the pixel values of the image patch-by-patch."""
        B, _, H = x.shape
        x = x.view(B, 3, self.num_patches, H).transpose(1, 2)
        x = x.flatten(-2)

        x = self.predictors(x)
        x = (
            x.view(B, self.num_patches, 12, self.out_features)
            .transpose(1, 2)
            .flatten(1, 2)
        )

        return x

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        x = cast(Tensor, item["latent"])
        channels = cast(list[int], item.get("decoder_channels", []))
        channels = [channel + 1 for channel in channels]
        mask = cast(Tensor | None, item.get("mask_decoder", None))

        x = self.embed_module(x)

        if self.mask_tokens_decoder:
            x = self.apply_mask_tokens(x, mask)
            x = self.encoder(x)
            x = self.norm(x)

        x = x[:, : self.num_patches]

        if self.satmae:
            x = self.predict_satmae_predictors(x, tuple(channels))
        elif self.multi_predictors:
            x = self.predict_multiple_predictors(x, tuple(channels))
        else:
            x = self.predict(x, tuple(channels))

        return x


class MaskedAutoencoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        sensor: str,
        bands: str,
        image_size: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        multi_conv: bool = False,
        satmae: bool = False,
        num_checkpoints_encoder: int = 0,
        num_checkpoints_decoder: int = 0,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 24,
        decoder_num_heads: int = 1,
        mask_tokens_encoder: bool = False,
        mask_tokens_decoder: bool = True,
        mask_tokens_reduction_encoder: bool = True,
        mask_tokens_reduction_decoder: bool = True,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            in_channels=12,  # IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            multi_conv=multi_conv,
            satmae=satmae,
            num_checkpoints=num_checkpoints_encoder,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_tokens_encoder=mask_tokens_encoder,
            mask_tokens_decoder=mask_tokens_decoder,
            mask_tokens_reduction_encoder=mask_tokens_reduction_encoder,
        )

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.num_patches,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            out_channels=12,  # IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            satmae=satmae,
            num_checkpoints=num_checkpoints_decoder,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            mask_tokens_decoder=mask_tokens_decoder,
            mask_tokens_reduction_decoder=mask_tokens_reduction_decoder,
        )

    def forward(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass of the model."""
        item["latent"] = self.encoder(item)
        item["pred"] = self.decoder(item)

        return item


class MaskedViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        sensor: str,
        bands: str,
        image_size: int,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=IN_CHANNELS[sensor][bands],
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dropout_attn=dropout_attn,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass of the model."""
        embedding = self.encoder(x, mask)

        return cast(Tensor, embedding)
