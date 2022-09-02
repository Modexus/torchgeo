"""MaskedVit."""

from typing import cast

import torch
from kornia.contrib.vit import FeedForward, TransformerEncoderBlock
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Module, Sequential

from .utils import (
    add_embed_encoding,
    get_encoding,
    get_encoding_masked,
    get_mask_tokens,
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
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a TransformerEncoder."""
        super().__init__()
        self.blocks = Sequential(
            *(
                TransformerEncoderBlock(
                    embed_dim, num_heads, dropout_rate, dropout_attn
                )
                for _ in range(depth)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return cast(Tensor, self.blocks(x))


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

    def forward(self, x: Tensor, channels: list[int] = []) -> Tensor:
        """Forward pass of the encoder embedding module.

        First embed the image to patches.
        Secondly, add the positional embeddings for each patch.
        Finally, add the channel embeddings if channels are passed.
        """
        x = self.embedder(x)

        B, H, PW, _ = x.shape
        x = x.view(B, H, PW**2).permute(0, 2, 1)  # BxCxPSxPS -> BxPxH

        *_, H = x.shape
        if self.channel_wise:
            x = x.reshape(-1, len(channels) * PW**2, H)

        x += get_encoding(
            embed_dim=H,
            num_patches=self.num_patches,
            channel_wise=self.channel_wise,
            embed_enc=self.mask_tokens_encoder,
            return_pos_enc=True,
            channels=tuple(channels),
            device=x.device,
        )

        return x


class MaskedEncoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
        mask_tokens_encoder: bool = False,
        mask_tokens_decoder: bool = False,
        mask_tokens_reduction_encoder: bool = True,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.embed_dim = embed_dim
        self.channel_wise = channel_wise
        self.use_mask_tokens_encoder = mask_tokens_encoder
        self.use_mask_tokens_decoder = mask_tokens_decoder
        self.mask_tokens_reduction_encoder = mask_tokens_reduction_encoder

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
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(init_weights)

    def apply_mask_tokens(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply the mask tokens to the input."""
        if self.use_mask_tokens_encoder:
            B, *_ = x.shape
            mask_tokens = get_mask_tokens(
                self.embed_dim,
                self.num_patches,
                channel_enc=True,
                embed_enc=True,
                channel_wise=self.channel_wise,
                device=x.device,
            ).repeat(B, 1, 1)

            if self.mask_tokens_reduction_encoder:
                if mask is not None:
                    x = reduce_mask_token(
                        x, mask, mask_tokens, self.num_patches, keep_unreduced=True
                    )
            else:
                x = torch.cat([mask_tokens, x], dim=1)

        if self.use_mask_tokens_decoder and mask is not None:
            x = add_embed_encoding(x, mask, self.num_patches, self.channel_wise)

        return x

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        x = cast(Tensor, item["input"])
        channels = cast(list[int], item.get("encoder_channels", []))
        channels = [channel + 1 for channel in channels]
        mask = cast(Tensor | None, item.get("mask", None))

        x = self.embed_module(x, channels)

        if mask is not None:
            x = x[:, ~mask]

        if self.use_mask_tokens_encoder or self.use_mask_tokens_decoder:
            x = self.apply_mask_tokens(x, mask)

        x = self.encoder(x)
        x = self.norm(x)

        if self.use_mask_tokens_encoder or (
            mask is None and self.use_mask_tokens_decoder
        ):
            x = x[:, : self.num_patches]

        return x


class DecoderEmbedding(Module):
    """Decoder embedding module."""

    def __init__(self, num_patches: int) -> None:
        """Initialize a new DecoderEmbedding module."""
        super().__init__()

        self.num_patches = num_patches

    def forward(self, x: Tensor, channels: list[int] = []) -> Tensor:
        """Embed the decoder input with channel encodings."""
        if len(channels):
            *_, H = x.shape
            x += get_encoding(
                embed_dim=H,
                num_patches=self.num_patches,
                channel_wise=True,
                embed_enc=True,
                channels=tuple(channels),
                device=x.device,
            )

        return x


class MaskedDecoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        out_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        mask_tokens_decoder: bool = False,
        mask_tokens_reduction_encoder: bool = False,
        mask_tokens_reduction_decoder: bool = True,
        depth: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.mask_tokens_decoder = mask_tokens_decoder
        self.mask_tokens_reduction_encoder = mask_tokens_reduction_encoder
        self.mask_tokens_reduction_decoder = mask_tokens_reduction_decoder
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.channel_wise = channel_wise

        out_features = patch_size**2
        if not channel_wise:
            out_features *= out_channels

        self.embed_module = DecoderEmbedding(num_patches)
        self.norm = LayerNorm(embed_dim)
        self.predictor = Sequential(
            self.norm, FeedForward(embed_dim, embed_dim, out_features)
        )

        if self.mask_tokens_decoder:
            self.encoder = TransformerEncoder(
                embed_dim, depth, num_heads, dropout_rate, dropout_attn
            )
            self.norm = LayerNorm(embed_dim)

        self.apply(init_weights)

    def apply_mask_tokens(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply mask tokens to the decoder input."""
        if mask is not None:
            B, *_ = x.shape
            mask_tokens = get_mask_tokens(
                self.embed_dim,
                self.num_patches,
                channel_enc=False,
                embed_enc=True,
                channel_wise=self.channel_wise,
                device=x.device,
            ).repeat(B, 1, 1)

            # Select only the special embed tokens
            if not self.mask_tokens_reduction_encoder:
                mask = mask.view(-1, self.num_patches)
                mask = (~mask).sum(dim=0) == 0
                x = x[:, : int((~mask).sum())]

            # Re-apply positional encoding
            x += get_encoding_masked(
                mask, self.num_patches, self.embed_dim, self.channel_wise
            )

            if self.mask_tokens_reduction_decoder:
                x = reduce_mask_token(x, mask, mask_tokens, self.num_patches)
            else:
                x = torch.cat([mask_tokens, x], dim=1)

        return x

    def predict(self, x: Tensor, channels: list[int]) -> Tensor:
        """Predict the pixel values of the image patch-by-patch."""
        if len(channels) == 0:
            return cast(Tensor, self.predictor(x))

        x = x.repeat(1, len(channels), 1)
        x = self.embed_module(x, channels)
        x = self.predictor(x)

        return x

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        x = cast(Tensor, item["latent"])
        channels = cast(list[int], item.get("decoder_channels", []))
        channels = [channel + 1 for channel in channels]
        mask = cast(Tensor | None, item.get("mask", None))

        if self.mask_tokens_decoder:
            x = self.apply_mask_tokens(x, mask)

            x = self.encoder(x)
            x = self.norm(x)

            x = x[:, : self.num_patches]

        x = self.predict(x, channels)

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
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
        decoder_depth: int = 24,
        decoder_num_heads: int = 1,
        mask_tokens_encoder: bool = False,
        mask_tokens_decoder: bool = True,
        mask_tokens_reduction_encoder: bool = True,
        mask_tokens_reduction_decoder: bool = True,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        embed_dim = (embed_dim // 4 // 4) * 4 * 4

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            in_channels=IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dropout_attn=dropout_attn,
            mask_tokens_encoder=mask_tokens_encoder,
            mask_tokens_decoder=mask_tokens_decoder,
            mask_tokens_reduction_encoder=mask_tokens_reduction_encoder,
        )

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.num_patches,
            embed_dim=embed_dim,
            out_channels=IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            mask_tokens_decoder=mask_tokens_decoder,
            mask_tokens_reduction_encoder=mask_tokens_reduction_encoder,
            mask_tokens_reduction_decoder=mask_tokens_reduction_decoder,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
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
