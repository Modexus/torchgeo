# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg
from .farseg import FarSeg
from .fcn import FCN
from .fcsiam import FCSiamConc, FCSiamDiff
from .maehivit import MaskedAutoencoderHiViT
from .maevit import MaskedAutoencoderViT, MaskedViT
from .rcf import RCF
from .resnet import resnet18, resnet50
from .tile2vec_resnet import ResNet18

__all__ = (
    "ChangeMixin",
    "ChangeStar",
    "ChangeStarFarSeg",
    "FarSeg",
    "FCN",
    "FCSiamConc",
    "FCSiamDiff",
    "RCF",
    "resnet50",
    "resnet18",
    "ResNet18",
    "MaskedAutoencoderViT",
    "MaskedViT",
    "MaskedAutoencoderHiViT",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
