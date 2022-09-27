"""."""
# %%
import glob
import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from ffcv.fields import NDArrayField
from ffcv.writer import DatasetWriter
from rasterio.enums import Resampling
from torch import Tensor
from tqdm import tqdm

from torchgeo.datasets import BigEarthNet

# %%
DATA_DIR = "/scratch/users/mike/data"
BIGEARTHNET_DIR = "BigEarthNet"
# %%
class BigEarthNetNumpy(BigEarthNet):
    """BigEarthNet but returns numpy arrays instead of torch tensors."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        load_target: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance."""
        super().__init__(
            root, split, bands, num_classes, transforms, load_target, download, checksum
        )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)

        if self.load_target:
            label = self._load_target(index)

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, label)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []

        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays = np.stack(images, axis=-1)

        return arrays

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        if self.bands == "s2":
            folder = self.folders[index]["s2"]
        else:
            folder = self.folders[index]["s1"]

        path = glob.glob(os.path.join(folder, "*.json"))[0]
        with open(path) as f:
            labels = json.load(f)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            indices_optional = [self.label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]

        target = np.zeros(self.num_classes, dtype=np.dtype("int8"))
        target[indices] = 1
        return target


# %%
ds = BigEarthNetNumpy(
    os.path.join(DATA_DIR, BIGEARTHNET_DIR), bands="all", split="train"
)
# %%
for split in tqdm(["train", "val"]):
    ds = BigEarthNetNumpy(
        os.path.join(DATA_DIR, BIGEARTHNET_DIR), bands="all", split=split
    )
    write_path = os.path.join(DATA_DIR, f"FFCV_new/BigEarthNet_{split}.beton")
    writer = DatasetWriter(
        write_path,
        {
            "image": NDArrayField(shape=(120, 120, 14), dtype=np.dtype("int32")),
            "label": NDArrayField(shape=(19,), dtype=np.dtype("int8")),
        },
        num_workers=8,
    )
    writer.from_indexed_dataset(ds)
# %%
