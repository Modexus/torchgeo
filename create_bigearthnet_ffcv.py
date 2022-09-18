"""."""
# %%
import glob
import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from ffcv.fields import NDArrayField, FloatField
from ffcv.writer import DatasetWriter
from rasterio.enums import Resampling
from torch import Tensor
from tqdm import tqdm

from torchgeo.datasets import BigEarthNet

# %%
DATA_DIR = "/scratch/users/mike/data"
BIGEARTHNET_DIR = "BigEarthNetStacked"
# %%
class BigEarthNetNumpy(BigEarthNet):
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

    def _load_image(self, index: int) -> np.ndarray:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)

        if len(paths) == 1:
            with rasterio.open(paths[0]) as dataset:
                arrays = dataset.read(out_shape=self.image_size, out_dtype="int32")
            return arrays

        raise NotImplementedError("Only single band images are supported")

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

        target = np.zeros(self.num_classes, dtype=np.dtype("int32"))
        target[indices] = 1
        return target


# %%
band_mins = np.array(
    [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
band_maxs = np.array(
    [
        6.0,
        16.0,
        9859.0,
        12872.0,
        13163.0,
        14445.0,
        12477.0,
        12563.0,
        12289.0,
        15596.0,
        12183.0,
        9458.0,
        5897.0,
        5544.0,
    ],
    dtype=np.float32,
)
mins = band_mins[:, None, None]
maxs = band_maxs[:, None, None]


def preprocess(image: np.ndarray) -> np.ndarray:
    """Transform a single sample from the Dataset."""
    image = image.astype(np.float32)
    image = (image - mins) / (maxs - mins)
    image = np.clip(image, a_min=0.0, a_max=1.0)
    return image


# %%
ds = BigEarthNetNumpy(
    os.path.join(DATA_DIR, BIGEARTHNET_DIR),
    bands="s2",
    split="train",
    transforms=preprocess,
)
ds[0]
# %%
for split in tqdm(["train", "val", "test"]):
    ds = BigEarthNetNumpy(
        os.path.join(DATA_DIR, BIGEARTHNET_DIR),
        bands="s2",
        split=split,
        transforms=preprocess,
    )
    write_path = os.path.join(DATA_DIR, f"/FFCV/BigEarthNet_{split}.beton")
    writer = DatasetWriter(
        write_path,
        {
            "image": NDArrayField(shape=(14, 120, 120), dtype=np.dtype("float32")),
            "label": NDArrayField(shape=(19,), dtype=np.dtype("int32")),
        },
    )
    writer.from_indexed_dataset(ds)
# %%
