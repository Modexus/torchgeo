"""."""
# %%
import glob
import itertools
import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import rasterio
from ffcv.fields import NDArrayField
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from rasterio.enums import Resampling
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from torchgeo.datasets import BigEarthNet

# %%
DATA_DIR = "/scratch/users/mike/data"
BIGEARTHNET_DIR = "BigEarthNet"
# %%
mins = np.array(
    [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
maxs = np.array(
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
        self.i = 0

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

        image = image.astype(np.float32)
        image = (image - mins) / (maxs - mins)
        image = np.clip(image, a_min=0.0, a_max=1.0)

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

        if len(paths) == 1:
            with rasterio.open(paths[0]) as dataset:
                arrays = dataset.read(
                    out_shape=self.image_size, out_dtype="int32"
                ).transpose(1, 2, 0)
        else:
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

    def _load_folders(self) -> list[Dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]

        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        folders = [
            {
                "s1": os.path.join(self.root, dir_s1, pair[1]),
                "s2": os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]

        if self.i == 0 and self.split == "train":
            filename = self.splits_metadata["val"]["filename"]
            dir_s1 = self.metadata["s1"]["directory"]
            dir_s2 = self.metadata["s2"]["directory"]

            with open(os.path.join(self.root, filename)) as f:
                lines = f.read().strip().splitlines()
                pairs = [line.split(",") for line in lines]

            folders += [
                {
                    "s1": os.path.join(self.root, dir_s1, pair[1]),
                    "s2": os.path.join(self.root, dir_s2, pair[0]),
                }
                for pair in pairs
            ]

        return folders


# %%
ds = BigEarthNetNumpy(
    os.path.join(DATA_DIR, BIGEARTHNET_DIR),
    split="train",
    bands="all",
    load_target=True,
)
# %%
ds[390000]
# %%
write_path = os.path.join(DATA_DIR, "FFCV/BigEarthNet_trainval.beton")
writer = DatasetWriter(
    write_path,
    {
        "image": NDArrayField(shape=(120, 120, 14), dtype=np.dtype("float32")),
        "label": NDArrayField(shape=(19,), dtype=np.dtype("int8")),
    },
    num_workers=32,
)
writer.from_indexed_dataset(ds)
# %%
ds = BigEarthNetNumpy(
    os.path.join(DATA_DIR, BIGEARTHNET_DIR), split="test", bands="all", load_target=True
)
# %%
write_path = os.path.join(DATA_DIR, "FFCV/BigEarthNet_test.beton")
writer = DatasetWriter(
    write_path,
    {
        "image": NDArrayField(shape=(120, 120, 14), dtype=np.dtype("float32")),
        "label": NDArrayField(shape=(19,), dtype=np.dtype("int8")),
    },
    num_workers=8,
)
writer.from_indexed_dataset(ds)
# %%
