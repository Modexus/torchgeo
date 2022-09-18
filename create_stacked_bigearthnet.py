"""."""
# %%
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from torch import Tensor
from torchvision.transforms import Compose
from tqdm import tqdm

from torchgeo.datamodules import BigEarthNetDataModule
from torchgeo.datasets import BigEarthNet

# %%
path_to_stacked = "/scratch/users/mike/data/BigEarthNetStacked"
# %%
class BigEarthNetStacked(BigEarthNet):
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
                array = dataset.read(out_shape=self.image_size, out_dtype="int32")
            return torch.from_numpy(array)

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
                if dataset.profile["height"] == 120:
                    profile = dataset.profile
                images.append(array)
        arrays = np.stack(images, axis=0)

        profile["dtype"] = "int32"
        profile["count"] = 14

        dir, filename = path.split("/")[-2:]
        dir_path = os.path.join(path_to_stacked, "BigEarthNet-v1.0", dir)
        file_path = os.path.join(dir_path, filename)
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
        else:
            with rasterio.open(file_path, "w", **profile) as dataset:
                dataset.write(arrays)

        tensor = torch.from_numpy(arrays)
        return tensor


# %%
class BigEarthNetStackedDataModule(BigEarthNetDataModule):
    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        num_classes: int = 19,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 10,
        persistent_workers: bool = False,
        load_target: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root_dir,
            bands,
            num_classes,
            batch_size,
            num_workers,
            pin_memory,
            prefetch_factor,
            persistent_workers,
            load_target,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = BigEarthNetStacked(
            self.root_dir,
            split="train",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )
        self.val_dataset = BigEarthNetStacked(
            self.root_dir,
            split="val",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )
        self.test_dataset = BigEarthNetStacked(
            self.root_dir,
            split="test",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )


# %%
os.makedirs(path_to_stacked, exist_ok=True)
os.makedirs(os.path.join(path_to_stacked, "BigEarthNet-S1-v1.0"), exist_ok=True)
os.makedirs(os.path.join(path_to_stacked, "BigEarthNet-v1.0"), exist_ok=True)
# %%
dm = BigEarthNetStackedDataModule(
    "/scratch/users/mike/data/BigEarthNet", num_workers=32
)
dm.setup()
# %%
for _ in tqdm(dm.train_dataloader()):
    pass
for _ in tqdm(dm.val_dataloader()):
    pass
for _ in tqdm(dm.test_dataloader()):
    pass
