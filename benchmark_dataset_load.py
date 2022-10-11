"""Benchmark dataset load."""
# %%
from torchgeo.datasets import BigEarthNet
from torch.utils.data import DataLoader
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
import os
from torch.utils.benchmark import Timer
from tqdm import tqdm

# %%
batch_size = 64
num_workers = 0
num_batches_load = 50
data_root = "/data/users/mike/data/"
# %%
ds_no_target = BigEarthNet(
    root=os.path.join(data_root, "BigEarthNetStacked"), load_target=False
)
ds_target = BigEarthNet(
    root=os.path.join(data_root, "BigEarthNetStacked"), load_target=True
)
ffcv_pipeline = {
    "image": [NDArrayDecoder(), ToTensor()],
    "label": [NDArrayDecoder(), ToTensor()],
}
# %%
dl_no_label = DataLoader(
    dataset=ds_no_target, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
# %%
dl_with_label = DataLoader(
    dataset=ds_target, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
# %%
dl_ffcv = Loader(
    fname=os.path.join(data_root, "FFCV", "BigEarthNet_train.beton"),
    batch_size=batch_size,
    num_workers=num_workers,
    order=OrderOption.RANDOM,
    distributed=False,
    batches_ahead=2,
    pipelines=ffcv_pipeline,
)
# %%
def load_num_batches(num_batches, dataloader):
    i = 0
    for _ in tqdm(dataloader):
        if i >= num_batches:
            return
        i += 1


# %%
timer_no_label = Timer(
    stmt="load_num_batches(num_batches_load, dl_no_label)",
    globals=globals(),
    label="Load without label",
)
# %%
# timer_label = Timer(
#     stmt="load_num_batches(num_batches_load, dl_with_label)",
#     globals=globals(),
#     label="Load with label",
# )
# %%
timer_ffcv = Timer(
    stmt="load_num_batches(num_batches_load, dl_ffcv)", globals=globals(), label="Load ffcv"
)
# %%
print("Running timer:")
timer_no_label.timeit(number=1)
# %%
# timer_label.timeit(number=1)
# %%
timer_ffcv.timeit(number=1)
# %%
