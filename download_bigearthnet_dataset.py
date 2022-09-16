"""."""
# %%
from torchgeo.datasets import BigEarthNet
import ssl
# %%
ssl._create_default_https_context = ssl._create_unverified_context
# %%
ds = BigEarthNet(root="/scratch/users/mike/data/BigEarthNet", download=True, split="train")
ds = BigEarthNet(root="/scratch/users/mike/data/BigEarthNet", download=True, split="val")
ds = BigEarthNet(root="/scratch/users/mike/data/BigEarthNet", download=True, split="test")
