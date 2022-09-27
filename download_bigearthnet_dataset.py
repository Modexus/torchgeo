"""."""
# %%
import ssl

from torchgeo.datasets import BigEarthNet

# %%
ssl._create_default_https_context = ssl._create_unverified_context
# %%
ds = BigEarthNet(root="/scratch/users/mike/data/BigEarthNet", download=True, bands="s2")
