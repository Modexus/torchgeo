"""Test."""
# %%
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

# %%
convert_zero_checkpoint_to_fp32_state_dict(
    "/scratch/users/mike/experiments/mae_bigearthnet_train/mae-vit/epoch=800-step=52000.ckpt",
    "/scratch/users/mike/experiments/mae_bigearthnet_train/mae-vit/last.ckpt",
)
# %%
