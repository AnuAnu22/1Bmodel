# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Run once per session to install deps and mount Drive
# ═══════════════════════════════════════════════════════════════════════════════

# Install dependencies
# For TPU v5e-1:
# !pip install -q "jax[tpu]>=0.4.30" flax>=0.8.5 optax>=0.2.3 \
#     orbax-checkpoint datasets transformers matplotlib

# For T4 GPU (slower, use only if TPU unavailable):
# !pip install -q "jax[cuda12]>=0.4.30" flax>=0.8.5 optax>=0.2.3 \
#     orbax-checkpoint datasets transformers matplotlib

# Mount Google Drive (your checkpoints and logs live here — persist across sessions)
from google.colab import drive
drive.mount('/content/drive')

import jax
print("Devices:", jax.devices())
# TPU v5e-1 should show: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]
# T4 GPU should show:    [GpuDevice(id=0, ...)]


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Clone / upload your project files
# ═══════════════════════════════════════════════════════════════════════════════

# Option A: clone from GitHub (recommended — keeps your code versioned)
# !git clone https://github.com/YOUR_USERNAME/deepseek_1b.git /content/deepseek_1b

# Option B: upload the zip
# from google.colab import files
# files.upload()  # upload deepseek_1b.zip
# !unzip deepseek_1b.zip -d /content/

import sys
sys.path.insert(0, '/content/deepseek_1b')


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Verify everything imports and the model init works
# ═══════════════════════════════════════════════════════════════════════════════

import jax
import jax.numpy as jnp
import numpy as np
from config import ModelConfig
from model.model import DeepSeek1B

cfg   = ModelConfig()
model = DeepSeek1B(cfg, use_remat=True)
dummy = jnp.ones((1, 4), dtype=jnp.int32)

# Trace only (no memory allocation) — confirm model structure is valid
param_shapes = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
n = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(param_shapes))
print(f"Model OK — {n/1e9:.2f}B parameters")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Start (or resume) training
# ═══════════════════════════════════════════════════════════════════════════════

# This single call auto-detects the latest checkpoint in Drive and resumes.
# If no checkpoint exists, it starts fresh.
# Just re-run this cell after each Colab session restart.

import subprocess
result = subprocess.run(
    ["python", "/content/deepseek_1b/train/train_colab.py"],
    cwd="/content/deepseek_1b"
)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — View training charts (run anytime)
# ═══════════════════════════════════════════════════════════════════════════════

from IPython.display import Image, display
display(Image("/content/drive/MyDrive/deepseek1b_run/training_curves.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Inspect a checkpoint manually (optional debug)
# ═══════════════════════════════════════════════════════════════════════════════

import json
from pathlib import Path

log_path = Path("/content/drive/MyDrive/deepseek1b_run/train_log.jsonl")
if log_path.exists():
    lines = log_path.read_text().strip().splitlines()
    last_10 = [json.loads(l) for l in lines[-10:]]
    print("Last 10 log entries:")
    for entry in last_10:
        print(f"  step {entry['step']:>7,} | loss {entry['loss']:.4f} | "
              f"ppl {entry['ppl']:.1f} | {entry['tokens_B']:.2f}B tokens")

# Check which checkpoints exist
ckpt_dir = Path("/content/drive/MyDrive/deepseek1b_run/checkpoints")
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("step_*"))
    print(f"\nSaved checkpoints ({len(ckpts)}):")
    for c in ckpts:
        meta = json.loads((c / "meta.json").read_text())
        print(f"  {c.name}  —  {meta['tokens_seen']/1e9:.2f}B tokens")
