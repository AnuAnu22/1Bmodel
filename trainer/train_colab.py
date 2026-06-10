"""
train_colab.py v2 — Resumable training for Google Colab TPU v5e-1.

Fixed vs v1:
  1. bfloat16 activations: forward in bf16, params/grads/opt-state in float32.
     Required to fit 1B model on 16 GB HBM at seq=1024, batch=32.
  2. train_step is a single @jax.jit that uses jax.lax.scan for accumulation.
     v1 had a bare Python loop that retraced the graph every micro-batch.
  3. step_rng is passed to model.apply as rngs={'dropout': rng}.
  4. optax.clip_by_global_norm(1.0) added to the optimizer chain.
  5. Checkpoint restore: full TrainState (params + opt_state + step) via orbax.
     v1 rebuilt the optimizer on resume, zeroing all momentum buffers.
  6. Dataset skip at document level (HF .skip(n)) — O(n_docs) not O(n_tokens).
  7. Gradient norm logged each step.
  8. Background prefetch thread keeps one batch ready while TPU computes.

Memory budget for single TPU v5e-1 core (16 GB HBM):
  float32 params:         ~4 GB
  Muon momentum (900M):   ~3.6 GB
  AdamW 2×moments (100M): ~0.8 GB
  Grad accumulator (scan): ~4 GB   ← scan reuses buffers, only 1 copy live
  Activations (remat):    ~0.5 GB
  Total:                  ~13 GB   ← fits in 16 GB

pmap / sharding (4-core) is not added here — get single-core stable first,
then add it for ~4× throughput improvement.
"""

from __future__ import annotations
import os, sys, time, json, math, queue, threading, shutil, itertools
from pathlib import Path
from typing import Dict, Any, Iterator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ModelConfig
from model.model import DeepSeek1B
from train.muon import muon_with_adamw_fallback, label_params
from train.trainer import cross_entropy_loss

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DRIVE_DIR      = Path("/content/drive/MyDrive/deepseek1b_run")
LOG_FILE       = DRIVE_DIR / "train_log.jsonl"
CHART_FILE     = DRIVE_DIR / "training_curves.png"
CKPT_DIR       = DRIVE_DIR / "checkpoints"

SEQ_LEN        = 1024
BATCH_SIZE     = 32        # global; each step trains on BATCH_SIZE × SEQ_LEN tokens
GRAD_ACCUM     = 4         # micro-batches per optimizer step
MICRO_BATCH    = BATCH_SIZE // GRAD_ACCUM   # = 8 per micro-step

TOTAL_TOKENS   = 20_000_000_000
WARMUP_STEPS   = 2_000
MAX_LR         = 3e-4
MIN_LR         = 3e-5
WEIGHT_DECAY   = 0.1
MUON_LR        = 0.02
GRAD_CLIP      = 1.0

CKPT_EVERY     = 500
CKPT_KEEP      = 3
LOG_EVERY      = 10
PREFETCH_SIZE  = 2         # batches to pre-load in background

DATASET_MIX = [
    ("HuggingFaceFW/fineweb",           "sample-10BT", 0.70),
    ("bigcode/the-stack-v2-train-smol-ids", None,      0.20),
    ("wikimedia/wikipedia",              "20231101.en", 0.10),
]
TOKENIZER_NAME = "gpt2"    # fallback; swap for "meta-llama/Llama-2-7b-hf" if available

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICES
# ═══════════════════════════════════════════════════════════════════════════════

devices     = jax.devices()
NUM_DEVICES = len(devices)
print(f"JAX backend: {jax.default_backend()}  |  Devices: {devices}")

# ═══════════════════════════════════════════════════════════════════════════════
# MIXED PRECISION HELPERS  (fix 1)
# ═══════════════════════════════════════════════════════════════════════════════

def to_bf16(params):
    """Cast all float32 leaves to bfloat16 for the forward pass."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
        params,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER  (fix 4 — gradient clipping added)
# ═══════════════════════════════════════════════════════════════════════════════

def make_optimizer(params, total_steps: int):
    """
    Build optimizer.  Structure:
      clip_by_global_norm  →  Muon (2D+ weights)  /  AdamW (embeddings, norms)

    The LR schedule is built into the AdamW chain.  When TrainState is restored
    from a checkpoint, opt_state.count is also restored, so the schedule
    automatically resumes at the correct LR without any manual step offset.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=MAX_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=total_steps,
        end_value=MIN_LR,
    )
    labels = label_params(params)
    inner  = muon_with_adamw_fallback(
        param_labels=labels,
        muon_lr=MUON_LR,
        adamw_lr=schedule,
        weight_decay=WEIGHT_DECAY,
    )
    # Clip BEFORE the optimizer sees the gradients
    return optax.chain(optax.clip_by_global_norm(GRAD_CLIP), inner)


def init_train_state(model: DeepSeek1B, rng: jax.Array):
    from flax.training import train_state
    dummy  = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy)
    total  = TOTAL_TOKENS // (SEQ_LEN * BATCH_SIZE)
    tx     = make_optimizer(params, total)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN STEP  (fixes 2, 3, 7)
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, donate_argnums=(0,))
def train_step(
    state,
    batch: jnp.ndarray,   # [GRAD_ACCUM, MICRO_BATCH, SEQ_LEN+1]
    rng: jax.Array,
) -> tuple:
    """
    JIT-compiled step with gradient accumulation via jax.lax.scan.

    jax.lax.scan means the accumulation loop is compiled as a single XLA op —
    no Python-level retracing on each micro-batch.  XLA also reuses the gradient
    buffer between micro-steps, keeping peak memory at ~1 copy of gradients.

    donate_argnums=0: XLA may reuse state's HBM buffer for the output new_state.
    """
    # Derive per-step RNG from the current step (deterministic across restarts)
    step_rng = jax.random.fold_in(rng, state.step)   # fix 3

    def loss_fn(params, micro_ids, micro_rng):
        inputs  = micro_ids[:, :-1]    # [M, seq]
        targets = micro_ids[:, 1:]     # [M, seq]

        # fix 1: forward pass in bfloat16 to halve activation memory
        logits, mtp_logits = state.apply_fn(
            to_bf16(params),
            inputs,
            rngs={"dropout": micro_rng},   # fix 3: rng actually passed
        )
        # Loss in float32 for numerical stability
        logits     = logits.astype(jnp.float32)
        mtp_logits = mtp_logits.astype(jnp.float32)
        main = cross_entropy_loss(logits[:, :-1], targets[:, :-1])
        mtp  = cross_entropy_loss(mtp_logits[:, :-1], targets[:, :-1])
        return main + 0.1 * mtp, main

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # ── Accumulate gradients with lax.scan  (fix 2) ───────────────────────────
    def accumulate(carry, xs):
        micro_ids, micro_rng = xs
        (loss, _), grads = grad_fn(state.params, micro_ids, micro_rng)
        # carry: sum of gradients so far; XLA reuses this buffer
        new_carry = jax.tree_util.tree_map(jnp.add, carry, grads)
        return new_carry, loss

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    rngs       = jax.random.split(step_rng, GRAD_ACCUM)   # one rng per micro-batch

    final_grads, per_micro_losses = jax.lax.scan(
        accumulate, zero_grads, (batch, rngs)
    )
    # Average gradients over accumulation steps
    final_grads = jax.tree_util.tree_map(lambda g: g / GRAD_ACCUM, final_grads)

    # Gradient norm before the optimizer clips them  (fix 7)
    grad_norm = optax.global_norm(final_grads)

    new_state = state.apply_gradients(grads=final_grads)

    metrics = {
        "loss":      per_micro_losses.mean(),
        "grad_norm": grad_norm,
    }
    return new_state, metrics

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING  (fix 5)
# ═══════════════════════════════════════════════════════════════════════════════

def _ckpt_path(step: int) -> Path:
    return CKPT_DIR / f"step_{step:08d}"


def save_checkpoint(state, step: int, tokens_seen: int,
                    docs_seen: int, history: list):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    final = _ckpt_path(step)
    tmp   = final.with_suffix(".tmp")

    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(tmp), state)

    meta = {"step": step, "tokens_seen": tokens_seen,
            "docs_seen": docs_seen, "ts": time.time()}
    (tmp / "meta.json").write_text(json.dumps(meta))

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    print(f"[ckpt] step {step:,}  →  {final.name}")

    # Prune old checkpoints
    all_ckpts = sorted(CKPT_DIR.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    for old in all_ckpts[:-CKPT_KEEP]:
        shutil.rmtree(old)

    _save_chart(history)


def load_latest_checkpoint(model: DeepSeek1B, rng: jax.Array):
    """
    Restore full TrainState from the latest Drive checkpoint.

    fix 5: orbax restores state.params, state.opt_state (Adam moments,
    Muon momentum buffers), and state.step from disk.  state.apply_fn and
    state.tx (Python callables) come from the freshly-built base_state.
    Result: the LR schedule resumes correctly because opt_state.count
    (the Adam internal step counter) is also restored.
    """
    if not CKPT_DIR.exists():
        print("[ckpt] No checkpoint dir — starting fresh")
        return init_train_state(model, rng), 0, 0, 0

    all_ckpts = sorted(CKPT_DIR.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if not all_ckpts:
        print("[ckpt] No checkpoints — starting fresh")
        return init_train_state(model, rng), 0, 0, 0

    latest = all_ckpts[-1]
    meta   = json.loads((latest / "meta.json").read_text())
    step, tokens_seen, docs_seen = meta["step"], meta["tokens_seen"], meta["docs_seen"]
    print(f"[ckpt] Resuming step {step:,}  ({tokens_seen/1e9:.2f}B tokens)")

    # Base state provides the pytree structure (apply_fn, tx);
    # orbax fills in params, opt_state, step from disk.
    base  = init_train_state(model, rng)
    ckptr = ocp.PyTreeCheckpointer()
    state = ckptr.restore(str(latest), item=base)

    total = TOTAL_TOKENS // (SEQ_LEN * BATCH_SIZE)
    # LR at restored step — just for logging; schedule comes from opt_state.count
    from train.muon import muon_with_adamw_fallback  # noqa
    import optax as _optax
    restored_lr = float(_optax.warmup_cosine_decay_schedule(
        0.0, MAX_LR, WARMUP_STEPS, total, MIN_LR)(step))
    print(f"[ckpt] LR at step {step}: {restored_lr:.2e}")

    return state, step, tokens_seen, docs_seen


def load_history() -> list:
    if not LOG_FILE.exists():
        return []
    return [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  (fix 6 — doc-level skip)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_tokenizer():
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def _token_stream(dataset, tokenizer) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (doc_index, seq_array) where seq_array has length SEQ_LEN+1.
    Packs documents end-to-end with EOS separators.
    Yields doc_index = how many documents have been consumed so far.
    """
    buf    = []
    eos    = tokenizer.eos_token_id or 0
    n_docs = 0

    for ex in dataset:
        text = ex.get("content") or ex.get("text") or ex.get("passage") or ""
        if not text:
            continue
        buf.extend(tokenizer.encode(text, add_special_tokens=False))
        buf.append(eos)
        n_docs += 1

        while len(buf) >= SEQ_LEN + 1:
            chunk = np.array(buf[:SEQ_LEN + 1], dtype=np.int32)
            buf   = buf[SEQ_LEN + 1:]
            yield n_docs, chunk


def make_batch_stream(skip_docs: int = 0) -> Iterator[tuple[dict, int]]:
    """
    Yield (batch_dict, docs_seen) where batch_dict["input_ids"] has
    shape [GRAD_ACCUM, MICRO_BATCH, SEQ_LEN+1].

    fix 6: skip at doc level — HuggingFace .skip(n) uses itertools.islice,
    which iterates through n documents WITHOUT tokenising them.  For typical
    doc sizes (~300 tokens) skipping 1M docs takes ~30 s, not minutes.
    """
    tokenizer  = _load_tokenizer()
    ds_list, probs = [], []
    for name, cfg_name, prob in DATASET_MIX:
        try:
            kw = dict(split="train", streaming=True, trust_remote_code=True)
            if cfg_name:
                kw["name"] = cfg_name
            ds_list.append(load_dataset(name, **kw))
            probs.append(prob)
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  WARNING {name}: {e}")

    if not ds_list:
        raise RuntimeError("No datasets loaded")

    total = sum(probs)
    probs = [p / total for p in probs]
    mixed = interleave_datasets(ds_list, probabilities=probs, seed=42)

    # Skip already-processed documents (fast, no tokenisation)
    if skip_docs > 0:
        print(f"[data] Skipping {skip_docs:,} documents...")
        mixed = mixed.skip(skip_docs)

    buf = []
    for doc_idx, seq in _token_stream(mixed, tokenizer):
        buf.append(seq)
        if len(buf) == BATCH_SIZE:
            arr = np.stack(buf).reshape(GRAD_ACCUM, MICRO_BATCH, SEQ_LEN + 1)
            yield {"input_ids": jnp.array(arr)}, skip_docs + doc_idx
            buf = []


# ── Prefetch thread  (fix 8) ──────────────────────────────────────────────────

def prefetch(iterator: Iterator, size: int = PREFETCH_SIZE) -> Iterator:
    """
    Run `iterator` in a daemon thread, pre-filling a queue of `size` items.
    The main thread just calls next() on the returned generator.
    """
    q = queue.Queue(maxsize=size)

    def _producer():
        try:
            for item in iterator:
                q.put(item)
        finally:
            q.put(None)  # sentinel

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            return
        yield item

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def _save_chart(history: list):
    if len(history) < 2:
        return
    steps  = [h["step"]       for h in history]
    losses = [h["loss"]       for h in history]
    gnorms = [h.get("grad_norm", 0) for h in history]
    lrs    = [h.get("lr", 0)  for h in history]
    tokB   = [h["tokens_B"]   for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("DeepSeek-1B Training", fontsize=12)

    axes[0].plot(tokB, losses, color="#2563eb", lw=0.8)
    axes[0].set_xlabel("Tokens (B)"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Tokens"); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, losses, color="#2563eb", lw=0.8)
    axes[1].set_xlabel("Step"); axes[1].set_title("Loss vs Step")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, gnorms, color="#dc2626", lw=0.8)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Grad norm")
    axes[2].set_title("Gradient Norm"); axes[2].grid(alpha=0.3)

    axes[3].plot(steps, lrs, color="#16a34a", lw=0.8)
    axes[3].set_xlabel("Step"); axes[3].set_ylabel("LR")
    axes[3].set_title("LR Schedule"); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(CHART_FILE), dpi=120, bbox_inches="tight")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    cfg   = ModelConfig()
    model = DeepSeek1B(cfg, use_remat=True)
    rng   = jax.random.PRNGKey(0)

    state, start_step, tokens_seen, docs_seen = load_latest_checkpoint(model, rng)
    history = load_history()

    total_steps = TOTAL_TOKENS // (SEQ_LEN * BATCH_SIZE)
    print(f"Total steps: {total_steps:,}  |  Resume from: {start_step:,}")
    print(f"Tokens seen: {tokens_seen/1e9:.2f}B  |  Docs seen: {docs_seen:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_stream  = make_batch_stream(skip_docs=docs_seen)
    data_stream = prefetch(raw_stream, size=PREFETCH_SIZE)

    # ── Compilation warmup ────────────────────────────────────────────────────
    # Force JIT compilation on a dummy batch before timing starts.
    print("Compiling train_step (first call triggers XLA compilation)...")
    t_compile = time.time()
    dummy_batch = jnp.zeros((GRAD_ACCUM, MICRO_BATCH, SEQ_LEN + 1), dtype=jnp.int32)
    _state, _ = train_step(state, dummy_batch, rng)
    # Don't use _state — just warm up the compiler; state is donated so use original
    # Re-init state since donation invalidated it
    state, start_step, tokens_seen, docs_seen = load_latest_checkpoint(model, rng)
    print(f"Compilation took {time.time() - t_compile:.1f}s")

    # ── Training loop ─────────────────────────────────────────────────────────
    step          = start_step
    loss_window   = []
    t0            = time.time()

    print("\n=== Training ===")
    for batch, docs_seen in data_stream:
        if step >= total_steps:
            break

        state, metrics = train_step(state, batch["input_ids"], rng)
        step        += 1
        tokens_seen += BATCH_SIZE * SEQ_LEN

        loss      = float(metrics["loss"])
        grad_norm = float(metrics["grad_norm"])
        loss_window.append(loss)
        if len(loss_window) > 100:
            loss_window.pop(0)

        if step % LOG_EVERY == 0:
            elapsed     = time.time() - t0
            tok_per_sec = LOG_EVERY * BATCH_SIZE * SEQ_LEN / max(elapsed, 1e-6)
            t0          = time.time()
            smooth      = sum(loss_window[-50:]) / len(loss_window[-50:])
            ppl         = math.exp(min(smooth, 20))

            # LR from the schedule at current step (for display only)
            lr_now = float(optax.warmup_cosine_decay_schedule(
                0.0, MAX_LR, WARMUP_STEPS, total_steps, MIN_LR)(step))

            record = {
                "step":      step,
                "loss":      round(loss, 4),
                "smooth":    round(smooth, 4),
                "ppl":       round(ppl, 2),
                "grad_norm": round(grad_norm, 4),
                "tokens_B":  round(tokens_seen / 1e9, 4),
                "lr":        lr_now,
                "tok_s":     int(tok_per_sec),
            }
            history.append(record)
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"step {step:>7,} | loss {loss:.4f} | smooth {smooth:.4f} | "
                f"ppl {ppl:.1f} | gn {grad_norm:.2f} | "
                f"{tokens_seen/1e9:.2f}B tok | lr {lr_now:.2e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )

        if step % CKPT_EVERY == 0:
            save_checkpoint(state, step, tokens_seen, docs_seen, history)

    save_checkpoint(state, step, tokens_seen, docs_seen, history)
    print(f"\n=== Done. Step {step:,} | {tokens_seen/1e9:.2f}B tokens ===")


if __name__ == "__main__":
    main()