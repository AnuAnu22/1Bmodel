"""
Trainer — TrainState, loss, and a JIT-compiled train step.

Loss:
    L = CrossEntropy(logits[:, :-1], tokens[:, 1:])
      + mtp_weight * CrossEntropy(mtp_logits[:, :-1], tokens[:, 1:])

The main and MTP heads predict the same shifted targets but from different
representations (main model vs lightweight MTP block after main model).

Usage:
    model = DeepSeek1B(config)
    state = create_train_state(model, config, rng, sample_batch)
    for batch in dataloader:
        state, metrics = train_step(state, batch)
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from config import ModelConfig
from model.model import DeepSeek1B
from train.muon import muon_with_adamw_fallback, label_params


# ── Loss ──────────────────────────────────────────────────────────────────────

def cross_entropy_loss(
    logits: jnp.ndarray,    # [batch, seq, vocab]
    targets: jnp.ndarray,   # [batch, seq]  integer token ids
) -> jnp.ndarray:
    """Mean cross-entropy over all (batch, position) pairs."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)        # [b, s, vocab]
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[..., None], axis=-1
    ).squeeze(-1)                                           # [b, s]
    return -target_log_probs.mean()


# ── TrainState ────────────────────────────────────────────────────────────────

class TrainState(train_state.TrainState):
    """Extends flax TrainState with a step counter (already included in base)."""
    pass


def create_train_state(
    model: DeepSeek1B,
    config: ModelConfig,
    rng: jax.Array,
    sample_ids: jnp.ndarray,       # [1, seq]  dummy input for param init
    muon_lr: float      = 0.02,
    adamw_lr: float     = 3e-4,
    weight_decay: float = 0.01,
) -> TrainState:
    """
    Initialise parameters and optimizer state.

    Uses the proper Muon / AdamW split based on param labels.
    """
    # Initialise params with a dummy forward pass
    params = model.init(rng, sample_ids)

    # Build param labels for the Muon / AdamW split
    labels = label_params(params)

    tx = muon_with_adamw_fallback(
        param_labels=labels,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ── Train step ────────────────────────────────────────────────────────────────

@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],   # {'input_ids': [batch, seq+1]}
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Single gradient update step. JIT-compiled.

    Expects input_ids of length seq+1; slices into:
        inputs  = input_ids[:, :-1]
        targets = input_ids[:, 1:]

    Returns:
        new_state : updated TrainState
        metrics   : dict with 'loss', 'main_loss', 'mtp_loss'
    """
    input_ids = batch["input_ids"]
    inputs    = input_ids[:, :-1]
    targets   = input_ids[:, 1:]

    def loss_fn(params: Any):
        logits, mtp_logits = state.apply_fn(params, inputs)

        # Shift: predict position t+1 from position t
        main_loss = cross_entropy_loss(logits[:, :-1], targets[:, :-1])
        mtp_loss  = cross_entropy_loss(mtp_logits[:, :-1], targets[:, :-1])

        # Combined (note: config is captured from outer scope via closure)
        # We can't easily access config here without passing it, so we use a
        # fixed weight; override by modifying this function for custom schedules.
        mtp_weight = 0.1
        total_loss = main_loss + mtp_weight * mtp_loss

        return total_loss, {"main_loss": main_loss, "mtp_loss": mtp_loss}

    (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    new_state = state.apply_gradients(grads=grads)

    metrics = {
        "loss":      loss,
        "main_loss": aux_metrics["main_loss"],
        "mtp_loss":  aux_metrics["mtp_loss"],
        "step":      state.step,
    }
    return new_state, metrics


# ── Perplexity helper ─────────────────────────────────────────────────────────

def compute_perplexity(loss: jnp.ndarray) -> jnp.ndarray:
    """Perplexity from mean cross-entropy loss."""
    return jnp.exp(loss)
    