"""
Muon Optimizer — optax-compatible implementation.

Reference: Jordan (2024) "Muon: An optimizer for hidden layers in neural networks"
           DeepSeek-V4 Algorithm 1 (hybrid NS with 10 iterations over two stages).

Rules:
  • 2D+ weight matrices → SGD momentum + Newton-Schulz orthogonalization
  • 1D / scalar params  → plain SGD momentum (biases, RMSNorm scales, position biases)

The NS iteration is split into two stages compiled via jax.lax.scan:
  Stage 1 (quintic): rapid convergence of singular values toward 1.
  Stage 2 (cubic):   stabilisation.

Then rescale the output to the correct RMS for a semi-orthogonal matrix in R^{n×m}:
    RMS_target = 1 / sqrt(max(n, m))

Usage:
    tx = muon_optimizer(muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    opt_state = tx.init(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
"""

from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


# ── Newton-Schulz orthogonalization ───────────────────────────────────────────

def newton_schulz(G: jnp.ndarray, steps: int = 10) -> jnp.ndarray:
    """
    Orthogonalize a 2D matrix G via two-stage Newton-Schulz polynomial iteration.

    Args:
        G     : [n, m]  input matrix (any real values, will be normalised first).
        steps : total NS iterations (split evenly between the two stages).

    Returns:
        [n, m]  approximately semi-orthogonal matrix, rescaled to correct RMS.
    """
    assert G.ndim == 2, f"newton_schulz expects a 2D matrix, got shape {G.shape}"
    n, m = G.shape

    # Work on the shorter dimension to keep A = G @ G.T small.
    transposed = n > m
    if transposed:
        G = G.T          # G is now [m, n] with m ≤ n

    # Normalise spectral norm to ≤ 1 for convergence guarantee.
    G = G / (jnp.linalg.norm(G) + 1e-7)

    # ── Stage 1: quintic polynomial — rapid convergence ───────────────────
    # Coefficients from Jordan (2024) / DeepSeek-V4 Algorithm 1.
    a1, b1, c1 = 3.4445, -4.7750, 2.0315

    def _stage1(G: jnp.ndarray, _: None):
        A = G @ G.T
        return a1 * G + b1 * (A @ G) + c1 * (A @ (A @ G)), None

    n1 = steps // 2
    G, _ = jax.lax.scan(_stage1, G, xs=None, length=n1)

    # ── Stage 2: cubic polynomial — stabilisation ─────────────────────────
    a2, b2 = 1.5, -0.5

    def _stage2(G: jnp.ndarray, _: None):
        A = G @ G.T
        return a2 * G + b2 * (A @ G), None

    n2 = steps - n1
    G, _ = jax.lax.scan(_stage2, G, xs=None, length=n2)

    # ── Rescale to target RMS ─────────────────────────────────────────────
    # A semi-orthogonal matrix in R^{r×c} has entries with RMS = 1/sqrt(max(r,c)).
    r, c = (m, n) if transposed else (n, m)   # original shape after possible transpose-back
    target_rms = 1.0 / jnp.sqrt(jnp.maximum(r, c))
    cur_rms    = jnp.sqrt(jnp.mean(G ** 2)) + 1e-8
    G = G * (target_rms / cur_rms)

    if transposed:
        G = G.T

    return G


# ── optax GradientTransformation ──────────────────────────────────────────────

class MuonState(NamedTuple):
    """Per-parameter momentum buffers."""
    momentum: Any   # same pytree structure as params


def muon_transform(
    momentum: float = 0.95,
    ns_steps: int   = 10,
    nesterov: bool  = True,
) -> optax.GradientTransformation:
    """
    Core Muon update (no learning-rate scaling — compose with optax.scale).

    For each parameter leaf:
      m  ←  momentum * m  +  g
      ĝ  =  (momentum * m + g)   if nesterov  else  m          [Nesterov look-ahead]
      if param is 2D+:  update = newton_schulz(reshape_2d(ĝ)).reshape_back()
      else:             update = ĝ                              [vanilla momentum]
    """

    def init_fn(params: Any) -> MuonState:
        return MuonState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(
        updates: Any,
        state: MuonState,
        params: Any = None,
    ):
        grads_flat,  treedef = jax.tree_util.tree_flatten(updates)
        mom_flat,    _       = jax.tree_util.tree_flatten(state.momentum)

        new_grads, new_mom = [], []
        for g, m in zip(grads_flat, mom_flat):
            new_m  = momentum * m + g
            g_hat  = (momentum * new_m + g) if nesterov else new_m

            if g_hat.ndim >= 2:
                # Flatten to 2D, orthogonalise, restore shape.
                shape   = g_hat.shape
                g_2d    = g_hat.reshape(shape[0], -1)
                g_orth  = newton_schulz(g_2d, steps=ns_steps)
                update  = g_orth.reshape(shape)
            else:
                update = g_hat   # 1D / scalar: plain momentum

            new_grads.append(update)
            new_mom.append(new_m)

        return (
            treedef.unflatten(new_grads),
            MuonState(momentum=treedef.unflatten(new_mom)),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def muon_optimizer(
    muon_lr: float      = 0.02,
    adamw_lr: float     = 3e-4,
    weight_decay: float = 0.01,
    momentum: float     = 0.95,
    ns_steps: int       = 10,
    nesterov: bool      = True,
) -> optax.GradientTransformation:
    """
    Full Muon optimizer: NS-momentum for 2D+ matrices, AdamW for everything else.

    Per the paper:
      • Muon (this transform) is applied to hidden-layer weight matrices.
      • AdamW is used for: embeddings, LM head, mHC params, RMSNorm scales.

    Implementation note:
      We apply Muon uniformly to ALL parameter leaves here. In practice, use
      optax.multi_transform with a label function to restrict Muon to 2D+
      hidden weights and AdamW to the rest.  The `muon_transform` already
      degrades gracefully to plain momentum for 1D params, so this is safe
      as a baseline — just slightly suboptimal for the excluded params.

    For a production split, see `muon_with_adamw_fallback` below.
    """
    return optax.chain(
        muon_transform(momentum=momentum, ns_steps=ns_steps, nesterov=nesterov),
        optax.scale(-muon_lr),
    )


def muon_with_adamw_fallback(
    param_labels: Any,           # pytree of 'muon' | 'adamw' strings
    muon_lr: float      = 0.02,
    adamw_lr: float     = 3e-4,
    weight_decay: float = 0.01,
    momentum: float     = 0.95,
    ns_steps: int       = 10,
) -> optax.GradientTransformation:
    """
    Proper hybrid: Muon for hidden 2D+ weights, AdamW for excluded params.

    param_labels is a pytree matching params with values 'muon' or 'adamw'.
    Build it with label_params() below.
    """
    return optax.multi_transform(
        {
            "muon": optax.chain(
                muon_transform(momentum=momentum, ns_steps=ns_steps),
                optax.scale(-muon_lr),
            ),
            "adamw": optax.chain(
                optax.add_decayed_weights(weight_decay),
                optax.adam(learning_rate=adamw_lr),
            ),
        },
        param_labels,
    )


def label_params(params: Any) -> Any:
    """
    Walk the param pytree and label each leaf 'muon' or 'adamw'.

    Labelling rules:
      'adamw': embeddings, lm_head, mhc (W_A/W_B/W_C), RMSNorm scales,
               position biases (1D), any param with ndim < 2.
      'muon':  everything else (hidden 2D+ weight matrices).
    """
    ADAMW_KEYWORDS = {"embedding", "embed_table", "lm_head", "W_A", "W_B",
                      "W_C", "scale", "bias_a", "bias_b", "bias_hca"}

    def _label(path, leaf):
        path_str = "/".join(str(k) for k in path)
        if leaf.ndim < 2:
            return "adamw"
        if any(kw in path_str for kw in ADAMW_KEYWORDS):
            return "adamw"
        return "muon"

    return jax.tree_util.tree_map_with_path(_label, params)
