"""
Rotary Position Embeddings (RoPE).

Following V4: RoPE is applied only to the LAST `rope_dims` dimensions of
each attention head, leaving the first (head_dim - rope_dims) dimensions
untouched. This matches the paper's "apply to last 64 of 512 dims".

Functions are pure JAX (no nn.Module) — called from attention modules.
"""

from __future__ import annotations
import jax.numpy as jnp


# ── Frequency table ────────────────────────────────────────────────────────────

def make_rope_embeds(
    positions: jnp.ndarray,    # [seq]  integer positions
    rope_dims: int,
    theta: float = 10_000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build (cos, sin) tables for the given positions.

    Returns:
        cos, sin : each [seq, rope_dims // 2]
    """
    half = rope_dims // 2
    # Frequency for each dimension pair: θ_i = 1 / theta^(2i / rope_dims)
    freqs = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    # Outer product: angle[t, i] = position[t] * freq[i]
    angles = jnp.outer(positions.astype(jnp.float32), freqs)   # [seq, half]
    return jnp.cos(angles), jnp.sin(angles)


# ── Rotation ───────────────────────────────────────────────────────────────────

def apply_rope(
    x: jnp.ndarray,                          # [batch, seq, n_heads, head_dim]
    cos: jnp.ndarray,                         # [seq, rope_dims // 2]
    sin: jnp.ndarray,                         # [seq, rope_dims // 2]
    rope_dims: int,
) -> jnp.ndarray:
    """
    Apply RoPE to the last `rope_dims` of the head dimension.

    Returns:
        [batch, seq, n_heads, head_dim]  with last rope_dims rotated.
    """
    half = rope_dims // 2

    # Split head dim into [pass-through | rotate]
    x_pass = x[..., :-rope_dims]           # [b, s, h, head_dim - rope_dims]
    x_rot  = x[..., -rope_dims:]           # [b, s, h, rope_dims]

    x1 = x_rot[..., :half]                 # [b, s, h, half]
    x2 = x_rot[..., half:]                 # [b, s, h, half]

    # cos/sin: [seq, half] → broadcast to [1, seq, 1, half]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]

    x_rotated = jnp.concatenate(
        [x1 * c - x2 * s,
         x1 * s + x2 * c],
        axis=-1,
    )                                       # [b, s, h, rope_dims]

    return jnp.concatenate([x_pass, x_rotated], axis=-1)


def apply_rope_1d(
    x: jnp.ndarray,                          # [batch, seq, head_dim]  (single head / compressed KV)
    cos: jnp.ndarray,                         # [seq, rope_dims // 2]
    sin: jnp.ndarray,
    rope_dims: int,
) -> jnp.ndarray:
    """
    RoPE for a tensor without an explicit head axis.
    Used for compressed KV entries (MQA single-head).

    Returns:
        [batch, seq, head_dim]
    """
    half = rope_dims // 2
    x_pass = x[..., :-rope_dims]
    x_rot  = x[..., -rope_dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]

    c = cos[None, :, :]    # [1, seq, half]
    s = sin[None, :, :]

    x_rotated = jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    return jnp.concatenate([x_pass, x_rotated], axis=-1)
    