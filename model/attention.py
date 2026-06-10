"""
Compressed Attention: CSA (Compressed Sparse Attention) and HCA (Heavily
Compressed Attention).

Both share the same skeleton:
  1.  Low-rank query projection  →  Q  [b, s, n_heads, head_dim]
  2.  KV compression  →  C_comp  [b, n_blocks, head_dim]   (K = V in MQA)
  3.  Sliding-window local KV    →  kv_local  [b, s, sw, head_dim]
  4.  RMSNorm on Q heads and KV entries (per paper §2.3.3)
  5.  RoPE on last rope_dims of Q and KV
  6.  Causal masking of compressed blocks
  7.  Joint softmax over [global_scores | local_scores]
  8.  Output projection  →  [b, s, hidden_dim]

CSA additions:
  - Dual-stream KV compression with learnable position biases
  - Lightning Indexer: low-rank indexer queries → top-k block selection
    (per-position sparse gather before the attention dot-product)
  - Validity mask: blocks gathered for positions where no valid past block
    exists are masked to -inf before the joint softmax.

HCA difference:
  - Single-stream compression, denser rate (m'=128 vs m=4)
  - Dense attention over ALL past compressed blocks (no Lightning Indexer)
  - Causal mask applied directly to attention scores

MQA: a single KV head is shared across all n_heads query heads.
     The compressed entry serves as BOTH key and value.

Flax note:
  All sub-modules and parameters are defined in setup(). The Bias helper
  module is used for learnable 1-D bias vectors (self.param cannot be called
  directly in non-compact methods).
"""

from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from config import ModelConfig
from model.norm import RMSNorm
from model.rope import make_rope_embeds, apply_rope, apply_rope_1d

NEG_INF = jnp.finfo(jnp.float32).min


# ── tiny helpers ───────────────────────────────────────────────────────────────

class _Bias(nn.Module):
    """Learnable 1-D bias vector. Wraps self.param so it can live in setup()."""
    size: int

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        return self.param("bias", nn.initializers.zeros, (self.size,))


def _build_sliding_window_kv(kv: jnp.ndarray, sw: int) -> jnp.ndarray:
    """
    Build causal sliding-window KV.

    For each position t, collects the sw most-recent entries ending at t.
    Positions before 0 are zero-padded. Causal by construction:
      local[b, t, sw-1, :] = kv[b, t, :]  (most recent = current position)
      local[b, t,  0,   :] = kv[b, t-sw+1, :]  (oldest in window, or 0 if padded)

    Returns:
        [batch, seq, sw, head_dim]
    """
    batch, seq, d = kv.shape
    padded  = jnp.pad(kv, ((0, 0), (sw - 1, 0), (0, 0)))  # [b, sw-1+seq, d]
    t_idx   = jnp.arange(seq)[None, :]                      # [1, seq]
    i_idx   = jnp.arange(sw)[:, None]                       # [sw, 1]
    indices = t_idx + i_idx                                  # [sw, seq]
    local   = padded[:, indices, :]                          # [b, sw, seq, d]
    return local.transpose(0, 2, 1, 3)                       # [b, seq, sw, d]


def _causal_block_mask(seq: int, n_blocks: int, rate: int) -> jnp.ndarray:
    """
    [seq, n_blocks] bool mask: True where block i is fully past at position t.
    Block i covers tokens [i*rate, (i+1)*rate). Accessible when (i+1)*rate <= t+1.
    """
    ends = (jnp.arange(n_blocks) + 1) * rate          # [n_blocks]
    pos  = jnp.arange(seq)                             # [seq]
    return ends[None, :] <= (pos[:, None] + 1)         # [seq, n_blocks]


# ── base class ─────────────────────────────────────────────────────────────────

class _CompressedAttentionBase(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        # Low-rank query: d → q_latent → n_heads * head_dim
        self.W_DQ        = nn.Dense(cfg.q_latent_dim,           use_bias=False)
        self.W_UQ        = nn.Dense(cfg.n_heads * cfg.head_dim, use_bias=False)
        self.q_norm      = RMSNorm()
        # Local (sliding-window) KV
        self.W_kv_local  = nn.Dense(cfg.head_dim, use_bias=False)
        self.kv_local_norm = RMSNorm()
        # Compressed-global KV norm (applied after compression)
        self.kv_comp_norm = RMSNorm()
        # Output projection
        self.W_out = nn.Dense(cfg.hidden_dim, use_bias=False)

    # ── queries ────────────────────────────────────────────────────────────────

    def _queries(
        self,
        x: jnp.ndarray,    # [b, s, d]
        cos: jnp.ndarray,
        sin: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg   = self.config
        b, s, _ = x.shape
        q = self.W_UQ(self.W_DQ(x)).reshape(b, s, cfg.n_heads, cfg.head_dim)
        # Per-head RMSNorm (vmap over batch, seq, head)
        q = jax.vmap(jax.vmap(jax.vmap(self.q_norm)))(q)
        return apply_rope(q, cos, sin, cfg.rope_dims)

    # ── sliding-window local KV ────────────────────────────────────────────────

    def _local_kv(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg = self.config
        kv  = self.kv_local_norm(self.W_kv_local(x))          # [b, s, d_h]
        kv  = apply_rope_1d(kv, cos, sin, cfg.rope_dims)
        return _build_sliding_window_kv(kv, cfg.sliding_window)  # [b, s, sw, d_h]

    # ── attention + projection ────────────────────────────────────────────────

    def _attend(
        self,
        q: jnp.ndarray,                   # [b, s, n, d_h]
        kv_global: jnp.ndarray,           # [b, K, d_h]  OR  [b, s, K, d_h]
        kv_local: jnp.ndarray,            # [b, s, sw, d_h]
        causal_mask: Optional[jnp.ndarray],  # [s, K] or None
        per_position: bool = False,
        valid_mask: Optional[jnp.ndarray] = None,  # [b, s, K] or None
    ) -> jnp.ndarray:
        """
        Joint softmax over concatenated global + local scores.

        per_position=True  → kv_global : [b, s, K, d_h]  (CSA per-query gather)
        per_position=False → kv_global : [b, K, d_h]      (HCA shared)

        valid_mask : [b, s, K] — True for blocks that are causally valid.
                     Blocks where valid_mask=False are set to NEG_INF before
                     the joint softmax, preventing future-token leakage when
                     top-k selection filled invalid slots with arbitrary indices.
        """
        scale = self.config.head_dim ** -0.5

        if per_position:
            scores_g = jnp.einsum("bsnd,bskd->bsnk", q, kv_global) * scale
            if valid_mask is not None:
                # valid_mask [b, s, K] → [b, s, 1, K]
                scores_g = jnp.where(valid_mask[:, :, None, :], scores_g, NEG_INF)
        else:
            scores_g = jnp.einsum("bsnd,bkd->bsnk", q, kv_global) * scale
            if causal_mask is not None:
                # causal_mask [s, K] → [1, s, 1, K]
                scores_g = jnp.where(causal_mask[None, :, None, :], scores_g, NEG_INF)

        scores_l = jnp.einsum("bsnd,bswd->bsnw", q, kv_local) * scale

        scores   = jnp.concatenate([scores_g, scores_l], axis=-1)
        weights  = jax.nn.softmax(scores, axis=-1)
        K_g      = scores_g.shape[-1]
        w_g, w_l = weights[..., :K_g], weights[..., K_g:]

        if per_position:
            out_g = jnp.einsum("bsnk,bskd->bsnd", w_g, kv_global)
        else:
            out_g = jnp.einsum("bsnk,bkd->bsnd", w_g, kv_global)

        out_l = jnp.einsum("bsnw,bswd->bsnd", w_l, kv_local)
        out   = (out_g + out_l).reshape(*q.shape[:2], -1)    # [b, s, n*d_h]
        return self.W_out(out)                                 # [b, s, d]

    def __call__(self, x, positions):
        raise NotImplementedError


# ── CSA ────────────────────────────────────────────────────────────────────────

class CSAttention(_CompressedAttentionBase):
    """
    Compressed Sparse Attention — dual-stream KV + Lightning Indexer top-k.
    """

    def setup(self) -> None:
        super().setup()
        cfg = self.config
        m   = cfg.csa_compression_rate
        # Dual-stream KV and logit projections
        self.W_KV_a  = nn.Dense(cfg.head_dim, use_bias=False)
        self.W_KV_b  = nn.Dense(cfg.head_dim, use_bias=False)
        self.W_Z_a   = nn.Dense(1, use_bias=False)
        self.W_Z_b   = nn.Dense(1, use_bias=False)
        # Learnable position biases [m]
        self.bias_a  = _Bias(m)
        self.bias_b  = _Bias(m)
        # Lightning Indexer
        self.W_DI    = nn.Dense(cfg.q_latent_dim,                              use_bias=False)
        self.W_UI    = nn.Dense(cfg.csa_indexer_heads * cfg.csa_indexer_dim,   use_bias=False)
        self.W_IK    = nn.Dense(cfg.csa_indexer_dim,                           use_bias=False)

    def _compress(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Dual-stream KV compression → C_comp : [b, n_valid, head_dim].
        n_valid = seq // csa_compression_rate  (complete past blocks only).
        """
        cfg = self.config
        m   = cfg.csa_compression_rate
        b, seq, _ = x.shape
        pad = (m - seq % m) % m

        # Stream a: project then pad
        C_a  = self.W_KV_a(x)                              # [b, seq, d_h]
        Z_a  = self.W_Z_a(C_a).squeeze(-1)                 # [b, seq]
        if pad > 0:
            C_a = jnp.pad(C_a, ((0,0),(0,pad),(0,0)))
            Z_a = jnp.pad(Z_a, ((0,0),(0,pad)))

        # Stream b: identical positions, independent projection
        C_b  = self.W_KV_b(x)
        Z_b  = self.W_Z_b(C_b).squeeze(-1)
        if pad > 0:
            C_b = jnp.pad(C_b, ((0,0),(0,pad),(0,0)))
            Z_b = jnp.pad(Z_b, ((0,0),(0,pad)))

        nb = (seq + pad) // m
        B_a = self.bias_a()   # [m]
        B_b = self.bias_b()

        # Reshape into blocks
        C_a_bl = C_a.reshape(b, nb, m, cfg.head_dim)       # [b, nb, m, d_h]
        C_b_bl = C_b.reshape(b, nb, m, cfg.head_dim)
        Z_a_bl = Z_a.reshape(b, nb, m) + B_a               # [b, nb, m]
        Z_b_bl = Z_b.reshape(b, nb, m) + B_b

        # Softmax over 2m logits → weighted sum per block
        w   = jax.nn.softmax(jnp.concatenate([Z_a_bl, Z_b_bl], axis=-1), axis=-1)
        w_a, w_b = w[..., :m], w[..., m:]
        C_comp = (jnp.einsum("bni,bnid->bnd", w_a, C_a_bl) +
                  jnp.einsum("bni,bnid->bnd", w_b, C_b_bl))   # [b, nb, d_h]

        # Trim to complete (causal) blocks; apply norm + RoPE
        n_valid = seq // m
        C_comp  = self.kv_comp_norm(C_comp[:, :n_valid, :])
        centres = ((jnp.arange(n_valid) + 0.5) * m).astype(jnp.int32)
        cos_c, sin_c = make_rope_embeds(centres, cfg.rope_dims, cfg.rope_theta)
        return apply_rope_1d(C_comp, cos_c, sin_c, cfg.rope_dims)

    def _index(
        self,
        x: jnp.ndarray,
        C_comp: jnp.ndarray,
        causal_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Lightning Indexer: top-k sparse block selection.

        Returns:
            C_sparse  : [b, s, k, d_h]
            valid_mask: [b, s, k]  True where the gathered block is causal-valid.
        """
        cfg = self.config
        b, seq, _ = x.shape
        n_blocks  = C_comp.shape[1]

        q_I = self.W_UI(self.W_DI(x)).reshape(
            b, seq, cfg.csa_indexer_heads, cfg.csa_indexer_dim
        )                                                      # [b, s, n_Ih, c_I]
        K_I = self.W_IK(C_comp)                               # [b, n_blocks, c_I]

        # Score: average over indexer heads
        scores = jnp.einsum(
            "bshc,bkc->bshk", q_I, K_I
        ) * (cfg.csa_indexer_dim ** -0.5)                     # [b, s, n_Ih, nb]
        scores = scores.mean(axis=2)                           # [b, s, nb]

        # Apply causal mask (block must be fully in the past)
        scores = jnp.where(causal_mask[None, :, :], scores, NEG_INF)

        k = min(cfg.csa_top_k, n_blocks)
        top_vals, top_idx = jax.lax.top_k(scores, k)          # [b, s, k]

        # Validity: True if the score was NOT NEG_INF (i.e., block was accessible)
        valid_mask = top_vals > (NEG_INF * 0.5)                # [b, s, k]

        def gather(c, idx):          # c:[nb,d_h], idx:[s,k] → [s,k,d_h]
            return c[idx]

        C_sparse = jax.vmap(gather)(C_comp, top_idx)           # [b, s, k, d_h]
        return C_sparse, valid_mask

    def __call__(
        self,
        x: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg = self.config
        _, seq, _ = x.shape
        cos, sin = make_rope_embeds(positions, cfg.rope_dims, cfg.rope_theta)

        q        = self._queries(x, cos, sin)
        C_comp   = self._compress(x)
        n_blocks = C_comp.shape[1]

        causal_mask       = _causal_block_mask(seq, n_blocks, cfg.csa_compression_rate)
        C_sparse, v_mask  = self._index(x, C_comp, causal_mask)
        kv_local          = self._local_kv(x, cos, sin)

        return self._attend(
            q, C_sparse, kv_local,
            causal_mask=None,
            per_position=True,
            valid_mask=v_mask,
        )


# ── HCA ────────────────────────────────────────────────────────────────────────

class HCAttention(_CompressedAttentionBase):
    """
    Heavily Compressed Attention — single-stream, dense over all past blocks.
    """

    def setup(self) -> None:
        super().setup()
        cfg = self.config
        self.W_KV  = nn.Dense(cfg.head_dim, use_bias=False)
        self.W_Z   = nn.Dense(1,            use_bias=False)
        self.bias  = _Bias(cfg.hca_compression_rate)

    def _compress(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        m   = cfg.hca_compression_rate
        b, seq, _ = x.shape
        pad = (m - seq % m) % m

        C = self.W_KV(x)
        Z = self.W_Z(C).squeeze(-1)
        if pad > 0:
            C = jnp.pad(C, ((0,0),(0,pad),(0,0)))
            Z = jnp.pad(Z, ((0,0),(0,pad)))

        nb = (seq + pad) // m
        B  = self.bias()
        C_bl = C.reshape(b, nb, m, cfg.head_dim)
        Z_bl = Z.reshape(b, nb, m) + B
        w    = jax.nn.softmax(Z_bl, axis=-1)
        C_comp = jnp.einsum("bni,bnid->bnd", w, C_bl)  # [b, nb, d_h]

        n_valid = seq // m
        C_comp  = self.kv_comp_norm(C_comp[:, :n_valid, :])
        centres = ((jnp.arange(n_valid) + 0.5) * m).astype(jnp.int32)
        cos_c, sin_c = make_rope_embeds(centres, cfg.rope_dims, cfg.rope_theta)
        return apply_rope_1d(C_comp, cos_c, sin_c, cfg.rope_dims)

    def __call__(
        self,
        x: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        cfg = self.config
        _, seq, _ = x.shape
        cos, sin = make_rope_embeds(positions, cfg.rope_dims, cfg.rope_theta)

        q        = self._queries(x, cos, sin)
        C_comp   = self._compress(x)
        n_blocks = C_comp.shape[1]
        c_mask   = _causal_block_mask(seq, n_blocks, cfg.hca_compression_rate)
        kv_local = self._local_kv(x, cos, sin)

        return self._attend(
            q, C_comp, kv_local,
            causal_mask=c_mask,
            per_position=False,
        )
