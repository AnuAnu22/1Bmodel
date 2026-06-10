"""
mHC — Manifold-Constrained Hyper-Connections.

Replaces every layer-to-layer residual connection.  Instead of the standard
    x_{l+1} = x_l + F_l(x_l)
we maintain an expanded residual state H of shape [batch, seq, n_hc, d] and
update it with three dynamically-generated matrices:

    A_l  : [batch, seq, n_hc]          mixing weights  (softmax, sum=1)
    B_l  : [batch, seq, n_hc, n_hc]    residual mix    (doubly stochastic via Sinkhorn)
    C_l  : [batch, seq, n_hc]          output weights  (softmax, sum=1)

Forward pass (per layer):
    x          = einsum('bsn,bsnd->bsd', A_l, H_l)        # mix n_hc channels → d
    block_out  = TransformerBlock(x)                        # attention + FFN (with internal residuals)
    H_{l+1}   = einsum('bsnm,bsmd->bsnd', B_l, H_l)       # mix residual channels
               + einsum('bsn,bsd->bsnd', C_l, block_out)   # inject block output

Simplification vs paper (Eqs 3-7):
  A, B, C are computed from mean(H, axis=n_hc), a d-dim summary.
  The paper may use a richer function of H; this is the efficient variant.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

from config import ModelConfig
from utils.sinkhorn import sinkhorn_normalize


class mHCMixing(nn.Module):
    """
    Generates the three dynamic mixing matrices (A, B, C) from H.
    Instantiated once per TransformerLayer.
    """
    config: ModelConfig

    @nn.compact
    def __call__(
        self, H: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            H : [batch, seq, n_hc, d]  current residual state.

        Returns:
            A : [batch, seq, n_hc]          softmax channel-mix weights
            B : [batch, seq, n_hc, n_hc]    doubly-stochastic residual mix
            C : [batch, seq, n_hc]          softmax output-injection weights
        """
        cfg = self.config
        n_hc, d = cfg.n_hc, cfg.hidden_dim

        # Summarise: mean over channels → [batch, seq, d]
        h_mean = H.mean(axis=-2)

        # A: [batch, seq, n_hc]
        A = jax.nn.softmax(nn.Dense(n_hc, use_bias=False, name="W_A")(h_mean), axis=-1)

        # B (log-domain, then Sinkhorn): [batch, seq, n_hc, n_hc]
        B_log = nn.Dense(n_hc * n_hc, use_bias=False, name="W_B")(h_mean)
        B_log = B_log.reshape(*h_mean.shape[:-1], n_hc, n_hc)
        B = sinkhorn_normalize(B_log, n_iters=cfg.sinkhorn_iters)

        # C: [batch, seq, n_hc]
        C = jax.nn.softmax(nn.Dense(n_hc, use_bias=False, name="W_C")(h_mean), axis=-1)

        return A, B, C


def mhc_init(
    tokens_embedded: jnp.ndarray,   # [batch, seq, d]
    n_hc: int,
) -> jnp.ndarray:
    """
    Initialise the mHC residual state H from the embedding layer output.
    Channel 0 gets the actual embeddings; remaining channels start as zeros.

    Returns:
        H : [batch, seq, n_hc, d]
    """
    batch, seq, d = tokens_embedded.shape
    zeros = jnp.zeros((batch, seq, n_hc - 1, d), dtype=tokens_embedded.dtype)
    return jnp.concatenate(
        [tokens_embedded[:, :, None, :], zeros], axis=2
    )   # [batch, seq, n_hc, d]


def mhc_update(
    H: jnp.ndarray,                  # [batch, seq, n_hc, d]
    block_out: jnp.ndarray,          # [batch, seq, d]  — output of Attn+FFN block
    A: jnp.ndarray,                  # [batch, seq, n_hc]
    B: jnp.ndarray,                  # [batch, seq, n_hc, n_hc]
    C: jnp.ndarray,                  # [batch, seq, n_hc]
) -> jnp.ndarray:
    """
    Apply the mHC state update: H_new = B @ H + C ⊗ block_out.
    Pure function, no parameters.

    Returns:
        H_new : [batch, seq, n_hc, d]
    """
    # Mix existing residual channels
    H_mixed = jnp.einsum("bsnm,bsmd->bsnd", B, H)         # [b, s, n_hc, d]
    # Inject new block output into all channels
    H_inject = jnp.einsum("bsn,bsd->bsnd", C, block_out)  # [b, s, n_hc, d]
    return H_mixed + H_inject


def mix_channels(
    H: jnp.ndarray,    # [batch, seq, n_hc, d]
    A: jnp.ndarray,    # [batch, seq, n_hc]
) -> jnp.ndarray:
    """
    Produce the transformer block input by mixing the n_hc residual channels.

    Returns:
        x : [batch, seq, d]
    """
    return jnp.einsum("bsn,bsnd->bsd", A, H)
