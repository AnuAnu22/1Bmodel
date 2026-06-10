"""
TransformerLayer — one full block of the dense DeepSeek-V4 architecture.

Each layer:
  1. Generates dynamic mixing matrices A, B, C from the mHC residual state H.
  2. Mixes the n_hc residual channels → transformer block input x.
  3. Runs a pre-norm attention sub-block (with its own internal residual).
  4. Runs a pre-norm FFN sub-block (with its own internal residual).
  5. Updates H via the mHC outer rule:
       H_new = B @ H + C ⊗ block_out

The mHC replaces the layer-to-layer residual; the sub-block residuals (add
after attention and FFN) remain, as they are internal to the transformer block.

Gradient checkpointing is applied via nn.remat — each layer recomputes its
activations during backprop rather than storing them, trading 2× FLOPS for
O(1) activation memory per layer.
"""

from __future__ import annotations
import flax.linen as nn
import jax.numpy as jnp

from config import ModelConfig
from model.norm import RMSNorm
from model.mhc import mHCMixing, mix_channels, mhc_update
from model.ffn import SwiGLUFFN
from model.attention import CSAttention, HCAttention


class TransformerLayer(nn.Module):
    config: ModelConfig
    layer_type: str    # 'csa' | 'hca'

    def setup(self) -> None:
        cfg = self.config
        self.mhc = mHCMixing(cfg)
        self.attention = (
            CSAttention(cfg) if self.layer_type == "csa" else HCAttention(cfg)
        )
        self.ffn      = SwiGLUFFN(cfg)
        self.attn_norm = RMSNorm()
        self.ffn_norm  = RMSNorm()

    def __call__(
        self,
        H: jnp.ndarray,          # [batch, seq, n_hc, hidden_dim]
        positions: jnp.ndarray,  # [seq]  integer positions
    ) -> jnp.ndarray:
        """
        Returns:
            H_new : [batch, seq, n_hc, hidden_dim]
        """
        # ── mHC: generate dynamic matrices and mix channels ────────────────
        A, B, C = self.mhc(H)
        x = mix_channels(H, A)                                # [b, s, d]

        # ── Attention sub-block (pre-norm, with internal residual) ─────────
        x = x + self.attention(self.attn_norm(x), positions)  # [b, s, d]

        # ── FFN sub-block (pre-norm, with internal residual) ───────────────
        block_out = x + self.ffn(self.ffn_norm(x))            # [b, s, d]

        # ── mHC outer update (replaces layer-to-layer residual) ───────────
        return mhc_update(H, block_out, A, B, C)              # [b, s, n_hc, d]


# Gradient-checkpointed variant — use this for actual training.
# Recomputes activations on the backward pass; saves ~(n_layers - 1) × layer
# activation memory at the cost of one extra forward pass per layer.
RematTransformerLayer = nn.remat(TransformerLayer, prevent_cse=False)
