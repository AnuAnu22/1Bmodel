"""
MTPHead — Multi-Token Prediction auxiliary head.

After the main model produces its final hidden states h : [b, s, d], the MTP
head applies one lightweight transformer block (same config) and projects to
vocab to predict the NEXT token at every position.

The auxiliary loss is:
    L_mtp = CrossEntropy(mtp_logits[:, :-1], targets[:, 1:])

This is added to the main loss weighted by config.mtp_loss_weight.

The embedding matrix is intentionally NOT tied here — pass the embed_table
if you want weight tying (see DeepSeek1B for how to thread it through).
During a future optimisation pass, wire in the shared embed_table.
"""

from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn

from config import ModelConfig
from model.norm import RMSNorm
from model.attention import HCAttention   # HCA is simpler; suitable for MTP
from model.ffn import SwiGLUFFN


class MTPHead(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        # Lightweight single transformer block
        self.attn      = HCAttention(cfg)
        self.ffn       = SwiGLUFFN(cfg)
        self.attn_norm = RMSNorm()
        self.ffn_norm  = RMSNorm()
        self.out_norm  = RMSNorm()
        # Projection to vocab (TODO: tie with main embedding for efficiency)
        self.lm_head   = nn.Dense(cfg.vocab_size, use_bias=False)

    def __call__(
        self,
        h: jnp.ndarray,          # [batch, seq, hidden_dim]  final hidden states
        positions: jnp.ndarray,  # [seq]
    ) -> jnp.ndarray:
        """
        Returns:
            mtp_logits : [batch, seq, vocab_size]
        """
        # Single-block transformer (pre-norm, with residuals)
        h = h + self.attn(self.attn_norm(h), positions)
        h = h + self.ffn(self.ffn_norm(h))
        h = self.out_norm(h)
        return self.lm_head(h)   # [b, s, vocab]
        