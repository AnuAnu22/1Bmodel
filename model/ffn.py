"""
SwiGLU Feed-Forward Network.

    hidden = SiLU(W_gate(x)) * W_up(x)
    out    = W_down(hidden)

No bias on any projection (standard for LLM pretraining).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn

from config import ModelConfig


class SwiGLUFFN(nn.Module):
    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.W_gate = nn.Dense(cfg.ffn_dim, use_bias=False)
        self.W_up   = nn.Dense(cfg.ffn_dim, use_bias=False)
        self.W_down = nn.Dense(cfg.hidden_dim, use_bias=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x : [batch, seq, hidden_dim]
        Returns:
            [batch, seq, hidden_dim]
        """
        return self.W_down(jax.nn.silu(self.W_gate(x)) * self.W_up(x))
        