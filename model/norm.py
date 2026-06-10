"""
RMSNorm — used everywhere standard LayerNorm would appear.
Per-element scale (gamma), no bias, no mean subtraction.
"""

from __future__ import annotations
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x : [..., d]  — any leading batch/sequence dims.
        Returns:
            [..., d]  normalised and scaled.
        """
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,))
        # RMS norm: x / sqrt(mean(x^2) + eps)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * scale
        