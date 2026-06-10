"""Tests for model/ffn.py — SwiGLU feed-forward network."""

import jax
import jax.numpy as jnp
import numpy as np

from config import get_test_config
from model.ffn import SwiGLUFFN


def test_output_shape():
    cfg   = get_test_config()
    rng   = jax.random.PRNGKey(0)
    x     = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
    model = SwiGLUFFN(cfg)
    params = model.init(rng, x)
    out    = model.apply(params, x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_no_nan_or_inf():
    cfg   = get_test_config()
    rng   = jax.random.PRNGKey(1)
    x     = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
    model = SwiGLUFFN(cfg)
    params = model.init(rng, x)
    out    = model.apply(params, x)
    assert jnp.all(jnp.isfinite(out)), "Output contains NaN or Inf"


def test_not_identity():
    """FFN should transform its input (not trivially return it)."""
    cfg    = get_test_config()
    rng    = jax.random.PRNGKey(2)
    x      = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
    model  = SwiGLUFFN(cfg)
    params = model.init(rng, x)
    out    = model.apply(params, x)
    assert not jnp.allclose(out, x), "FFN output is identical to input"


def test_gradient_flows():
    cfg    = get_test_config()
    rng    = jax.random.PRNGKey(3)
    x      = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
    model  = SwiGLUFFN(cfg)
    params = model.init(rng, x)

    def loss(p):
        return model.apply(p, x).sum()

    grads = jax.grad(loss)(params)
    # Flatten and check all leaves are finite and non-zero
    leaves = jax.tree_util.tree_leaves(grads)
    for g in leaves:
        assert jnp.all(jnp.isfinite(g)), "Gradient contains NaN/Inf"
        assert jnp.any(g != 0.0), "A gradient leaf is identically zero"


def test_large_input_stays_finite():
    """Check numerical stability under large input magnitudes."""
    cfg    = get_test_config()
    rng    = jax.random.PRNGKey(4)
    x      = jax.random.normal(rng, (2, 16, cfg.hidden_dim)) * 100.0
    model  = SwiGLUFFN(cfg)
    params = model.init(rng, x)
    out    = model.apply(params, x)
    assert jnp.all(jnp.isfinite(out)), "Output NaN/Inf under large input"
    