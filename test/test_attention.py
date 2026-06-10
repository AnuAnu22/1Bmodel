"""
Tests for model/attention.py — CSA and HCA.

Key checks:
  • Output shape
  • No NaN / Inf
  • Causal property: output at position t is unchanged when only tokens
    at positions > t are modified (no future leakage)
  • Gradient flow through both attention types
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import get_test_config
from model.attention import CSAttention, HCAttention


@pytest.fixture(scope="module")
def cfg():
    return get_test_config()


@pytest.fixture(scope="module")
def x(cfg):
    rng = jax.random.PRNGKey(50)
    return jax.random.normal(rng, (2, 16, cfg.hidden_dim))


@pytest.fixture(scope="module")
def pos():
    return jnp.arange(16)


# ── helpers ────────────────────────────────────────────────────────────────────

def _forward(model_cls, cfg, x, pos):
    model  = model_cls(cfg)
    params = model.init(jax.random.PRNGKey(99), x, pos)
    out    = model.apply(params, x, pos)
    return params, out


def _check_causality(model_cls, cfg, pos):
    """
    Create two inputs that differ only at positions seq//2 onwards.
    Outputs at positions < seq//2 must be identical.
    """
    rng   = jax.random.PRNGKey(77)
    batch = 2
    seq   = 16
    split = seq // 2

    x1 = jax.random.normal(rng, (batch, seq, cfg.hidden_dim))
    # Perturb only the second half
    x2 = x1.at[:, split:, :].set(
        jax.random.normal(jax.random.PRNGKey(78), (batch, seq - split, cfg.hidden_dim))
    )

    model  = model_cls(cfg)
    params = model.init(jax.random.PRNGKey(79), x1, pos)

    out1 = model.apply(params, x1, pos)
    out2 = model.apply(params, x2, pos)

    # First half must be identical
    np.testing.assert_allclose(
        np.array(out1[:, :split, :]),
        np.array(out2[:, :split, :]),
        atol=1e-5,
        err_msg=f"{model_cls.__name__}: causal violation — future tokens leaked into past",
    )


# ── CSA ────────────────────────────────────────────────────────────────────────

class TestCSAttention:

    def test_output_shape(self, cfg, x, pos):
        _, out = _forward(CSAttention, cfg, x, pos)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_no_nan(self, cfg, x, pos):
        _, out = _forward(CSAttention, cfg, x, pos)
        assert jnp.all(jnp.isfinite(out)), "CSA output has NaN/Inf"

    def test_causal(self, cfg, pos):
        _check_causality(CSAttention, cfg, pos)

    def test_gradient_flow(self, cfg, x, pos):
        model  = CSAttention(cfg)
        params = model.init(jax.random.PRNGKey(60), x, pos)

        def loss(p):
            return model.apply(p, x, pos).sum()

        grads = jax.grad(loss)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves), \
            "CSA gradient has NaN/Inf"

    def test_output_not_constant(self, cfg, x, pos):
        """Output should not be the same vector at every position."""
        _, out = _forward(CSAttention, cfg, x, pos)
        # Variance across sequence positions should be > 0
        assert jnp.var(out, axis=1).mean() > 0.0


# ── HCA ────────────────────────────────────────────────────────────────────────

class TestHCAttention:

    def test_output_shape(self, cfg, x, pos):
        _, out = _forward(HCAttention, cfg, x, pos)
        assert out.shape == x.shape

    def test_no_nan(self, cfg, x, pos):
        _, out = _forward(HCAttention, cfg, x, pos)
        assert jnp.all(jnp.isfinite(out)), "HCA output has NaN/Inf"

    def test_causal(self, cfg, pos):
        _check_causality(HCAttention, cfg, pos)

    def test_gradient_flow(self, cfg, x, pos):
        model  = HCAttention(cfg)
        params = model.init(jax.random.PRNGKey(70), x, pos)

        def loss(p):
            return model.apply(p, x, pos).sum()

        grads = jax.grad(loss)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves), \
            "HCA gradient has NaN/Inf"

    def test_output_not_constant(self, cfg, x, pos):
        _, out = _forward(HCAttention, cfg, x, pos)
        assert jnp.var(out, axis=1).mean() > 0.0


# ── Shared: different inputs → different outputs ───────────────────────────────

@pytest.mark.parametrize("model_cls", [CSAttention, HCAttention])
def test_different_inputs_different_outputs(model_cls):
    cfg = get_test_config()
    pos = jnp.arange(16)
    rng = jax.random.PRNGKey(80)

    x1     = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
    x2     = jax.random.normal(jax.random.PRNGKey(81), (2, 16, cfg.hidden_dim))
    model  = model_cls(cfg)
    params = model.init(jax.random.PRNGKey(82), x1, pos)

    out1 = model.apply(params, x1, pos)
    out2 = model.apply(params, x2, pos)

    assert not jnp.allclose(out1, out2, atol=1e-6), \
        f"{model_cls.__name__}: different inputs produced identical outputs"
        