"""
Tests for model/model.py — full DeepSeek1B forward pass.

Covers:
  • Output shapes for logits and mtp_logits
  • No NaN in outputs
  • Finite scalar loss
  • All parameter gradients are finite and non-None
  • Causal property end-to-end (most important integration test)
  • TransformerLayer (layer.py) in isolation
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import get_test_config
from model.model import DeepSeek1B
from model.layer import TransformerLayer
from model.mhc import mhc_init
from train.trainer import cross_entropy_loss


@pytest.fixture(scope="module")
def cfg():
    return get_test_config()


@pytest.fixture(scope="module")
def model(cfg):
    # use_remat=False for simpler tracebacks during testing
    return DeepSeek1B(cfg, use_remat=False)


@pytest.fixture(scope="module")
def tokens(cfg):
    rng = jax.random.PRNGKey(100)
    return jax.random.randint(rng, (2, 16), 0, cfg.vocab_size)


@pytest.fixture(scope="module")
def params(model, tokens):
    rng = jax.random.PRNGKey(101)
    return model.init(rng, tokens)


# ── Output shape & validity ────────────────────────────────────────────────────

class TestModelForward:

    def test_logits_shape(self, model, params, tokens, cfg):
        logits, _ = model.apply(params, tokens)
        assert logits.shape == (2, 16, cfg.vocab_size), \
            f"logits shape: {logits.shape}"

    def test_mtp_logits_shape(self, model, params, tokens, cfg):
        _, mtp = model.apply(params, tokens)
        assert mtp.shape == (2, 16, cfg.vocab_size), \
            f"mtp_logits shape: {mtp.shape}"

    def test_no_nan_in_logits(self, model, params, tokens):
        logits, mtp = model.apply(params, tokens)
        assert jnp.all(jnp.isfinite(logits)), "logits has NaN/Inf"
        assert jnp.all(jnp.isfinite(mtp)),    "mtp_logits has NaN/Inf"

    def test_logits_not_constant(self, model, params, tokens):
        logits, _ = model.apply(params, tokens)
        # Variance across vocab should be non-zero
        assert jnp.var(logits).item() > 0.0, "logits are constant"


# ── Loss ──────────────────────────────────────────────────────────────────────

class TestLoss:

    def test_loss_is_finite_scalar(self, model, params, tokens):
        logits, _ = model.apply(params, tokens)
        targets    = tokens        # same shape for a quick check
        loss = cross_entropy_loss(logits, targets)
        assert loss.ndim == 0,            "Loss should be a scalar"
        assert jnp.isfinite(loss).item(), "Loss is NaN or Inf"

    def test_loss_is_positive(self, model, params, tokens):
        logits, _ = model.apply(params, tokens)
        loss = cross_entropy_loss(logits, tokens)
        assert loss.item() > 0.0, "Cross-entropy loss should be positive"

    def test_loss_below_random_baseline(self, cfg, model, params, tokens):
        """
        With random weights the loss should be near log(vocab_size) ≈ log(128).
        Anything wildly above that indicates a bug (e.g. all logits = -inf).
        """
        logits, _ = model.apply(params, tokens)
        loss = cross_entropy_loss(logits, tokens)
        upper_bound = 2.0 * jnp.log(cfg.vocab_size)
        assert loss.item() < upper_bound, \
            f"Loss {loss:.3f} > 2×log(vocab) {upper_bound:.3f} — likely a bug"


# ── Gradients ─────────────────────────────────────────────────────────────────

class TestGradients:

    def test_all_gradients_finite(self, model, params, tokens):
        def loss_fn(p):
            logits, mtp = model.apply(p, tokens)
            return (cross_entropy_loss(logits, tokens) +
                    0.1 * cross_entropy_loss(mtp, tokens))

        grads = jax.grad(loss_fn)(params)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf)), \
                f"Gradient contains NaN/Inf for leaf with shape {leaf.shape}"

    def test_gradients_not_all_zero(self, model, params, tokens):
        def loss_fn(p):
            logits, _ = model.apply(p, tokens)
            return cross_entropy_loss(logits, tokens)

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(jnp.any(g != 0.0).item() for g in leaves)
        assert any_nonzero, "All gradients are zero — no signal flowing"


# ── Causality (end-to-end) ─────────────────────────────────────────────────────

class TestCausality:

    def test_full_model_causal(self, cfg, model, params):
        """
        Tokens at positions ≥ split must not affect logits at positions < split.
        This is the definitive integration test for attention causality.
        """
        rng   = jax.random.PRNGKey(200)
        batch = 2
        seq   = 16
        split = seq // 2

        tokens1 = jax.random.randint(rng, (batch, seq), 0, cfg.vocab_size)
        tokens2 = tokens1.at[:, split:].set(
            jax.random.randint(jax.random.PRNGKey(201),
                               (batch, seq - split), 0, cfg.vocab_size)
        )

        logits1, _ = model.apply(params, tokens1)
        logits2, _ = model.apply(params, tokens2)

        np.testing.assert_allclose(
            np.array(logits1[:, :split, :]),
            np.array(logits2[:, :split, :]),
            atol=1e-4,
            err_msg="Causal violation: future tokens affected past logits",
        )


# ── TransformerLayer in isolation ─────────────────────────────────────────────

@pytest.mark.parametrize("layer_type", ["csa", "hca"])
def test_transformer_layer(layer_type, cfg):
    rng   = jax.random.PRNGKey(300)
    batch, seq = 2, 16
    H     = mhc_init(
        jax.random.normal(rng, (batch, seq, cfg.hidden_dim)), cfg.n_hc
    )
    pos   = jnp.arange(seq)
    layer = TransformerLayer(cfg, layer_type=layer_type)
    params = layer.init(rng, H, pos)
    H_new  = layer.apply(params, H, pos)

    assert H_new.shape == H.shape, f"Shape mismatch for {layer_type}"
    assert jnp.all(jnp.isfinite(H_new)), f"NaN/Inf in {layer_type} layer output"
    