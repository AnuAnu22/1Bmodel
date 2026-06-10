"""Tests for model/mhc.py — mHC dynamic mixing and state update."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import get_test_config
from model.mhc import mHCMixing, mhc_init, mhc_update, mix_channels


@pytest.fixture(scope="module")
def cfg():
    return get_test_config()


@pytest.fixture(scope="module")
def H(cfg):
    """Random mHC residual state : [2, 16, n_hc, d]."""
    rng = jax.random.PRNGKey(10)
    return jax.random.normal(rng, (2, 16, cfg.n_hc, cfg.hidden_dim))


class TestMHCInit:

    def test_shape(self, cfg):
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
        H = mhc_init(x, cfg.n_hc)
        assert H.shape == (2, 16, cfg.n_hc, cfg.hidden_dim)

    def test_channel0_equals_embed(self, cfg):
        rng = jax.random.PRNGKey(1)
        x = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
        H = mhc_init(x, cfg.n_hc)
        np.testing.assert_array_equal(np.array(H[:, :, 0, :]), np.array(x))

    def test_other_channels_zero(self, cfg):
        rng = jax.random.PRNGKey(2)
        x = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
        H = mhc_init(x, cfg.n_hc)
        assert jnp.all(H[:, :, 1:, :] == 0.0)


class TestMHCMixing:

    def test_output_shapes(self, cfg, H):
        model = mHCMixing(cfg)
        rng   = jax.random.PRNGKey(20)
        params = model.init(rng, H)
        A, B, C = model.apply(params, H)

        assert A.shape == (2, 16, cfg.n_hc),                    "A shape"
        assert B.shape == (2, 16, cfg.n_hc, cfg.n_hc),          "B shape"
        assert C.shape == (2, 16, cfg.n_hc),                    "C shape"

    def test_A_is_probability_distribution(self, cfg, H):
        model  = mHCMixing(cfg)
        params = model.init(jax.random.PRNGKey(21), H)
        A, _, _ = model.apply(params, H)
        row_sums = A.sum(axis=-1)
        np.testing.assert_allclose(np.array(row_sums), np.ones_like(row_sums), atol=1e-5)

    def test_C_is_probability_distribution(self, cfg, H):
        model  = mHCMixing(cfg)
        params = model.init(jax.random.PRNGKey(22), H)
        _, _, C = model.apply(params, H)
        row_sums = C.sum(axis=-1)
        np.testing.assert_allclose(np.array(row_sums), np.ones_like(row_sums), atol=1e-5)

    def test_B_is_doubly_stochastic(self, cfg, H):
        model  = mHCMixing(cfg)
        params = model.init(jax.random.PRNGKey(23), H)
        _, B, _ = model.apply(params, H)
        # B : [batch, seq, n_hc, n_hc] — check one slice
        B_slice = B[0, 0]    # [n_hc, n_hc]
        row_sums = B_slice.sum(axis=-1)
        col_sums = B_slice.sum(axis=-2)
        np.testing.assert_allclose(np.array(row_sums), np.ones(cfg.n_hc), atol=1e-3)
        np.testing.assert_allclose(np.array(col_sums), np.ones(cfg.n_hc), atol=1e-3)

    def test_no_nan(self, cfg, H):
        model  = mHCMixing(cfg)
        params = model.init(jax.random.PRNGKey(24), H)
        A, B, C = model.apply(params, H)
        for t, name in [(A, "A"), (B, "B"), (C, "C")]:
            assert jnp.all(jnp.isfinite(t)), f"{name} contains NaN/Inf"


class TestMHCUpdate:

    def test_output_shape(self, cfg, H):
        rng   = jax.random.PRNGKey(30)
        model = mHCMixing(cfg)
        params = model.init(rng, H)
        A, B, C = model.apply(params, H)

        block_out = jax.random.normal(rng, (2, 16, cfg.hidden_dim))
        H_new = mhc_update(H, block_out, A, B, C)
        assert H_new.shape == H.shape

    def test_gradients_flow(self, cfg, H):
        """Ensure gradients reach H through the mHC update."""
        rng   = jax.random.PRNGKey(31)
        model = mHCMixing(cfg)
        params = model.init(rng, H)

        def loss(H_in):
            A, B, C = model.apply(params, H_in)
            block_out = jnp.ones((2, 16, cfg.hidden_dim))
            H_new = mhc_update(H_in, block_out, A, B, C)
            return H_new.sum()

        grad = jax.grad(loss)(H)
        assert jnp.all(jnp.isfinite(grad)), "Gradient through mHC_update has NaN/Inf"
        assert jnp.any(grad != 0.0), "Gradient is identically zero — no signal"


class TestMixChannels:

    def test_output_shape(self, cfg, H):
        A = jax.random.normal(jax.random.PRNGKey(40), (2, 16, cfg.n_hc))
        A = jax.nn.softmax(A, axis=-1)
        x = mix_channels(H, A)
        assert x.shape == (2, 16, cfg.hidden_dim)
        