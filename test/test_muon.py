"""
Tests for train/muon.py — Newton-Schulz orthogonalization and Muon optimizer.

Key checks:
  • NS output is approximately semi-orthogonal
  • NS handles non-square matrices (tall and wide)
  • Optimizer state is initialised with correct structure
  • A single update step changes the parameters
  • Momentum accumulates across steps
  • 1D params use plain momentum (no NS)
  • No NaN anywhere
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from train.muon import newton_schulz, muon_transform, muon_optimizer


# ── Newton-Schulz ─────────────────────────────────────────────────────────────

class TestNewtonSchulz:

    def _near_orthogonal(self, G: jnp.ndarray, atol: float = 0.1) -> bool:
        """
        For a semi-orthogonal matrix in R^{n x m} (n >= m), G.T @ G ≈ c * I.
        Check that the off-diagonal entries are small relative to the diagonal.
        """
        n, m = G.shape
        M = G.T @ G if n >= m else G @ G.T   # should be close to scalar * I
        diag     = jnp.diag(M)
        off_diag = M - jnp.diag(diag)
        return jnp.abs(off_diag).max() < atol * jnp.abs(diag).mean()

    def test_square_matrix(self):
        rng = jax.random.PRNGKey(0)
        G   = jax.random.normal(rng, (8, 8))
        G_orth = newton_schulz(G, steps=10)
        assert G_orth.shape == (8, 8)
        assert jnp.all(jnp.isfinite(G_orth)), "NS output has NaN/Inf"
        assert self._near_orthogonal(G_orth), "Square NS output not near-orthogonal"

    def test_tall_matrix(self):
        rng = jax.random.PRNGKey(1)
        G   = jax.random.normal(rng, (16, 4))
        G_orth = newton_schulz(G, steps=10)
        assert G_orth.shape == (16, 4)
        assert jnp.all(jnp.isfinite(G_orth))
        assert self._near_orthogonal(G_orth)

    def test_wide_matrix(self):
        rng = jax.random.PRNGKey(2)
        G   = jax.random.normal(rng, (4, 16))
        G_orth = newton_schulz(G, steps=10)
        assert G_orth.shape == (4, 16)
        assert jnp.all(jnp.isfinite(G_orth))
        assert self._near_orthogonal(G_orth)

    def test_rms_scaling(self):
        """Output RMS should match 1/sqrt(max(n, m))."""
        rng = jax.random.PRNGKey(3)
        n, m = 16, 4
        G   = jax.random.normal(rng, (n, m))
        G_orth = newton_schulz(G, steps=10)
        target_rms = 1.0 / jnp.sqrt(max(n, m))
        actual_rms = jnp.sqrt(jnp.mean(G_orth ** 2))
        np.testing.assert_allclose(
            float(actual_rms), float(target_rms), rtol=0.1,
            err_msg="NS output RMS does not match target"
        )

    def test_no_nan_with_zero_matrix(self):
        G = jnp.zeros((4, 4))
        # Near-zero matrix: norm is ~0, should not produce NaN
        G_orth = newton_schulz(G + 1e-8, steps=10)
        assert jnp.all(jnp.isfinite(G_orth))

    def test_few_steps_still_finite(self):
        rng = jax.random.PRNGKey(4)
        G   = jax.random.normal(rng, (8, 4))
        G_orth = newton_schulz(G, steps=2)
        assert jnp.all(jnp.isfinite(G_orth))


# ── Muon transform ────────────────────────────────────────────────────────────

class TestMuonTransform:

    @pytest.fixture
    def simple_params(self):
        rng = jax.random.PRNGKey(10)
        return {
            "W": jax.random.normal(rng, (8, 4)),    # 2D — gets NS
            "b": jax.random.normal(rng, (4,)),       # 1D — plain momentum
        }

    @pytest.fixture
    def simple_grads(self, simple_params):
        rng = jax.random.PRNGKey(11)
        return jax.tree_util.tree_map(
            lambda p: jax.random.normal(rng, p.shape),
            simple_params,
        )

    def test_init_creates_momentum_zeros(self, simple_params):
        tx    = muon_transform()
        state = tx.init(simple_params)
        for leaf in jax.tree_util.tree_leaves(state.momentum):
            assert jnp.all(leaf == 0.0), "Initial momentum should be zero"

    def test_update_returns_correct_shapes(self, simple_params, simple_grads):
        tx    = muon_transform()
        state = tx.init(simple_params)
        updates, new_state = tx.update(simple_grads, state)
        for k in simple_params:
            assert updates[k].shape == simple_params[k].shape, \
                f"Update shape mismatch for {k}"

    def test_update_no_nan(self, simple_params, simple_grads):
        tx    = muon_transform()
        state = tx.init(simple_params)
        updates, _ = tx.update(simple_grads, state)
        for k, u in updates.items():
            assert jnp.all(jnp.isfinite(u)), f"Update for {k} has NaN/Inf"

    def test_params_change_after_step(self, simple_params, simple_grads):
        tx    = muon_transform()
        state = tx.init(simple_params)
        updates, _ = tx.update(simple_grads, state)
        new_params = optax.apply_updates(simple_params, updates)
        for k in simple_params:
            assert not jnp.allclose(new_params[k], simple_params[k]), \
                f"Params[{k}] unchanged after optimizer step"

    def test_momentum_buffer_accumulates(self, simple_params, simple_grads):
        """
        The momentum buffer should accumulate across steps:
          step 1: m = 0.95*0 + g  = g
          step 2: m = 0.95*g + g  = 1.95*g

        Note: Newton-Schulz normalises its input (G / ‖G‖) before iterating,
        so the *scale* of the buffer does not change the NS output direction —
        identical gradient directions at both steps yield identical updates.
        The correct invariant to test is the buffer, not the update values.
        """
        tx    = muon_transform(momentum=0.95)
        state = tx.init(simple_params)

        _, state1 = tx.update(simple_grads, state)
        _, state2 = tx.update(simple_grads, state1)

        m1 = state1.momentum["W"]
        m2 = state2.momentum["W"]

        # After step 1: buffer == g
        np.testing.assert_allclose(
            np.array(m1), np.array(simple_grads["W"]), rtol=1e-5,
            err_msg="Momentum buffer after step 1 should equal the gradient"
        )
        # After step 2: buffer == 0.95*g + g = 1.95*g
        expected_m2 = 0.95 * simple_grads["W"] + simple_grads["W"]
        np.testing.assert_allclose(
            np.array(m2), np.array(expected_m2), rtol=1e-5,
            err_msg="Momentum buffer after step 2 should be 0.95*g + g"
        )

    def test_1d_param_uses_plain_momentum(self, simple_params, simple_grads):
        """
        1D params use vanilla Nesterov momentum (no NS transformation).

        With old_m=0, g=g, momentum=0.95, nesterov=True:
          new_m = 0.95 * 0 + g = g
          g_hat = 0.95 * new_m + g = 0.95*g + g = (0.95 + 1) * g = 1.95 * g
        """
        tx    = muon_transform(momentum=0.95, nesterov=True)
        state = tx.init(simple_params)
        g     = simple_grads["b"]
        updates, _ = tx.update(simple_grads, state)
        u = updates["b"]
        # Expected: (momentum + 1) * g
        expected = (0.95 + 1.0) * g
        np.testing.assert_allclose(
            np.array(u), np.array(expected), rtol=1e-5,
            err_msg="1D param update should follow plain Nesterov momentum"
        )


# ── Full muon_optimizer (with scale) ─────────────────────────────────────────

def test_muon_optimizer_applies_lr():
    rng    = jax.random.PRNGKey(20)
    params = {"W": jax.random.normal(rng, (8, 4))}
    grads  = {"W": jax.random.normal(jax.random.PRNGKey(21), (8, 4))}
    lr     = 0.01

    tx     = muon_optimizer(muon_lr=lr)
    state  = tx.init(params)
    updates, _ = tx.update(grads, state)

    # Updates should be scaled by lr
    update_rms = jnp.sqrt(jnp.mean(updates["W"] ** 2))
    # Target RMS of NS output (before lr): 1/sqrt(max(8,4)) = 1/sqrt(8) ≈ 0.354
    # After scale(-lr): 0.01 * 0.354 ≈ 0.00354
    target_rms = lr / jnp.sqrt(8.0)
    np.testing.assert_allclose(
        float(update_rms), float(target_rms), rtol=0.2,
        err_msg="Optimizer update RMS differs significantly from expected"
    )
    