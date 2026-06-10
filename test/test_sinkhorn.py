"""Tests for utils/sinkhorn.py — doubly stochastic projection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.sinkhorn import sinkhorn_normalize


def _is_doubly_stochastic(P: jnp.ndarray, atol: float = 1e-4) -> bool:
    """Check rows and columns each sum to 1 (within tolerance)."""
    row_sums = P.sum(axis=-1)
    col_sums = P.sum(axis=-2)
    ones = jnp.ones_like(row_sums)
    return (
        jnp.allclose(row_sums, ones, atol=atol)
        and jnp.allclose(col_sums, ones, atol=atol)
    )


class TestSinkhorn:

    def test_square_matrix_is_doubly_stochastic(self):
        rng = jax.random.PRNGKey(0)
        log_alpha = jax.random.normal(rng, (4, 4))
        P = sinkhorn_normalize(log_alpha, n_iters=20)
        assert _is_doubly_stochastic(P), "Output should be doubly stochastic"

    def test_non_negative(self):
        rng = jax.random.PRNGKey(1)
        log_alpha = jax.random.normal(rng, (3, 3))
        P = sinkhorn_normalize(log_alpha, n_iters=20)
        assert jnp.all(P >= 0), "All entries should be non-negative"

    def test_batched_input(self):
        """Works on arbitrary leading batch / sequence dimensions."""
        rng = jax.random.PRNGKey(2)
        # Shape: [batch, seq, n_hc, n_hc]  — the actual mHC use case
        log_alpha = jax.random.normal(rng, (2, 8, 4, 4))
        P = sinkhorn_normalize(log_alpha, n_iters=20)
        assert P.shape == (2, 8, 4, 4)
        # Check a slice
        assert _is_doubly_stochastic(P[0, 3], atol=1e-3)

    def test_identity_init_stays_uniform(self):
        """Starting from a uniform log-matrix should give a uniform output."""
        log_alpha = jnp.zeros((4, 4))
        P = sinkhorn_normalize(log_alpha, n_iters=20)
        expected = jnp.ones((4, 4)) / 4.0
        np.testing.assert_allclose(np.array(P), np.array(expected), atol=1e-5)

    def test_no_nan_or_inf(self):
        rng = jax.random.PRNGKey(3)
        # Large values that would overflow without log-space normalisation
        log_alpha = jax.random.normal(rng, (6, 6)) * 100.0
        P = sinkhorn_normalize(log_alpha, n_iters=20)
        assert jnp.all(jnp.isfinite(P)), "No NaN or Inf in output"

    def test_few_iters_still_converges(self):
        """Even 3 iterations (test config value) produce a near-valid result."""
        rng = jax.random.PRNGKey(4)
        log_alpha = jax.random.normal(rng, (4, 4))
        P = sinkhorn_normalize(log_alpha, n_iters=3)
        # Looser tolerance for fewer iterations
        assert _is_doubly_stochastic(P, atol=1e-2)
        