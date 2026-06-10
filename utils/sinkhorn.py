"""
Sinkhorn-Knopp normalisation — used by mHC to project B_l onto the
doubly-stochastic manifold (rows AND columns sum to 1, all entries ≥ 0).

Operates entirely in log-space for numerical stability. Uses jax.lax.scan
so the loop compiles to a single XLA while-loop rather than being unrolled.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp


def sinkhorn_normalize(
    log_alpha: jnp.ndarray,
    n_iters: int = 20,
) -> jnp.ndarray:
    """
    Project an arbitrary matrix onto the doubly-stochastic manifold.

    Args:
        log_alpha : [..., n, n]  Raw log-domain matrix (any real values).
        n_iters   : Number of alternating row/column normalisations.
                    20 is the paper's value; 3–5 suffices for tests.

    Returns:
        [..., n, n]  Doubly-stochastic matrix P where
                     P[..., i, :].sum() == 1  and  P[..., :, j].sum() == 1.
    """
    def _step(la: jnp.ndarray, _: None):
        # Row normalise  (axis=-1 sums across columns → each row sums to 1)
        la = la - jax.scipy.special.logsumexp(la, axis=-1, keepdims=True)
        # Column normalise (axis=-2 sums across rows → each col sums to 1)
        la = la - jax.scipy.special.logsumexp(la, axis=-2, keepdims=True)
        return la, None

    result, _ = jax.lax.scan(_step, log_alpha, xs=None, length=n_iters)
    return jnp.exp(result)
    