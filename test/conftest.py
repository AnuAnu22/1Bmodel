"""
Shared pytest fixtures — imported automatically by pytest from all test files
in the same directory.

Path bootstrap: inserts the project root (deepseek_1b/) into sys.path so
`from config import ...` etc. resolve correctly regardless of where pytest is
invoked from.
"""

from __future__ import annotations
import sys
import os

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── imports (after path fix) ──────────────────────────────────────────────────
import pytest
import jax
import jax.numpy as jnp

from config import get_test_config, ModelConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg() -> ModelConfig:
    """Tiny model config — fast on CPU, fits in <500 MB RAM."""
    return get_test_config()


@pytest.fixture(scope="session")
def rng() -> jax.Array:
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def batch_size() -> int:
    return 2


@pytest.fixture(scope="session")
def seq_len() -> int:
    """Must be a multiple of csa_compression_rate (4) AND hca_compression_rate (8)."""
    return 16


@pytest.fixture(scope="session")
def dummy_tokens(cfg, rng, batch_size, seq_len) -> jnp.ndarray:
    """Random integer token ids : [batch, seq]."""
    return jax.random.randint(rng, (batch_size, seq_len), 0, cfg.vocab_size)


@pytest.fixture(scope="session")
def dummy_hidden(cfg, rng, batch_size, seq_len) -> jnp.ndarray:
    """Random float hidden states : [batch, seq, hidden_dim]."""
    return jax.random.normal(rng, (batch_size, seq_len, cfg.hidden_dim))


@pytest.fixture(scope="session")
def positions(seq_len) -> jnp.ndarray:
    return jnp.arange(seq_len)
    