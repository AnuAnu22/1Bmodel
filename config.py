"""
ModelConfig — single source of truth for all architectural hyperparameters.

Scaled to ~1B dense parameters. Swap in TestConfig (defined at the bottom)
for unit-test runs that need to be fast on CPU.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # ── Core ──────────────────────────────────────────────────────────────────
    vocab_size: int = 32000
    hidden_dim: int = 2048          # d  — scaled for ~1B total params
    n_layers: int = 18
    max_seq_len: int = 2048

    # ── mHC (Manifold-Constrained Hyper-Connections) ──────────────────────────
    # Replaces every layer-to-layer residual connection.
    # H has shape [batch, seq, n_hc, hidden_dim] — the expanded residual stream.
    n_hc: int = 4
    sinkhorn_iters: int = 20        # Sinkhorn-Knopp iterations for B_l projection

    # ── Attention (shared between CSA and HCA) ────────────────────────────────
    n_heads: int = 8                # query heads (MQA: single shared KV head)
    head_dim: int = 256             # d_h  → n_heads * head_dim = 2048 = hidden_dim
    q_latent_dim: int = 512         # bottleneck for low-rank Q projection
    rope_dims: int = 32             # last N dims of each head get RoPE
    rope_theta: float = 10_000.0
    sliding_window: int = 128       # local causal window (# past tokens)

    # ── CSA (Compressed Sparse Attention) ────────────────────────────────────
    # Dual-stream token-level compression + Lightning Indexer top-k selection.
    csa_compression_rate: int = 4   # m  — tokens per compressed block
    csa_indexer_heads: int = 16      # n_I_h
    csa_indexer_dim: int = 128       # c_I  — indexer head dimension
    csa_top_k: int = 256            # sparse blocks selected per query position

    # ── HCA (Heavily Compressed Attention) ────────────────────────────────────
    # Single-stream heavy compression, dense attention over all past blocks.
    hca_compression_rate: int = 128  # m'

    # ── Dense FFN (SwiGLU) ────────────────────────────────────────────────────
    # ~2.75× hidden_dim, standard for ~1B SwiGLU LLMs.
    ffn_dim: int = 5632

    # ── MTP (Multi-Token Prediction) ─────────────────────────────────────────
    mtp_steps: int = 1              # auxiliary prediction steps (paper uses 1)
    mtp_loss_weight: float = 0.1    # weight on MTP auxiliary loss

    # ── Misc ──────────────────────────────────────────────────────────────────
    dropout_rate: float = 0.0       # 0 for pre-training; set for fine-tuning

    # ── Layer-type schedule ───────────────────────────────────────────────────
    def get_layer_types(self) -> list:
        """
        Returns a list of 'csa' or 'hca' per layer, following the V4 pattern:
          - Layers 0–1 : HCA  (no Lightning Indexer; long-range dense compression)
          - Layers 2+  : alternating CSA / HCA  (starting with CSA)
        """
        out: List[str] = []
        for i in range(self.n_layers):
            if i < 2:
                out.append("hca")
            else:
                out.append("csa" if (i - 2) % 2 == 0 else "hca")
        return out

    @property
    def total_attn_dim(self) -> int:
        return self.n_heads * self.head_dim


# ── Tiny config for unit tests ─────────────────────────────────────────────────
# Designed to fit in <1 GB of RAM and run each test in milliseconds on CPU.
def get_test_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=64,
        n_hc=2,
        sinkhorn_iters=3,       # fewer iters for speed
        n_heads=2,
        head_dim=16,
        q_latent_dim=16,
        rope_dims=4,
        rope_theta=10_000.0,
        sliding_window=8,
        csa_compression_rate=4,
        csa_indexer_heads=2,
        csa_indexer_dim=8,
        csa_top_k=4,
        hca_compression_rate=8,
        ffn_dim=64,
        mtp_steps=1,
        mtp_loss_weight=0.1,
    )
