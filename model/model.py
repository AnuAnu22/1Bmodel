"""
DeepSeek1B — full 1B dense model assembly.

Forward pass:
    1. Embed input tokens  →  x : [b, s, d]
    2. Init mHC state      →  H : [b, s, n_hc, d]   (channel 0 = embed, rest = 0)
    3. n_layers TransformerLayer passes  (alternating HCA / CSA per config)
    4. Extract primary channel H[:, :, 0, :], apply final RMSNorm
    5. Main LM head  →  logits   [b, s, vocab]
    6. MTP head      →  mtp_logits [b, s, vocab]

Returns (logits, mtp_logits).

Loss (computed in trainer, not here):
    L = CE(logits, targets) + mtp_weight * CE(mtp_logits, targets)

Gradient checkpointing:
    Set use_remat=True (default) to use RematTransformerLayer.
    Disable for debugging to get clearer tracebacks.

Weight tying (TODO):
    Embedding and LM-head weights are currently independent.
    Wire them together by passing embed_table into lm_head for a 33M param saving.
"""

from __future__ import annotations
from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn

from config import ModelConfig
from model.norm import RMSNorm
from model.mhc import mhc_init
from model.layer import TransformerLayer, RematTransformerLayer
from model.mtp import MTPHead


class DeepSeek1B(nn.Module):
    config: ModelConfig
    use_remat: bool = True   # gradient checkpointing per-layer

    def setup(self) -> None:
        cfg = self.config
        layer_types = cfg.get_layer_types()
        LayerCls = RematTransformerLayer if self.use_remat else TransformerLayer

        self.embedding  = nn.Embed(
            cfg.vocab_size, cfg.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.layers = [
            LayerCls(cfg, layer_type=lt, name=f"layer_{i}")
            for i, lt in enumerate(layer_types)
        ]
        self.final_norm = RMSNorm()
        # TODO: tie with embedding for 33M param saving
        self.lm_head    = nn.Dense(cfg.vocab_size, use_bias=False,
                                   kernel_init=nn.initializers.normal(0.02))
        self.mtp_head   = MTPHead(cfg)

    def __call__(
        self,
        input_ids: jnp.ndarray,           # [batch, seq]  int tokens
        positions: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            logits     : [batch, seq, vocab_size]
            mtp_logits : [batch, seq, vocab_size]
        """
        cfg = self.config
        batch, seq = input_ids.shape

        if positions is None:
            positions = jnp.arange(seq)

        # ── Embed ─────────────────────────────────────────────────────────
        x = self.embedding(input_ids)          # [b, s, d]

        # ── Init mHC residual state ───────────────────────────────────────
        H = mhc_init(x, cfg.n_hc)             # [b, s, n_hc, d]

        # ── Transformer layers ────────────────────────────────────────────
        for layer in self.layers:
            H = layer(H, positions)

        # ── Readout: primary channel + final norm ─────────────────────────
        h = self.final_norm(H[:, :, 0, :])    # [b, s, d]

        # ── LM head ───────────────────────────────────────────────────────
        logits     = self.lm_head(h)           # [b, s, vocab]

        # ── MTP auxiliary head ────────────────────────────────────────────
        mtp_logits = self.mtp_head(h, positions)  # [b, s, vocab]

        return logits, mtp_logits
        