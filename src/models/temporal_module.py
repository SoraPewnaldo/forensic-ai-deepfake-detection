import torch
import torch.nn as nn
import math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class TemporalModule(nn.Module):
    """
    Multi-Head Self-Attention over T fused frame embeddings.

    V2 Design (MASTER_DESIGN.md):
    - Input: (B, T, d_model) - concatenated spatial + frequency features
    - Learnable positional encoding
    - 4 attention heads
    - Mean-pool over T to produce final video-level representation (B, d_model)

    Key insight: Not all frames contribute equally. Frames with stronger
    manipulation evidence (e.g., boundary frames of splice edits) should be
    weighted higher. Self-attention learns this weighting automatically.
    """

    def __init__(self):
        super().__init__()
        d_model = config.architecture.temporal.d_model   # 1024
        n_heads = config.architecture.temporal.n_heads   # 4
        T = config.architecture.temporal.frames          # 16 or 32

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # Learnable positional encoding (not sinusoidal - allows the model
        # to learn forensically relevant positional priors, e.g., first/last frames)
        self.pos_embed = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Standard PyTorch Transformer Block (MHA -> Add&Norm -> FFN -> Add&Norm)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )

        print(f"[INFO] Temporal Module: {n_heads}-head SA on T={T} frames, d_model={d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) - fused frame features
        Returns:
            video_repr: (B, d_model) - mean-pooled temporal representation
        """
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Full Transformer Block
        x = self.transformer_block(x)

        # Mean pool over time dimension -> video-level representation
        return x.mean(dim=1)   # (B, d_model)
