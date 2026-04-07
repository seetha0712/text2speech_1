"""
Duration Predictor for VITS2.

Simple deterministic duration predictor that predicts log-durations
from text encoder hidden states. More stable than the stochastic
version for training from scratch.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Deterministic Duration Predictor.

    Predicts phoneme durations from text encoder hidden states using
    a stack of conv layers. Simpler and more stable than the stochastic
    version, especially for training from scratch.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        n_flows: int = 4,  # unused, kept for config compat
        cond_channels: int = 0,
    ):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
                padding=kernel_size // 2,
            )
            for i in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, in_channels, 1)
        else:
            self.cond_proj = None

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            x: Text encoder hidden states (B, H, T_text)
            x_mask: Text mask (B, 1, T_text)
            w: Ground truth durations (B, 1, T_text) — only during training
            cond: Speaker conditioning
            reverse: If True, predict durations for inference
            noise_scale: Unused, kept for API compat

        Returns:
            During training: L2 duration prediction loss
            During inference: predicted durations (B, 1, T_text)
        """
        x = x.detach()  # Stop gradients from duration predictor to text encoder

        if cond is not None and self.cond_proj is not None:
            x = x + self.cond_proj(cond)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.proj(x) * x_mask  # (B, 1, T_text)

        if not reverse:
            # Training: compute L2 loss against ground truth log-durations
            assert w is not None
            log_w = torch.log(w.clamp(min=1.0))
            loss = F.mse_loss(x, log_w, reduction='none')
            loss = (loss * x_mask).sum() / x_mask.sum()
            return loss
        else:
            # Inference: predict durations
            w = torch.exp(x) * x_mask
            return w
