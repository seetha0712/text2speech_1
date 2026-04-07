"""
Stochastic Duration Predictor for VITS2.

Predicts phoneme durations using a flow-based model,
enabling natural variation in speech rhythm.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Log(nn.Module):
    """Safe logarithm."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp(min=1e-5))


class ElementwiseAffine(nn.Module):
    """Elementwise affine transformation."""
    def __init__(self, channels: int):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            return x * torch.exp(-self.logs) - self.m, -self.logs.sum()
        return (x + self.m) * torch.exp(self.logs), self.logs.sum()


class ConvFlow(nn.Module):
    """Convolutional coupling layer for duration predictor flow."""

    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, n_bins: int = 10):
        super().__init__()
        self.half_channels = in_channels // 2
        self.n_bins = n_bins

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(filter_channels) for _ in range(3)])
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (n_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h)
            h = h.transpose(1, 2)
            h = norm(h)
            h = h.transpose(1, 2)
            h = F.gelu(h)

        h = self.proj(h)
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # (B, C, T, bins)

        # Simplified linear spline (for stability)
        w = torch.softmax(h[..., :self.n_bins], dim=-1)
        b_param = h[..., self.n_bins:2*self.n_bins]

        # Piecewise linear transform
        cum_w = torch.cumsum(w, dim=-1)
        cum_w = F.pad(cum_w, (1, 0), value=0.0)

        if not reverse:
            # Find bin
            x1_expanded = x1.unsqueeze(-1)
            bin_idx = (cum_w[:, :, :, :-1] <= x1_expanded).sum(dim=-1).clamp(0, self.n_bins - 1)
            bin_idx_oh = F.one_hot(bin_idx, self.n_bins).float()

            w_sel = (w * bin_idx_oh).sum(-1)
            b_sel = (b_param * bin_idx_oh).sum(-1)

            x1 = x1 * w_sel + b_sel
            logdet = torch.log(w_sel.clamp(min=1e-5)).sum(dim=(1, 2))
        else:
            # Approximate inverse
            x1_expanded = x1.unsqueeze(-1)
            target = x1_expanded - b_param
            x1 = (target / w.clamp(min=1e-5)).mean(dim=-1)
            logdet = torch.zeros(x.size(0), device=x.device)

        x = torch.cat([x0, x1], dim=1)
        if mask is not None:
            x = x * mask

        return x, logdet


class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor.

    Uses a flow-based model to predict phoneme durations with natural variation.
    During training, it learns the distribution of durations.
    During inference, it samples from this distribution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        n_flows: int = 4,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, hidden_channels, 1)
        else:
            self.cond_proj = None

        # Duration flow
        self.log = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, hidden_channels, kernel_size))

        # Post-net for training (encodes ground truth durations)
        self.post_pre = nn.Conv1d(1, hidden_channels, 1)
        self.post_proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.post_convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(3)
        ])
        self.post_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(3)])
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for _ in range(n_flows):
            self.post_flows.append(ConvFlow(2, hidden_channels, kernel_size))

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
            reverse: If True, sample durations for inference

        Returns:
            During training: negative log-likelihood loss
            During inference: predicted durations (B, 1, T_text)
        """
        x = self.pre(x.detach())
        if cond is not None and self.cond_proj is not None:
            x = x + self.cond_proj(cond)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x * x_mask)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = F.gelu(x)
            x = self.dropout(x)

        x = self.proj(x) * x_mask
        h = x

        if not reverse:
            # Training: compute NLL of ground truth durations
            assert w is not None
            e_q = torch.randn_like(w) * x_mask
            z_q = e_q

            logdet_q = 0
            for flow in self.post_flows:
                if isinstance(flow, ElementwiseAffine):
                    # Expand to 2 channels
                    pass
                else:
                    z_q_2ch = torch.cat([h, z_q], dim=1)
                    z_q_2ch, ld = flow(z_q_2ch, x_mask)
                    z_q = z_q_2ch[:, h.size(1):, :]
                    logdet_q += ld

            # Log duration
            log_w = self.log(w)
            z_in = torch.cat([log_w, z_q], dim=1)

            logdet_total = 0
            for flow in self.flows:
                if isinstance(flow, ElementwiseAffine):
                    z_in, ld = flow(z_in)
                    logdet_total += ld
                else:
                    z_in, ld = flow(z_in, x_mask)
                    logdet_total += ld

            # NLL loss
            z0, z1 = z_in.split(1, dim=1)
            logp = -0.5 * (math.log(2 * math.pi) + z0 ** 2) * x_mask
            nll = -(logp.sum() + logdet_total - logdet_q)
            return nll / (x_mask.sum() + 1e-8)

        else:
            # Inference: sample durations
            z = torch.randn(x.size(0), 2, x.size(2), device=x.device) * noise_scale
            for flow in reversed(self.flows):
                if isinstance(flow, ElementwiseAffine):
                    z, _ = flow(z, reverse=True)
                else:
                    z, _ = flow(z, x_mask, reverse=True)

            log_w = z[:, 0:1, :]
            w = torch.exp(log_w) * x_mask
            return w


import math  # noqa: E402 — needed for math.log in forward
