"""
Core building blocks for VITS2 architecture.

Includes:
- WaveNet residual blocks
- Multi-head attention
- Feed-forward networks
- Normalizing flow coupling layers
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """Channel-first layer normalization."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)  # (B, C, T)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(
        self,
        channels: int,
        n_heads: int,
        dropout: float = 0.0,
        window_size: Optional[int] = 4,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

        if window_size is not None:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(n_heads, 2 * window_size + 1) * 0.02
            )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, T = x.shape

        qkv = self.qkv(x)  # (B, 3*C, T)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h d) t -> b h t d", h=self.n_heads)
        k = rearrange(k, "b (h d) t -> b h t d", h=self.n_heads)
        v = rearrange(v, "b (h d) t -> b h t d", h=self.n_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Relative positional bias
        if self.window_size is not None and T > 0:
            rel_pos_bias = self._get_relative_position_bias(T)
            attn = attn + rel_pos_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b (h d) t")
        return self.out_proj(out)

    def _get_relative_position_bias(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, device=self.rel_pos_emb.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.window_size, self.window_size) + self.window_size
        return self.rel_pos_emb[:, rel_pos]


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(
        self,
        channels: int,
        filter_channels: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(filter_channels, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(channels, n_heads, dropout)
        self.norm1 = LayerNorm(channels)
        self.ffn = FeedForwardNetwork(channels, filter_channels, kernel_size, dropout)
        self.norm2 = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout(x) + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, mask.unsqueeze(1) if mask is not None else None)
        x = self.dropout(x) + residual

        return x


class WaveNetResBlock(nn.Module):
    """WaveNet-style residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.dilated_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.cond_layers = nn.ModuleList() if cond_channels > 0 else None

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2

            self.dilated_convs.append(
                nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation, padding=padding)
            )
            self.out_convs.append(nn.Conv1d(channels, 2 * channels, 1))
            self.norms.append(LayerNorm(channels))

            if cond_channels > 0:
                self.cond_layers.append(nn.Conv1d(cond_channels, 2 * channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            residual = x
            x = self.norms[i](x)
            x = self.dilated_convs[i](x)

            if cond is not None and self.cond_layers is not None:
                x = x + self.cond_layers[i](cond)

            # Gated activation
            xa, xb = x.chunk(2, dim=1)
            x = torch.tanh(xa) * torch.sigmoid(xb)

            # Output projection + residual
            x = self.out_convs[i](x)
            skip, gate = x.chunk(2, dim=1)
            output = output + skip
            x = (gate + residual) / math.sqrt(2.0)

        if mask is not None:
            output = output * mask.unsqueeze(1)

        return output


class ResidualCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        cond_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(
            hidden_channels, kernel_size, dilation_rate, n_layers, cond_channels
        )
        out_channels = self.half_channels if mean_only else self.half_channels * 2
        self.post = nn.Conv1d(hidden_channels, out_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x0, x1 = x.split(self.half_channels, dim=1)

        h = self.pre(x0)
        h = self.enc(h, mask, cond)
        stats = self.post(h)

        if not self.mean_only:
            m, log_s = stats.split(self.half_channels, dim=1)
        else:
            m = stats
            log_s = torch.zeros_like(m)

        if reverse:
            x1 = (x1 - m) * torch.exp(-log_s)
            logdet = None
        else:
            x1 = m + x1 * torch.exp(log_s)
            logdet = torch.sum(log_s, dim=(1, 2))

        x = torch.cat([x0, x1], dim=1)
        if mask is not None:
            x = x * mask.unsqueeze(1)

        return x, logdet


class ResidualCouplingBlock(nn.Module):
    """Stack of residual coupling layers forming a normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    cond_channels, mean_only=True,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logdet_total = 0

        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, mask, cond)
                # Flip channels between layers
                x = torch.flip(x, dims=[1])
                if logdet is not None:
                    logdet_total = logdet_total + logdet
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, dims=[1])
                x, _ = flow(x, mask, cond, reverse=True)

        return x, logdet_total if not reverse else None
