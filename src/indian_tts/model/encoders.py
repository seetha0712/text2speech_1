"""
Text and Posterior Encoders for VITS2.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from indian_tts.model.modules import (
    TransformerEncoderLayer,
    WaveNetResBlock,
    LayerNorm,
)


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder.

    Converts phoneme ID sequences into hidden representations
    and predicts prior distribution parameters (mu, sigma) for the latent space.
    """

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout: float = 0.1,
        out_channels: int = 192,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Phoneme embedding
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(hidden_channels)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_channels, filter_channels, n_heads, kernel_size, dropout
            )
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(hidden_channels)

        # Project to prior distribution parameters
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Phoneme IDs (B, T_text)
            x_lengths: Lengths of phoneme sequences (B,)

        Returns:
            x: Encoder output (B, H, T_text)
            m_p: Prior mean (B, C, T_text)
            logs_p: Prior log-variance (B, C, T_text)
            x_mask: Mask (B, 1, T_text)
        """
        # Create mask
        x_mask = self._sequence_mask(x_lengths, x.size(1)).unsqueeze(1)  # (B, 1, T)

        # Embed phonemes
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # (B, T, H)
        x = x.transpose(1, 2)  # (B, H, T)

        # Add positional encoding
        x = x + self.pos_enc(x)
        x = x * x_mask

        # Transformer layers
        attn_mask = x_mask.squeeze(1)  # (B, T)
        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.norm(x)
        x = x * x_mask

        # Project to prior parameters
        stats = self.proj(x) * x_mask
        m_p, logs_p = stats.split(self.out_channels, dim=1)

        return x, m_p, logs_p, x_mask

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = lengths.max().item()
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, channels: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 2))  # (1, C, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :, : x.size(2)]


class PosteriorEncoder(nn.Module):
    """
    WaveNet-based posterior encoder.

    Encodes mel-spectrogram into latent representation z
    with reparameterization trick.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(
            hidden_channels, kernel_size, dilation_rate, n_layers, cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Linear spectrogram (B, F, T_spec)
            x_lengths: Spectrogram lengths (B,)
            cond: Speaker conditioning (B, cond_channels, 1)

        Returns:
            z: Sampled latent (B, C, T_spec)
            m_q: Posterior mean (B, C, T_spec)
            logs_q: Posterior log-variance (B, C, T_spec)
            x_mask: Mask (B, 1, T_spec)
        """
        x_mask = self._sequence_mask(x_lengths, x.size(2)).unsqueeze(1)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask.squeeze(1), cond)
        stats = self.proj(x) * x_mask
        m_q, logs_q = stats.split(self.out_channels, dim=1)

        # Reparameterization
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        z = z * x_mask

        return z, m_q, logs_q, x_mask

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = lengths.max().item()
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()
