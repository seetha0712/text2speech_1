"""
Multi-Period Discriminator (MPD) for VITS2 adversarial training.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorP(nn.Module):
    """Single-period sub-discriminator."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period

        norm_fn = nn.utils.parametrizations.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_fn(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_fn(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_fn(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_fn(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0))),
            norm_fn(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0))),
        ])
        self.conv_post = norm_fn(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (B, 1, T)

        Returns:
            score: Discriminator score
            feature_maps: Intermediate feature maps for feature matching loss
        """
        feature_maps = []
        B, C, T = x.shape

        # Reshape to 2D with period
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad

        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)
        x = x.flatten(1, -1)

        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator.

    Evaluates audio at different periodic resolutions to capture
    both fine-grained and coarse temporal patterns.
    """

    def __init__(self, periods: List[int] = None, use_spectral_norm: bool = False):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]

        self.discriminators = nn.ModuleList([
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods
        ])

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Args:
            y: Real audio (B, 1, T)
            y_hat: Generated audio (B, 1, T)

        Returns:
            real_scores: List of discriminator scores for real audio
            fake_scores: List of discriminator scores for fake audio
            real_feature_maps: List of feature maps for real audio
            fake_feature_maps: List of feature maps for fake audio
        """
        real_scores = []
        fake_scores = []
        real_fmaps = []
        fake_fmaps = []

        for disc in self.discriminators:
            r_score, r_fmap = disc(y)
            f_score, f_fmap = disc(y_hat)
            real_scores.append(r_score)
            fake_scores.append(f_score)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)

        return real_scores, fake_scores, real_fmaps, fake_fmaps
