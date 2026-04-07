"""
HiFi-GAN based decoder (generator) for VITS2.

Converts latent representations to raw audio waveforms.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """HiFi-GAN Residual Block Type 1."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[List[int]] = None):
        super().__init__()
        if dilations is None:
            dilations = [[1, 3, 5], [1, 3, 5]]

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d_group in dilations:
            for d in d_group:
                padding = (kernel_size * d - d) // 2
                self.convs1.append(
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding)
                    )
                )
            for _ in d_group:
                self.convs2.append(
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            nn.utils.parametrize.remove_parametrizations(c, "weight")
        for c in self.convs2:
            nn.utils.parametrize.remove_parametrizations(c, "weight")


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN generator adapted for VITS2 decoder.

    Takes latent z and upsamples to waveform through transposed convolutions
    and residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 192,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        cond_channels: int = 0,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # Conditioning projection
        if cond_channels > 0:
            self.cond_proj = nn.Conv1d(cond_channels, upsample_initial_channel, 1)
        else:
            self.cond_proj = None

        # Upsample layers
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, k, stride=u, padding=(k - u) // 2
                    )
                )
            )
            ch = ch // 2

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, [d]))

        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)
        self.conv_post = nn.utils.parametrizations.weight_norm(self.conv_post)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Latent representation (B, C, T_latent)
            cond: Speaker conditioning (B, cond_channels, 1) or None

        Returns:
            Audio waveform (B, 1, T_audio)
        """
        x = self.conv_pre(x)

        if cond is not None and self.cond_proj is not None:
            x = x + self.cond_proj(cond)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)

            # Sum residual blocks
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for up in self.ups:
            nn.utils.parametrize.remove_parametrizations(up, "weight")
        for rb in self.resblocks:
            rb.remove_weight_norm()
        nn.utils.parametrize.remove_parametrizations(self.conv_post, "weight")
