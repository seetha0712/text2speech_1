"""
Loss functions for VITS2 training.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def generator_loss(disc_fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss (hinge)."""
    loss = 0
    for dg in disc_fake_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_fake_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discriminator adversarial loss."""
    real_loss = 0
    fake_loss = 0
    for dr, dg in zip(disc_real_outputs, disc_fake_outputs):
        real_loss += torch.mean((1 - dr) ** 2)
        fake_loss += torch.mean(dg ** 2)
    return real_loss, fake_loss


def feature_matching_loss(
    real_feature_maps: List[List[torch.Tensor]],
    fake_feature_maps: List[List[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature matching loss between discriminator feature maps."""
    loss = 0
    for real_fmaps, fake_fmaps in zip(real_feature_maps, fake_feature_maps):
        for real_fm, fake_fm in zip(real_fmaps, fake_fmaps):
            loss += F.l1_loss(fake_fm, real_fm.detach())
    return loss


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence loss between posterior and prior.

    KL(q(z|x) || p(z|c)) where:
    - q is the posterior (from audio)
    - p is the prior (from text)
    """
    z_mask = z_mask.unsqueeze(1) if z_mask.dim() == 2 else z_mask

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)

    kl = kl * z_mask
    kl = kl.sum() / z_mask.sum()
    return kl


def mel_loss(
    y_mel: torch.Tensor,
    y_hat_mel: torch.Tensor,
) -> torch.Tensor:
    """L1 mel-spectrogram reconstruction loss."""
    return F.l1_loss(y_hat_mel, y_mel)
