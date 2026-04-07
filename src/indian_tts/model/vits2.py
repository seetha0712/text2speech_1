"""
VITS2: End-to-end Text-to-Speech model.

Combines:
- Text Encoder (Transformer)
- Posterior Encoder (WaveNet)
- Normalizing Flow
- HiFi-GAN Decoder
- Stochastic Duration Predictor
- Monotonic Alignment Search

For multi-speaker Indian TTS with male/female voice support.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from indian_tts.model.encoders import TextEncoder, PosteriorEncoder
from indian_tts.model.decoder import HiFiGANGenerator
from indian_tts.model.duration_predictor import StochasticDurationPredictor
from indian_tts.model.modules import ResidualCouplingBlock
from indian_tts.model.monotonic_align import maximum_path


@dataclass
class VITS2Config:
    """Configuration for VITS2 model."""

    # Text Encoder
    n_vocab: int = 256
    text_enc_hidden: int = 192
    text_enc_filter: int = 768
    text_enc_heads: int = 2
    text_enc_layers: int = 6
    text_enc_kernel: int = 3
    text_enc_dropout: float = 0.1

    # Posterior Encoder
    spec_channels: int = 513  # n_fft // 2 + 1
    latent_channels: int = 192
    posterior_hidden: int = 192
    posterior_kernel: int = 5
    posterior_dilation: int = 1
    posterior_layers: int = 16

    # Flow
    flow_hidden: int = 192
    flow_kernel: int = 5
    flow_dilation: int = 1
    flow_layers: int = 4
    flow_n_flows: int = 4

    # Decoder
    upsample_initial_channel: int = 512
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    # Duration Predictor
    dp_hidden: int = 192
    dp_kernel: int = 3
    dp_dropout: float = 0.5
    dp_n_flows: int = 4
    dp_filter_channels: int = 256

    # Multi-Speaker
    n_speakers: int = 2
    speaker_embed_dim: int = 256

    # Audio
    segment_size: int = 8192
    hop_length: int = 256

    @classmethod
    def from_yaml(cls, cfg: dict) -> "VITS2Config":
        """Create config from parsed YAML dictionary."""
        model_cfg = cfg.get("model", {})
        audio_cfg = cfg.get("audio", {})

        te = model_cfg.get("text_encoder", {})
        pe = model_cfg.get("posterior_encoder", {})
        fl = model_cfg.get("flow", {})
        dec = model_cfg.get("decoder", {})
        dp = model_cfg.get("duration_predictor", {})

        return cls(
            n_vocab=te.get("n_vocab", 256),
            text_enc_hidden=te.get("hidden_channels", 192),
            text_enc_filter=te.get("filter_channels", 768),
            text_enc_heads=te.get("n_heads", 2),
            text_enc_layers=te.get("n_layers", 6),
            text_enc_kernel=te.get("kernel_size", 3),
            text_enc_dropout=te.get("dropout", 0.1),
            spec_channels=pe.get("in_channels", 513),
            latent_channels=pe.get("out_channels", 192),
            posterior_hidden=pe.get("hidden_channels", 192),
            posterior_kernel=pe.get("kernel_size", 5),
            posterior_dilation=pe.get("dilation_rate", 1),
            posterior_layers=pe.get("n_layers", 16),
            flow_hidden=fl.get("hidden_channels", 192),
            flow_kernel=fl.get("kernel_size", 5),
            flow_dilation=fl.get("dilation_rate", 1),
            flow_layers=fl.get("n_layers", 4),
            flow_n_flows=fl.get("n_flows", 4),
            upsample_initial_channel=dec.get("upsample_initial_channel", 512),
            upsample_rates=dec.get("upsample_rates", [8, 8, 2, 2]),
            upsample_kernel_sizes=dec.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            resblock_kernel_sizes=dec.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=dec.get("resblock_dilation_sizes", [[1, 3, 5]] * 3),
            dp_hidden=dp.get("hidden_channels", 192),
            dp_kernel=dp.get("kernel_size", 3),
            dp_dropout=dp.get("dropout", 0.5),
            dp_n_flows=dp.get("n_flows", 4),
            dp_filter_channels=dp.get("filter_channels", 256),
            n_speakers=model_cfg.get("n_speakers", 2),
            speaker_embed_dim=model_cfg.get("speaker_embed_dim", 256),
            segment_size=audio_cfg.get("segment_size", 8192),
            hop_length=audio_cfg.get("hop_length", 256),
        )


class VITS2(nn.Module):
    """
    VITS2: Variational Inference with adversarial learning
    for end-to-end Text-to-Speech (v2).

    Multi-speaker variant with Indian male/female voice support.
    """

    def __init__(self, config: VITS2Config):
        super().__init__()
        self.config = config

        # Speaker embedding
        if config.n_speakers > 1:
            self.speaker_emb = nn.Embedding(config.n_speakers, config.speaker_embed_dim)
        else:
            self.speaker_emb = None

        gin_channels = config.speaker_embed_dim if config.n_speakers > 1 else 0

        # Text Encoder
        self.text_encoder = TextEncoder(
            n_vocab=config.n_vocab,
            hidden_channels=config.text_enc_hidden,
            filter_channels=config.text_enc_filter,
            n_heads=config.text_enc_heads,
            n_layers=config.text_enc_layers,
            kernel_size=config.text_enc_kernel,
            dropout=config.text_enc_dropout,
            out_channels=config.latent_channels,
        )

        # Posterior Encoder
        self.posterior_encoder = PosteriorEncoder(
            in_channels=config.spec_channels,
            hidden_channels=config.posterior_hidden,
            out_channels=config.latent_channels,
            kernel_size=config.posterior_kernel,
            dilation_rate=config.posterior_dilation,
            n_layers=config.posterior_layers,
            cond_channels=gin_channels,
        )

        # Normalizing Flow
        self.flow = ResidualCouplingBlock(
            channels=config.latent_channels,
            hidden_channels=config.flow_hidden,
            kernel_size=config.flow_kernel,
            dilation_rate=config.flow_dilation,
            n_layers=config.flow_layers,
            n_flows=config.flow_n_flows,
            cond_channels=gin_channels,
        )

        # Decoder (HiFi-GAN)
        self.decoder = HiFiGANGenerator(
            in_channels=config.latent_channels,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            cond_channels=gin_channels,
        )

        # Stochastic Duration Predictor
        self.duration_predictor = StochasticDurationPredictor(
            in_channels=config.text_enc_hidden,
            hidden_channels=config.dp_hidden,
            kernel_size=config.dp_kernel,
            dropout=config.dp_dropout,
            n_flows=config.dp_n_flows,
            cond_channels=gin_channels,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            text: Phoneme IDs (B, T_text)
            text_lengths: Text lengths (B,)
            spec: Linear spectrogram (B, F, T_spec)
            spec_lengths: Spectrogram lengths (B,)
            speaker_ids: Speaker IDs (B,) — 0=male, 1=female

        Returns:
            Dictionary of outputs for loss computation
        """
        # Speaker conditioning
        g = None
        if self.speaker_emb is not None and speaker_ids is not None:
            g = self.speaker_emb(speaker_ids).unsqueeze(-1)  # (B, spk_dim, 1)

        # Text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # Posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(spec, spec_lengths, g)

        # Normalizing flow (posterior → prior)
        z_p, logdet = self.flow(z, y_mask.squeeze(1), g)

        # Monotonic Alignment Search
        with torch.no_grad():
            # Compute alignment cost
            s_p_sq_r = torch.exp(-2 * logs_p)  # (B, C, T_text)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * s_p_sq_r.transpose(1, 2), z_p ** 2)
            neg_cent3 = torch.matmul(s_p_sq_r.transpose(1, 2) * m_p.transpose(1, 2), z_p)
            neg_cent4 = torch.sum(-0.5 * m_p ** 2 * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # (B, 1, T_text) + ...

            # Reshape for alignment
            neg_cent = neg_cent.squeeze(1) if neg_cent.dim() == 4 else neg_cent
            if neg_cent.dim() == 3:
                neg_cent = neg_cent.transpose(1, 2)  # (B, T_text, T_spec)
            else:
                neg_cent = neg_cent.squeeze(1)

            # Ensure correct shape
            if neg_cent.dim() == 2:
                neg_cent = neg_cent.unsqueeze(0)

            # Alignment mask
            attn_mask = x_mask.transpose(1, 2) * y_mask  # (B, T_text, T_spec)

            attn = maximum_path(neg_cent, attn_mask)

        # Compute durations from alignment
        w = attn.sum(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, T_text)

        # Duration predictor loss
        dp_loss = self.duration_predictor(x, x_mask, w, g)

        # Expand prior to spectrogram length using alignment
        m_p_expanded = torch.matmul(m_p, attn)  # (B, C, T_spec)
        logs_p_expanded = torch.matmul(logs_p, attn)

        # Random segment for decoder training
        z_slice, ids_slice = self._rand_slice_segments(
            z, spec_lengths, self.config.segment_size // self.config.hop_length
        )

        # Decode to waveform
        o = self.decoder(z_slice, g)

        return {
            "audio": o,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p_expanded,
            "logs_p": logs_p_expanded,
            "m_q": m_q,
            "logs_q": logs_q,
            "dp_loss": dp_loss,
            "attn": attn,
        }

    @torch.no_grad()
    def infer(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference: Generate speech from text.

        Args:
            text: Phoneme IDs (B, T_text)
            text_lengths: Text lengths (B,)
            speaker_id: Speaker ID (B,) — 0=male, 1=female
            noise_scale: Controls expressiveness (0=monotone, 1=expressive)
            noise_scale_w: Controls duration variation
            length_scale: Controls speaking speed (>1 = slower, <1 = faster)

        Returns:
            Generated audio waveform (B, 1, T_audio)
        """
        # Speaker conditioning
        g = None
        if self.speaker_emb is not None and speaker_id is not None:
            g = self.speaker_emb(speaker_id).unsqueeze(-1)

        # Text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(text, text_lengths)

        # Predict durations
        w = self.duration_predictor(x, x_mask, cond=g, reverse=True, noise_scale=noise_scale_w)
        w = w * length_scale
        w_ceil = torch.clamp_min(torch.ceil(w), 1).long()

        # Compute output length
        y_lengths = w_ceil.squeeze(1).sum(dim=1)
        y_max_len = y_lengths.max().item()

        # Create alignment from predicted durations
        y_mask = self._sequence_mask(y_lengths, y_max_len).unsqueeze(1)
        attn = self._generate_path(w_ceil.squeeze(1), y_mask.squeeze(1))

        # Expand prior
        m_p = torch.matmul(m_p, attn)
        logs_p = torch.matmul(logs_p, attn)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow inverse (prior → posterior)
        z, _ = self.flow(z_p, y_mask.squeeze(1), g, reverse=True)

        # Decode to audio
        o = self.decoder(z * y_mask, g)
        return o

    def _rand_slice_segments(
        self, x: torch.Tensor, lengths: torch.Tensor, segment_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly slice segments from the latent for decoder training."""
        B, C, T = x.shape
        ids_start = torch.zeros(B, dtype=torch.long, device=x.device)
        for i in range(B):
            max_start = max(0, lengths[i].item() - segment_size)
            ids_start[i] = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0

        segments = torch.zeros(B, C, segment_size, device=x.device, dtype=x.dtype)
        for i in range(B):
            start = ids_start[i]
            end = min(start + segment_size, T)
            seg_len = end - start
            segments[i, :, :seg_len] = x[i, :, start:end]

        return segments, ids_start

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device)
        return (ids < lengths.unsqueeze(1)).float()

    @staticmethod
    def _generate_path(durations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate alignment path from durations."""
        B, T_text = durations.shape
        T_spec = mask.shape[1] if mask.dim() == 2 else mask.shape[2]

        path = torch.zeros(B, T_text, T_spec, device=durations.device)
        for b in range(B):
            pos = 0
            for t in range(T_text):
                dur = durations[b, t].item()
                if dur > 0 and pos < T_spec:
                    end = min(pos + dur, T_spec)
                    path[b, t, pos:end] = 1.0
                    pos = end
        return path
