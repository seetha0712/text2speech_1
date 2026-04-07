"""
Audio processing utilities for Indian TTS.

Handles:
- Audio loading and resampling
- Spectrogram computation (linear + mel)
- Audio normalization
- Silence trimming
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio


class AudioProcessor:
    """Audio feature extraction and processing."""

    def __init__(
        self,
        sampling_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        max_wav_value: float = 32768.0,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax or sampling_rate / 2.0
        self.max_wav_value = max_wav_value

        # Pre-compute mel filterbank
        self._mel_basis = None
        self._hann_window = {}

    def load_audio(self, path: str) -> torch.Tensor:
        """Load and resample audio file to target sampling rate."""
        audio, sr = torchaudio.load(path)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        return audio.squeeze(0)  # (T,)

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio

    def trim_silence(
        self, audio: torch.Tensor, threshold_db: float = -40.0, frame_length: int = 2048
    ) -> torch.Tensor:
        """Trim leading and trailing silence."""
        # Convert to energy
        energy = audio.unfold(0, frame_length, frame_length // 4).pow(2).mean(-1)
        threshold = 10 ** (threshold_db / 10)

        # Find non-silent frames
        active = energy > threshold
        if not active.any():
            return audio

        first = active.nonzero()[0].item() * (frame_length // 4)
        last = (active.nonzero()[-1].item() + 1) * (frame_length // 4) + frame_length
        return audio[first:min(last, len(audio))]

    def get_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute linear spectrogram.

        Args:
            audio: Waveform (T,)

        Returns:
            Linear spectrogram (F, T_spec) where F = n_fft // 2 + 1
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Get or create hann window
        device_key = str(audio.device)
        if device_key not in self._hann_window:
            self._hann_window[device_key] = torch.hann_window(
                self.win_length, device=audio.device
            )
        window = self._hann_window[device_key]

        # Pad audio
        pad_amount = (self.n_fft - self.hop_length) // 2
        audio = torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode="reflect")

        # STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )

        spec = spec.abs().squeeze(0)  # (F, T)

        # Log scale
        spec = torch.clamp(spec, min=1e-5)

        return spec

    def get_mel(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert linear spectrogram to mel spectrogram.

        Args:
            spec: Linear spectrogram (F, T)

        Returns:
            Mel spectrogram (n_mels, T)
        """
        if self._mel_basis is None:
            import librosa
            mel_np = librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self._mel_basis = torch.from_numpy(mel_np).float()

        mel_basis = self._mel_basis.to(spec.device)
        mel = torch.matmul(mel_basis, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def spec_to_mel(self, spec: torch.Tensor) -> torch.Tensor:
        """Alias for get_mel."""
        return self.get_mel(spec)

    @classmethod
    def from_config(cls, config: dict) -> "AudioProcessor":
        """Create AudioProcessor from config dictionary."""
        audio_cfg = config.get("audio", config)
        return cls(
            sampling_rate=audio_cfg.get("sampling_rate", 22050),
            n_fft=audio_cfg.get("n_fft", 1024),
            hop_length=audio_cfg.get("hop_length", 256),
            win_length=audio_cfg.get("win_length", 1024),
            n_mels=audio_cfg.get("n_mels", 80),
            mel_fmin=audio_cfg.get("mel_fmin", 0.0),
            mel_fmax=audio_cfg.get("mel_fmax", None),
            max_wav_value=audio_cfg.get("max_wav_value", 32768.0),
        )
