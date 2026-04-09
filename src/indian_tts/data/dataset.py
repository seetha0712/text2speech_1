"""
PyTorch Dataset and Collator for Indian TTS training.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from indian_tts.data.audio import AudioProcessor
from indian_tts.text.processing import text_to_sequence


class TTSDataset(Dataset):
    """
    TTS Dataset.

    Expects a manifest file with lines in the format:
        <audio_path>|<speaker_id>|<text>

    Speaker IDs:
        0 = Male (Indian)
        1 = Female (Indian)
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        max_text_length: int = 300,
        max_audio_length: int = 220500,
        min_audio_length: int = 11025,
    ):
        self.audio_processor = audio_processor
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length

        self.data = self._load_manifest(manifest_path)

    def _load_manifest(self, path: str) -> List[Dict]:
        """Load and parse manifest file."""
        data = []
        base_dir = os.path.dirname(path)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue

                audio_path = parts[0]
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(base_dir, audio_path)

                speaker_id = int(parts[1])
                text = parts[2]

                if len(text) > self.max_text_length:
                    continue

                data.append({
                    "audio_path": audio_path,
                    "speaker_id": speaker_id,
                    "text": text,
                })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Load and process audio
        audio = self.audio_processor.load_audio(item["audio_path"])
        audio = self.audio_processor.normalize_audio(audio)

        # Enforce length limits
        if len(audio) > self.max_audio_length:
            audio = audio[: self.max_audio_length]
        if len(audio) < self.min_audio_length:
            # Pad short audio
            audio = torch.nn.functional.pad(audio, (0, self.min_audio_length - len(audio)))

        # Compute spectrogram
        spec = self.audio_processor.get_spec(audio)

        # Convert text to phoneme IDs
        text_ids = torch.LongTensor(text_to_sequence(item["text"]))

        return {
            "text": text_ids,
            "text_length": len(text_ids),
            "spec": spec,
            "spec_length": spec.shape[1],
            "audio": audio,
            "speaker_id": item["speaker_id"],
            "raw_text": item["text"],  # Original text for BERT
        }


class TTSCollator:
    """Collate function for batching TTS samples with padding."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max lengths
        max_text_len = max(b["text_length"] for b in batch)
        max_spec_len = max(b["spec_length"] for b in batch)
        max_audio_len = max(len(b["audio"]) for b in batch)

        B = len(batch)
        spec_channels = batch[0]["spec"].shape[0]

        # Initialize padded tensors
        text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(B, dtype=torch.long)
        spec_padded = torch.zeros(B, spec_channels, max_spec_len)
        spec_lengths = torch.zeros(B, dtype=torch.long)
        audio_padded = torch.zeros(B, max_audio_len)
        speaker_ids = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            tl = b["text_length"]
            sl = b["spec_length"]
            al = len(b["audio"])

            text_padded[i, :tl] = b["text"]
            text_lengths[i] = tl
            spec_padded[i, :, :sl] = b["spec"]
            spec_lengths[i] = sl
            audio_padded[i, :al] = b["audio"]
            speaker_ids[i] = b["speaker_id"]

        # Collect raw texts for BERT
        raw_texts = [b.get("raw_text", "") for b in batch]

        return {
            "text": text_padded,
            "text_lengths": text_lengths,
            "spec": spec_padded,
            "spec_lengths": spec_lengths,
            "audio": audio_padded,
            "speaker_ids": speaker_ids,
            "raw_texts": raw_texts,
        }
