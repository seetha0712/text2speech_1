"""
Inference pipeline for Indian TTS.

Supports:
- Male and female Indian voice synthesis
- Speed control
- Expressiveness control
- Batch inference
- ONNX export for production deployment

Usage:
    # CLI
    python -m indian_tts.inference --text "Hello world" --voice female --output out.wav

    # Python API
    from indian_tts.inference import IndianTTS
    tts = IndianTTS("outputs/checkpoints/checkpoint_final.pt")
    tts.synthesize("Hello world", voice="female", output_path="out.wav")
"""

import argparse
import os
import time
from typing import Literal, Optional

import numpy as np
import torch
import soundfile as sf

from indian_tts.model.vits2 import VITS2, VITS2Config
from indian_tts.text.processing import text_to_sequence
from indian_tts.data.audio import AudioProcessor


VOICE_MAP = {
    "male": 0,
    "female": 1,
    "m": 0,
    "f": 1,
    0: 0,
    1: 1,
}


class IndianTTS:
    """
    High-level inference API for Indian TTS.

    Example:
        tts = IndianTTS("path/to/checkpoint.pt")

        # Generate speech
        audio = tts.synthesize("Namaste, how are you?", voice="female")

        # Save to file
        tts.synthesize("Good morning!", voice="male", output_path="greeting.wav")

        # Control parameters
        tts.synthesize(
            "This is a test.",
            voice="female",
            speed=1.2,           # 20% faster
            expressiveness=0.8,  # More expressive
        )
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to training checkpoint
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Build model from saved config
        config = ckpt["config"]
        model_config = VITS2Config.from_yaml(config)
        self.model = VITS2(model_config).to(self.device)
        self.model.load_state_dict(ckpt["generator"])
        self.model.eval()

        # Remove weight norm for faster inference
        self.model.decoder.remove_weight_norm()

        # Audio processor
        self.audio_processor = AudioProcessor.from_config(config)
        self.sampling_rate = config["audio"]["sampling_rate"]

        # Model info
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model loaded: {n_params:.2f}M parameters on {self.device}")

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        voice: Literal["male", "female", "m", "f"] = "female",
        speed: float = 1.0,
        expressiveness: float = 0.667,
        duration_variation: float = 0.8,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Input text to speak
            voice: Voice selection — "male" or "female"
            speed: Speaking speed multiplier (0.5=slow, 1.0=normal, 2.0=fast)
            expressiveness: Controls voice expressiveness (0=monotone, 1=very expressive)
            duration_variation: Controls timing variation (0=rigid, 1=natural)
            output_path: If provided, save audio to this path

        Returns:
            Audio waveform as numpy array
        """
        start_time = time.time()

        # Process text
        phoneme_ids = text_to_sequence(text)
        text_tensor = torch.LongTensor([phoneme_ids]).to(self.device)
        text_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)

        # Speaker
        speaker_id = VOICE_MAP.get(voice, 1)
        sid_tensor = torch.LongTensor([speaker_id]).to(self.device)

        # Generate
        audio = self.model.infer(
            text=text_tensor,
            text_lengths=text_lengths,
            speaker_id=sid_tensor,
            noise_scale=expressiveness,
            noise_scale_w=duration_variation,
            length_scale=1.0 / max(speed, 0.1),
        )

        audio_np = audio.squeeze().cpu().numpy()

        elapsed = time.time() - start_time
        duration = len(audio_np) / self.sampling_rate
        rtf = elapsed / duration if duration > 0 else 0

        print(f"Generated {duration:.2f}s audio in {elapsed:.3f}s (RTF: {rtf:.3f})")

        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio_np, self.sampling_rate)
            print(f"Saved to {output_path}")

        return audio_np

    @torch.no_grad()
    def synthesize_batch(
        self,
        texts: list,
        voice: Literal["male", "female"] = "female",
        speed: float = 1.0,
        expressiveness: float = 0.667,
    ) -> list:
        """
        Synthesize speech for multiple texts.

        Args:
            texts: List of input texts
            voice: Voice selection
            speed: Speaking speed
            expressiveness: Voice expressiveness

        Returns:
            List of audio waveforms as numpy arrays
        """
        results = []
        for text in texts:
            audio = self.synthesize(
                text, voice=voice, speed=speed, expressiveness=expressiveness
            )
            results.append(audio)
        return results

    def export_onnx(self, output_path: str, opset_version: int = 17):
        """
        Export model to ONNX format for production deployment.

        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        print(f"Exporting to ONNX: {output_path}")

        # Create dummy inputs
        dummy_text = torch.randint(0, 100, (1, 50)).to(self.device)
        dummy_lengths = torch.LongTensor([50]).to(self.device)
        dummy_sid = torch.LongTensor([0]).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            (dummy_text, dummy_lengths, dummy_sid),
            output_path,
            opset_version=opset_version,
            input_names=["text", "text_lengths", "speaker_id"],
            output_names=["audio"],
            dynamic_axes={
                "text": {0: "batch", 1: "text_len"},
                "audio": {0: "batch", 2: "audio_len"},
            },
        )
        print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Indian TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="female", choices=["male", "female"])
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.5-2.0)")
    parser.add_argument("--expressiveness", type=float, default=0.667, help="Expressiveness (0-1)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    tts = IndianTTS(args.checkpoint, device=args.device)
    tts.synthesize(
        text=args.text,
        voice=args.voice,
        speed=args.speed,
        expressiveness=args.expressiveness,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
