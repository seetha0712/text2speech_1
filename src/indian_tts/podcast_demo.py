"""
Podcast demo script for Indian TTS validation.

Generates the SAME 2-minute AI podcast conversation at every training stage
so you can directly hear the quality improvement across iterations.

The script alternates between a male host (Arjun) and female host (Priya)
discussing AI in India.

Usage:
    python -m indian_tts.podcast_demo --checkpoint outputs/checkpoints/checkpoint_step_1000.pt
    python -m indian_tts.podcast_demo --checkpoint outputs/checkpoints/checkpoint_step_50000.pt
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf


# ============================================================================
# The Podcast Script (~2 minutes when spoken)
# Each line is (speaker, text). speaker: "male" = Arjun, "female" = Priya
# ============================================================================

PODCAST_SCRIPT = [
    ("female", "Welcome to AI India, the podcast where we explore how artificial intelligence is transforming our country. I am Priya."),
    ("male", "And I am Arjun. Today we are talking about something really exciting. The rise of Indian AI startups."),
    ("female", "That is right, Arjun. India now has over three hundred AI startups, and that number is growing every single month."),
    ("male", "What I find really interesting is that many of these companies are solving uniquely Indian problems. Like agriculture, healthcare in rural areas, and education."),
    ("female", "Absolutely. Take for example an AI system that can detect crop diseases just by looking at a photo taken on a farmer's mobile phone. This is saving thousands of farmers from losing their harvest."),
    ("male", "And in healthcare, AI models are now screening for conditions like diabetic retinopathy and tuberculosis in areas where there are very few doctors available."),
    ("female", "The language barrier is another big challenge that AI is helping with. India has twenty two official languages and hundreds of dialects."),
    ("male", "Exactly. And that is precisely why building speech technology like text to speech systems in Indian languages is so important. We cannot rely only on English."),
    ("female", "Speaking of which, the progress in Indian language AI has been remarkable. Models can now understand and generate speech in Hindi, Tamil, Bengali, and many more."),
    ("male", "The government has also been supportive with initiatives to build open source datasets for Indian languages. This is a game changer."),
    ("female", "So what do you think is next for AI in India, Arjun?"),
    ("male", "I believe we will see AI becoming a part of everyday life. From voice assistants that truly understand Indian accents, to AI tutors that teach children in their mother tongue."),
    ("female", "That is a beautiful vision. And it all starts with building the right foundation, the right data, the right models, and the right talent."),
    ("male", "Could not agree more. India has the talent, and now we are building the tools."),
    ("female", "That is all for today's episode of AI India. Thank you for listening, and we will see you next week."),
    ("male", "Goodbye everyone, and keep innovating!"),
]


def generate_podcast(
    checkpoint_path: str,
    output_dir: str,
    device: Optional[str] = None,
    speed: float = 1.0,
    expressiveness: float = 0.667,
    silence_between_lines_ms: int = 500,
    silence_between_speakers_ms: int = 700,
) -> str:
    """
    Generate the full podcast audio from a checkpoint.

    Produces:
    - Individual line WAVs (for debugging)
    - Combined full podcast WAV (for listening)

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Where to save audio files
        device: Device to use
        speed: Speaking speed
        expressiveness: Voice expressiveness
        silence_between_lines_ms: Silence between same-speaker lines
        silence_between_speakers_ms: Silence between speaker changes

    Returns:
        Path to the combined podcast WAV file
    """
    from indian_tts.inference import IndianTTS

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    tts = IndianTTS(checkpoint_path, device=device)
    sr = tts.sampling_rate

    # Silence segments
    silence_same = np.zeros(int(sr * silence_between_lines_ms / 1000), dtype=np.float32)
    silence_switch = np.zeros(int(sr * silence_between_speakers_ms / 1000), dtype=np.float32)

    # Generate each line
    all_audio_segments = []
    prev_speaker = None
    total_gen_time = 0

    print(f"\nGenerating podcast ({len(PODCAST_SCRIPT)} lines)...\n")
    print("-" * 60)

    for i, (speaker, text) in enumerate(PODCAST_SCRIPT):
        name = "Arjun" if speaker == "male" else "Priya"

        start = time.time()
        audio = tts.synthesize(
            text=text,
            voice=speaker,
            speed=speed,
            expressiveness=expressiveness,
        )
        gen_time = time.time() - start
        total_gen_time += gen_time

        duration = len(audio) / sr
        print(f"  [{name:5s}] ({duration:.1f}s, gen:{gen_time:.1f}s) {text[:55]}...")

        # Save individual line
        line_path = os.path.join(output_dir, f"line_{i:02d}_{speaker}.wav")
        sf.write(line_path, audio, sr)

        # Add silence before this line
        if prev_speaker is not None:
            if speaker != prev_speaker:
                all_audio_segments.append(silence_switch)
            else:
                all_audio_segments.append(silence_same)

        all_audio_segments.append(audio)
        prev_speaker = speaker

    # Combine into full podcast
    full_audio = np.concatenate(all_audio_segments)
    total_duration = len(full_audio) / sr

    podcast_path = os.path.join(output_dir, "podcast_full.wav")
    sf.write(podcast_path, full_audio, sr)

    print("-" * 60)
    print(f"\nPodcast generated!")
    print(f"  Duration:       {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"  Generation time: {total_gen_time:.1f} seconds")
    print(f"  Real-time factor: {total_gen_time/total_duration:.2f}x")
    print(f"  Full podcast:   {podcast_path}")
    print(f"  Individual lines: {output_dir}/line_*.wav")

    return podcast_path


def generate_podcast_for_stage(
    stage_num: int,
    checkpoint_path: str,
    output_base_dir: str,
    device: Optional[str] = None,
) -> str:
    """Generate podcast for a specific training stage."""
    output_dir = os.path.join(output_base_dir, f"podcast_stage_{stage_num}")
    print(f"\n{'=' * 60}")
    print(f"  PODCAST DEMO — Stage {stage_num}")
    print(f"  Using checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'=' * 60}")

    return generate_podcast(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate AI podcast demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/podcast", help="Output directory")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--expressiveness", type=float, default=0.667)
    args = parser.parse_args()

    generate_podcast(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
        speed=args.speed,
        expressiveness=args.expressiveness,
    )


if __name__ == "__main__":
    main()
