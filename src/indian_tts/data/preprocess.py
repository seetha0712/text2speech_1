"""
Data preprocessing pipeline for Indian TTS.

Downloads and prepares ONLY legally safe datasets:
1. Mozilla Common Voice — Indian English subset (CC-0, public domain)
2. Google FLEURS — Indian English (CC-BY 4.0)

NO research-only or non-commercial datasets are used.

Usage:
    # Download all safe datasets and prepare for training
    python -m indian_tts.data.preprocess --output data/

    # Download only Common Voice (largest source)
    python -m indian_tts.data.preprocess --source common_voice --output data/

    # Limit hours to save disk space
    python -m indian_tts.data.preprocess --source common_voice --max-hours 10 --output data/
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


# ===========================================================================
# License Information
# ===========================================================================
DATASET_LICENSES = {
    "common_voice": {
        "name": "Mozilla Common Voice (English — Indian accent subset)",
        "license": "CC-0 1.0 Universal (Public Domain Dedication)",
        "url": "https://commonvoice.mozilla.org/en/datasets",
        "commercial_use": True,
        "attribution_required": False,
        "notes": "Speakers voluntarily donated voice recordings to the public domain.",
    },
    "fleurs": {
        "name": "Google FLEURS (en_in — Indian English)",
        "license": "CC-BY 4.0 International",
        "url": "https://huggingface.co/datasets/google/fleurs",
        "commercial_use": True,
        "attribution_required": True,
        "notes": "Attribution: Google FLEURS dataset, CC-BY 4.0.",
    },
}


def print_license_info():
    """Print license information for all datasets used."""
    print("\n" + "=" * 70)
    print("DATASET LICENSE INFORMATION")
    print("=" * 70)
    for key, info in DATASET_LICENSES.items():
        print(f"\n  [{key}]")
        print(f"  Name:       {info['name']}")
        print(f"  License:    {info['license']}")
        print(f"  Commercial: {'YES' if info['commercial_use'] else 'NO'}")
        print(f"  Attribution: {'Required' if info['attribution_required'] else 'Not required'}")
        print(f"  URL:        {info['url']}")
    print("\n" + "=" * 70 + "\n")


# ===========================================================================
# Common Voice Downloader (CC-0 — Public Domain)
# ===========================================================================

def download_common_voice(
    output_dir: str,
    max_hours: float = 20.0,
    target_sr: int = 22050,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    min_upvotes: int = 2,
) -> List[Dict]:
    """
    Download Mozilla Common Voice — Indian English accent subset.

    License: CC-0 1.0 (Public Domain). Free for any use including commercial.

    Filters for:
    - Indian accent (self-reported by speakers)
    - Male and female speakers (using gender metadata)
    - Validated clips with upvotes (quality filter)
    - Duration between 1-15 seconds

    Args:
        output_dir: Where to save processed audio files
        max_hours: Maximum hours of data to download per gender
        target_sr: Target sampling rate for saved files
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        min_upvotes: Minimum upvotes for quality filtering

    Returns:
        List of manifest entries [{audio_path, speaker_id, text, duration}]
    """
    try:
        from datasets import load_dataset
        import torchaudio
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Run: pip install datasets torchaudio")
        return []

    print("\n[1/3] Downloading Common Voice — Indian English (CC-0)...")
    print(f"      Max hours per gender: {max_hours}")

    cv_dir = os.path.join(output_dir, "common_voice")
    male_dir = os.path.join(cv_dir, "male")
    female_dir = os.path.join(cv_dir, "female")
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    # Load with streaming to avoid downloading the entire 100GB+ dataset
    print("      Loading dataset (streaming mode)...")

    # Try multiple Common Voice versions (newer versions may change format)
    cv_configs = [
        ("mozilla-foundation/common_voice_17_0", "en"),
        ("mozilla-foundation/common_voice_16_1", "en"),
        ("mozilla-foundation/common_voice_16_0", "en"),
    ]

    ds = None
    for dataset_id, lang in cv_configs:
        try:
            print(f"      Trying {dataset_id}...")
            ds = load_dataset(
                dataset_id,
                lang,
                split="train",
                streaming=True,
            )
            # Test that we can iterate
            _ = next(iter(ds))
            print(f"      Using {dataset_id}")
            break
        except Exception as e:
            print(f"      {dataset_id} failed: {type(e).__name__}: {e}")
            ds = None
            continue

    if ds is None:
        print("      ERROR: Could not load any Common Voice version.")
        print("      You may need to accept the dataset terms on HuggingFace:")
        print("      https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("      Then run: huggingface-cli login")
        return []

    entries = []
    male_seconds = 0.0
    female_seconds = 0.0
    max_seconds = max_hours * 3600
    skipped_no_accent = 0
    skipped_no_gender = 0
    skipped_quality = 0
    skipped_duration = 0
    processed = 0

    for i, sample in enumerate(ds):
        # Check if we have enough for both genders
        if male_seconds >= max_seconds and female_seconds >= max_seconds:
            break

        # --- Filter: Indian accent ---
        accent = (sample.get("accent") or "").lower().strip()
        if "india" not in accent:
            skipped_no_accent += 1
            continue

        # --- Filter: Known gender ---
        gender = (sample.get("gender") or "").lower().strip()
        if gender == "male_masculine" or gender == "male":
            gender = "male"
        elif gender == "female_feminine" or gender == "female":
            gender = "female"
        else:
            skipped_no_gender += 1
            continue

        # Skip if this gender already has enough data
        if gender == "male" and male_seconds >= max_seconds:
            continue
        if gender == "female" and female_seconds >= max_seconds:
            continue

        # --- Filter: Quality (upvotes) ---
        up_votes = sample.get("up_votes", 0) or 0
        down_votes = sample.get("down_votes", 0) or 0
        if up_votes < min_upvotes or down_votes > up_votes:
            skipped_quality += 1
            continue

        # --- Extract audio ---
        audio_data = sample["audio"]
        sr = audio_data["sampling_rate"]
        array = np.array(audio_data["array"], dtype=np.float32)
        duration = len(array) / sr

        # --- Filter: Duration ---
        if duration < min_duration or duration > max_duration:
            skipped_duration += 1
            continue

        # --- Filter: Text ---
        text = (sample.get("sentence") or "").strip()
        if len(text) < 3:
            continue

        # --- Resample if needed ---
        if sr != target_sr:
            import torch
            audio_tensor = torch.from_numpy(array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            array = resampler(audio_tensor).squeeze(0).numpy()

        # --- Normalize audio ---
        max_val = np.abs(array).max()
        if max_val > 0:
            array = array / max_val * 0.95

        # --- Save ---
        speaker_id = 0 if gender == "male" else 1
        wav_dir = male_dir if gender == "male" else female_dir
        filename = f"cv_{processed:06d}.wav"
        filepath = os.path.join(wav_dir, filename)
        sf.write(filepath, array, target_sr)

        entries.append({
            "audio_path": filepath,
            "speaker_id": speaker_id,
            "text": text,
            "duration": len(array) / target_sr,
        })

        if gender == "male":
            male_seconds += duration
        else:
            female_seconds += duration
        processed += 1

        if processed % 200 == 0:
            print(
                f"      Processed {processed} clips | "
                f"Male: {male_seconds/3600:.2f}h | "
                f"Female: {female_seconds/3600:.2f}h"
            )

    print(f"\n      Common Voice Results:")
    print(f"        Total clips: {processed}")
    print(f"        Male:   {male_seconds/3600:.2f} hours")
    print(f"        Female: {female_seconds/3600:.2f} hours")
    print(f"        Skipped (no Indian accent): {skipped_no_accent}")
    print(f"        Skipped (no gender label):  {skipped_no_gender}")
    print(f"        Skipped (low quality):      {skipped_quality}")
    print(f"        Skipped (bad duration):     {skipped_duration}")

    return entries


# ===========================================================================
# Google FLEURS Downloader (CC-BY 4.0)
# ===========================================================================

def download_fleurs(
    output_dir: str,
    target_sr: int = 22050,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> List[Dict]:
    """
    Download Google FLEURS — Indian English (en_in).

    License: CC-BY 4.0. Free for commercial use with attribution.
    Attribution: "Google FLEURS dataset, licensed under CC-BY 4.0."

    The en_in subset contains ~10-15 hours of Indian English speech
    with gender labels.

    Args:
        output_dir: Where to save processed audio files
        target_sr: Target sampling rate

    Returns:
        List of manifest entries
    """
    try:
        from datasets import load_dataset
        import torchaudio
        import torch
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Run: pip install datasets torchaudio")
        return []

    print("\n[2/3] Downloading FLEURS — Indian English (CC-BY 4.0)...")

    fleurs_dir = os.path.join(output_dir, "fleurs")
    male_dir = os.path.join(fleurs_dir, "male")
    female_dir = os.path.join(fleurs_dir, "female")
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    entries = []

    for split in ["train", "validation", "test"]:
        print(f"      Loading {split} split...")
        try:
            ds = load_dataset("google/fleurs", "en_in", split=split)
        except Exception as e:
            print(f"      Warning: Could not load {split}: {e}")
            continue

        for i, sample in enumerate(ds):
            audio_data = sample["audio"]
            sr = audio_data["sampling_rate"]
            array = np.array(audio_data["array"], dtype=np.float32)
            duration = len(array) / sr

            if duration < min_duration or duration > max_duration:
                continue

            text = (sample.get("transcription") or "").strip()
            if len(text) < 3:
                continue

            # FLEURS has gender: 0=male, 1=female
            gender_val = sample.get("gender", -1)
            if gender_val == 0:
                speaker_id = 0
                wav_dir = male_dir
            elif gender_val == 1:
                speaker_id = 1
                wav_dir = female_dir
            else:
                continue

            # Resample
            if sr != target_sr:
                audio_tensor = torch.from_numpy(array).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                array = resampler(audio_tensor).squeeze(0).numpy()

            # Normalize
            max_val = np.abs(array).max()
            if max_val > 0:
                array = array / max_val * 0.95

            filename = f"fleurs_{split}_{i:05d}.wav"
            filepath = os.path.join(wav_dir, filename)
            sf.write(filepath, array, target_sr)

            entries.append({
                "audio_path": filepath,
                "speaker_id": speaker_id,
                "text": text,
                "duration": len(array) / target_sr,
            })

    male_count = sum(1 for e in entries if e["speaker_id"] == 0)
    female_count = sum(1 for e in entries if e["speaker_id"] == 1)
    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"\n      FLEURS Results:")
    print(f"        Total clips: {len(entries)}")
    print(f"        Male:   {male_count} clips")
    print(f"        Female: {female_count} clips")
    print(f"        Total:  {total_hours:.2f} hours")

    return entries


# ===========================================================================
# Manifest Creation
# ===========================================================================

def create_manifests(
    entries: List[Dict],
    output_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """Create train/val/test manifest files."""
    random.seed(seed)
    random.shuffle(entries)

    n = len(entries)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_entries = entries[:n_test]
    val_entries = entries[n_test : n_test + n_val]
    train_entries = entries[n_test + n_val :]

    def write_manifest(path: str, data: List[Dict]):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# audio_path|speaker_id|text\n")
            for entry in data:
                f.write(f"{entry['audio_path']}|{entry['speaker_id']}|{entry['text']}\n")

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    test_path = os.path.join(output_dir, "test.txt")

    write_manifest(train_path, train_entries)
    write_manifest(val_path, val_entries)
    write_manifest(test_path, test_entries)

    print(f"\nManifests created: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
    return train_path, val_path, test_path


def write_attribution_file(output_dir: str, sources_used: List[str]):
    """Write attribution file (required for CC-BY licensed data)."""
    attr_path = os.path.join(output_dir, "ATTRIBUTION.md")
    with open(attr_path, "w") as f:
        f.write("# Dataset Attribution\n\n")
        f.write("This model was trained using the following datasets:\n\n")

        if "common_voice" in sources_used:
            f.write("## Mozilla Common Voice\n")
            f.write("- License: CC-0 1.0 Universal (Public Domain)\n")
            f.write("- URL: https://commonvoice.mozilla.org/\n")
            f.write("- No attribution required (public domain)\n\n")

        if "fleurs" in sources_used:
            f.write("## Google FLEURS\n")
            f.write("- License: CC-BY 4.0 International\n")
            f.write("- URL: https://huggingface.co/datasets/google/fleurs\n")
            f.write("- Attribution: Google FLEURS dataset by Google Research,\n")
            f.write("  licensed under Creative Commons Attribution 4.0 International.\n\n")

    print(f"Attribution file written to: {attr_path}")


# ===========================================================================
# Sample Data (for pipeline testing)
# ===========================================================================

def create_sample_data(output_dir: str, n_samples: int = 20) -> str:
    """Create synthetic sample data for testing the pipeline without downloading."""
    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(os.path.join(sample_dir, "wav"), exist_ok=True)

    sample_texts = [
        "Hello, welcome to our Indian text to speech system.",
        "The weather in Mumbai is hot and humid today.",
        "Please enter your Aadhaar number for verification.",
        "The train from Delhi to Jaipur departs at nine thirty.",
        "Good morning, how are you doing today?",
        "The price of this item is five hundred rupees.",
        "Namaste, welcome to the conference.",
        "The meeting has been scheduled for tomorrow afternoon.",
        "Please call me back at your earliest convenience.",
        "The exam results will be announced next week.",
        "India is a diverse country with many languages.",
        "The stock market opened with positive gains today.",
        "She completed her engineering from IIT Bombay.",
        "The monsoon season brings relief from the summer heat.",
        "Our new office is located near the metro station.",
        "The project deadline has been extended by two weeks.",
        "We need to submit the report before Friday.",
        "The restaurant serves excellent South Indian cuisine.",
        "Cricket is the most popular sport in India.",
        "Technology is transforming education across the country.",
    ]

    entries = []
    sr = 22050

    for i in range(n_samples):
        duration = random.uniform(1.0, 5.0)
        t = np.linspace(0, duration, int(sr * duration))
        freq = random.uniform(200, 400)
        audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        audio *= np.linspace(0.5, 1.0, len(audio)).astype(np.float32)

        speaker_id = i % 2
        filename = f"sample_{i:04d}.wav"
        filepath = os.path.join(sample_dir, "wav", filename)
        sf.write(filepath, audio, sr)

        text = sample_texts[i % len(sample_texts)]
        entries.append({
            "audio_path": filepath,
            "speaker_id": speaker_id,
            "text": text,
            "duration": duration,
        })

    return create_manifests(entries, output_dir, val_ratio=0.1, test_ratio=0.1)[0]


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Indian English TTS training data (license-safe)"
    )
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["common_voice", "fleurs", "sample", "all"],
        help="Data source(s) to download",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=20.0,
        help="Max hours per gender for Common Voice (default: 20)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=22050,
        help="Target sampling rate (default: 22050)",
    )
    parser.add_argument(
        "--min-upvotes",
        type=int,
        default=2,
        help="Minimum upvotes for Common Voice quality filter (default: 2)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Print license info
    print_license_info()

    if args.source == "sample":
        print("Creating sample data for pipeline testing...")
        create_sample_data(args.output)
        print("\nDone! Use --source all to download real Indian English data.")
        return

    all_entries = []
    sources_used = []

    # --- Common Voice (CC-0) ---
    if args.source in ("common_voice", "all"):
        entries = download_common_voice(
            args.output,
            max_hours=args.max_hours,
            target_sr=args.target_sr,
            min_upvotes=args.min_upvotes,
        )
        all_entries.extend(entries)
        if entries:
            sources_used.append("common_voice")

    # --- FLEURS (CC-BY 4.0) ---
    if args.source in ("fleurs", "all"):
        entries = download_fleurs(args.output, target_sr=args.target_sr)
        all_entries.extend(entries)
        if entries:
            sources_used.append("fleurs")

    # --- Create manifests ---
    if all_entries:
        print("\n" + "=" * 70)
        print("[3/3] Creating training manifests...")
        create_manifests(all_entries, args.output)
        write_attribution_file(args.output, sources_used)

        # Statistics
        male_entries = [e for e in all_entries if e["speaker_id"] == 0]
        female_entries = [e for e in all_entries if e["speaker_id"] == 1]
        male_hours = sum(e["duration"] for e in male_entries) / 3600
        female_hours = sum(e["duration"] for e in female_entries) / 3600

        print(f"\n{'=' * 70}")
        print(f"DATASET READY FOR TRAINING")
        print(f"{'=' * 70}")
        print(f"  Total samples: {len(all_entries)}")
        print(f"  Male samples:   {len(male_entries)} ({male_hours:.2f} hours)")
        print(f"  Female samples: {len(female_entries)} ({female_hours:.2f} hours)")
        print(f"  Total duration: {male_hours + female_hours:.2f} hours")
        print(f"\n  Manifest files:")
        print(f"    Train: {args.output}/train.txt")
        print(f"    Val:   {args.output}/val.txt")
        print(f"    Test:  {args.output}/test.txt")
        print(f"\n  All data is CC-0 or CC-BY 4.0 licensed.")
        print(f"  See {args.output}/ATTRIBUTION.md for details.")
        print(f"{'=' * 70}")
    else:
        print("\nERROR: No data downloaded. Check your internet connection and try again.")
        print("You may need to accept the Common Voice terms on HuggingFace first:")
        print("  https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")


if __name__ == "__main__":
    main()
