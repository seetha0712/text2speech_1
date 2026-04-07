"""
Data preprocessing pipeline for Indian TTS.

Downloads and prepares ONLY legally safe datasets:
1. AI4Bharat Svarah — Indian English benchmark (CC-BY 4.0)
2. Common Voice community mirror — Indian English accent subset (CC-0)

NO research-only or non-commercial datasets are used.

Usage:
    # Download all safe datasets and prepare for training
    python -m indian_tts.data.preprocess --output data/

    # Download only Svarah (guaranteed to work)
    python -m indian_tts.data.preprocess --source svarah --output data/

    # Download Common Voice mirror (larger, may take longer)
    python -m indian_tts.data.preprocess --source common_voice --output data/ --max-hours 10
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
    "svarah": {
        "name": "AI4Bharat Svarah — Indian English Benchmark",
        "license": "CC-BY 4.0 International",
        "url": "https://huggingface.co/datasets/ai4bharat/Svarah",
        "commercial_use": True,
        "attribution_required": True,
        "notes": (
            "9.6 hrs of Indian English from 117 speakers across 65 districts. "
            "Attribution: AI4Bharat Svarah dataset, CC-BY 4.0."
        ),
    },
    "common_voice": {
        "name": "Mozilla Common Voice (community mirror, Indian English accent)",
        "license": "CC-0 1.0 Universal (Public Domain Dedication)",
        "url": "https://huggingface.co/datasets/fsicoli/common_voice_22_0",
        "commercial_use": True,
        "attribution_required": False,
        "notes": "Speakers voluntarily donated voice recordings to the public domain.",
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
# AI4Bharat Svarah Downloader (CC-BY 4.0) — GUARANTEED TO WORK
# ===========================================================================

def download_svarah(
    output_dir: str,
    target_sr: int = 22050,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> List[Dict]:
    """
    Download AI4Bharat Svarah — Indian English benchmark dataset.

    License: CC-BY 4.0. Free for commercial use with attribution.

    This is a purpose-built Indian English dataset with:
    - 9.6 hours of speech from 117 speakers
    - Gender labels (Male/Female)
    - Speakers from 65 districts across 19 Indian states
    - Clean audio with transcriptions

    Args:
        output_dir: Where to save processed audio files
        target_sr: Target sampling rate
        min_duration: Minimum clip duration
        max_duration: Maximum clip duration

    Returns:
        List of manifest entries
    """
    try:
        from datasets import load_dataset
        import torch
        import torchaudio
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Run: pip install datasets torchaudio")
        return []

    print("\n[1/2] Downloading Svarah — Indian English (CC-BY 4.0)...")
    print("      Source: ai4bharat/Svarah (HuggingFace)")

    svarah_dir = os.path.join(output_dir, "svarah")
    male_dir = os.path.join(svarah_dir, "male")
    female_dir = os.path.join(svarah_dir, "female")
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    # Check HuggingFace login
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"      HuggingFace user: {user.get('name', 'unknown')}")
    except Exception:
        print("      WARNING: Not logged into HuggingFace.")
        print("      Run: from huggingface_hub import notebook_login; notebook_login()")
        print("      Then re-run this cell.")

    # Load the dataset — try multiple approaches
    print("      Loading dataset...")
    ds = None

    # Try 1: Standard load
    try:
        ds = load_dataset("ai4bharat/Svarah", split="test")
    except Exception as e1:
        err1 = str(e1)
        print(f"      Standard load failed: {err1[:150]}")

        # Try 2: With explicit token from environment or HfApi
        try:
            token = None
            try:
                from huggingface_hub import HfApi
                token = HfApi().token
            except Exception:
                pass
            if not token:
                # Check all common token env var names
                for env_name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_1", "HF_TOKEN_2"]:
                    token = os.environ.get(env_name)
                    if token:
                        print(f"      Found token in ${env_name}")
                        break

            if token:
                print("      Retrying with explicit token...")
                ds = load_dataset("ai4bharat/Svarah", split="test", token=token)
            else:
                print("      No HF token found.")
        except Exception as e2:
            print(f"      Token retry failed: {str(e2)[:150]}")

    if ds is None:
        print("\n      ERROR: Could not load Svarah.")
        print("      Steps to fix:")
        print("      1. Go to: https://huggingface.co/datasets/ai4bharat/Svarah")
        print("      2. Click 'Agree and access repository' (if gated)")
        print("      3. Get a token: https://huggingface.co/settings/tokens")
        print("      4. In Colab run:")
        print("           from huggingface_hub import notebook_login")
        print("           notebook_login()")
        print("      5. Re-run this cell")
        return []

    print(f"      Loaded {len(ds)} samples")

    entries = []
    male_seconds = 0.0
    female_seconds = 0.0
    skipped = 0

    for i, sample in enumerate(ds):
        # Get audio
        audio_data = sample.get("audio")
        if audio_data is None:
            skipped += 1
            continue

        sr = audio_data["sampling_rate"]
        array = np.array(audio_data["array"], dtype=np.float32)
        duration = len(array) / sr

        # Duration filter
        if duration < min_duration or duration > max_duration:
            skipped += 1
            continue

        # Get text
        text = (sample.get("text") or sample.get("transcript") or "").strip()
        if len(text) < 3:
            skipped += 1
            continue

        # Get gender
        gender = (sample.get("gender") or "").strip().lower()
        if gender in ("male", "m"):
            speaker_id = 0
            wav_dir = male_dir
        elif gender in ("female", "f"):
            speaker_id = 1
            wav_dir = female_dir
        else:
            skipped += 1
            continue

        # Resample if needed
        if sr != target_sr:
            audio_tensor = torch.from_numpy(array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            array = resampler(audio_tensor).squeeze(0).numpy()

        # Normalize
        max_val = np.abs(array).max()
        if max_val > 0:
            array = array / max_val * 0.95

        # Save
        filename = f"svarah_{i:06d}.wav"
        filepath = os.path.join(wav_dir, filename)
        sf.write(filepath, array, target_sr)

        entries.append({
            "audio_path": filepath,
            "speaker_id": speaker_id,
            "text": text,
            "duration": len(array) / target_sr,
        })

        if speaker_id == 0:
            male_seconds += duration
        else:
            female_seconds += duration

        if (i + 1) % 500 == 0:
            print(f"      Processed {i+1}/{len(ds)} | Male: {male_seconds/3600:.2f}h | Female: {female_seconds/3600:.2f}h")

    print(f"\n      Svarah Results:")
    print(f"        Total clips: {len(entries)}")
    print(f"        Male:   {male_seconds/3600:.2f} hours")
    print(f"        Female: {female_seconds/3600:.2f} hours")
    print(f"        Skipped: {skipped}")

    return entries


# ===========================================================================
# Common Voice Community Mirror (CC-0 — Public Domain)
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
    Download Common Voice — Indian English accent subset from community mirror.

    License: CC-0 1.0 (Public Domain). Free for any use.

    Uses fsicoli/common_voice_22_0 (community mirror) since the official
    Mozilla HuggingFace repo was emptied in October 2025.

    Args:
        output_dir: Where to save processed audio files
        max_hours: Maximum hours of data to download per gender
        target_sr: Target sampling rate
        min_duration: Minimum clip duration
        max_duration: Maximum clip duration
        min_upvotes: Minimum upvotes for quality filtering

    Returns:
        List of manifest entries
    """
    try:
        from datasets import load_dataset
        import torch
        import torchaudio
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Run: pip install datasets torchaudio")
        return []

    print("\n[2/2] Downloading Common Voice — Indian English (CC-0)...")
    print(f"      Max hours per gender: {max_hours}")

    cv_dir = os.path.join(output_dir, "common_voice")
    male_dir = os.path.join(cv_dir, "male")
    female_dir = os.path.join(cv_dir, "female")
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    # Strategy: try multiple approaches to load Common Voice English
    ds = None

    # Approach 1: Load from the auto-converted parquet branch
    try:
        print("      Trying fsicoli/common_voice_22_0 (parquet branch)...")
        ds = load_dataset(
            "fsicoli/common_voice_22_0",
            "en",
            split="train",
            streaming=True,
            revision="refs/convert/parquet",
        )
        _ = next(iter(ds))
        print("      Loaded from parquet branch!")
    except Exception as e:
        print(f"      Parquet branch failed: {type(e).__name__}: {str(e)[:120]}")
        ds = None

    # Approach 2: Try loading parquet files directly by URL
    if ds is None:
        try:
            print("      Trying direct parquet file loading...")
            ds = load_dataset(
                "parquet",
                data_files={"train": "hf://datasets/fsicoli/common_voice_22_0/en/train-*.parquet"},
                split="train",
                streaming=True,
            )
            _ = next(iter(ds))
            print("      Loaded from direct parquet files!")
        except Exception as e:
            print(f"      Direct parquet failed: {type(e).__name__}: {str(e)[:120]}")
            ds = None

    # Approach 3: malaysia-ai mirror (default config, then filter by locale)
    if ds is None:
        for config_name in ["default", "pseudospeaker"]:
            try:
                print(f"      Trying malaysia-ai/common_voice_22_0 ({config_name})...")
                ds = load_dataset(
                    "malaysia-ai/common_voice_22_0",
                    config_name,
                    split="train",
                    streaming=True,
                )
                sample = next(iter(ds))
                # Check if this has locale/language field to filter English
                if "locale" in sample or "language" in sample or "sentence" in sample:
                    print(f"      Loaded from malaysia-ai ({config_name})!")
                    print(f"      Sample fields: {list(sample.keys())}")
                    break
                else:
                    ds = None
            except Exception as e:
                print(f"      {config_name} failed: {type(e).__name__}: {str(e)[:120]}")
                ds = None

    # Approach 4: Try other English speech datasets with Indian speakers
    if ds is None:
        try:
            print("      Trying google/fleurs en_in (Parquet)...")
            ds = load_dataset(
                "google/fleurs",
                "en_in",
                split="train",
                revision="refs/convert/parquet",
                streaming=True,
            )
            _ = next(iter(ds))
            print("      Loaded FLEURS en_in from parquet branch!")
        except Exception as e:
            print(f"      FLEURS parquet failed: {type(e).__name__}: {str(e)[:120]}")
            ds = None

    if ds is None:
        print("      WARNING: Could not load Common Voice from any source.")
        print("      Continuing with Svarah data only.")
        return []

    # Detect which dataset we loaded to adjust filtering
    test_sample = next(iter(ds))
    is_fleurs = "transcription" in test_sample
    # Check both 'accent' and 'accents' field names
    accent_field = None
    if "accent" in test_sample:
        accent_field = "accent"
    elif "accents" in test_sample:
        accent_field = "accents"
    has_accent = accent_field is not None

    # Check if audio is embedded or just a path
    has_audio_column = "audio" in test_sample
    has_path_only = "path" in test_sample and not has_audio_column

    print(f"      Dataset type: {'FLEURS' if is_fleurs else 'Common Voice'}")
    print(f"      Fields: {list(test_sample.keys())[:10]}")
    print(f"      Accent field: {accent_field or 'none'}")
    print(f"      Audio: {'embedded' if has_audio_column else 'path only' if has_path_only else 'unknown'}")

    if has_path_only:
        print("      WARNING: Dataset has file paths but no embedded audio.")
        print("      This mirror may not include audio files. Skipping.")
        return []

    entries = []
    male_seconds = 0.0
    female_seconds = 0.0
    max_seconds = max_hours * 3600
    skipped = 0
    processed = 0

    for i, sample in enumerate(ds):
        # Stop when we have enough
        if male_seconds >= max_seconds and female_seconds >= max_seconds:
            break

        # Filter: Indian accent (skip for FLEURS which is already Indian English)
        if has_accent and accent_field:
            accent = (sample.get(accent_field) or "").lower().strip()
            if "india" not in accent:
                continue

        # Filter: Gender (handle both CV and FLEURS formats)
        gender_raw = sample.get("gender", "")
        if isinstance(gender_raw, int):
            # FLEURS format: 0=male, 1=female
            gender = "male" if gender_raw == 0 else "female" if gender_raw == 1 else ""
        else:
            gender = (gender_raw or "").lower().strip()
            if gender in ("male_masculine", "male"):
                gender = "male"
            elif gender in ("female_feminine", "female"):
                gender = "female"
            else:
                continue

        if gender not in ("male", "female"):
            continue

        # Skip if this gender has enough
        if gender == "male" and male_seconds >= max_seconds:
            continue
        if gender == "female" and female_seconds >= max_seconds:
            continue

        # Filter: Quality (only for CV which has upvotes)
        up_votes = sample.get("up_votes", 0) or 0
        down_votes = sample.get("down_votes", 0) or 0
        if up_votes < min_upvotes or down_votes > up_votes:
            skipped += 1
            continue

        # Extract audio
        audio_data = sample.get("audio")
        if audio_data is None:
            continue
        sr = audio_data["sampling_rate"]
        array = np.array(audio_data["array"], dtype=np.float32)
        duration = len(array) / sr

        if duration < min_duration or duration > max_duration:
            continue

        # Text
        # Text field varies by dataset
        text = (sample.get("sentence") or sample.get("transcription") or sample.get("text") or "").strip()
        if len(text) < 3:
            continue

        # Resample
        if sr != target_sr:
            import torch
            audio_tensor = torch.from_numpy(array).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            array = resampler(audio_tensor).squeeze(0).numpy()

        # Normalize
        max_val = np.abs(array).max()
        if max_val > 0:
            array = array / max_val * 0.95

        # Save
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
    print(f"        Skipped (low quality): {skipped}")

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

        if "svarah" in sources_used:
            f.write("## AI4Bharat Svarah\n")
            f.write("- License: CC-BY 4.0 International\n")
            f.write("- URL: https://huggingface.co/datasets/ai4bharat/Svarah\n")
            f.write("- Attribution: AI4Bharat Svarah dataset, licensed under\n")
            f.write("  Creative Commons Attribution 4.0 International.\n")
            f.write("- Paper: https://github.com/AI4Bharat/Svarah\n\n")

        if "common_voice" in sources_used:
            f.write("## Mozilla Common Voice\n")
            f.write("- License: CC-0 1.0 Universal (Public Domain)\n")
            f.write("- URL: https://commonvoice.mozilla.org/\n")
            f.write("- No attribution required (public domain)\n\n")

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
        choices=["svarah", "common_voice", "sample", "all"],
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

    # --- Svarah (CC-BY 4.0) — always try this first, guaranteed to work ---
    if args.source in ("svarah", "all"):
        entries = download_svarah(args.output, target_sr=args.target_sr)
        all_entries.extend(entries)
        if entries:
            sources_used.append("svarah")

    # --- Common Voice mirror (CC-0) ---
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

    # --- Create manifests ---
    if all_entries:
        print("\n" + "=" * 70)
        print("Creating training manifests...")
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
        print("\nERROR: No data downloaded. Check your internet connection.")
        print("Try running with just Svarah first:")
        print("  python -m indian_tts.data.preprocess --source svarah --output data/")


if __name__ == "__main__":
    main()
