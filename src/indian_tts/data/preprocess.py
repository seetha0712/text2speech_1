"""
Data preprocessing pipeline for Indian TTS.

Downloads, processes, and prepares training data from:
1. IndicTTS dataset (IIT Madras)
2. Common Voice (Indian English subset)
3. Custom recordings

Usage:
    python -m indian_tts.data.preprocess --config configs/base_config.yaml --output data/
"""

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import numpy as np


def download_indic_tts(output_dir: str, languages: List[str] = None) -> str:
    """
    Download IndicTTS dataset from IIT Madras.

    The IndicTTS dataset contains high-quality recordings of
    Indian English speakers (male and female).

    Note: This requires manual download from:
    https://www.iitm.ac.in/donlab/tts/database.php

    This function sets up the expected directory structure.
    """
    if languages is None:
        languages = ["english"]

    dataset_dir = os.path.join(output_dir, "indic_tts")
    os.makedirs(dataset_dir, exist_ok=True)

    readme = f"""# IndicTTS Dataset Setup

## Manual Download Required

The IndicTTS dataset requires registration. Please:

1. Visit: https://www.iitm.ac.in/donlab/tts/database.php
2. Register and download the Indian English dataset
3. Extract the files to: {dataset_dir}

Expected structure:
    {dataset_dir}/
    ├── english/
    │   ├── male/
    │   │   ├── wav/
    │   │   │   ├── eng_m_0001.wav
    │   │   │   ├── eng_m_0002.wav
    │   │   │   └── ...
    │   │   └── txt/
    │   │       ├── eng_m_0001.txt
    │   │       └── ...
    │   └── female/
    │       ├── wav/
    │       └── txt/

## Alternative: Common Voice

You can also use Mozilla Common Voice (Indian English):
    python -m indian_tts.data.preprocess --source common_voice --output data/
"""
    readme_path = os.path.join(dataset_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)

    # Create expected directory structure
    for lang in languages:
        for gender in ["male", "female"]:
            os.makedirs(os.path.join(dataset_dir, lang, gender, "wav"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, lang, gender, "txt"), exist_ok=True)

    return dataset_dir


def download_common_voice(output_dir: str, max_hours: float = 20.0) -> str:
    """
    Download Common Voice Indian English subset using HuggingFace datasets.

    Args:
        output_dir: Where to save processed data
        max_hours: Maximum hours of data to download

    Returns:
        Path to downloaded data directory
    """
    dataset_dir = os.path.join(output_dir, "common_voice")
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        from datasets import load_dataset

        print("Downloading Common Voice Indian English subset...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        male_dir = os.path.join(dataset_dir, "male", "wav")
        female_dir = os.path.join(dataset_dir, "female", "wav")
        os.makedirs(male_dir, exist_ok=True)
        os.makedirs(female_dir, exist_ok=True)

        manifest = []
        total_duration = 0.0
        max_seconds = max_hours * 3600

        for i, sample in enumerate(ds):
            if total_duration >= max_seconds:
                break

            # Filter for Indian English accents
            accent = sample.get("accent", "")
            if accent and "india" not in accent.lower():
                continue

            audio = sample["audio"]
            sr = audio["sampling_rate"]
            array = np.array(audio["array"], dtype=np.float32)
            duration = len(array) / sr

            # Skip very short or very long
            if duration < 1.0 or duration > 15.0:
                continue

            # Determine gender (if available)
            gender = sample.get("gender", "")
            if gender == "male":
                speaker_id = 0
                wav_dir = male_dir
            elif gender == "female":
                speaker_id = 1
                wav_dir = female_dir
            else:
                # Assign alternating if unknown
                speaker_id = i % 2
                wav_dir = male_dir if speaker_id == 0 else female_dir

            # Save audio
            filename = f"cv_{i:06d}.wav"
            filepath = os.path.join(wav_dir, filename)
            sf.write(filepath, array, sr)

            manifest.append({
                "audio_path": filepath,
                "speaker_id": speaker_id,
                "text": sample["sentence"],
                "duration": duration,
            })

            total_duration += duration

            if i % 100 == 0:
                print(f"  Processed {i} samples, {total_duration/3600:.2f} hours...")

        print(f"Downloaded {len(manifest)} samples, {total_duration/3600:.2f} hours total")

        # Save manifest
        manifest_path = os.path.join(dataset_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return dataset_dir

    except ImportError:
        print("WARNING: 'datasets' package not installed. Cannot download Common Voice.")
        print("Install with: pip install datasets")

        # Create placeholder
        placeholder = os.path.join(dataset_dir, "DOWNLOAD_INSTRUCTIONS.md")
        with open(placeholder, "w") as f:
            f.write("""# Common Voice Download

Install the datasets library and run:
    pip install datasets
    python -m indian_tts.data.preprocess --source common_voice --output data/

Or download manually from:
    https://commonvoice.mozilla.org/en/datasets
    (Select 'English' and filter for Indian accents)
""")
        return dataset_dir


def process_indic_tts(raw_dir: str, output_dir: str, target_sr: int = 22050) -> List[Dict]:
    """
    Process IndicTTS raw data into training format.

    Args:
        raw_dir: Path to raw IndicTTS data
        output_dir: Path to save processed data
        target_sr: Target sampling rate

    Returns:
        List of manifest entries
    """
    import librosa

    manifest = []
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for gender in ["male", "female"]:
        speaker_id = 0 if gender == "male" else 1
        wav_dir = os.path.join(raw_dir, "english", gender, "wav")
        txt_dir = os.path.join(raw_dir, "english", gender, "txt")

        if not os.path.exists(wav_dir):
            print(f"  Skipping {gender}: {wav_dir} not found")
            continue

        wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
        out_wav_dir = os.path.join(processed_dir, gender)
        os.makedirs(out_wav_dir, exist_ok=True)

        for wav_file in wav_files:
            wav_path = os.path.join(wav_dir, wav_file)
            txt_file = wav_file.replace(".wav", ".txt")
            txt_path = os.path.join(txt_dir, txt_file)

            if not os.path.exists(txt_path):
                continue

            # Read text
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            # Load and process audio
            try:
                audio, sr = librosa.load(wav_path, sr=target_sr)
            except Exception as e:
                print(f"  Error loading {wav_path}: {e}")
                continue

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=30)

            duration = len(audio) / target_sr
            if duration < 0.5 or duration > 15.0:
                continue

            # Save processed audio
            out_path = os.path.join(out_wav_dir, wav_file)
            sf.write(out_path, audio, target_sr)

            manifest.append({
                "audio_path": out_path,
                "speaker_id": speaker_id,
                "text": text,
                "duration": duration,
            })

    return manifest


def create_manifests(
    entries: List[Dict],
    output_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Create train/val/test manifest files.

    Args:
        entries: List of data entries
        output_dir: Where to save manifests
        val_ratio: Fraction for validation
        test_ratio: Fraction for test

    Returns:
        Paths to (train, val, test) manifest files
    """
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

    print(f"Created manifests: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
    return train_path, val_path, test_path


def create_sample_data(output_dir: str, n_samples: int = 20) -> str:
    """
    Create synthetic sample data for testing the pipeline.

    Generates sine-wave audio with random text to verify
    the entire training pipeline works before using real data.
    """
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
        # Generate dummy audio (sine wave with varying frequency)
        duration = random.uniform(1.0, 5.0)
        t = np.linspace(0, duration, int(sr * duration))
        freq = random.uniform(200, 400)
        audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Add some variation
        audio *= np.linspace(0.5, 1.0, len(audio)).astype(np.float32)

        speaker_id = i % 2  # Alternate male/female
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for Indian TTS")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument(
        "--source",
        type=str,
        default="sample",
        choices=["indic_tts", "common_voice", "sample", "all"],
        help="Data source to process",
    )
    parser.add_argument("--indic-tts-dir", type=str, default=None, help="Path to raw IndicTTS data")
    parser.add_argument("--max-hours", type=float, default=20.0, help="Max hours for Common Voice")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_entries = []

    if args.source in ("sample",):
        print("Creating sample data for pipeline testing...")
        create_sample_data(args.output)
        print("Done! Sample data created. Use --source indic_tts or common_voice for real data.")
        return

    if args.source in ("indic_tts", "all"):
        if args.indic_tts_dir:
            print("Processing IndicTTS data...")
            entries = process_indic_tts(args.indic_tts_dir, args.output)
            all_entries.extend(entries)
            print(f"  Processed {len(entries)} IndicTTS samples")
        else:
            print("Setting up IndicTTS directory structure...")
            download_indic_tts(args.output)
            print("  Please download the dataset manually (see instructions in data/indic_tts/README.md)")

    if args.source in ("common_voice", "all"):
        print("Downloading Common Voice Indian English...")
        cv_dir = download_common_voice(args.output, args.max_hours)
        manifest_path = os.path.join(cv_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                entries = json.load(f)
            all_entries.extend(entries)
            print(f"  Loaded {len(entries)} Common Voice samples")

    if all_entries:
        create_manifests(all_entries, args.output)
        print(f"\nTotal: {len(all_entries)} samples ready for training")

        # Print statistics
        male_count = sum(1 for e in all_entries if e["speaker_id"] == 0)
        female_count = sum(1 for e in all_entries if e["speaker_id"] == 1)
        total_hours = sum(e.get("duration", 0) for e in all_entries) / 3600
        print(f"  Male samples: {male_count}")
        print(f"  Female samples: {female_count}")
        print(f"  Total duration: {total_hours:.2f} hours")


if __name__ == "__main__":
    main()
