"""
Download and preprocess LIMMITS'24 Indian English dataset.

Dataset: SYSPIN/LIMMITS'24 (CC-BY 4.0)
Source: HuggingFace mirror at iAkashPaul/limmits-2024
Content: 80 hours of studio-quality Indian English (40 hrs male + 40 hrs female)
Speakers: 2 (1 male, 1 female)

Format inside tar.gz:
    English_F_*/
        en_f_00001.wav + en_f_00001.txt
        en_f_00002.wav + en_f_00002.txt
        ...
    English_M_*/
        en_m_00001.wav + en_m_00001.txt
        ...

Audio: WAV, 48kHz, 24-bit (studio quality)
Text: One .txt file per utterance

Usage:
    python -m indian_tts.data.download_limmits --output /path/to/data
"""

import argparse
import os
import tarfile
import subprocess
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


def download_limmits(output_dir: str, target_sr: int = 22050) -> Tuple[str, str, str]:
    """
    Download and preprocess LIMMITS'24 Indian English data.

    Args:
        output_dir: Base directory for processed data
        target_sr: Target sampling rate (22050 for VITS2)

    Returns:
        (train_path, val_path, test_path) manifest file paths
    """
    raw_dir = os.path.join(output_dir, "limmits_raw")
    processed_dir = os.path.join(output_dir, "limmits")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # -------------------------------------------------------
    # Step 1: Download tar.gz files from HuggingFace
    # -------------------------------------------------------
    print("=" * 60)
    print("  LIMMITS'24 Indian English Dataset")
    print("  License: CC-BY 4.0 (commercial OK)")
    print("  80 hours: 40 hrs male + 40 hrs female")
    print("=" * 60)

    hf_repo = "iAkashPaul/limmits-2024"
    tar_files = {
        "female": "English_F.tar.gz",
        "male": "English_M.tar.gz",
    }

    for gender, filename in tar_files.items():
        tar_path = os.path.join(raw_dir, filename)
        if os.path.exists(tar_path):
            print(f"\n  {filename} already downloaded.")
        else:
            print(f"\n  Downloading {filename} (~10 GB)...")
            try:
                subprocess.run([
                    "huggingface-cli", "download",
                    hf_repo, filename,
                    "--repo-type", "dataset",
                    "--local-dir", raw_dir,
                ], check=True)
                print(f"  Downloaded: {tar_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  huggingface-cli failed. Trying Python download...")
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=raw_dir,
                )
                print(f"  Downloaded: {tar_path}")

    # -------------------------------------------------------
    # Step 2: Extract tar.gz files
    # -------------------------------------------------------
    for gender, filename in tar_files.items():
        tar_path = os.path.join(raw_dir, filename)
        extract_marker = os.path.join(raw_dir, f".{gender}_extracted")

        if os.path.exists(extract_marker):
            print(f"\n  {filename} already extracted.")
            continue

        print(f"\n  Extracting {filename}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)

        # Mark as extracted
        Path(extract_marker).touch()
        print(f"  Extracted to {raw_dir}/")

    # -------------------------------------------------------
    # Step 3: Find all WAV + TXT pairs
    # -------------------------------------------------------
    print("\n  Scanning for audio/text pairs...")

    entries = []
    for gender, prefix in [("male", "en_m"), ("female", "en_f")]:
        speaker_id = 0 if gender == "male" else 1
        wav_files = sorted(Path(raw_dir).rglob(f"{prefix}_*.wav"))

        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            text = txt_path.read_text(encoding="utf-8").strip()
            if len(text) < 3:
                continue

            entries.append({
                "wav_path": str(wav_path),
                "speaker_id": speaker_id,
                "gender": gender,
                "text": text,
            })

    male_count = sum(1 for e in entries if e["speaker_id"] == 0)
    female_count = sum(1 for e in entries if e["speaker_id"] == 1)
    print(f"  Found {len(entries)} pairs: {male_count} male, {female_count} female")

    # -------------------------------------------------------
    # Step 4: Process audio (resample + normalize + save)
    # -------------------------------------------------------
    print(f"\n  Processing audio (resampling to {target_sr} Hz)...")

    processed_entries = []
    os.makedirs(os.path.join(processed_dir, "male"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "female"), exist_ok=True)

    for i, entry in enumerate(entries):
        try:
            audio, sr = sf.read(entry["wav_path"])
        except Exception as e:
            continue

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Resample if needed
        if sr != target_sr:
            try:
                import torchaudio
                import torch
                t = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.transforms.Resample(sr, target_sr)(t).squeeze(0).numpy()
            except ImportError:
                # Fallback: simple decimation (not ideal but works)
                ratio = target_sr / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

        duration = len(audio) / target_sr

        # Skip very short or very long
        if duration < 0.5 or duration > 15.0:
            continue

        # Normalize
        mx = np.abs(audio).max()
        if mx > 0:
            audio = audio / mx * 0.95

        # Save processed audio
        out_path = os.path.join(processed_dir, entry["gender"], f"limmits_{i:06d}.wav")
        sf.write(out_path, audio, target_sr)

        processed_entries.append({
            "audio_path": out_path,
            "speaker_id": entry["speaker_id"],
            "text": entry["text"],
            "duration": duration,
        })

        if (i + 1) % 2000 == 0:
            m = sum(1 for e in processed_entries if e["speaker_id"] == 0)
            f = sum(1 for e in processed_entries if e["speaker_id"] == 1)
            m_hrs = sum(e["duration"] for e in processed_entries if e["speaker_id"] == 0) / 3600
            f_hrs = sum(e["duration"] for e in processed_entries if e["speaker_id"] == 1) / 3600
            print(f"    {i+1}/{len(entries)} | Male: {m} ({m_hrs:.1f}h) | Female: {f} ({f_hrs:.1f}h)")

    # -------------------------------------------------------
    # Step 5: Create train/val/test splits
    # -------------------------------------------------------
    print(f"\n  Creating manifests...")

    random.seed(42)
    random.shuffle(processed_entries)

    n = len(processed_entries)
    n_test = max(50, int(n * 0.02))
    n_val = max(50, int(n * 0.02))

    test_entries = processed_entries[:n_test]
    val_entries = processed_entries[n_test : n_test + n_val]
    train_entries = processed_entries[n_test + n_val :]

    def write_manifest(path, data):
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

    # Write attribution
    attr_path = os.path.join(output_dir, "ATTRIBUTION.md")
    with open(attr_path, "w") as f:
        f.write("# Dataset Attribution\n\n")
        f.write("## SYSPIN LIMMITS'24 Indian English\n")
        f.write("- License: CC-BY 4.0 International\n")
        f.write("- Source: SPIRE Lab, IISc Bangalore\n")
        f.write("- URL: https://syspin.iisc.ac.in/datasets\n")
        f.write("- HuggingFace mirror: https://huggingface.co/datasets/iAkashPaul/limmits-2024\n")

    # Statistics
    male_train = [e for e in train_entries if e["speaker_id"] == 0]
    female_train = [e for e in train_entries if e["speaker_id"] == 1]
    male_hrs = sum(e["duration"] for e in male_train) / 3600
    female_hrs = sum(e["duration"] for e in female_train) / 3600

    print(f"\n{'=' * 60}")
    print(f"  DATASET READY")
    print(f"{'=' * 60}")
    print(f"  Total: {len(processed_entries)} utterances")
    print(f"  Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")
    print(f"  Male:   {len(male_train)} train ({male_hrs:.1f} hours)")
    print(f"  Female: {len(female_train)} train ({female_hrs:.1f} hours)")
    print(f"  Total:  {male_hrs + female_hrs:.1f} hours")
    print(f"\n  Manifests: {train_path}")
    print(f"  Attribution: {attr_path}")
    print(f"{'=' * 60}")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Download LIMMITS'24 Indian English data")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--target-sr", type=int, default=22050)
    args = parser.parse_args()

    download_limmits(args.output, args.target_sr)


if __name__ == "__main__":
    main()
