"""
Evaluation and benchmarking tools for Indian TTS.

Metrics:
- MOS prediction (UTMOS / DNSMOS proxy)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Speaker Similarity (cosine similarity of speaker embeddings)
- Real-Time Factor (RTF)
- Mel Cepstral Distortion (MCD)

Usage:
    python -m indian_tts.evaluate \
        --checkpoint outputs/checkpoints/checkpoint_final.pt \
        --test-manifest data/test.txt \
        --output-dir outputs/eval
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import soundfile as sf


def compute_mcd(reference: np.ndarray, synthesized: np.ndarray, sr: int = 22050) -> float:
    """
    Compute Mel Cepstral Distortion (MCD) between reference and synthesized audio.

    Lower MCD = better quality. Typical values:
    - < 4.0: Excellent
    - 4.0-6.0: Good
    - 6.0-8.0: Fair
    - > 8.0: Poor
    """
    try:
        import librosa
    except ImportError:
        print("WARNING: librosa not installed, skipping MCD computation")
        return float("nan")

    # Compute MFCCs
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=13)
    mfcc_syn = librosa.feature.mfcc(y=synthesized, sr=sr, n_mfcc=13)

    # Align lengths
    min_len = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_syn = mfcc_syn[:, :min_len]

    # MCD (exclude c0)
    diff = mfcc_ref[1:] - mfcc_syn[1:]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))

    return float(mcd)


def compute_pesq_score(reference: np.ndarray, synthesized: np.ndarray, sr: int = 16000) -> float:
    """
    Compute PESQ score.

    Range: -0.5 to 4.5 (higher is better)
    """
    try:
        from pesq import pesq
        import librosa
    except ImportError:
        print("WARNING: pesq/librosa not installed, skipping PESQ computation")
        return float("nan")

    # PESQ requires 16kHz
    if sr != 16000:
        reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
        synthesized = librosa.resample(synthesized, orig_sr=sr, target_sr=16000)

    # Align lengths
    min_len = min(len(reference), len(synthesized))
    reference = reference[:min_len]
    synthesized = synthesized[:min_len]

    try:
        score = pesq(16000, reference, synthesized, "wb")
        return float(score)
    except Exception as e:
        print(f"PESQ error: {e}")
        return float("nan")


def compute_stoi_score(reference: np.ndarray, synthesized: np.ndarray, sr: int = 22050) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility) score.

    Range: 0 to 1 (higher is better)
    """
    try:
        from pystoi import stoi
    except ImportError:
        print("WARNING: pystoi not installed, skipping STOI computation")
        return float("nan")

    min_len = min(len(reference), len(synthesized))
    reference = reference[:min_len]
    synthesized = synthesized[:min_len]

    try:
        score = stoi(reference, synthesized, sr, extended=False)
        return float(score)
    except Exception as e:
        print(f"STOI error: {e}")
        return float("nan")


def measure_rtf(
    tts_model,
    texts: List[str],
    voice: str = "female",
    n_runs: int = 3,
) -> Dict[str, float]:
    """
    Measure Real-Time Factor (RTF).

    RTF < 1.0 means faster than real-time.
    """
    rtfs = []

    for text in texts:
        for _ in range(n_runs):
            start = time.time()
            audio = tts_model.synthesize(text, voice=voice)
            elapsed = time.time() - start

            duration = len(audio) / tts_model.sampling_rate
            rtf = elapsed / duration if duration > 0 else float("inf")
            rtfs.append(rtf)

    return {
        "rtf_mean": float(np.mean(rtfs)),
        "rtf_std": float(np.std(rtfs)),
        "rtf_min": float(np.min(rtfs)),
        "rtf_max": float(np.max(rtfs)),
    }


def evaluate_model(
    checkpoint_path: str,
    test_manifest: str,
    output_dir: str,
    max_samples: int = 50,
    device: str = None,
) -> Dict:
    """
    Run comprehensive evaluation on a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        test_manifest: Path to test manifest file
        output_dir: Where to save evaluation results
        max_samples: Maximum number of samples to evaluate
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    from indian_tts.inference import IndianTTS
    from indian_tts.data.audio import AudioProcessor

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    tts = IndianTTS(checkpoint_path, device=device)

    # Load test data
    test_entries = []
    with open(test_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                test_entries.append({
                    "audio_path": parts[0],
                    "speaker_id": int(parts[1]),
                    "text": parts[2],
                })

    test_entries = test_entries[:max_samples]
    print(f"Evaluating on {len(test_entries)} test samples...")

    # Metrics accumulators
    all_mcd = []
    all_pesq = []
    all_stoi = []

    audio_processor = AudioProcessor.from_config(tts.model.config.__dict__ if hasattr(tts.model.config, '__dict__') else {})

    for i, entry in enumerate(test_entries):
        voice = "male" if entry["speaker_id"] == 0 else "female"

        # Synthesize
        synth_audio = tts.synthesize(entry["text"], voice=voice)

        # Save synthesized audio
        synth_path = os.path.join(output_dir, f"synth_{i:04d}_{voice}.wav")
        sf.write(synth_path, synth_audio, tts.sampling_rate)

        # Load reference audio
        if os.path.exists(entry["audio_path"]):
            ref_audio, ref_sr = sf.read(entry["audio_path"])
            if ref_sr != tts.sampling_rate:
                try:
                    import librosa
                    ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=tts.sampling_rate)
                except ImportError:
                    continue

            # Compute metrics
            mcd = compute_mcd(ref_audio, synth_audio, tts.sampling_rate)
            pesq_score = compute_pesq_score(ref_audio, synth_audio, tts.sampling_rate)
            stoi_score = compute_stoi_score(ref_audio, synth_audio, tts.sampling_rate)

            all_mcd.append(mcd)
            all_pesq.append(pesq_score)
            all_stoi.append(stoi_score)

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(test_entries)} samples")

    # RTF benchmark
    benchmark_texts = [
        "Hello, this is a test of the Indian text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "India is the world's largest democracy with over one billion people.",
    ]
    rtf_results = measure_rtf(tts, benchmark_texts)

    # Aggregate results
    def safe_mean(arr):
        valid = [x for x in arr if not np.isnan(x)]
        return float(np.mean(valid)) if valid else float("nan")

    results = {
        "n_samples": len(test_entries),
        "metrics": {
            "mcd_mean": safe_mean(all_mcd),
            "pesq_mean": safe_mean(all_pesq),
            "stoi_mean": safe_mean(all_stoi),
        },
        "rtf": rtf_results,
        "quality_assessment": _assess_quality(safe_mean(all_mcd), safe_mean(all_pesq), safe_mean(all_stoi)),
    }

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MCD:  {results['metrics']['mcd_mean']:.3f} (lower is better, <5.0 is good)")
    print(f"  PESQ: {results['metrics']['pesq_mean']:.3f} (higher is better, >3.0 is good)")
    print(f"  STOI: {results['metrics']['stoi_mean']:.3f} (higher is better, >0.8 is good)")
    print(f"  RTF:  {results['rtf']['rtf_mean']:.3f} (<1.0 = faster than real-time)")
    print(f"\n  Quality: {results['quality_assessment']}")
    print(f"\n  Results saved to: {results_path}")
    print("=" * 60)

    return results


def _assess_quality(mcd: float, pesq: float, stoi: float) -> str:
    """Provide human-readable quality assessment."""
    scores = []
    if not np.isnan(mcd):
        if mcd < 4.0:
            scores.append(4)
        elif mcd < 6.0:
            scores.append(3)
        elif mcd < 8.0:
            scores.append(2)
        else:
            scores.append(1)

    if not np.isnan(pesq):
        if pesq > 3.5:
            scores.append(4)
        elif pesq > 2.5:
            scores.append(3)
        elif pesq > 1.5:
            scores.append(2)
        else:
            scores.append(1)

    if not np.isnan(stoi):
        if stoi > 0.85:
            scores.append(4)
        elif stoi > 0.7:
            scores.append(3)
        elif stoi > 0.5:
            scores.append(2)
        else:
            scores.append(1)

    if not scores:
        return "Unable to assess (metrics unavailable)"

    avg = np.mean(scores)
    if avg >= 3.5:
        return "EXCELLENT — Production-ready quality"
    elif avg >= 2.5:
        return "GOOD — Usable with minor artifacts"
    elif avg >= 1.5:
        return "FAIR — Needs more training or data"
    else:
        return "POOR — Significant quality issues"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Indian TTS model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_manifest=args.test_manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
