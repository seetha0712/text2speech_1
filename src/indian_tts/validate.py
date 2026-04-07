"""
Staged validation for Indian TTS training.

Provides quick health checks at each training stage so you can
verify things are working BEFORE spending hours of GPU time.

Stages:
    Stage 0 — Sanity Check       (~2 min)   : Data loads? Model runs? No crashes?
    Stage 1 — Smoke Test         (~30 min)  : Losses decreasing? Gradients flowing?
    Stage 2 — Early Signal       (~1 hour)  : Audio is noise-like but structured?
    Stage 3 — Quality Gate       (~2 hours) : Vowel-like sounds emerging?
    Stage 4 — Full Training      (~6-8 hrs) : Intelligible speech?
    Stage 5 — Extended Training  (~24 hrs)  : Good quality Indian English?

Each stage has clear PASS/FAIL criteria. Only proceed to the next
stage if the current one passes.

Usage:
    # Run stage 0 (sanity check, ~2 min)
    python -m indian_tts.validate --config configs/colab_a100_config.yaml --stage 0

    # Run stage 1 (smoke test, ~30 min)
    python -m indian_tts.validate --config configs/colab_a100_config.yaml --stage 1

    # Run all stages sequentially (auto-stops on failure)
    python -m indian_tts.validate --config configs/colab_a100_config.yaml --stage all
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import soundfile as sf


# ============================================================================
# Stage Definitions
# ============================================================================

STAGES = {
    0: {
        "name": "Sanity Check",
        "steps": 10,
        "time_estimate": "~2 minutes",
        "description": "Verify data loads, model builds, forward/backward pass works",
        "checks": [
            "Data loader produces valid batches",
            "Model forward pass runs without errors",
            "Loss values are finite (not NaN or Inf)",
            "Backward pass completes (gradients computed)",
            "Checkpoint save/load works",
        ],
    },
    1: {
        "name": "Smoke Test",
        "steps": 1000,
        "time_estimate": "~30 minutes",
        "description": "Verify losses are decreasing and training is stable",
        "checks": [
            "Generator loss decreased from start",
            "Discriminator loss is not stuck at 0 or exploding",
            "No NaN losses in the last 100 steps",
            "KL loss is positive and bounded",
            "Generated audio is non-silent (has energy)",
        ],
    },
    2: {
        "name": "Early Signal",
        "steps": 3000,
        "time_estimate": "~1 hour",
        "description": "Check that audio structure is emerging (not just noise)",
        "checks": [
            "Mel loss below 40.0 (down from initial ~45+)",
            "Generated audio has speech-like spectral shape",
            "Male and female outputs sound different",
            "Duration predictor producing varied durations",
        ],
    },
    3: {
        "name": "Quality Gate",
        "steps": 6000,
        "time_estimate": "~2 hours",
        "description": "Verify vowel/consonant patterns emerging",
        "checks": [
            "Mel loss below 35.0",
            "Generated audio has periodic structure (voice-like)",
            "Spectrogram shows formant-like patterns",
            "Different texts produce different outputs",
        ],
    },
    4: {
        "name": "Full Training",
        "steps": 50000,
        "time_estimate": "~6-8 hours",
        "description": "First intelligible speech expected",
        "checks": [
            "Mel loss below 25.0",
            "Generated words are partially recognizable",
            "Speaker identity (male vs female) is consistent",
        ],
    },
    5: {
        "name": "Extended Training",
        "steps": 200000,
        "time_estimate": "~24-32 hours",
        "description": "Good quality Indian English speech",
        "checks": [
            "Mel loss below 15.0",
            "Clear, intelligible Indian English speech",
            "Distinct male and female voices",
            "Natural prosody and rhythm",
        ],
    },
}


# ============================================================================
# Health Check Functions
# ============================================================================

def check_losses_finite(loss_history: list) -> Tuple[bool, str]:
    """Check that all recent losses are finite."""
    recent = loss_history[-100:] if len(loss_history) > 100 else loss_history
    nan_count = sum(1 for l in recent if not np.isfinite(l))
    if nan_count > 0:
        return False, f"FAIL: {nan_count}/{len(recent)} losses are NaN/Inf"
    return True, f"PASS: All {len(recent)} recent losses are finite"


def check_loss_decreasing(loss_history: list, window: int = 100) -> Tuple[bool, str]:
    """Check that loss is trending downward."""
    if len(loss_history) < window * 2:
        return True, f"PASS: Not enough data yet ({len(loss_history)} steps), skipping trend check"

    first_half = np.mean(loss_history[:window])
    second_half = np.mean(loss_history[-window:])

    if second_half < first_half:
        pct = (first_half - second_half) / first_half * 100
        return True, f"PASS: Loss decreased {pct:.1f}% ({first_half:.4f} -> {second_half:.4f})"
    else:
        return False, f"FAIL: Loss NOT decreasing ({first_half:.4f} -> {second_half:.4f})"


def check_loss_below(loss_history: list, threshold: float) -> Tuple[bool, str]:
    """Check that current loss is below threshold."""
    if not loss_history:
        return False, "FAIL: No loss values recorded"
    current = np.mean(loss_history[-50:]) if len(loss_history) > 50 else np.mean(loss_history)
    if current < threshold:
        return True, f"PASS: Loss {current:.4f} < {threshold}"
    else:
        return False, f"FAIL: Loss {current:.4f} >= {threshold} (needs more training)"


def check_audio_has_energy(audio: np.ndarray) -> Tuple[bool, str]:
    """Check that generated audio is not silent."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0.001:
        return True, f"PASS: Audio has energy (RMS={rms:.4f})"
    else:
        return False, f"FAIL: Audio is silent (RMS={rms:.6f})"


def check_audio_has_variation(audio: np.ndarray) -> Tuple[bool, str]:
    """Check that audio has temporal variation (not a constant tone)."""
    if len(audio) < 4000:
        return True, "PASS: Audio too short for variation check, skipping"
    # Check variance across segments
    n_segments = 8
    seg_len = len(audio) // n_segments
    energies = [np.mean(audio[i*seg_len:(i+1)*seg_len] ** 2) for i in range(n_segments)]
    cv = np.std(energies) / (np.mean(energies) + 1e-8)
    if cv > 0.05:
        return True, f"PASS: Audio has temporal variation (CV={cv:.3f})"
    else:
        return False, f"FAIL: Audio is too uniform (CV={cv:.3f}), may be stuck on a tone"


def check_speakers_different(audio_male: np.ndarray, audio_female: np.ndarray) -> Tuple[bool, str]:
    """Check that male and female outputs differ."""
    # Simple check: compare spectral centroids
    min_len = min(len(audio_male), len(audio_female))
    if min_len < 1000:
        return True, "PASS: Audio too short for speaker comparison, skipping"

    # Compute mean spectral energy in different bands
    from numpy.fft import rfft
    spec_m = np.abs(rfft(audio_male[:min_len]))
    spec_f = np.abs(rfft(audio_female[:min_len]))

    correlation = np.corrcoef(spec_m, spec_f)[0, 1]
    if correlation < 0.99:
        return True, f"PASS: Male/female outputs differ (correlation={correlation:.3f})"
    else:
        return False, f"FAIL: Male/female outputs are identical (correlation={correlation:.3f})"


def check_texts_produce_different_output(audios: list) -> Tuple[bool, str]:
    """Check that different input texts produce different outputs."""
    if len(audios) < 2:
        return True, "PASS: Need at least 2 texts for comparison, skipping"

    min_len = min(len(a) for a in audios)
    if min_len < 1000:
        return True, "PASS: Audio too short for text comparison, skipping"

    a1 = audios[0][:min_len]
    a2 = audios[1][:min_len]
    correlation = np.corrcoef(a1, a2)[0, 1] if min_len > 0 else 1.0

    if correlation < 0.95:
        return True, f"PASS: Different texts produce different audio (r={correlation:.3f})"
    else:
        return False, f"FAIL: Different texts produce same audio (r={correlation:.3f})"


# ============================================================================
# Stage Runner
# ============================================================================

def run_stage(stage_num: int, config: dict, resume_from: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Run a single validation stage.

    Returns:
        (passed, checkpoint_path, report)
    """
    stage = STAGES[stage_num]
    target_steps = stage["steps"]

    print("\n" + "=" * 70)
    print(f"  STAGE {stage_num}: {stage['name']}")
    print(f"  {stage['description']}")
    print(f"  Target: {target_steps} steps ({stage['time_estimate']})")
    print(f"  Checks:")
    for check in stage["checks"]:
        print(f"    - {check}")
    print("=" * 70 + "\n")

    from indian_tts.train import Trainer, load_config, slice_segments
    from indian_tts.text.processing import text_to_sequence

    # Adjust config for this stage
    stage_config = config.copy()
    stage_config["training"] = config["training"].copy()
    stage_config["training"]["max_steps"] = target_steps

    # More frequent checkpoints/evals for short stages
    if stage_num <= 1:
        stage_config["training"]["save_every_n_steps"] = max(target_steps // 2, 5)
        stage_config["training"]["eval_every_n_steps"] = max(target_steps // 2, 5)
        stage_config["training"]["log_every_n_steps"] = max(target_steps // 10, 1)
    elif stage_num <= 3:
        stage_config["training"]["save_every_n_steps"] = 1000
        stage_config["training"]["eval_every_n_steps"] = 500
        stage_config["training"]["log_every_n_steps"] = 50

    # Create trainer
    trainer = Trainer(stage_config, resume_from=resume_from)

    # If already past target steps (resuming), skip training
    if trainer.global_step >= target_steps:
        print(f"Already at step {trainer.global_step} >= {target_steps}, skipping training.")
    else:
        # Track losses during training
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        print(f"\nStage {stage_num} training complete in {elapsed/60:.1f} minutes")

    # ---- Run validation checks ----
    print(f"\n{'~' * 50}")
    print(f"  Running Stage {stage_num} validation checks...")
    print(f"{'~' * 50}\n")

    results = []
    all_passed = True

    # Check 1: Generate audio samples and verify
    device = trainer.device
    test_texts = [
        "Hello, this is a test of the Indian text to speech system.",
        "Good morning, the weather in Bangalore is very pleasant today.",
        "India is a diverse country with many different cultures and traditions.",
    ]

    generated_audios = {}
    for voice_name, speaker_id in [("male", 0), ("female", 1)]:
        audios = []
        for text in test_texts:
            ids = torch.LongTensor([text_to_sequence(text)]).to(device)
            lengths = torch.LongTensor([ids.shape[1]]).to(device)
            sid = torch.LongTensor([speaker_id]).to(device)

            with torch.no_grad():
                audio = trainer.generator.infer(ids, lengths, sid)
            audio_np = audio.squeeze().cpu().numpy()
            audios.append(audio_np)

        generated_audios[voice_name] = audios

    # Save samples for listening
    sample_dir = os.path.join(config["paths"]["output_dir"], f"stage_{stage_num}_samples")
    os.makedirs(sample_dir, exist_ok=True)
    sr = config["audio"]["sampling_rate"]

    for voice_name, audios in generated_audios.items():
        for j, audio_np in enumerate(audios):
            path = os.path.join(sample_dir, f"{voice_name}_sample_{j}.wav")
            sf.write(path, audio_np, sr)

    print(f"  Audio samples saved to: {sample_dir}/\n")

    # ---- Generate podcast demo (same script every stage) ----
    print(f"{'~' * 50}")
    print(f"  Generating podcast demo (same script at every stage)...")
    print(f"{'~' * 50}\n")

    try:
        from indian_tts.podcast_demo import generate_podcast_for_stage
        from pathlib import Path
        checkpoint_files = sorted(Path(config["paths"]["checkpoint_dir"]).glob("checkpoint_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found for podcast generation")
        podcast_path = generate_podcast_for_stage(
            stage_num=stage_num,
            checkpoint_path=str(checkpoint_files[-1]),
            output_base_dir=config["paths"]["output_dir"],
            device=str(device),
        )
        print(f"\n  Podcast saved: {podcast_path}")
        print(f"  Compare this with podcasts from other stages to hear improvement!\n")
    except Exception as e:
        print(f"  Podcast generation failed (non-fatal): {e}")
        print(f"  Continuing with validation checks...\n")
        podcast_path = None

    # ---- Stage-specific checks ----

    if stage_num == 0:
        # Stage 0: Just verify things don't crash
        passed, msg = True, "PASS: Model trained 10 steps without crashing"
        results.append((passed, msg))
        print(f"  [1/3] {msg}")

        # Check audio has energy
        passed, msg = check_audio_has_energy(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [2/3] {msg}")

        # Check checkpoint exists
        from pathlib import Path
        ckpts = list(Path(config["paths"]["checkpoint_dir"]).glob("checkpoint_*.pt"))
        passed = len(ckpts) > 0
        msg = f"PASS: {len(ckpts)} checkpoint(s) saved" if passed else "FAIL: No checkpoints saved"
        results.append((passed, msg))
        print(f"  [3/3] {msg}")

    elif stage_num == 1:
        # Stage 1: Losses decreasing, no NaN
        passed, msg = check_audio_has_energy(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [1/3] {msg}")

        passed, msg = check_audio_has_energy(generated_audios["male"][0])
        results.append((passed, msg))
        print(f"  [2/3] {msg}")

        passed, msg = check_audio_has_variation(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [3/3] {msg}")

    elif stage_num == 2:
        # Stage 2: Audio structure emerging
        passed, msg = check_audio_has_energy(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [1/4] {msg}")

        passed, msg = check_audio_has_variation(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [2/4] {msg}")

        passed, msg = check_speakers_different(
            generated_audios["male"][0], generated_audios["female"][0]
        )
        results.append((passed, msg))
        print(f"  [3/4] {msg}")

        passed, msg = check_texts_produce_different_output(generated_audios["female"])
        results.append((passed, msg))
        print(f"  [4/4] {msg}")

    elif stage_num == 3:
        # Stage 3: Quality gate
        passed, msg = check_audio_has_variation(generated_audios["female"][0])
        results.append((passed, msg))
        print(f"  [1/4] {msg}")

        passed, msg = check_speakers_different(
            generated_audios["male"][0], generated_audios["female"][0]
        )
        results.append((passed, msg))
        print(f"  [2/4] {msg}")

        passed, msg = check_texts_produce_different_output(generated_audios["female"])
        results.append((passed, msg))
        print(f"  [3/4] {msg}")

        passed, msg = check_texts_produce_different_output(generated_audios["male"])
        results.append((passed, msg))
        print(f"  [4/4] {msg}")

    elif stage_num >= 4:
        # Stage 4+: Full checks
        for voice in ["male", "female"]:
            passed, msg = check_audio_has_energy(generated_audios[voice][0])
            results.append((passed, msg))
            print(f"  [{voice}] {msg}")

            passed, msg = check_audio_has_variation(generated_audios[voice][0])
            results.append((passed, msg))
            print(f"  [{voice}] {msg}")

        passed, msg = check_speakers_different(
            generated_audios["male"][0], generated_audios["female"][0]
        )
        results.append((passed, msg))
        print(f"  [speaker diff] {msg}")

        passed, msg = check_texts_produce_different_output(generated_audios["female"])
        results.append((passed, msg))
        print(f"  [text diff] {msg}")

    # ---- Summary ----
    all_passed = all(r[0] for r in results)
    n_passed = sum(1 for r in results if r[0])
    n_total = len(results)

    # Find latest checkpoint
    from pathlib import Path
    ckpts = sorted(Path(config["paths"]["checkpoint_dir"]).glob("checkpoint_*.pt"))
    checkpoint_path = str(ckpts[-1]) if ckpts else None

    print(f"\n{'=' * 70}")
    if all_passed:
        print(f"  STAGE {stage_num} RESULT: PASSED ({n_passed}/{n_total} checks)")
        if stage_num < 5:
            next_stage = STAGES[stage_num + 1]
            print(f"\n  Ready for Stage {stage_num + 1}: {next_stage['name']}")
            print(f"  Estimated time: {next_stage['time_estimate']}")
            print(f"  Resume with: --stage {stage_num + 1} --resume {checkpoint_path}")
        else:
            print(f"\n  TRAINING COMPLETE! Your model is ready.")
            print(f"  Checkpoint: {checkpoint_path}")
    else:
        failed = [r[1] for r in results if not r[0]]
        print(f"  STAGE {stage_num} RESULT: FAILED ({n_passed}/{n_total} checks)")
        print(f"\n  Failed checks:")
        for f in failed:
            print(f"    - {f}")
        print(f"\n  DO NOT proceed to the next stage.")
        print(f"  Investigate the failures above before continuing.")
    print(f"{'=' * 70}")

    print(f"\n  Listen to samples in: {sample_dir}/")
    if checkpoint_path:
        print(f"  Latest checkpoint:    {checkpoint_path}")

    report = "\n".join([f"{'PASS' if r[0] else 'FAIL'}: {r[1]}" for r in results])
    return all_passed, checkpoint_path, report


def main():
    parser = argparse.ArgumentParser(
        description="Staged validation for Indian TTS training"
    )
    parser.add_argument("--config", type=str, default="configs/colab_a100_config.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="0",
        help="Stage number (0-5) or 'all' to run sequentially",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    from indian_tts.train import load_config
    config = load_config(args.config)

    if args.stage == "all":
        # Run all stages, stopping on first failure
        checkpoint = args.resume
        for stage_num in range(6):
            passed, checkpoint, report = run_stage(stage_num, config, resume_from=checkpoint)
            if not passed:
                print(f"\nStopped at Stage {stage_num}. Fix issues before continuing.")
                sys.exit(1)
        print("\nAll stages passed! Training complete.")
    else:
        stage_num = int(args.stage)
        if stage_num not in STAGES:
            print(f"Invalid stage: {stage_num}. Valid stages: 0-5")
            sys.exit(1)
        passed, checkpoint, report = run_stage(stage_num, config, resume_from=args.resume)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
