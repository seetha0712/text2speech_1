"""
Training script for Indian TTS (VITS2).

Usage:
    python -m indian_tts.train --config configs/base_config.yaml

Features:
    - Multi-speaker training (male/female Indian voices)
    - Mixed precision (FP16)
    - Gradient accumulation
    - TensorBoard logging
    - Checkpoint saving/resuming
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from indian_tts.model.vits2 import VITS2, VITS2Config
from indian_tts.model.discriminator import MultiPeriodDiscriminator
from indian_tts.data.dataset import TTSDataset, TTSCollator
from indian_tts.data.audio import AudioProcessor
from indian_tts.losses import (
    generator_loss,
    discriminator_loss,
    feature_matching_loss,
    kl_loss,
    mel_loss,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slice_segments(x: torch.Tensor, ids_start: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Slice audio segments matching the decoder output."""
    B = x.shape[0]
    segments = torch.zeros(B, segment_size, device=x.device, dtype=x.dtype)
    for i in range(B):
        start = ids_start[i].item() * 256  # hop_length
        end = start + segment_size
        if end <= x.shape[1]:
            segments[i] = x[i, start:end]
        else:
            available = x.shape[1] - start
            if available > 0:
                segments[i, :available] = x[i, start:]
    return segments


class Trainer:
    """VITS2 Trainer."""

    def __init__(self, config: dict, resume_from: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config["training"]
        set_seed(train_cfg["seed"])

        # Build model
        model_config = VITS2Config.from_yaml(config)
        self.generator = VITS2(model_config).to(self.device)
        self.discriminator = MultiPeriodDiscriminator(
            periods=config["model"]["discriminator"]["periods"],
            use_spectral_norm=config["model"]["discriminator"].get("use_spectral_norm", False),
        ).to(self.device)

        # Optimizers
        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Schedulers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=train_cfg["lr_decay"]
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=train_cfg["lr_decay"]
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["fp16"] and self.device.type == "cuda")
        self.fp16 = train_cfg["fp16"]

        # Audio processor
        self.audio_processor = AudioProcessor.from_config(config)

        # Data
        self._setup_data()

        # Logging
        log_dir = config["paths"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Checkpointing
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.epoch = 0

        # Loss weights
        self.loss_weights = train_cfg["loss_weights"]

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

        # Print model info
        g_params = sum(p.numel() for p in self.generator.parameters()) / 1e6
        d_params = sum(p.numel() for p in self.discriminator.parameters()) / 1e6
        print(f"Generator parameters: {g_params:.2f}M")
        print(f"Discriminator parameters: {d_params:.2f}M")
        print(f"Device: {self.device}")
        print(f"FP16: {self.fp16}")

    def _setup_data(self):
        """Setup data loaders."""
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]

        train_dataset = TTSDataset(
            manifest_path=data_cfg["training_files"],
            audio_processor=self.audio_processor,
            max_text_length=data_cfg["max_text_length"],
            max_audio_length=data_cfg["max_audio_length"],
            min_audio_length=data_cfg["min_audio_length"],
        )

        val_dataset = TTSDataset(
            manifest_path=data_cfg["validation_files"],
            audio_processor=self.audio_processor,
            max_text_length=data_cfg["max_text_length"],
            max_audio_length=data_cfg["max_audio_length"],
            min_audio_length=data_cfg["min_audio_length"],
        )

        collator = TTSCollator()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=collator,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=False,
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    def train(self):
        """Main training loop."""
        train_cfg = self.config["training"]
        max_steps = train_cfg["max_steps"]

        print(f"\nStarting training for {max_steps} steps...")
        start_time = time.time()

        while self.global_step < max_steps:
            self.epoch += 1
            self._train_epoch()

        elapsed = time.time() - start_time
        print(f"\nTraining complete! {max_steps} steps in {elapsed/3600:.2f} hours")
        self._save_checkpoint("final")

    def _train_epoch(self):
        """Train for one epoch."""
        train_cfg = self.config["training"]
        max_steps = train_cfg["max_steps"]

        self.generator.train()
        self.discriminator.train()

        for batch in self.train_loader:
            if self.global_step >= max_steps:
                break

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            losses = self._train_step(batch)

            self.global_step += 1

            # Logging
            if self.global_step % train_cfg["log_every_n_steps"] == 0:
                self._log_training(losses)

            # Evaluation
            if self.global_step % train_cfg["eval_every_n_steps"] == 0:
                self._evaluate()

            # Checkpointing
            if self.global_step % train_cfg["save_every_n_steps"] == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            # LR scheduling
            self.scheduler_g.step()
            self.scheduler_d.step()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        segment_size = self.config["audio"]["segment_size"]
        grad_clip = self.config["training"]["grad_clip_norm"]

        # --- Discriminator step ---
        self.optim_d.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.fp16):
            outputs = self.generator(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                spec=batch["spec"],
                spec_lengths=batch["spec_lengths"],
                speaker_ids=batch["speaker_ids"],
            )

            y_hat = outputs["audio"]  # Generated audio
            ids_slice = outputs["ids_slice"]

            # Slice real audio to match
            y = slice_segments(batch["audio"], ids_slice, segment_size)
            y = y.unsqueeze(1)  # (B, 1, T)

        # Discriminator forward
        y_hat_detach = y_hat.detach()
        with torch.amp.autocast("cuda", enabled=self.fp16):
            real_scores, fake_scores, real_fmaps, fake_fmaps = self.discriminator(y, y_hat_detach)
            loss_d_real, loss_d_fake = discriminator_loss(real_scores, fake_scores)
            loss_d = loss_d_real + loss_d_fake

        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.optim_d)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), grad_clip)
        self.scaler.step(self.optim_d)

        # --- Generator step ---
        self.optim_g.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.fp16):
            # Re-run discriminator with generator gradients
            real_scores, fake_scores, real_fmaps, fake_fmaps = self.discriminator(y, y_hat)

            # Losses
            loss_gen = generator_loss(fake_scores)
            loss_fm = feature_matching_loss(real_fmaps, fake_fmaps)

            # Mel loss
            y_mel = self.audio_processor.get_mel(
                self.audio_processor.get_spec(y.squeeze(1).reshape(-1))
            ) if y.shape[-1] > 0 else torch.zeros(1, device=self.device)
            y_hat_mel = self.audio_processor.get_mel(
                self.audio_processor.get_spec(y_hat.squeeze(1).reshape(-1))
            ) if y_hat.shape[-1] > 0 else torch.zeros(1, device=self.device)
            loss_mel = mel_loss(y_mel, y_hat_mel)

            # KL loss
            loss_kl = kl_loss(
                outputs["z_p"],
                outputs["logs_q"],
                outputs["m_p"],
                outputs["logs_p"],
                outputs["y_mask"],
            )

            # Duration predictor loss
            loss_dp = outputs["dp_loss"]

            # Total generator loss
            w = self.loss_weights
            loss_g = (
                w["generator"] * loss_gen
                + w["feature_matching"] * loss_fm
                + w["mel"] * loss_mel
                + w["kl"] * loss_kl
                + w["duration"] * loss_dp
            )

        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.optim_g)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip)
        self.scaler.step(self.optim_g)
        self.scaler.update()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_gen": loss_gen.item(),
            "loss_fm": loss_fm.item(),
            "loss_mel": loss_mel.item(),
            "loss_kl": loss_kl.item(),
            "loss_dp": loss_dp.item() if isinstance(loss_dp, torch.Tensor) else loss_dp,
        }

    @torch.no_grad()
    def _evaluate(self):
        """Run evaluation."""
        self.generator.eval()
        total_mel_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = self.generator(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                spec=batch["spec"],
                spec_lengths=batch["spec_lengths"],
                speaker_ids=batch["speaker_ids"],
            )

            total_mel_loss += outputs.get("dp_loss", torch.tensor(0.0)).item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = total_mel_loss / n_batches
            self.writer.add_scalar("val/loss", avg_loss, self.global_step)
            print(f"  [Step {self.global_step}] Val loss: {avg_loss:.4f}")

        # Generate sample audio
        self._generate_samples()

        self.generator.train()

    @torch.no_grad()
    def _generate_samples(self):
        """Generate sample audio for monitoring."""
        sample_texts = [
            "Hello, this is a test of the Indian text to speech system.",
            "The weather in Bangalore is pleasant throughout the year.",
        ]
        from indian_tts.text.processing import text_to_sequence

        for i, text in enumerate(sample_texts):
            for speaker_id in [0, 1]:  # Male and Female
                ids = torch.LongTensor([text_to_sequence(text)]).to(self.device)
                lengths = torch.LongTensor([ids.shape[1]]).to(self.device)
                sid = torch.LongTensor([speaker_id]).to(self.device)

                audio = self.generator.infer(ids, lengths, sid)
                tag = f"sample/{['male', 'female'][speaker_id]}_{i}"
                self.writer.add_audio(
                    tag, audio.squeeze().cpu(), self.global_step,
                    sample_rate=self.config["audio"]["sampling_rate"],
                )

    def _log_training(self, losses: Dict[str, float]):
        """Log training metrics."""
        for k, v in losses.items():
            self.writer.add_scalar(f"train/{k}", v, self.global_step)

        lr = self.optim_g.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", lr, self.global_step)

        print(
            f"  [Step {self.global_step}] "
            f"G: {losses['loss_g']:.4f} | D: {losses['loss_d']:.4f} | "
            f"Mel: {losses['loss_mel']:.4f} | KL: {losses['loss_kl']:.4f} | "
            f"LR: {lr:.6f}"
        )

    def _save_checkpoint(self, tag: str):
        """Save training checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
        }, path)
        print(f"  Saved checkpoint: {path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, path: str):
        """Resume from checkpoint."""
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optim_g.load_state_dict(ckpt["optim_g"])
        self.optim_d.load_state_dict(ckpt["optim_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]

        print(f"  Resumed from step {self.global_step}")

    def _cleanup_checkpoints(self):
        """Keep only the latest N checkpoints."""
        keep = self.config["training"]["keep_last_n_checkpoints"]
        ckpts = sorted(Path(self.checkpoint_dir).glob("checkpoint_step_*.pt"))
        if len(ckpts) > keep:
            for ckpt in ckpts[:-keep]:
                ckpt.unlink()


def main():
    parser = argparse.ArgumentParser(description="Train Indian TTS model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Trainer(config, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
