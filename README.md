# Indian TTS — Custom Text-to-Speech Model

A custom VITS2-based Text-to-Speech system producing natural **Indian-sounding male and female voices**, built from scratch.

## Architecture

**VITS2** (Variational Inference with adversarial learning for end-to-end TTS, v2):
- Transformer-based text encoder with Indian English phoneme support
- WaveNet posterior encoder
- Normalizing flow for latent space mapping
- HiFi-GAN decoder for high-fidelity waveform generation
- Stochastic duration predictor for natural speech rhythm
- Multi-speaker conditioning (male/female)

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .

# Required: eSpeak-ng for phonemization
# Ubuntu/Debian:
sudo apt-get install espeak-ng
# macOS:
brew install espeak-ng
```

### 2. Prepare Data

```bash
# Generate sample data to test the pipeline
python -m indian_tts.data.preprocess --source sample --output data/

# Use real data (IndicTTS from IIT Madras)
python -m indian_tts.data.preprocess --source indic_tts --indic-tts-dir /path/to/indic_tts --output data/

# Use Common Voice (auto-download Indian English subset)
python -m indian_tts.data.preprocess --source common_voice --max-hours 20 --output data/
```

### 3. Train

```bash
# Train with default config
python -m indian_tts.train --config configs/base_config.yaml

# Resume from checkpoint
python -m indian_tts.train --config configs/base_config.yaml --resume outputs/checkpoints/checkpoint_step_100000.pt
```

### 4. Inference

```bash
# CLI
python -m indian_tts.inference \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --text "Hello, welcome to our text to speech system." \
    --voice female \
    --output output.wav

# Python API
from indian_tts.inference import IndianTTS

tts = IndianTTS("outputs/checkpoints/checkpoint_final.pt")
audio = tts.synthesize("Namaste, how are you?", voice="female")
tts.synthesize("Good morning!", voice="male", output_path="greeting.wav")
```

### 5. Web Demo

```bash
python app.py --checkpoint outputs/checkpoints/checkpoint_final.pt --share
```

### 6. Evaluate

```bash
python -m indian_tts.evaluate \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --test-manifest data/test.txt \
    --output-dir outputs/eval
```

## Voice Controls

| Parameter | Range | Description |
|-----------|-------|-------------|
| `voice` | "male" / "female" | Select Indian voice |
| `speed` | 0.5 — 2.0 | Speaking speed (1.0 = normal) |
| `expressiveness` | 0.0 — 1.0 | Prosodic variation (0.667 = default) |
| `duration_variation` | 0.0 — 1.0 | Timing naturalness (0.8 = default) |

## Data Sources

| Dataset | Description | Access |
|---------|-------------|--------|
| **IndicTTS** (IIT Madras) | Studio-quality Indian English M/F | [Register](https://www.iitm.ac.in/donlab/tts/database.php) |
| **Common Voice** | Crowdsourced Indian English | [Download](https://commonvoice.mozilla.org/) |
| **IndicVoices** | 22 Indian languages, 7000+ hrs | [HuggingFace](https://huggingface.co/datasets/ai4bharat/IndicVoices) |

## Project Structure

```
├── configs/
│   └── base_config.yaml        # Model & training configuration
├── src/indian_tts/
│   ├── text/
│   │   ├── symbols.py          # Indian English phoneme set
│   │   └── processing.py       # G2P and text normalization
│   ├── data/
│   │   ├── audio.py            # Audio feature extraction
│   │   ├── dataset.py          # PyTorch dataset & collator
│   │   └── preprocess.py       # Data download & preparation
│   ├── model/
│   │   ├── vits2.py            # Main VITS2 model
│   │   ├── encoders.py         # Text & posterior encoders
│   │   ├── decoder.py          # HiFi-GAN decoder
│   │   ├── modules.py          # Core building blocks
│   │   ├── duration_predictor.py
│   │   ├── discriminator.py    # Multi-period discriminator
│   │   └── monotonic_align.py  # MAS alignment
│   ├── losses.py               # Training losses
│   ├── train.py                # Training pipeline
│   ├── inference.py            # Inference API
│   └── evaluate.py             # Evaluation & benchmarks
├── app.py                      # Gradio web demo
├── requirements.txt
└── FEASIBILITY_ANALYSIS.md     # Detailed feasibility study
```

## Hardware Requirements

| Tier | GPU | Training Time | Expected MOS |
|------|-----|---------------|-------------|
| MVP | 1x RTX 4090 / A100 | 3-5 days | 3.2-3.5 |
| Production | 4x A100 | 1-2 weeks | 3.7-4.0 |
| SOTA | 8x H100 | 2-4 weeks | 4.0-4.2 |

## License

Research use. See individual dataset licenses for data restrictions.
