# Indian TTS — Custom Indian English Text-to-Speech (Male & Female)

A VITS2-based Text-to-Speech system trained from scratch to produce **Indian English accented male and female voices**.

## Quickest Way to Start

**Open the Colab notebook** (requires Colab Pro+ with A100 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/train_indian_tts_colab.ipynb)

The notebook handles everything: data download, preprocessing, training, and inference.

## How It Works

1. **Downloads Indian English speech data** automatically (legally safe, no manual steps)
2. **Trains a VITS2 model** from scratch on the data
3. **Generates speech** with selectable male/female Indian English voices

### Training Data (Auto-Downloaded)

Only legally safe datasets with permissive licenses are used:

| Dataset | License | Commercial Use | What It Provides |
|---------|---------|---------------|-----------------|
| **Mozilla Common Voice** | **CC-0** (public domain) | Unrestricted | ~100-300 hrs Indian English, male & female |
| **Google FLEURS** (en_in) | **CC-BY 4.0** | Yes (with attribution) | ~10-15 hrs Indian English, male & female |

No research-only or non-commercial datasets are used. See `ATTRIBUTION.md` after download.

## Local Setup (non-Colab)

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .
sudo apt-get install espeak-ng   # Required for phonemization
```

### 2. Download Data

```bash
# Download all safe datasets (Common Voice + FLEURS)
python -m indian_tts.data.preprocess --source all --output data/ --max-hours 15

# Or just Common Voice (largest source)
python -m indian_tts.data.preprocess --source common_voice --output data/ --max-hours 20

# Or generate sample data to test the pipeline first
python -m indian_tts.data.preprocess --source sample --output data/
```

Note: Common Voice requires accepting terms on HuggingFace first:
https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

### 3. Train

```bash
# Train on A100 (use colab config for optimized settings)
python -m indian_tts.train --config configs/colab_a100_config.yaml

# Resume from checkpoint
python -m indian_tts.train --config configs/colab_a100_config.yaml \
    --resume outputs/checkpoints/checkpoint_step_50000.pt
```

### 4. Generate Speech

```bash
# CLI
python -m indian_tts.inference \
    --checkpoint outputs/checkpoints/checkpoint_final.pt \
    --text "Hello, welcome to our text to speech system." \
    --voice female \
    --output output.wav
```

```python
# Python API
from indian_tts.inference import IndianTTS

tts = IndianTTS("outputs/checkpoints/checkpoint_final.pt")
tts.synthesize("Good morning!", voice="male", output_path="greeting.wav")
tts.synthesize("How are you?", voice="female", output_path="greeting_f.wav")
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
| `voice` | `"male"` / `"female"` | Select Indian English voice |
| `speed` | 0.5 — 2.0 | Speaking speed (1.0 = normal) |
| `expressiveness` | 0.0 — 1.0 | Prosodic variation (0.667 = default) |
| `duration_variation` | 0.0 — 1.0 | Timing naturalness (0.8 = default) |

## Architecture

**VITS2** — end-to-end text-to-waveform:
- Transformer text encoder with Indian English phoneme support (retroflex consonants, aspirated stops)
- WaveNet posterior encoder
- Normalizing flow for latent space mapping
- HiFi-GAN decoder for waveform generation
- Stochastic duration predictor for natural rhythm
- Multi-speaker conditioning (male = speaker 0, female = speaker 1)

## Training Time (Colab Pro+ A100)

| Steps | Time | Quality |
|-------|------|---------|
| 50K | ~6-8 hours | Early results, some artifacts |
| 100K | ~12-16 hours | Decent quality |
| 200K | ~24-32 hours | Good quality |

## Project Structure

```
├── notebooks/
│   └── train_indian_tts_colab.ipynb   # One-click Colab notebook
├── configs/
│   ├── base_config.yaml               # Base configuration
│   └── colab_a100_config.yaml         # A100-optimized config
├── src/indian_tts/
│   ├── text/                          # Phoneme symbols & G2P
│   ├── data/                          # Data download, processing, dataset
│   ├── model/                         # VITS2 architecture
│   ├── losses.py                      # Training losses
│   ├── train.py                       # Training pipeline
│   ├── inference.py                   # Inference API
│   └── evaluate.py                    # Evaluation & benchmarks
├── app.py                             # Gradio web demo
├── requirements.txt
├── FEASIBILITY_ANALYSIS.md            # Detailed feasibility study
└── ATTRIBUTION.md                     # Data license attribution (auto-generated)
```

## License

Code: MIT. Training data licenses: CC-0 (Common Voice) and CC-BY 4.0 (FLEURS).
