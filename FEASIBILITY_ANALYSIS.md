# Indian TTS Custom Model — Feasibility Analysis

## Executive Summary

**Verdict: FEASIBLE with caveats.** Building a custom TTS model producing natural Indian-sounding male and female voices is achievable, but matching Qwen-TTS / XTTS-v2 / StyleTTS2 quality requires significant compute, quality data, and iterative tuning. Below is an honest assessment.

---

## 1. Feasibility Tiers

| Tier | Quality Level | Data Needed | Compute | Timeline |
|------|--------------|-------------|---------|----------|
| **MVP** | Intelligible, accented, slight artifacts | 5-10 hrs/speaker | 1× A100 (40GB), ~3-5 days | 2-4 weeks |
| **Production** | Natural, minimal artifacts, good prosody | 20-50 hrs/speaker | 4× A100, ~1-2 weeks | 2-3 months |
| **SOTA-matching** | Comparable to Qwen/XTTS quality | 100+ hrs/speaker + multi-speaker data | 8× H100, ~2-4 weeks | 4-6 months |

## 2. Architecture Selection

We recommend **VITS2** (Variational Inference with adversarial learning for end-to-end Text-to-Speech, v2) as the primary architecture:

### Why VITS2?
- **End-to-end**: Text → Waveform directly (no separate vocoder needed)
- **High quality**: Produces near-SOTA results with proper training
- **Multi-speaker support**: Native speaker embedding support for male/female voices
- **Open-source ecosystem**: Well-documented, active community
- **Reasonable compute**: Trainable on 1-4 GPUs (vs. VALL-E needing 16+ GPUs)
- **Proven for Indian voices**: Several community projects have adapted VITS for Hindi/Indian English

### Architecture Comparison

| Model | Quality | Compute Cost | Training Difficulty | Indian Voice Support |
|-------|---------|-------------|--------------------|--------------------|
| **VITS2** | ★★★★☆ | ★★★☆☆ (moderate) | ★★★☆☆ (moderate) | ★★★★☆ (good) |
| StyleTTS2 | ★★★★★ | ★★★★☆ (high) | ★★★★☆ (hard) | ★★★☆☆ (limited) |
| VALL-E | ★★★★★ | ★★★★★ (very high) | ★★★★★ (very hard) | ★★☆☆☆ (poor) |
| Tortoise | ★★★★☆ | ★★★★☆ (high) | ★★★☆☆ (moderate) | ★★★☆☆ (limited) |
| Bark | ★★★☆☆ | ★★★☆☆ (moderate) | ★★☆☆☆ (easy) | ★★☆☆☆ (poor) |

## 3. Data Requirements

### Recommended Datasets

#### Indian English
| Dataset | Hours | Speakers | Quality | Access |
|---------|-------|----------|---------|--------|
| **IndicTTS** (IIT Madras) | 10+ hrs/language | Multiple M/F | Studio quality, 48kHz | Free (research) |
| **IITM Indian English** | ~15 hrs | 2 (M+F) | Clean, 22kHz+ | Free |
| **Common Voice** (Indian English) | 50+ hrs | Many speakers | Variable quality | Free (CC-0) |
| **OpenSLR Indian datasets** | Various | Various | Good | Free |
| **LibriTTS** (Indian-origin speakers) | Subset available | Variable | Studio quality | Free |

#### Hindi / Indian Languages (for accent/prosody transfer)
| Dataset | Hours | Notes |
|---------|-------|-------|
| **IndicVoices** | 7000+ hrs across 22 languages | Massive, diverse |
| **MUCS 2021** | Hindi read speech | Clean |
| **Kathbath** | Hindi conversational | Natural prosody |

### Data Quality Requirements
- **Sampling rate**: 22,050 Hz minimum (44,100 Hz preferred)
- **Bit depth**: 16-bit minimum
- **SNR**: > 30 dB (clean recordings)
- **Format**: WAV (uncompressed)
- **Transcription accuracy**: > 98%
- **Per-speaker consistency**: Same mic, room, session conditions

## 4. Compute Requirements

### Minimum Viable (MVP)
- **GPU**: 1× NVIDIA A100 40GB (or 1× RTX 4090 24GB)
- **RAM**: 64 GB
- **Storage**: 500 GB SSD
- **Training time**: ~3-5 days (200K steps)
- **Cloud cost**: ~$50-150 (spot instances)

### Production Quality
- **GPU**: 4× NVIDIA A100 80GB
- **RAM**: 128 GB
- **Storage**: 1 TB NVMe
- **Training time**: ~7-14 days (800K steps)
- **Cloud cost**: ~$500-2000

### SOTA-Matching
- **GPU**: 8× NVIDIA H100
- **RAM**: 256 GB
- **Training time**: ~2-4 weeks (1M+ steps)
- **Cloud cost**: ~$3000-10000

## 5. Key Technical Challenges

### Indian Voice Specific
1. **Phoneme inventory**: Indian English has unique phonemes (retroflex consonants ʈ, ɖ, ɳ) not in CMU/ARPAbet
2. **Prosody patterns**: Indian English has distinct intonation patterns influenced by native languages
3. **Code-switching**: Indian speakers frequently mix Hindi/regional words into English
4. **Vowel system**: Different vowel qualities compared to American/British English
5. **Rhythm**: Syllable-timed (vs. stress-timed in American English)

### General TTS Challenges
1. **Long utterances**: Attention collapse on sentences > 200 characters
2. **Robustness**: Avoiding mispronunciations, skipped words, repeats
3. **Speaker similarity**: Maintaining consistent voice identity
4. **Emotional range**: Natural expressiveness without monotone output

## 6. Our Approach

### Phase 1: Data Pipeline
- Download and preprocess IndicTTS + Common Voice datasets
- Build Indian English phoneme set (extended IPA)
- Segment, normalize, and validate audio
- Create train/val/test splits per speaker

### Phase 2: Model Training
- VITS2 architecture with multi-speaker conditioning
- Custom Indian English G2P (grapheme-to-phoneme) module
- Speaker embeddings for male/female voice control
- Progressive training: base → fine-tune → speaker adaptation

### Phase 3: Inference & Deployment
- Fast inference with GPU/CPU support
- Voice selection API (male/female)
- Streaming support for real-time applications
- ONNX export for edge deployment

## 7. What "As Good As Qwen" Means

Qwen-TTS (CosyVoice) achieves:
- **MOS**: ~4.0-4.2 (out of 5.0)
- **Speaker similarity**: > 0.85 cosine similarity
- **Real-time factor**: < 0.5 on A100

Our realistic targets:
| Metric | MVP | Production | SOTA |
|--------|-----|-----------|------|
| MOS | 3.2-3.5 | 3.7-4.0 | 4.0-4.2 |
| Speaker similarity | 0.70 | 0.80 | 0.85+ |
| Real-time factor | 1.0 | 0.5 | 0.3 |

## 8. Recommendation

**Start with the MVP tier** using VITS2 + IndicTTS data. This gives you:
- Working Indian male/female voices in 2-4 weeks
- A foundation to iterate and improve
- Clear benchmarks to measure progress toward SOTA

Then progressively scale data, compute, and model complexity to reach production quality.

---

*This analysis is based on the current state of TTS research as of early 2026.*
