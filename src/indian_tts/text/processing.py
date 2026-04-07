"""
Text processing pipeline for Indian English TTS.

Handles:
1. Text normalization (numbers, abbreviations, etc.)
2. Grapheme-to-Phoneme (G2P) conversion with Indian English rules
3. Phoneme-to-ID sequence conversion
"""

import re
from typing import List, Optional

from num2words import num2words

from indian_tts.text.symbols import SYMBOL_TO_ID, BOS, EOS, SPACE


# --- Text Normalization ---

# Common Indian English abbreviations
ABBREVIATIONS = {
    "mr.": "mister",
    "mrs.": "missus",
    "dr.": "doctor",
    "sr.": "senior",
    "jr.": "junior",
    "st.": "saint",
    "govt.": "government",
    "pvt.": "private",
    "ltd.": "limited",
    "rs.": "rupees",
    "rs": "rupees",
    "crore": "crore",
    "lakh": "lakh",
    "km": "kilometers",
    "kg": "kilograms",
}

# Indian number system markers
INDIAN_NUMBERS = {
    "lakh": 100000,
    "lakhs": 100000,
    "crore": 10000000,
    "crores": 10000000,
}


def normalize_numbers(text: str) -> str:
    """Convert numbers to words using Indian English conventions."""

    def _expand_number(match):
        num = match.group(0)
        try:
            # Handle decimal numbers
            if "." in num:
                return num2words(float(num), lang="en_IN")
            else:
                return num2words(int(num), lang="en_IN")
        except (ValueError, OverflowError):
            return num

    # Handle currency (₹ or Rs.)
    text = re.sub(r"₹\s*(\d+)", r"\1 rupees", text)

    # Handle percentages
    text = re.sub(r"(\d+)%", r"\1 percent", text)

    # Expand remaining numbers
    text = re.sub(r"\d+\.?\d*", _expand_number, text)

    return text


def normalize_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    text_lower = text.lower()
    for abbr, expansion in ABBREVIATIONS.items():
        text_lower = text_lower.replace(abbr, expansion)
    return text_lower


def clean_text(text: str) -> str:
    """Full text normalization pipeline."""
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize abbreviations
    text = normalize_abbreviations(text)
    # Normalize numbers
    text = normalize_numbers(text)
    # Remove unsupported characters (keep basic punctuation and letters)
    text = re.sub(r"[^\w\s.,!?;:\-']", "", text)
    # Collapse whitespace again
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- Grapheme-to-Phoneme ---

class IndianEnglishG2P:
    """
    Grapheme-to-Phoneme converter for Indian English.

    Uses phonemizer (eSpeak-ng backend) with Indian English rules,
    falling back to rule-based conversion for common patterns.
    """

    # Indian English pronunciation rules (common patterns)
    RULES = {
        # Retroflex 't' and 'd' are common in Indian English
        "th": "t̪",   # Dental rather than fricative for many speakers
        # 'v' and 'w' often merge in Indian English
        # Rhotic - Indian English is generally rhotic
    }

    def __init__(self, use_espeak: bool = True):
        self.use_espeak = use_espeak
        self._phonemizer = None

    @property
    def phonemizer(self):
        if self._phonemizer is None and self.use_espeak:
            try:
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                self._backend = EspeakBackend(
                    language="en-in",  # Indian English
                    preserve_punctuation=True,
                    with_stress=True,
                )
                self._separator = Separator(
                    phone=" ",
                    word=" _ ",
                    syllable="",
                )
                self._phonemizer = True
            except Exception:
                self._phonemizer = False
        return self._phonemizer

    def __call__(self, text: str) -> str:
        """Convert text to phoneme string."""
        text = clean_text(text)

        if self.phonemizer and self._phonemizer is True:
            try:
                from phonemizer import phonemize
                phonemes = phonemize(
                    text,
                    language="en-in",
                    backend="espeak",
                    strip=True,
                    preserve_punctuation=True,
                    with_stress=True,
                )
                return phonemes
            except Exception:
                pass

        # Fallback: basic rule-based (for environments without eSpeak)
        return self._rule_based_g2p(text)

    def _rule_based_g2p(self, text: str) -> str:
        """Simple rule-based fallback G2P."""
        # This is a simplified fallback - eSpeak is strongly preferred
        phonemes = []
        for char in text.lower():
            if char in SYMBOL_TO_ID:
                phonemes.append(char)
            elif char.isalpha():
                # Pass through letters as-is (basic fallback)
                phonemes.append(char)
            elif char == " ":
                phonemes.append(SPACE)
        return "".join(phonemes)


# Singleton G2P instance
_g2p = IndianEnglishG2P()


def text_to_phonemes(text: str) -> str:
    """Convert text to phoneme string."""
    return _g2p(text)


def text_to_sequence(text: str) -> List[int]:
    """Convert text string to sequence of phoneme IDs."""
    phoneme_str = text_to_phonemes(text)
    sequence = [SYMBOL_TO_ID[BOS]]
    for p in phoneme_str:
        if p in SYMBOL_TO_ID:
            sequence.append(SYMBOL_TO_ID[p])
    sequence.append(SYMBOL_TO_ID[EOS])
    return sequence


def cleaned_text_to_sequence(cleaned_text: str) -> List[int]:
    """Convert already-phonemized text to sequence of IDs."""
    sequence = [SYMBOL_TO_ID[BOS]]
    for p in cleaned_text:
        if p in SYMBOL_TO_ID:
            sequence.append(SYMBOL_TO_ID[p])
    sequence.append(SYMBOL_TO_ID[EOS])
    return sequence
