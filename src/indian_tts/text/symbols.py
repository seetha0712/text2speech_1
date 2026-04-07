"""
Phoneme symbol set for Indian English TTS.

Extends standard IPA with Indian-specific phonemes:
- Retroflex consonants (ʈ, ɖ, ɳ, ɽ)
- Aspirated stops (pʰ, tʰ, kʰ, bʱ, dʱ, gʱ)
- Indian vowels and nasalization
"""

# Special symbols
PAD = "_"
BOS = "^"  # Beginning of sentence
EOS = "$"  # End of sentence
SPACE = " "

# Standard IPA consonants (pulmonic)
PLOSIVES = list("pbtdkg")
NASALS = list("mnɲŋ")
FRICATIVES = list("fvθðszʃʒhɦ")
APPROXIMANTS = list("jw")
LIQUIDS = ["l", "ɹ", "r"]

# Indian-specific consonants
RETROFLEX = ["ʈ", "ɖ", "ɳ", "ɽ"]  # Retroflex stops, nasal, flap
ASPIRATED = ["pʰ", "tʰ", "ʈʰ", "kʰ", "bʱ", "dʱ", "ɖʱ", "gʱ", "tʃʰ", "dʒʱ"]
AFFRICATES = ["tʃ", "dʒ"]

# Vowels (including Indian English vowels)
MONOPHTHONGS = ["iː", "ɪ", "eː", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ə", "ɐ"]
DIPHTHONGS = ["aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ"]
NASALIZED = ["ã", "ĩ", "ũ", "ẽ", "õ"]  # Nasalized vowels (Hindi influence)

# Suprasegmentals
STRESS = ["ˈ", "ˌ"]  # Primary and secondary stress
LENGTH = ["ː"]
TONE = []  # Not typically needed for Indian English

# Punctuation (mapped to prosodic features)
PUNCTUATION = [".", ",", "?", "!", ";", ":", "-", "…"]

# Build the complete symbol set
_special = [PAD, BOS, EOS, SPACE]
_consonants = PLOSIVES + NASALS + FRICATIVES + APPROXIMANTS + LIQUIDS
_indian_consonants = RETROFLEX + ASPIRATED + AFFRICATES
_vowels = MONOPHTHONGS + DIPHTHONGS + NASALIZED
_prosody = STRESS + LENGTH
_punctuation = PUNCTUATION

symbols = (
    _special
    + _consonants
    + _indian_consonants
    + _vowels
    + _prosody
    + _punctuation
)

# Remove duplicates while preserving order
seen = set()
unique_symbols = []
for s in symbols:
    if s not in seen:
        seen.add(s)
        unique_symbols.append(s)
symbols = unique_symbols

# Mappings
SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(symbols)}

NUM_SYMBOLS = len(symbols)
