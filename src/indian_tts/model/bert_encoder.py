"""
BERT feature extractor for Bert-VITS2.

Uses Microsoft's DeBERTa-v3-base (MIT license) to extract contextual
text embeddings that improve prosody, intonation, and natural energy
variation in the generated speech.

The BERT model is FROZEN during TTS training — only the projection
layer is trained. This keeps training fast and stable.

License: MIT (DeBERTa-v3-base from Microsoft)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class BertFeatureExtractor(nn.Module):
    """
    Extract contextual text features using DeBERTa-v3-base.

    The BERT model understands sentence structure, so greetings
    naturally get different features than factual statements,
    questions get different features than answers, etc.

    This enables natural prosody variation without explicit style tags.
    """

    def __init__(
        self,
        bert_model_name: str = "microsoft/deberta-v3-base",
        output_dim: int = 192,
        freeze_bert: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert

        # Load pre-trained DeBERTa (MIT license)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # 768 for base

        # Project BERT features to VITS2 hidden dimension
        self.proj = nn.Linear(self.bert_hidden_size, output_dim)

        # Freeze BERT weights (only train the projection)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Extract BERT features for a batch of text strings.

        Args:
            texts: List of raw text strings (B,)
            device: Target device

        Returns:
            BERT features (B, bert_hidden, T_bert) — projected to output_dim
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Extract BERT features
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(**encoded)
        else:
            outputs = self.bert(**encoded)

        # Last hidden state: (B, T_bert, 768)
        hidden = outputs.last_hidden_state

        # Project to VITS2 dimension: (B, T_bert, output_dim)
        projected = self.proj(hidden)

        # Transpose to channel-first: (B, output_dim, T_bert)
        return projected.transpose(1, 2)

    def get_features_aligned(
        self,
        texts: List[str],
        phoneme_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract BERT features and align them to phoneme sequence length.

        Since BERT uses wordpiece tokens and VITS2 uses phonemes,
        we interpolate BERT features to match the phoneme length.

        Args:
            texts: List of raw text strings (B,)
            phoneme_lengths: Length of phoneme sequences (B,)
            device: Target device

        Returns:
            Aligned BERT features (B, output_dim, T_phoneme)
        """
        # Get BERT features: (B, output_dim, T_bert)
        bert_features = self.forward(texts, device)

        # Interpolate to match phoneme length for each sample
        B = len(texts)
        max_phone_len = phoneme_lengths.max().item()
        aligned = torch.zeros(B, self.output_dim, max_phone_len, device=device)

        for i in range(B):
            phone_len = phoneme_lengths[i].item()
            # Interpolate: (1, output_dim, T_bert) -> (1, output_dim, phone_len)
            feat = bert_features[i : i + 1, :, :]
            feat_interp = torch.nn.functional.interpolate(
                feat, size=phone_len, mode="linear", align_corners=False
            )
            aligned[i, :, :phone_len] = feat_interp.squeeze(0)

        return aligned
