"""
Monotonic Alignment Search (MAS) for VITS2.

Finds the optimal monotonic alignment between text and audio
using dynamic programming. This is used to determine phoneme
durations during training.
"""

import torch
import numpy as np
from typing import Tuple


@torch.no_grad()
def maximum_path(neg_log_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Find the maximum probability monotonic alignment path.

    Uses Viterbi-like dynamic programming to find the best
    alignment between text encoder output and posterior encoder output.

    Args:
        neg_log_prob: Negative log probability matrix (B, T_text, T_spec)
        mask: Alignment mask (B, T_text, T_spec)

    Returns:
        path: Binary alignment matrix (B, T_text, T_spec)
    """
    device = neg_log_prob.device
    dtype = neg_log_prob.dtype

    neg_log_prob = neg_log_prob.cpu().numpy()
    mask = mask.cpu().numpy().astype(bool)

    B, T_text, T_spec = neg_log_prob.shape
    path = np.zeros_like(neg_log_prob)

    for b in range(B):
        path[b] = _compute_path(neg_log_prob[b], mask[b])

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def _compute_path(neg_log_prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute optimal alignment path for a single sample."""
    T_text, T_spec = neg_log_prob.shape
    path = np.zeros((T_text, T_spec), dtype=np.float32)

    # Find actual lengths from mask
    t_text_len = mask.any(axis=1).sum()
    t_spec_len = mask.any(axis=0).sum()

    if t_text_len == 0 or t_spec_len == 0:
        return path

    # DP table
    Q = np.full((t_text_len, t_spec_len), -np.inf, dtype=np.float64)

    # Initialize first row
    Q[0, 0] = neg_log_prob[0, 0]
    for j in range(1, t_spec_len):
        Q[0, j] = Q[0, j - 1] + neg_log_prob[0, j]

    # Fill DP table
    for i in range(1, t_text_len):
        for j in range(i, t_spec_len):
            # Can only come from same text position or previous text position
            Q[i, j] = neg_log_prob[i, j] + max(Q[i - 1, j - 1], Q[i, j - 1])

    # Backtrace
    i = t_text_len - 1
    j = t_spec_len - 1
    path[i, j] = 1.0

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif Q[i - 1, j - 1] >= Q[i, j - 1]:
            i -= 1
            j -= 1
        else:
            j -= 1
        path[i, j] = 1.0

    return path
