"""
Token Descriptor — converts VQ-VAE token sequences into rich textual
descriptions for direct LLM classification.

Instead of:  VQ-VAE tokens → Transformer → symbolic text → LLM classify
We do:       VQ-VAE tokens → Statistical text description → LLM classify

This removes the Translator model entirely. The VQ-VAE codebook tokens are
treated as a learned "motion vocabulary", and we describe each sequence using:
  - Token inventory (which tokens, how many, percentages)
  - Temporal structure (start/middle/end patterns, transitions)
  - Statistical features (entropy, diversity, burst detection)
  - N-gram patterns (common bigrams/trigrams)

The resulting text is fed directly to the LLM for zero-shot classification.
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# Special token IDs (same as imu_tokenizer.py)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}


def _strip_special(tokens: List[int]) -> List[int]:
    """Remove BOS, EOS, PAD, UNK from token sequence."""
    return [t for t in tokens if t not in SPECIAL_TOKENS]


def _token_name(token_id: int) -> str:
    """Human-readable name for a codebook token."""
    return f"imu_tok_{token_id}"


# =========================================================================
# Core statistics
# =========================================================================

def token_frequencies(tokens: List[int]) -> Dict[int, int]:
    """Count occurrences of each token."""
    return dict(Counter(tokens))


def token_percentages(tokens: List[int]) -> Dict[int, float]:
    """Percentage of total for each token."""
    n = len(tokens)
    if n == 0:
        return {}
    counts = Counter(tokens)
    return {t: round(c / n * 100, 1) for t, c in counts.most_common()}


def token_entropy(tokens: List[int]) -> float:
    """Shannon entropy of the token distribution (bits)."""
    n = len(tokens)
    if n == 0:
        return 0.0
    counts = Counter(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def token_diversity(tokens: List[int], codebook_size: int = 1028) -> float:
    """Fraction of codebook used (0-1)."""
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / codebook_size, 4)


# =========================================================================
# Temporal analysis
# =========================================================================

def split_thirds(tokens: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Split sequence into start / middle / end thirds."""
    n = len(tokens)
    t1 = n // 3
    t2 = 2 * n // 3
    return tokens[:t1], tokens[t1:t2], tokens[t2:]


def top_tokens_in_segment(tokens: List[int], top_k: int = 3) -> List[Tuple[str, float]]:
    """Return top-K tokens with their percentage in a segment."""
    if not tokens:
        return []
    counts = Counter(tokens)
    n = len(tokens)
    return [
        (_token_name(t), round(c / n * 100, 1))
        for t, c in counts.most_common(top_k)
    ]


def transition_counts(tokens: List[int]) -> Dict[Tuple[int, int], int]:
    """Count bigram transitions."""
    trans = Counter()
    for i in range(len(tokens) - 1):
        trans[(tokens[i], tokens[i + 1])] += 1
    return dict(trans)


def top_bigrams(tokens: List[int], top_k: int = 5) -> List[Tuple[str, int]]:
    """Top-K most common bigram transitions."""
    trans = transition_counts(tokens)
    sorted_trans = sorted(trans.items(), key=lambda x: -x[1])[:top_k]
    return [
        (f"{_token_name(a)}→{_token_name(b)}", c)
        for (a, b), c in sorted_trans
    ]


def top_trigrams(tokens: List[int], top_k: int = 3) -> List[Tuple[str, int]]:
    """Top-K most common trigram patterns."""
    if len(tokens) < 3:
        return []
    tri = Counter()
    for i in range(len(tokens) - 2):
        tri[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
    sorted_tri = sorted(tri.items(), key=lambda x: -x[1])[:top_k]
    return [
        (f"{_token_name(a)}→{_token_name(b)}→{_token_name(c)}", cnt)
        for (a, b, c), cnt in sorted_tri
    ]


def detect_bursts(tokens: List[int], min_run: int = 3) -> List[Tuple[str, int, int]]:
    """Find runs of the same token (bursts). Returns (token_name, start_pos, length)."""
    if not tokens:
        return []
    bursts = []
    current = tokens[0]
    start = 0
    length = 1
    for i in range(1, len(tokens)):
        if tokens[i] == current:
            length += 1
        else:
            if length >= min_run:
                bursts.append((_token_name(current), start, length))
            current = tokens[i]
            start = i
            length = 1
    if length >= min_run:
        bursts.append((_token_name(current), start, length))
    return bursts


def self_repetition_rate(tokens: List[int]) -> float:
    """Fraction of consecutive pairs that are the same token (0.0 = no repeats)."""
    if len(tokens) < 2:
        return 0.0
    same = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
    return round(same / (len(tokens) - 1), 3)


# =========================================================================
# Main descriptor
# =========================================================================

class TokenDescriptor:
    """
    Converts a VQ-VAE token sequence into a rich text description
    suitable for direct LLM classification.

    Args:
        codebook_size: Number of VQ-VAE codebook entries.
        top_k_tokens: Number of top tokens to list in the description.
        top_k_bigrams: Number of top bigrams to include.
        include_temporal: Whether to include start/middle/end breakdown.
        include_trigrams: Whether to include trigram patterns.
        include_bursts: Whether to include burst detection.
    """

    def __init__(
        self,
        codebook_size: int = 1028,
        top_k_tokens: int = 10,
        top_k_bigrams: int = 5,
        top_k_trigrams: int = 3,
        include_temporal: bool = True,
        include_trigrams: bool = True,
        include_bursts: bool = True,
    ):
        self.codebook_size = codebook_size
        self.top_k_tokens = top_k_tokens
        self.top_k_bigrams = top_k_bigrams
        self.top_k_trigrams = top_k_trigrams
        self.include_temporal = include_temporal
        self.include_trigrams = include_trigrams
        self.include_bursts = include_bursts

    def describe(self, tokens: List[int]) -> str:
        """
        Generate a comprehensive textual description of a token sequence.

        Args:
            tokens: VQ-VAE token IDs (may include BOS/EOS).

        Returns:
            Multi-line text description for LLM consumption.
        """
        clean = _strip_special(tokens)
        if not clean:
            return "EMPTY SEQUENCE (no tokens)"

        lines = []

        # --- Overview ---
        n_unique = len(set(clean))
        entropy = token_entropy(clean)
        diversity = token_diversity(clean, self.codebook_size)
        rep_rate = self_repetition_rate(clean)

        lines.append("TOKEN SEQUENCE OVERVIEW:")
        lines.append(f"  Total tokens: {len(clean)}")
        lines.append(f"  Unique tokens: {n_unique} out of {self.codebook_size} codebook entries")
        lines.append(f"  Codebook usage: {diversity:.1%}")
        lines.append(f"  Shannon entropy: {entropy:.2f} bits")
        lines.append(f"  Self-repetition rate: {rep_rate:.1%} (consecutive same-token pairs)")

        # --- Top tokens ---
        pcts = token_percentages(clean)
        top_items = list(pcts.items())[:self.top_k_tokens]
        lines.append(f"\nTOP {len(top_items)} TOKENS (by frequency):")
        for tok_id, pct in top_items:
            count = Counter(clean)[tok_id]
            lines.append(f"  {_token_name(tok_id):15s}: {count:4d} occurrences ({pct:5.1f}%)")

        # --- Temporal structure ---
        if self.include_temporal and len(clean) >= 6:
            start, middle, end = split_thirds(clean)
            lines.append("\nTEMPORAL STRUCTURE (start / middle / end):")
            for label, seg in [("Start ", start), ("Middle", middle), ("End   ", end)]:
                top = top_tokens_in_segment(seg, top_k=3)
                seg_entropy = token_entropy(seg)
                top_str = ", ".join(f"{name}={pct:.0f}%" for name, pct in top)
                lines.append(f"  {label} ({len(seg)} tokens, entropy={seg_entropy:.2f}): {top_str}")

        # --- Transition patterns ---
        bigrams = top_bigrams(clean, top_k=self.top_k_bigrams)
        if bigrams:
            lines.append(f"\nTOP TRANSITIONS (bigrams):")
            for pattern, count in bigrams:
                lines.append(f"  {pattern}: {count} times")

        # --- Trigram patterns ---
        if self.include_trigrams:
            trigrams = top_trigrams(clean, top_k=self.top_k_trigrams)
            if trigrams:
                lines.append(f"\nTOP TRIGRAM PATTERNS:")
                for pattern, count in trigrams:
                    lines.append(f"  {pattern}: {count} times")

        # --- Bursts ---
        if self.include_bursts:
            bursts = detect_bursts(clean, min_run=3)
            if bursts:
                lines.append(f"\nREPEATED TOKEN BURSTS (same token ≥3 in a row):")
                # Show top 5 longest bursts
                bursts_sorted = sorted(bursts, key=lambda x: -x[2])[:5]
                for name, pos, length in bursts_sorted:
                    fraction = pos / len(clean)
                    region = "start" if fraction < 0.33 else "middle" if fraction < 0.67 else "end"
                    lines.append(f"  {name} × {length} at position {pos} ({region} of sequence)")

        # --- Sequence fingerprint (first/last 5) ---
        if len(clean) >= 10:
            first5 = " ".join(_token_name(t) for t in clean[:5])
            last5 = " ".join(_token_name(t) for t in clean[-5:])
            lines.append(f"\nSEQUENCE EDGES:")
            lines.append(f"  First 5: {first5}")
            lines.append(f"  Last 5:  {last5}")

        return "\n".join(lines)

    def describe_batch(self, token_batch: List[List[int]]) -> List[str]:
        """Describe a batch of token sequences."""
        return [self.describe(tokens) for tokens in token_batch]
