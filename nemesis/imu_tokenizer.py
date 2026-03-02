"""
IMU Tokenizer — converts raw IMU signals into discrete token sequences.

Two strategies:
  1. Simple Binning: quantize each channel value into bins (fast, no training)
  2. VQ-VAE: learn a codebook of motion primitives (better, requires training)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from nemesis.config import IMUTokenizerConfig


# ============================================================================
# Special Tokens
# ============================================================================
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


# ============================================================================
# Windowing — slice raw IMU stream into overlapping windows
# ============================================================================

def window_imu_stream(
    data: np.ndarray,
    window_size: int = 25,
    overlap: int = 5,
) -> np.ndarray:
    """
    Slice a (T, C) IMU array into overlapping windows of shape (N, W, C).

    Args:
        data: Raw IMU data, shape (timesteps, channels)
        window_size: Number of samples per window
        overlap: Number of overlapping samples between consecutive windows

    Returns:
        windows: shape (num_windows, window_size, channels)
    """
    step = window_size - overlap
    T, C = data.shape
    num_windows = max(1, (T - window_size) // step + 1)

    windows = np.zeros((num_windows, window_size, C), dtype=data.dtype)
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        segment = data[start:end]
        windows[i, :len(segment)] = segment

    return windows


# ============================================================================
# Strategy 1: Simple Binning Tokenizer
# ============================================================================

class BinningTokenizer:
    """
    Quantizes each IMU window into a single token by:
      1. Computing summary statistics (mean, std, min, max) per channel
      2. Hashing/binning the feature vector into a codebook index
    """

    def __init__(self, config: IMUTokenizerConfig = IMUTokenizerConfig()):
        self.config = config
        self.codebook_size = config.codebook_size
        self.num_channels = config.num_channels
        # Feature statistics: mean, std, min, max per channel → 4 * C features
        self.feature_dim = 4 * config.num_channels
        # Random projection for hashing (deterministic seed)
        rng = np.random.RandomState(42)
        self.projection = rng.randn(self.feature_dim, 16).astype(np.float32)

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract statistical features from a single window (W, C)."""
        return np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ])

    def _hash_to_token(self, features: np.ndarray) -> int:
        """Hash a feature vector to a codebook index."""
        projected = features @ self.projection  # (16,)
        # Binary hash → integer
        binary = (projected > 0).astype(np.int32)
        hash_val = sum(b << i for i, b in enumerate(binary))
        return (hash_val % self.codebook_size) + 4  # offset by special tokens

    def tokenize(self, data: np.ndarray) -> List[int]:
        """
        Tokenize raw IMU data (T, C) into a list of token IDs.

        Returns:
            tokens: [BOS, tok_1, tok_2, ..., tok_N, EOS]
        """
        windows = window_imu_stream(
            data,
            self.config.window_size,
            self.config.window_overlap,
        )
        tokens = [BOS_TOKEN]
        for window in windows:
            feat = self._extract_features(window)
            token = self._hash_to_token(feat)
            tokens.append(token)
        tokens.append(EOS_TOKEN)
        return tokens

    def batch_tokenize(self, batch: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of IMU sequences and return padded tensor + mask.

        Returns:
            token_ids: (B, max_len) LongTensor
            attention_mask: (B, max_len) BoolTensor (True = valid)
        """
        all_tokens = [self.tokenize(seq) for seq in batch]
        max_len = max(len(t) for t in all_tokens)

        token_ids = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, tokens in enumerate(all_tokens):
            token_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[i, :len(tokens)] = True

        return token_ids, mask


# ============================================================================
# Strategy 2: VQ-VAE Tokenizer (Trainable)
# ============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer — maps continuous embeddings to nearest
    codebook entry. Used inside the VQ-VAE to learn discrete IMU tokens.
    """

    def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        # Initialize codebook with uniform distribution
        self.codebook.weight.data.uniform_(
            -1.0 / codebook_size, 1.0 / codebook_size
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, N, D) continuous encoder output

        Returns:
            quantized: (B, N, D) quantized embeddings
            indices: (B, N) codebook indices
            vq_loss: scalar commitment + codebook loss
        """
        B, N, D = z.shape

        # Flatten for distance computation
        flat_z = z.reshape(-1, D)  # (B*N, D)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.codebook.weight.t()
        )

        # Nearest codebook entry
        indices = distances.argmin(dim=1)  # (B*N,)
        quantized = self.codebook(indices).reshape(B, N, D)

        # Losses
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        indices = indices.reshape(B, N)

        return quantized, indices, vq_loss


class IMUEncoder(nn.Module):
    """
    Encodes a single IMU window (W, C) into a continuous embedding (D,).
    Uses 1D convolution over the time axis.
    """

    def __init__(self, num_channels: int = 6, embedding_dim: int = 64, window_size: int = 25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, W, C) — batch of N windows, each W×C

        Returns:
            z: (B, N, D) — one embedding per window
        """
        B, N, W, C = x.shape
        x = x.reshape(B * N, W, C)     # (B*N, W, C)
        x = x.permute(0, 2, 1)          # (B*N, C, W) for Conv1d
        x = self.conv(x)                # (B*N, D, W)
        x = self.pool(x).squeeze(-1)    # (B*N, D)
        return x.reshape(B, N, -1)      # (B, N, D)


class IMUDecoder(nn.Module):
    """
    Reconstructs IMU window from quantized embedding (for VQ-VAE training).
    """

    def __init__(self, num_channels: int = 6, embedding_dim: int = 64, window_size: int = 25):
        super().__init__()
        self.window_size = window_size
        self.num_channels = num_channels

        self.fc = nn.Linear(embedding_dim, 64 * window_size)
        self.deconv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D)

        Returns:
            recon: (B, N, W, C)
        """
        B, N, D = z.shape
        x = self.fc(z.reshape(B * N, D))      # (B*N, 64*W)
        x = x.reshape(B * N, 64, self.window_size)  # (B*N, 64, W)
        x = self.deconv(x)                    # (B*N, C, W)
        x = x.permute(0, 2, 1)                # (B*N, W, C)
        return x.reshape(B, N, self.window_size, self.num_channels)


class VQVAE_Tokenizer(nn.Module):
    """
    Full VQ-VAE for IMU tokenization.

    Training: reconstruct IMU windows → learn meaningful codebook.
    Inference: encode → quantize → return codebook indices as tokens.
    """

    def __init__(self, config: IMUTokenizerConfig = IMUTokenizerConfig()):
        super().__init__()
        self.config = config
        self.encoder = IMUEncoder(config.num_channels, config.vq_embedding_dim, config.window_size)
        self.vq = VectorQuantizer(config.codebook_size, config.vq_embedding_dim)
        self.decoder = IMUDecoder(config.num_channels, config.vq_embedding_dim, config.window_size)

    def forward(self, windows: torch.Tensor):
        """
        Args:
            windows: (B, N, W, C) batched IMU windows

        Returns:
            recon: (B, N, W, C) reconstructed windows
            indices: (B, N) codebook token IDs
            vq_loss: scalar VQ loss
        """
        z = self.encoder(windows)                    # (B, N, D)
        quantized, indices, vq_loss = self.vq(z)     # (B, N, D), (B, N), scalar
        recon = self.decoder(quantized)              # (B, N, W, C)

        # Offset indices by special tokens
        indices = indices + 4  # PAD=0, BOS=1, EOS=2, UNK=3

        return recon, indices, vq_loss

    @torch.no_grad()
    def tokenize(self, data: np.ndarray) -> List[int]:
        """
        Tokenize raw IMU data (T, C) into discrete tokens.

        Returns:
            [BOS, tok_1, tok_2, ..., tok_N, EOS]
        """
        self.eval()
        windows = window_imu_stream(
            data, self.config.window_size, self.config.window_overlap
        )
        # (1, N, W, C)
        windows_t = torch.tensor(windows, dtype=torch.float32).unsqueeze(0)
        z = self.encoder(windows_t)
        _, indices, _ = self.vq(z)
        tokens = [BOS_TOKEN] + (indices[0] + 4).tolist() + [EOS_TOKEN]
        return tokens

    @torch.no_grad()
    def batch_tokenize(self, batch: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch and return padded tensors.

        Returns:
            token_ids: (B, max_len) LongTensor
            attention_mask: (B, max_len) BoolTensor
        """
        self.eval()
        all_tokens = [self.tokenize(seq) for seq in batch]
        max_len = max(len(t) for t in all_tokens)

        token_ids = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, tokens in enumerate(all_tokens):
            token_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[i, :len(tokens)] = True

        return token_ids, mask


# ============================================================================
# Utility: Generate synthetic IMU data for testing
# ============================================================================

def generate_synthetic_imu(
    activity: str = "walking",
    duration_sec: float = 5.0,
    sampling_rate: int = 50,
    num_channels: int = 6,
) -> np.ndarray:
    """
    Generate fake IMU data for testing. Returns (T, C) array.

    Activities: walking, running, sitting, jumping, waving
    """
    T = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, T)

    patterns = {
        "walking": {
            "freq": 2.0, "amp": np.array([1.0, 0.3, 9.8, 0.5, 0.2, 0.1]),
        },
        "running": {
            "freq": 3.5, "amp": np.array([2.5, 0.8, 9.8, 1.2, 0.6, 0.3]),
        },
        "sitting": {
            "freq": 0.1, "amp": np.array([0.05, 0.05, 9.8, 0.01, 0.01, 0.01]),
        },
        "jumping": {
            "freq": 1.5, "amp": np.array([1.5, 0.5, 12.0, 2.0, 0.8, 0.4]),
        },
        "waving": {
            "freq": 2.5, "amp": np.array([3.0, 2.0, 9.8, 4.0, 3.0, 0.5]),
        },
    }

    p = patterns.get(activity, patterns["walking"])
    signal = np.outer(np.sin(2 * math.pi * p["freq"] * t), p["amp"])
    noise = np.random.randn(T, num_channels) * 0.1
    return (signal + noise).astype(np.float32)
