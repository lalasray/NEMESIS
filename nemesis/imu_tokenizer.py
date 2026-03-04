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
from typing import Dict, List, Tuple, Optional

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
    Quantizes each IMU window into a single token by extracting rich
    statistical + spectral features and hashing them into a codebook index.

    Features per channel:
      - mean, std, min, max (basic stats)
      - energy (sum of squares)
      - zero-crossing rate
      - dominant frequency (FFT)
      - spectral energy ratio (low vs high freq)
      - inter-quartile range
      - signal magnitude area
    """

    def __init__(self, config: IMUTokenizerConfig = IMUTokenizerConfig()):
        self.config = config
        self.codebook_size = config.codebook_size
        self.num_channels = config.num_channels
        # 10 features per channel + cross-channel correlations
        num_cross = config.num_channels * (config.num_channels - 1) // 2
        self.feature_dim = 10 * config.num_channels + num_cross
        # Larger random projection for better discrimination
        rng = np.random.RandomState(42)
        # Use multiple independent hash functions for locality-sensitive hashing
        self.num_hash_bits = min(20, int(np.log2(config.codebook_size)) + 2)
        self.projection = rng.randn(self.feature_dim, self.num_hash_bits).astype(np.float32)

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract rich statistical + spectral features from a single window (W, C)."""
        C = window.shape[1]
        features = []

        for c in range(C):
            col = window[:, c]
            # Basic stats
            features.append(col.mean())
            features.append(col.std() + 1e-8)
            features.append(col.min())
            features.append(col.max())
            # Energy
            features.append(np.sum(col ** 2) / len(col))
            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(col - col.mean())))) / (2.0 * len(col))
            features.append(zcr)
            # Inter-quartile range
            q75, q25 = np.percentile(col, [75, 25])
            features.append(q75 - q25)
            # FFT-based features
            fft_vals = np.abs(np.fft.rfft(col))
            freqs = np.fft.rfftfreq(len(col), d=1.0 / self.config.sampling_rate)
            # Dominant frequency
            if len(fft_vals) > 1:
                dom_idx = np.argmax(fft_vals[1:]) + 1  # skip DC
                features.append(freqs[dom_idx])
            else:
                features.append(0.0)
            # Spectral energy ratio: low freq (<5Hz) vs high freq (>5Hz)
            low_mask = freqs <= 5.0
            high_mask = freqs > 5.0
            low_energy = np.sum(fft_vals[low_mask] ** 2) + 1e-8
            high_energy = np.sum(fft_vals[high_mask] ** 2) + 1e-8
            features.append(low_energy / (low_energy + high_energy))
            # Signal magnitude area contribution
            features.append(np.sum(np.abs(col)) / len(col))

        # Cross-channel correlations (captures multi-axis patterns)
        for i in range(C):
            for j in range(i + 1, C):
                corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
                features.append(corr if np.isfinite(corr) else 0.0)

        return np.array(features, dtype=np.float32)

    def _hash_to_token(self, features: np.ndarray) -> int:
        """Hash a feature vector to a codebook index using LSH."""
        # Normalize features to unit variance for better hashing
        norm = np.linalg.norm(features) + 1e-8
        features_norm = features / norm
        projected = features_norm @ self.projection  # (num_hash_bits,)
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
#
# Architecture inspired by github.com/lalasray/VQVAE:
#   - Conv1d encoder with stride-2 downsampling + residual blocks
#   - VectorQuantizerEMA with exponential moving average codebook updates
#   - Conv1d decoder with ConvTranspose1d upsampling + residual blocks
#   - Pre-VQ 1×1 convolution to project to embedding dim
#   - Channels-first format: (B, C, T) throughout
#   - Training normalised by data variance, iteration-based
# ============================================================================


class IMUResidual(nn.Module):
    """Single residual block for 1-D IMU signals."""

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels, num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(num_residual_hiddens, num_hiddens,
                      kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class IMUResidualStack(nn.Module):
    """Stack of residual blocks."""

    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self._layers = nn.ModuleList([
            IMUResidual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class IMUEncoder(nn.Module):
    """
    Conv1d encoder with stride-2 downsampling and residual stack.
    Input:  (B, C_in, T)
    Output: (B, num_hiddens, T//4)
    """

    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens // 2,
                                 kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv1d(num_hiddens // 2, num_hiddens,
                                 kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv1d(num_hiddens, num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._residual_stack = IMUResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        return self._residual_stack(x)


class IMUDecoder(nn.Module):
    """
    Conv1d decoder with ConvTranspose1d upsampling and residual stack.
    Input:  (B, embedding_dim, T//4)
    Output: (B, C_out, T)
    """

    def __init__(self, in_channels: int, num_hiddens: int,
                 num_residual_layers: int, num_residual_hiddens: int,
                 out_channels: int = 6):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels, num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._residual_stack = IMUResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose1d(
            num_hiddens, num_hiddens // 2,
            kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose1d(
            num_hiddens // 2, out_channels,
            kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization with Exponential Moving Average codebook updates.

    EMA is much more stable than gradient-based codebook learning:
    - Codebook vectors track the running mean of assigned encoder outputs
    - Laplace smoothing prevents dead codes
    - Only commitment loss is back-propagated through the encoder
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()

        # EMA state
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: (B, D, T') — encoder output in channels-first format

        Returns:
            loss: VQ commitment loss (scalar)
            quantized: (B, D, T') — quantized tensor (straight-through)
            perplexity: codebook usage metric (scalar)
            encoding_indices: (B, T') — codebook indices
        """
        # BCHW-style: (B, D, T') → (B, T', D)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape  # (B, T', D)

        # Flatten: (B*T', D)
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Distances: ||z - e||^2
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # EMA codebook update (only during training)
        if self.training:
            self._ema_cluster_size = (
                self._ema_cluster_size * self._decay
                + (1 - self._decay) * torch.sum(encodings, 0)
            )

            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Commitment loss (encoder → codebook)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity (codebook usage metric)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Back to channels-first: (B, T', D) → (B, D, T')
        quantized = quantized.permute(0, 2, 1).contiguous()

        # Reshape indices: (B*T', 1) → (B, T')
        encoding_indices = encoding_indices.squeeze(1).view(input_shape[0], input_shape[1])

        return loss, quantized, perplexity, encoding_indices


class VQVAE_Tokenizer(nn.Module):
    """
    Full VQ-VAE for IMU tokenization.

    Architecture (channels-first throughout):
        IMU (B, C, T) → Encoder → pre_vq_conv → VQ-EMA → Decoder → recon (B, C, T)

    MUST BE PRE-TRAINED before use:
      1. Train with train_vqvae() on the dataset's raw IMU data (unsupervised)
      2. This learns a codebook of motion primitives via reconstruction
      3. Then freeze and use tokenize() during RL training

    Hyperparameters (from reference):
      - num_hiddens=64, num_residual_hiddens=64, num_residual_layers=3
      - embedding_dim=128, num_embeddings=1028
      - commitment_cost=0.25, decay=0.99
      - input: channels-first (B, C, T)
    """

    def __init__(self, config: IMUTokenizerConfig = IMUTokenizerConfig()):
        super().__init__()
        self.config = config

        num_hiddens = config.vq_num_hiddens
        num_residual_hiddens = config.vq_num_residual_hiddens
        num_residual_layers = config.vq_num_residual_layers
        embedding_dim = config.vq_embedding_dim
        num_embeddings = config.codebook_size

        self._encoder = IMUEncoder(
            config.num_channels, num_hiddens,
            num_residual_layers, num_residual_hiddens)

        # Project encoder output to embedding dim before quantisation
        self._pre_vq_conv = nn.Conv1d(
            num_hiddens, embedding_dim, kernel_size=1, stride=1)

        self._vq_vae = VectorQuantizerEMA(
            num_embeddings, embedding_dim,
            config.vq_commitment_cost, config.vq_decay)

        self._decoder = IMUDecoder(
            embedding_dim, num_hiddens,
            num_residual_layers, num_residual_hiddens,
            out_channels=config.num_channels)

        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T) — channels-first raw IMU

        Returns:
            vq_loss: scalar VQ commitment loss
            x_recon: (B, C, T) reconstructed IMU
            perplexity: codebook usage metric
            encoding_indices: (B, T') codebook indices (T' = T//4)
        """
        z = self._encoder(x)              # (B, num_hiddens, T//4)
        z = self._pre_vq_conv(z)          # (B, embedding_dim, T//4)
        loss, quantized, perplexity, indices = self._vq_vae(z)
        x_recon = self._decoder(quantized) # (B, C, T)
        return loss, x_recon, perplexity, indices

    def train_vqvae(
        self,
        imu_data_list: List[np.ndarray],
        num_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
        input_length: Optional[int] = None,
        patience: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """
        Pre-train the VQ-VAE on raw IMU data (unsupervised reconstruction).

        Follows the reference training procedure:
          - Channels-first input (B, C, T)
          - Reconstruction loss normalised by data variance
          - Iteration-based training (cycles through dataloader)
          - Perplexity tracking for codebook health

        Args:
            imu_data_list: List of (T, C) raw IMU sequences
            num_epochs: Number of full passes over the data
            batch_size: Batch size
            lr: Learning rate
            device: 'cpu' or 'cuda'
            input_length: Crop/pad sequences to this length.
                          If None, uses config.window_size * 4 (to match encoder stride).
            verbose: Print progress

        Returns:
            Training metrics dict
        """
        import random

        self.to(device)
        self.train()

        # Determine input length (must be divisible by 4 for stride-2 × 2)
        if input_length is None:
            input_length = max(64, (self.config.window_size // 4) * 4)
            # Use median length for variable-length data, exact length for fixed
            sample_lengths = [d.shape[0] for d in imu_data_list[:100]]
            sample_len = int(np.median(sample_lengths)) if sample_lengths else 128
            if sample_len >= 64:
                input_length = (sample_len // 4) * 4  # round down to multiple of 4
            else:
                input_length = 64

        if verbose:
            print(f"[VQ-VAE] Preparing data: {len(imu_data_list)} sequences, "
                  f"input_length={input_length}")

        # Prepare data: (N, C, T) channels-first
        all_segments = []
        for imu_data in imu_data_list:
            T, C = imu_data.shape
            # Slice into non-overlapping segments of input_length
            for start in range(0, T - input_length + 1, input_length):
                segment = imu_data[start:start + input_length]  # (T, C)
                all_segments.append(segment.T)  # (C, T) channels-first

            # If sequence is exactly input_length, we already got it
            # Also handle shorter sequences by padding
            if T == input_length:
                continue
            elif T < input_length:
                padded = np.zeros((input_length, C), dtype=imu_data.dtype)
                padded[:T] = imu_data
                all_segments.append(padded.T)

        all_segments = np.array(all_segments, dtype=np.float32)  # (N, C, T)
        n_segments = len(all_segments)

        # Compute data variance for loss normalisation (reference approach)
        data_variance = np.var(all_segments)
        if data_variance < 1e-8:
            data_variance = 1.0

        if verbose:
            print(f"[VQ-VAE] Total segments: {n_segments}, "
                  f"shape: {all_segments.shape}, "
                  f"data_variance: {data_variance:.6f}")
            print(f"[VQ-VAE] Training for {num_epochs} epochs, "
                  f"batch_size={batch_size}, lr={lr}")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=False)

        train_recon_errors = []
        train_perplexities = []
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        metrics_log = []

        for epoch in range(num_epochs):
            # Shuffle
            perm = list(range(n_segments))
            random.shuffle(perm)

            epoch_recon = []
            epoch_perplexity = []

            for start in range(0, n_segments, batch_size):
                end = min(start + batch_size, n_segments)
                batch_idx = perm[start:end]
                if len(batch_idx) < 2:
                    continue

                data = torch.tensor(
                    all_segments[batch_idx], dtype=torch.float32, device=device
                )  # (B, C, T)

                optimizer.zero_grad()
                vq_loss, data_recon, perplexity, _ = self.forward(data)

                # Match output length to input (in case of rounding)
                if data_recon.shape[-1] != data.shape[-1]:
                    min_len = min(data_recon.shape[-1], data.shape[-1])
                    data_recon = data_recon[..., :min_len]
                    data = data[..., :min_len]

                recon_error = F.mse_loss(data_recon, data) / data_variance
                loss = recon_error + vq_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                train_recon_errors.append(recon_error.item())
                train_perplexities.append(perplexity.item())
                epoch_recon.append(recon_error.item())
                epoch_perplexity.append(perplexity.item())

            avg_recon = np.mean(epoch_recon) if epoch_recon else 0
            avg_perplexity = np.mean(epoch_perplexity) if epoch_perplexity else 0
            avg_loss = avg_recon + np.mean([e for e in epoch_recon]) if epoch_recon else 0

            metrics_log.append({
                "epoch": epoch + 1,
                "recon_error": float(avg_recon),
                "perplexity": float(avg_perplexity),
            })

            # Early stopping check
            if avg_recon < best_loss - 1e-5:
                best_loss = avg_recon
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % max(1, num_epochs // 20) == 0 or epoch == 0):
                print(f"  Epoch {epoch+1:3d}/{num_epochs} — "
                      f"recon_error={avg_recon:.5f}, "
                      f"perplexity={avg_perplexity:.1f}, "
                      f"best={best_loss:.5f}, "
                      f"patience={patience_counter}/{patience}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  [VQ-VAE] Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} epochs)")
                break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"[VQ-VAE] Restored best weights (recon_error={best_loss:.5f})")

        self._trained = True
        self.eval()

        # Final perplexity check
        final_perplexity = np.mean(train_perplexities[-50:]) if train_perplexities else 0

        if verbose:
            print(f"[VQ-VAE] Training complete! Best recon_error: {best_loss:.5f}")
            print(f"[VQ-VAE] Final perplexity: {final_perplexity:.1f} "
                  f"(ideal ≈ {self.config.codebook_size}, "
                  f"higher = more codebook usage)")

        return {
            "epochs": num_epochs,
            "best_recon_error": best_loss,
            "final_perplexity": final_perplexity,
            "log": metrics_log,
        }

    def save_pretrained(self, path: str):
        """Save the pre-trained VQ-VAE weights."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "trained": self._trained,
        }, path)
        print(f"[VQ-VAE] Saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "VQVAE_Tokenizer":
        """Load a pre-trained VQ-VAE."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model._trained = checkpoint.get("trained", True)
        model.to(device)
        model.eval()
        print(f"[VQ-VAE] Loaded from {path}")
        return model

    @torch.no_grad()
    def tokenize(self, data: np.ndarray) -> List[int]:
        """
        Tokenize raw IMU data (T, C) into discrete tokens.

        The encoder downsamples by 4× so a (128, 6) input yields 32 tokens.

        Returns:
            [BOS, tok_1, tok_2, ..., tok_N, EOS]
            where each tok_i = codebook_index + 4  (offset by special tokens)
        """
        self.eval()
        T, C = data.shape
        # Pad to multiple of 4 if needed
        pad_len = (4 - T % 4) % 4
        if pad_len > 0:
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')

        # (T, C) → (1, C, T) channels-first
        x = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            x = x.to(next(self.parameters()).device)

        _, _, _, indices = self.forward(x)  # indices: (1, T')
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
