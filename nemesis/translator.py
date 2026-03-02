"""
Translator Model — Transformer Encoder-Decoder that maps
IMU token sequences to neuro-symbolic token sequences.

This is the core policy network that gets trained with:
  1. Supervised pretraining (IMU → symbolic templates)
  2. RL fine-tuning (OpenAI reward signal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from nemesis.config import TranslatorConfig


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# Translator Model
# ============================================================================

class TranslatorModel(nn.Module):
    """
    Transformer Encoder-Decoder for IMU→Symbolic translation.

    Encoder: processes discrete IMU token IDs
    Decoder: autoregressively generates neuro-symbolic token IDs
    """

    def __init__(self, config: TranslatorConfig = TranslatorConfig()):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.src_embedding = nn.Embedding(config.src_vocab_size, config.d_model,
                                          padding_idx=config.pad_token_id)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model,
                                          padding_idx=config.pad_token_id)
        self.src_pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.tgt_pos_enc = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # --- Transformer ---
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        # --- Output projection ---
        self.output_proj = nn.Linear(config.d_model, config.tgt_vocab_size)

        # --- Value head (for RL) ---
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for embeddings and projections."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (teacher-forced).

        Args:
            src_ids: (B, S) source IMU token IDs
            tgt_ids: (B, T) target symbolic token IDs (shifted right)
            src_key_padding_mask: (B, S) True where padded
            tgt_key_padding_mask: (B, T) True where padded

        Returns:
            logits: (B, T, tgt_vocab_size) — token logits
            values: (B, T, 1) — value estimates (for RL)
        """
        # Embed source
        src_emb = self.src_pos_enc(self.src_embedding(src_ids))  # (B, S, D)
        tgt_emb = self.tgt_pos_enc(self.tgt_embedding(tgt_ids))  # (B, T, D)

        # Causal mask for decoder
        T = tgt_ids.size(1)
        tgt_causal_mask = self._generate_square_subsequent_mask(T, tgt_ids.device)

        # Invert padding masks for PyTorch Transformer (True = ignore)
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask  # True → padded
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = ~tgt_key_padding_mask

        # Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, D)

        logits = self.output_proj(output)   # (B, T, V)
        values = self.value_head(output)    # (B, T, 1)

        return logits, values

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation.

        Args:
            src_ids: (B, S) source token IDs
            src_mask: (B, S) attention mask (True = valid)
            max_len: maximum output length
            temperature: sampling temperature
            top_k: if > 0, top-k sampling
            sample: if False, greedy decoding

        Returns:
            generated_ids: (B, T) generated token IDs
            log_probs: (B, T) log probabilities of each token (for RL)
        """
        B = src_ids.size(0)
        device = src_ids.device

        # Encode source
        src_emb = self.src_pos_enc(self.src_embedding(src_ids))
        src_key_padding_mask = ~src_mask if src_mask is not None else None

        # Start with BOS
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        log_probs_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            tgt_emb = self.tgt_pos_enc(self.tgt_embedding(generated))
            T = generated.size(1)
            tgt_causal_mask = self._generate_square_subsequent_mask(T, device)

            output = self.transformer(
                src=src_emb,
                tgt=tgt_emb,
                tgt_mask=tgt_causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            # Get logits for the last position
            last_logits = self.output_proj(output[:, -1, :])  # (B, V)

            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = last_logits.topk(top_k, dim=-1)
                threshold = top_vals[:, -1].unsqueeze(-1)
                last_logits[last_logits < threshold] = float('-inf')

            # Sample or greedy
            probs = F.softmax(last_logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, 1)  # (B, 1)
            else:
                next_token = last_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Log probability
            log_prob = F.log_softmax(last_logits, dim=-1)
            token_log_prob = log_prob.gather(1, next_token)  # (B, 1)
            log_probs_list.append(token_log_prob)

            # Mask finished sequences
            next_token = next_token.masked_fill(finished.unsqueeze(1), 0)  # PAD
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        log_probs = torch.cat(log_probs_list, dim=1)  # (B, T)
        return generated, log_probs

    def get_log_probs_and_values(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and values for given (src, tgt) pairs.
        Used during RL training to compute policy gradient.

        Returns:
            log_probs: (B, T) log prob of each target token
            values: (B, T) value estimates
            entropy: (B, T) entropy at each position
        """
        logits, values = self.forward(
            src_ids, tgt_ids,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        values = values.squeeze(-1)  # (B, T)

        # Log probs and entropy
        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)
        probs_all = F.softmax(logits, dim=-1)

        # Gather log probs for the actual target tokens
        # Shift: logits at position t predict token at t+1
        # But we pass tgt_ids already shifted, so logits[t] → tgt_ids[t]
        log_probs = log_probs_all.gather(2, tgt_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
        entropy = -(probs_all * log_probs_all).sum(dim=-1)  # (B, T)

        return log_probs, values, entropy

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TranslatorModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)
