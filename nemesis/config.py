"""
NEMESIS Configuration — centralises all hyperparameters and paths.
"""

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_PATH = os.path.join(PROJECT_ROOT, "apikey.txt")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def load_api_key(filepath: str = API_KEY_PATH) -> str:
    with open(filepath, "r") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# IMU Tokenizer (VQ-VAE)
# ---------------------------------------------------------------------------
@dataclass
class IMUTokenizerConfig:
    # Number of IMU channels (e.g. accel_x/y/z + gyro_x/y/z = 6)
    num_channels: int = 6
    # Window size in samples for each token
    window_size: int = 25
    # Overlap between windows (samples)
    window_overlap: int = 5
    # VQ-VAE codebook size (number of discrete tokens / embeddings)
    codebook_size: int = 1028
    # VQ-VAE embedding dimension (post pre_vq_conv)
    vq_embedding_dim: int = 128
    # VQ-VAE encoder/decoder hidden channels
    vq_num_hiddens: int = 64
    # VQ-VAE residual block hidden channels
    vq_num_residual_hiddens: int = 64
    # VQ-VAE residual layers
    vq_num_residual_layers: int = 3
    # VQ-VAE commitment cost (β in the loss)
    vq_commitment_cost: float = 0.25
    # VQ-VAE EMA decay for codebook updates
    vq_decay: float = 0.99
    # Number of special tokens: PAD, BOS, EOS, UNK
    num_special_tokens: int = 4
    # Sampling rate of the IMU sensor (Hz)
    sampling_rate: int = 50


MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory_store")


# ---------------------------------------------------------------------------
# OpenAI Classifier
# ---------------------------------------------------------------------------
@dataclass
class ClassifierConfig:
    # OpenAI model for classification
    model: str = "gpt-4.1-mini"
    # Reward for correct classification
    correct_reward: float = 1.0
    # Reward for partially correct (e.g. walking vs walking_upstairs)
    partial_reward: float = 0.3
    # Reward for wrong classification (auto-calibrated before eval)
    wrong_reward: float = -1.0


# ---------------------------------------------------------------------------
# Hierarchical Memory
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    # SQLite database path
    db_path: str = os.path.join(MEMORY_DIR, "memory.db")
    # VQ-VAE codebook size (must match IMUTokenizerConfig)
    codebook_size: int = 1028
    # Number of few-shot neighbours to retrieve
    top_k: int = 5
    # Minimum confidence to store an inference in short-term memory
    store_threshold: float = 0.6
    # Minimum confidence + neighbour agreement to promote to long-term
    promote_threshold: float = 0.85


# ---------------------------------------------------------------------------
# Online Learning (Prototype Refinement + Prompt Tuning)
# ---------------------------------------------------------------------------
@dataclass
class LearnerConfig:
    # Prototype EMA learning rate (attract correct)
    prototype_lr: float = 0.05
    # Prototype repel rate (push away wrong)
    prototype_repel_lr: float = 0.02
    # Effectiveness boost rate for helpful neighbours
    effectiveness_boost_lr: float = 0.15
    # Effectiveness decay rate for misleading neighbours
    effectiveness_decay_lr: float = 0.10
    # Exponent on effectiveness in retrieval reranking
    effectiveness_alpha: float = 0.3
    # Weight on prototype similarity in retrieval reranking
    prototype_beta: float = 0.2
    # Number of learning epochs (passes over training data)
    learn_epochs: int = 100
