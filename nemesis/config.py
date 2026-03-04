"""
NEMESIS Configuration — centralizes all hyperparameters and paths.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_PATH = os.path.join(PROJECT_ROOT, "apikey.txt")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory_store")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def load_api_key(filepath: str = API_KEY_PATH) -> str:
    with open(filepath, "r") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# IMU Tokenizer
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
    # Quantization bins (for simple binning fallback)
    num_bins: int = 256
    # Sampling rate of the IMU sensor (Hz)
    sampling_rate: int = 50


# ---------------------------------------------------------------------------
# Translator (Transformer Encoder-Decoder)
# ---------------------------------------------------------------------------
@dataclass
class TranslatorConfig:
    # Input vocabulary = codebook_size + special tokens
    src_vocab_size: int = 1032  # 1028 + 4 special
    # Output vocabulary = neuro-symbolic tokens
    tgt_vocab_size: int = 256
    # Model dimensions
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1024
    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


# ---------------------------------------------------------------------------
# Neuro-Symbolic Language
# ---------------------------------------------------------------------------
@dataclass
class NeuroSymbolicConfig:
    # Maximum number of symbolic statements per output
    max_statements: int = 20
    # Predicate types in the language
    predicates: List[str] = field(default_factory=lambda: [
        "MOTION", "POSTURE", "GAIT", "GESTURE",
        "ROTATION", "IMPACT", "STILLNESS", "TRANSITION", "CONTEXT"
    ])
    # Body parts / limbs
    limbs: List[str] = field(default_factory=lambda: [
        "left_arm", "right_arm", "left_leg", "right_leg",
        "torso", "head", "left_hand", "right_hand",
        "left_foot", "right_foot", "pelvis", "spine"
    ])
    # Motion types
    motion_types: List[str] = field(default_factory=lambda: [
        "swing", "lift", "lower", "extend", "flex",
        "rotate", "shake", "press", "hold", "release"
    ])
    # Intensity levels
    intensity_levels: List[str] = field(default_factory=lambda: [
        "low", "medium", "high", "explosive"
    ])
    # Gait patterns
    gait_patterns: List[str] = field(default_factory=lambda: [
        "stride", "shuffle", "stomp", "tiptoe", "limp", "jog", "sprint"
    ])
    # Posture states
    posture_states: List[str] = field(default_factory=lambda: [
        "upright", "leaning", "crouching", "sitting",
        "lying", "bending", "kneeling"
    ])


# ---------------------------------------------------------------------------
# Reinforcement Learning
# ---------------------------------------------------------------------------
@dataclass
class RLConfig:
    # Learning rate for policy gradient
    lr: float = 3e-4
    # Discount factor
    gamma: float = 0.99
    # PPO clip range
    clip_epsilon: float = 0.2
    # Entropy coefficient (encourages exploration)
    entropy_coef: float = 0.08
    # Entropy decay per epoch (multiply by this each epoch)
    entropy_decay: float = 0.95
    # Minimum entropy coefficient
    entropy_coef_min: float = 0.02
    # Value loss coefficient
    value_coef: float = 0.5
    # Max gradient norm
    max_grad_norm: float = 0.5
    # Batch size for RL updates
    batch_size: int = 16
    # Number of RL epochs per batch
    ppo_epochs: int = 4
    # Reward scaling
    reward_scale: float = 1.0
    # Baseline (moving average of rewards)
    baseline_momentum: float = 0.95
    # OpenAI model for reward computation
    reward_model: str = "gpt-4.1-mini"
    # Temperature for reward model
    reward_temperature: float = 0.3
    # LR warmup steps
    warmup_steps: int = 5
    # LR schedule: 'cosine' or 'constant'
    lr_schedule: str = "cosine"
    # Total expected PPO updates (for cosine schedule)
    total_updates: int = 100
    # Use classification-style reward (single API call, pick from options)
    classify_reward: bool = True
    # Reward for correct classification
    correct_reward: float = 1.0
    # Reward for partially correct (e.g. walking vs walking_upstairs)
    partial_reward: float = 0.3
    # Reward for wrong classification.
    # Set to None to auto-calibrate from class weights and number of classes
    # via reward_fn.calibrate_rewards(). Formula ensures that always predicting
    # any single class yields E[reward] <= 0 under class-balanced sampling.
    wrong_reward: float = -1.0
    # Running reward normalization
    normalize_rewards: bool = True


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    # ChromaDB persistent directory
    vector_db_path: str = os.path.join(MEMORY_DIR, "vector_db")
    # SQLite knowledge graph path
    knowledge_db_path: str = os.path.join(MEMORY_DIR, "knowledge.db")
    # Embedding dimension for vector memory
    embedding_dim: int = 256
    # Number of nearest neighbors to retrieve
    top_k: int = 5
    # Confidence threshold for knowledge graph hits
    confidence_threshold: float = 0.85
    # Maximum memory entries before pruning
    max_entries: int = 100_000
    # Collection name in ChromaDB
    collection_name: str = "nemesis_patterns"
