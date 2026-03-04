"""
NEMESIS Pipeline — VQ-VAE token description → LLM activity classification.

    IMU stream ──► VQ-VAE Tokenizer ──► Token Statistics ──► LLM ──► Activity
                    (424K params)        (frequencies,
                                          entropy,
                                          transitions,
                                          bursts, ...)

The VQ-VAE learns a codebook of motion primitives (unsupervised).
Token sequences are described statistically and classified by an LLM
in a zero-shot manner — no Transformer or RL training needed.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nemesis.config import (
    IMUTokenizerConfig, ClassifierConfig, load_api_key, CHECKPOINTS_DIR,
)
from nemesis.imu_tokenizer import VQVAE_Tokenizer
from nemesis.token_descriptor import TokenDescriptor
from nemesis.classifier import OpenAIClassifier


# ============================================================================
# Classification Result
# ============================================================================

@dataclass
class ClassificationResult:
    """Result of one VQ-VAE → descriptor → LLM classification."""
    descriptor_text: str
    activity: str
    reward: float
    imu_tokens: List[int]


# ============================================================================
# NEMESIS Pipeline
# ============================================================================

class NemesisPipeline:
    """
    Full NEMESIS pipeline: raw IMU → VQ-VAE tokens → statistical
    description → LLM classification.
    """

    def __init__(
        self,
        imu_config: IMUTokenizerConfig = IMUTokenizerConfig(),
        classifier_config: ClassifierConfig = ClassifierConfig(),
        device: str = "cpu",
    ):
        self.device = device
        self.imu_config = imu_config
        self.classifier_config = classifier_config

        # --- Components ---
        self.tokenizer = VQVAE_Tokenizer(imu_config)
        self.descriptor = TokenDescriptor(codebook_size=imu_config.codebook_size)
        self.classifier = OpenAIClassifier(classifier_config)

        print(f"[NEMESIS] Pipeline initialised (descriptor mode)")
        trained = self.tokenizer.is_trained
        status = "trained" if trained else "NEEDS PRE-TRAINING"
        print(f"  VQ-VAE Tokenizer: {status}")
        print(f"  Codebook: {imu_config.codebook_size} entries, {imu_config.vq_embedding_dim}-dim")
        print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_activity_options(self, options: List[str]):
        """Set the valid activity labels for LLM classification."""
        self.classifier.set_activity_options(options)
        print(f"[NEMESIS] Activity options set: {len(options)} classes")

    def set_sensor_context(
        self,
        num_channels: int = 6,
        channel_names: List[str] = None,
        sampling_rate: int = 50,
        window_duration_sec: float = 2.56,
    ):
        """Set IMU sensor context included in LLM prompts."""
        if channel_names is None:
            channel_names = [f"ch_{i}" for i in range(num_channels)]
        ctx = {
            "num_channels": num_channels,
            "channel_names": channel_names,
            "sampling_rate": sampling_rate,
            "window_duration_sec": window_duration_sec,
        }
        self.classifier.set_sensor_context(ctx)
        print(f"[NEMESIS] Sensor context: {num_channels}ch @ {sampling_rate}Hz, "
              f"{window_duration_sec:.2f}s windows")

    # ------------------------------------------------------------------
    # VQ-VAE pre-training
    # ------------------------------------------------------------------

    def pretrain_tokenizer(
        self,
        imu_data_list: List[np.ndarray],
        num_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
    ):
        """
        Pre-train the VQ-VAE tokenizer on raw IMU data (unsupervised).

        Learns a codebook of motion primitives by reconstructing IMU signals
        — no labels needed.

        Args:
            imu_data_list: List of (T, C) raw IMU sequences
            num_epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Early stopping patience
        """
        print("\n--- VQ-VAE Pre-training (unsupervised) ---")
        print(f"  Learning motion primitives from {len(imu_data_list)} IMU sequences")

        metrics = self.tokenizer.train_vqvae(
            imu_data_list=imu_data_list,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            device=self.device,
            patience=patience,
        )

        vqvae_path = os.path.join(CHECKPOINTS_DIR, "vqvae_pretrained.pt")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        self.tokenizer.save_pretrained(vqvae_path)

        return metrics

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        imu_batch: List[np.ndarray],
        ground_truths: Optional[List[str]] = None,
        class_weights: Optional[List[float]] = None,
        max_workers: int = 8,
    ) -> List[ClassificationResult]:
        """
        Classify a batch of IMU segments.

        Steps:
          1. Tokenize all samples with VQ-VAE (serial, fast)
          2. Generate statistical text descriptions (serial, fast)
          3. Classify all with OpenAI in PARALLEL (ThreadPoolExecutor)
          4. Compute rewards (for evaluation metrics)

        Args:
            imu_batch: List of (T, C) raw IMU arrays
            ground_truths: Optional activity labels for scoring
            class_weights: Optional per-sample class weights
            max_workers: Parallel API threads

        Returns:
            List of ClassificationResult
        """
        B = len(imu_batch)
        if ground_truths is None:
            ground_truths = [None] * B
        if class_weights is None:
            class_weights = [1.0] * B

        # Step 1: Tokenize
        all_tokens = []
        for imu_data in imu_batch:
            tokens = self.tokenizer.tokenize(imu_data)
            all_tokens.append(tokens)

        # Step 2: Describe
        all_descriptions = self.descriptor.describe_batch(all_tokens)

        # Step 3: Parallel LLM classification
        activities = self.classifier.classify_batch(
            all_descriptions, max_workers=max_workers
        )

        # Step 4: Compute rewards
        results = []
        for i in range(B):
            gt = ground_truths[i]
            cw = class_weights[i]
            activity = activities[i]

            if gt is not None:
                reward = self.classifier.compute_reward(activity, gt, class_weight=cw)
            else:
                reward = 0.0

            results.append(ClassificationResult(
                descriptor_text=all_descriptions[i],
                activity=activity,
                reward=reward,
                imu_tokens=all_tokens[i],
            ))

        return results
