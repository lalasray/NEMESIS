"""
NEMESIS Pipeline — VQ-VAE token description + few-shot memory → LLM classification.

    IMU stream ──► VQ-VAE ──► Token Stats ──► Memory Query ──► LLM ──► Activity
                 (424K)     (frequencies,    (K nearest      (few-shot
                             entropy,         histograms)     grounded)
                             transitions)

The VQ-VAE learns a codebook of motion primitives (unsupervised).
At inference the K most similar past examples (by token histogram
cosine similarity) are retrieved from hierarchical memory and
injected as few-shot context, grounding the opaque token IDs.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nemesis.config import (
    IMUTokenizerConfig, ClassifierConfig, MemoryConfig, LearnerConfig,
    load_api_key, CHECKPOINTS_DIR,
)
from nemesis.imu_tokenizer import VQVAE_Tokenizer
from nemesis.token_descriptor import TokenDescriptor
from nemesis.classifier import OpenAIClassifier
from nemesis.memory import MemoryStore


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
        memory_config: MemoryConfig = MemoryConfig(),
        learner_config: LearnerConfig = LearnerConfig(),
        device: str = "cpu",
    ):
        self.device = device
        self.imu_config = imu_config
        self.classifier_config = classifier_config
        self.memory_config = memory_config
        self.learner_config = learner_config

        # --- Components ---
        self.tokenizer = VQVAE_Tokenizer(imu_config)
        self.descriptor = TokenDescriptor(codebook_size=imu_config.codebook_size)
        self.classifier = OpenAIClassifier(classifier_config)
        self.memory = MemoryStore(memory_config, learner_config)

        # Metadata for memory queries (set via set_sensor_context)
        self._dataset: str = ""
        self._imu_position: str = ""
        self._sampling_rate: int = 0
        self._num_channels: int = 0
        self._session_id: str = f"session_{int(__import__('time').time())}"

        print(f"[NEMESIS] Pipeline initialised (few-shot memory mode)")
        trained = self.tokenizer.is_trained
        status = "trained" if trained else "NEEDS PRE-TRAINING"
        print(f"  VQ-VAE Tokenizer: {status}")
        print(f"  Codebook: {imu_config.codebook_size} entries, {imu_config.vq_embedding_dim}-dim")
        print(f"  Memory: {self.memory.count()} entries")
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
        dataset: str = "",
        imu_position: str = "",
    ):
        """Set IMU sensor context included in LLM prompts + memory queries."""
        if channel_names is None:
            channel_names = [f"ch_{i}" for i in range(num_channels)]
        ctx = {
            "num_channels": num_channels,
            "channel_names": channel_names,
            "sampling_rate": sampling_rate,
            "window_duration_sec": window_duration_sec,
        }
        self.classifier.set_sensor_context(ctx)
        self._dataset = dataset
        self._imu_position = imu_position
        self._sampling_rate = sampling_rate
        self._num_channels = num_channels
        print(f"[NEMESIS] Sensor context: {num_channels}ch @ {sampling_rate}Hz, "
              f"{window_duration_sec:.2f}s windows")
        if dataset:
            print(f"  Dataset: {dataset}, IMU position: {imu_position}")

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

    def bootstrap_memory(
        self,
        tokens_list: List[List[int]],
        labels: List[str],
    ):
        """Populate long-term memory from labeled training data."""
        self.memory.bootstrap(
            tokens_list=tokens_list,
            labels=labels,
            dataset=self._dataset,
            imu_position=self._imu_position,
            sampling_rate=self._sampling_rate,
            num_channels=self._num_channels,
            session_id=self._session_id,
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_epoch(
        self,
        tokens_list: List[List[int]],
        descriptions: List[str],
        ground_truths: List[str],
        max_workers: int = 8,
        batch_size: int = 32,
    ) -> Dict:
        """
        One learning epoch: classify training samples, then update
        prototypes + effectiveness scores based on results.

        This is where actual parameter updates happen:
          - Prototype centroids shift via EMA
          - Effectiveness scores change for helpful/misleading entries

        No LLM fine-tuning or gradient backprop — but the retrieval
        system itself learns which examples to prefer.

        Args:
            tokens_list: Pre-tokenized training samples.
            descriptions: Pre-computed descriptor texts.
            ground_truths: Ground-truth activity labels (full descriptions).
            max_workers: Parallel LLM threads.
            batch_size: Samples per batch.

        Returns:
            Dict with epoch-level metrics (accuracy, corrections made).
        """
        from nemesis.memory import _token_histogram
        from concurrent.futures import ThreadPoolExecutor, as_completed

        N = len(tokens_list)
        correct = 0
        total = 0
        updates = 0

        # Process in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_tokens = tokens_list[start:end]
            batch_descs = descriptions[start:end]
            batch_gts = ground_truths[start:end]
            B = len(batch_tokens)

            # Query neighbours for each sample
            all_neighbours = []
            for tokens in batch_tokens:
                neighbours = self.memory.query(
                    tokens=tokens,
                    dataset=self._dataset,
                    imu_position=self._imu_position,
                    sampling_rate=self._sampling_rate,
                )
                all_neighbours.append(neighbours)

            # Parallel LLM classification
            activities = [None] * B

            def _classify_one(idx):
                desc = batch_descs[idx]
                neighbours = all_neighbours[idx]
                self.classifier.set_few_shot_examples(neighbours)
                cache_key = hash(("learn", desc, str(neighbours)))
                if cache_key in self.classifier._cache:
                    return idx, self.classifier._cache[cache_key]
                activity = self.classifier._call_llm(desc)
                self.classifier._cache[cache_key] = activity
                return idx, activity

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_classify_one, i) for i in range(B)]
                for future in as_completed(futures):
                    idx, activity = future.result()
                    activities[idx] = activity

            # Update learning parameters
            for i in range(B):
                gt = batch_gts[i]
                pred = activities[i]
                reward = self.classifier.compute_reward(pred, gt)
                is_correct = reward >= 0.8
                if is_correct:
                    correct += 1
                total += 1

                # Extract neighbour IDs and labels for learning update
                neighbours = all_neighbours[i]
                nb_ids = [n.get("entry_id", -1) for n in neighbours]
                nb_labels = [n["activity"] for n in neighbours]

                self.memory.update_learning(
                    sample_tokens=batch_tokens[i],
                    neighbour_ids=nb_ids,
                    neighbour_labels=nb_labels,
                    predicted_activity=pred,
                    ground_truth=gt,
                    is_correct=is_correct,
                )
                updates += 1

            if total % (batch_size * 4) < batch_size:
                acc = correct / max(total, 1)
                print(f"    [{total}/{N}] Learning acc: {acc:.1%}")

        # Save learned parameters
        self.memory.save_learning()
        # Invalidate index so next query uses updated scores
        self.memory._index_meta = None

        acc = correct / max(total, 1)
        stats = self.memory.learning_stats()
        print(f"  [Learn] Epoch done: {correct}/{total} = {acc:.1%}, "
              f"updates={updates}")
        if "effectiveness" in stats:
            e = stats["effectiveness"]
            print(f"  [Learn] Effectiveness: mean={e['mean']:.3f}, "
                  f"min={e['min']:.3f}, max={e['max']:.3f}")

        return {"accuracy": acc, "correct": correct, "total": total,
                "updates": updates, "learning_stats": stats}

    def classify_batch(
        self,
        imu_batch: List[np.ndarray],
        ground_truths: Optional[List[str]] = None,
        class_weights: Optional[List[float]] = None,
        max_workers: int = 8,
        store_inferences: bool = False,
    ) -> List[ClassificationResult]:
        """
        Classify a batch of IMU segments with few-shot memory retrieval.

        Steps:
          1. Tokenize all samples with VQ-VAE (serial, fast)
          2. Generate statistical text descriptions (serial, fast)
          3. Query memory for K nearest examples per sample
          4. Classify all with OpenAI in PARALLEL (few-shot grounded)
          5. Compute rewards and optionally store inferences

        Args:
            imu_batch: List of (T, C) raw IMU arrays
            ground_truths: Optional activity labels for scoring
            class_weights: Optional per-sample class weights
            max_workers: Parallel API threads
            store_inferences: If True, store high-confidence results in memory

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

        # Step 3: Query memory for few-shot neighbours per sample
        all_neighbours = []
        for tokens in all_tokens:
            neighbours = self.memory.query(
                tokens=tokens,
                dataset=self._dataset,
                imu_position=self._imu_position,
                sampling_rate=self._sampling_rate,
            )
            all_neighbours.append(neighbours)

        # Step 4: Parallel LLM classification with per-sample few-shot
        from concurrent.futures import ThreadPoolExecutor, as_completed

        activities = [None] * B

        def _classify_one(idx):
            desc = all_descriptions[idx]
            neighbours = all_neighbours[idx]
            # Set per-sample few-shot examples (thread-local via direct call)
            cache_key = hash(("fewshot", desc, str(neighbours)))
            if cache_key in self.classifier._cache:
                return idx, self.classifier._cache[cache_key]
            # Build per-sample prompt by temporarily injecting examples
            self.classifier.set_few_shot_examples(neighbours)
            activity = self.classifier._call_llm(desc)
            self.classifier._cache[cache_key] = activity
            return idx, activity

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_classify_one, i) for i in range(B)]
            for future in as_completed(futures):
                idx, activity = future.result()
                activities[idx] = activity

        # Step 5: Compute rewards + optionally store
        results = []
        for i in range(B):
            gt = ground_truths[i]
            cw = class_weights[i]
            activity = activities[i]

            if gt is not None:
                reward = self.classifier.compute_reward(activity, gt, class_weight=cw)
            else:
                reward = 0.0

            # Store high-confidence inferences in memory
            if store_inferences:
                self.memory.store_inference(
                    tokens=all_tokens[i],
                    predicted_activity=activity,
                    confidence=max(0.0, reward),
                    dataset=self._dataset,
                    imu_position=self._imu_position,
                    sampling_rate=self._sampling_rate,
                    num_channels=self._num_channels,
                    session_id=self._session_id,
                    neighbours=all_neighbours[i],
                )

            results.append(ClassificationResult(
                descriptor_text=all_descriptions[i],
                activity=activity,
                reward=reward,
                imu_tokens=all_tokens[i],
            ))

        return results
