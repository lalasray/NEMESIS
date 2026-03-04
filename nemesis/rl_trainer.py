"""
RL Trainer — trains the Translator model using reinforcement learning
with OpenAI API as the reward function.

Key insight: we have NO ground truth for the neuro-symbolic output.
The symbolic representation is a FREE latent space that the Translator
learns to use in whatever way maximizes downstream activity recognition.

Training loop:
  1. Translator generates neuro-symbolic output from IMU tokens
  2. Output is sent to OpenAI which CLASSIFIES it into one of the known activities
  3. Classification result compared to ground truth → binary reward
  4. Policy gradient (PPO) updates Translator to produce symbolic outputs
     that lead OpenAI to the correct activity classification

v2 improvements:
  - Classification-style reward: 1 API call instead of 2 (less noise, faster)
  - Running reward normalization for stable gradients
  - LR warmup + cosine decay
  - Entropy coefficient annealing (high→low for exploration→exploitation)
  - Partial credit for related activities
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from openai import OpenAI

from nemesis.config import RLConfig, load_api_key, CHECKPOINTS_DIR
from nemesis.translator import TranslatorModel
from nemesis.neuro_symbolic import SymbolicVocab


# ============================================================================
# Activity Similarity Map (for partial credit)
# ============================================================================

# Groups of related activities — members get partial credit against each other
ACTIVITY_GROUPS = {
    "locomotion": {"walking", "walking_upstairs", "walking_downstairs",
                   "walking forward on a flat surface at a normal steady pace",
                   "walking up a flight of stairs at a steady pace",
                   "walking down a flight of stairs at a steady pace",
                   "walk",
                   "the person is walking around the room at a natural pace"},
    "stationary": {"sitting", "standing", "laying", "stand", "sit", "lie",
                   "sitting still on a chair in a relaxed position",
                   "standing upright in place without moving",
                   "lying down flat on a surface in a resting position",
                   "the person is standing upright without locomotion",
                   "the person is sitting still on a chair",
                   "the person is lying down on a deckchair"},
}


def _activities_related(a: str, b: str) -> bool:
    """Check if two activities belong to the same group."""
    a_lower, b_lower = a.lower().strip(), b.lower().strip()
    for group in ACTIVITY_GROUPS.values():
        a_in = any(a_lower in m or m in a_lower for m in group)
        b_in = any(b_lower in m or m in b_lower for m in group)
        if a_in and b_in:
            return True
    return False


# ============================================================================
# Reward Function via OpenAI (Classification-style)
# ============================================================================

class OpenAIRewardFunction:
    """
    Classification-style reward computation (v2):

    Instead of 2 API calls (interpret + match), we do 1 call:
      - Present the symbolic text + list of possible activities
      - OpenAI picks the most likely activity
      - Binary reward: correct=1.0, related=0.3, wrong=-0.1

    This gives:
      - 50% fewer API calls (faster + cheaper)
      - Less reward noise (no semantic similarity scoring variance)
      - Cleaner gradient signal for PPO
    """

    def __init__(self, config: RLConfig = RLConfig(), activity_options: List[str] = None):
        self.config = config
        self.client = OpenAI(api_key=load_api_key())
        self.activity_options = activity_options or []
        # Sensor context for richer prompts
        self.sensor_context: Optional[Dict] = None
        # Cache: symbolic_text_hash → (reward, predicted_activity)
        self._cache: Dict[str, Tuple[float, str]] = {}

    def set_activity_options(self, options: List[str]):
        """Set the valid activity options for classification."""
        self.activity_options = list(set(options))

    def set_sensor_context(self, ctx: Dict):
        """Set IMU sensor context for richer prompts."""
        self.sensor_context = ctx

    def compute_reward(
        self,
        symbolic_text: str,
        ground_truth: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Score a neuro-symbolic output by classifying it into one of the
        known activities, then comparing to ground truth.

        Returns:
            reward: float
            activity: OpenAI's predicted activity
        """
        # Check cache first
        cache_key = hash(symbolic_text)
        if cache_key in self._cache and ground_truth is not None:
            cached_activity = self._cache[cache_key][1]
            reward = self._compute_classification_reward(cached_activity, ground_truth)
            return reward, cached_activity

        # Single API call: classify the symbolic text
        if self.config.classify_reward and self.activity_options:
            activity = self._classify_symbolic(symbolic_text)
        else:
            activity = self._interpret_symbolic(symbolic_text)

        # Cache the prediction
        self._cache[cache_key] = (0.0, activity)

        # Compute reward
        if ground_truth is not None:
            if self.config.classify_reward:
                reward = self._compute_classification_reward(activity, ground_truth)
            else:
                reward = self._compute_match_reward(activity, ground_truth)
        else:
            reward = 0.1 if self._is_parseable(symbolic_text) else 0.0

        return reward, activity

    def _classify_symbolic(self, symbolic_text: str) -> str:
        """
        Single-step classification with rich sensor context.

        The prompt now includes:
          - What kind of sensor data this came from (IMU, channels, rate)
          - Duration of the recording window
          - The neuro-symbolic representation
          - The fixed list of possible activities to choose from

        This gives OpenAI maximum context to make a good classification.
        """
        # Build sensor context section
        if self.sensor_context:
            ctx = self.sensor_context
            channels_str = ", ".join(ctx.get("channel_names", []))
            sensor_section = (
                f"SENSOR SETUP:\n"
                f"  Device: IMU (Inertial Measurement Unit) worn on the body\n"
                f"  Channels: {ctx['num_channels']} ({channels_str})\n"
                f"  Sampling rate: {ctx['sampling_rate']} Hz\n"
                f"  Window duration: {ctx['window_duration_sec']:.2f} seconds\n\n"
            )
        else:
            sensor_section = (
                "SENSOR SETUP:\n"
                "  Device: IMU (Inertial Measurement Unit) worn on the body\n"
                "  Channels: 6 (accelerometer xyz + gyroscope xyz)\n\n"
            )

        options_str = "\n".join(
            f"  {i+1}. {opt}" for i, opt in enumerate(self.activity_options)
        )

        prompt = (
            f"{sensor_section}"
            "The following neuro-symbolic description was generated by a neural network "
            "that processed the raw IMU sensor data. Each symbolic statement describes "
            "detected motion patterns, postures, gait characteristics, and intensities "
            "from the sensor readings.\n\n"
            "NEURO-SYMBOLIC DESCRIPTION:\n"
            f"{symbolic_text}\n\n"
            "Given the above sensor context and symbolic description, the person was "
            "performing ONE of these activities:\n"
            f"{options_str}\n\n"
            "Which activity best matches the symbolic description? "
            "Respond with ONLY the activity text (copy exactly from the list), nothing else."
        )

        try:
            response = self.client.responses.create(
                model=self.config.reward_model,
                input=prompt,
                temperature=0.1,  # very low for consistent classification
            )
            predicted = response.output_text.strip().lower()

            # Match to closest option
            best_match = self._match_to_option(predicted)
            return best_match
        except Exception as e:
            print(f"[RewardFunction] OpenAI classification error: {e}")
            return "unknown"

    def _match_to_option(self, predicted: str) -> str:
        """Match OpenAI's response to the closest activity option."""
        pred_lower = predicted.lower().strip()
        # Exact match
        for opt in self.activity_options:
            if opt.lower() == pred_lower:
                return opt
        # Substring match
        for opt in self.activity_options:
            if opt.lower() in pred_lower or pred_lower in opt.lower():
                return opt
        # Word overlap match
        pred_words = set(pred_lower.split())
        best_overlap = 0
        best_opt = self.activity_options[0] if self.activity_options else "unknown"
        for opt in self.activity_options:
            opt_words = set(opt.lower().split())
            overlap = len(pred_words & opt_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_opt = opt
        return best_opt

    def _compute_classification_reward(
        self, predicted: str, ground_truth: str, class_weight: float = 1.0,
    ) -> float:
        """
        Classification reward with asymmetric class weighting.

        Only CORRECT predictions are scaled by class_weight (boosting rare
        classes). Wrong predictions get a flat penalty regardless of class.

        Call ``calibrate_rewards()`` before training to auto-set wrong_reward
        so that no single-class strategy is profitable.

        Rewards:
          - Correct:  +correct_reward * class_weight
          - Related:  +partial_reward  (flat)
          - Wrong:    wrong_reward     (flat, NOT class-weighted)

        Args:
            class_weight: Multiplier for this sample's class (inverse frequency).
                          Higher for minority classes.
        """
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()

        # Exact or near-exact match
        if pred_lower == gt_lower or pred_lower in gt_lower or gt_lower in pred_lower:
            return self.config.correct_reward * class_weight

        # Check word overlap — if most words match, it's correct
        pred_words = set(pred_lower.split())
        gt_words = set(gt_lower.split())
        if pred_words and gt_words:
            overlap = len(pred_words & gt_words)
            union = len(pred_words | gt_words)
            if overlap / union > 0.5:
                return self.config.correct_reward * class_weight

        # Related activity (partial credit)
        if _activities_related(predicted, ground_truth):
            return self.config.partial_reward

        # Wrong: flat penalty (NOT scaled by class_weight)
        return self.config.wrong_reward

    def batch_compute_rewards(
        self,
        symbolic_texts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[Tuple[float, str]]:
        """Compute rewards for a batch of symbolic outputs."""
        if ground_truths is None:
            ground_truths = [None] * len(symbolic_texts)

        results = []
        for text, gt in zip(symbolic_texts, ground_truths):
            results.append(self.compute_reward(text, gt))
            time.sleep(0.05)  # Rate limiting (lighter since single call now)

        return results

    def calibrate_rewards(
        self,
        n_classes: int,
        class_weights: Dict[int, float],
        margin: float = 0.1,
    ) -> float:
        """
        Auto-calibrate ``wrong_reward`` so that no single-class prediction
        strategy is profitable under class-balanced sampling.

        For K classes with balanced batches, always predicting class c yields:

            E[r] = (1/K) * correct_reward * w_c + ((K-1)/K) * wrong_reward

        Setting wrong_reward = -correct_reward * max(w) / (K-1) makes the
        best single-class strategy break even (E[r] = 0).  We add a small
        margin to make it strictly negative.

        This is dataset-agnostic: works for any K and any weight distribution.

        Args:
            n_classes:     Number of classes in the dataset.
            class_weights: Dict {class_int: weight} (from dataset.get_class_weights()).
            margin:        Extra fraction to push E[r] below zero (default 10%).

        Returns:
            The computed wrong_reward (also sets self.config.wrong_reward).
        """
        max_w = max(class_weights.values())
        # wrong_reward <= -correct * max_w / (K-1)
        wrong_reward = -self.config.correct_reward * max_w / max(n_classes - 1, 1)
        wrong_reward *= (1.0 + margin)  # small safety margin
        self.config.wrong_reward = round(wrong_reward, 4)

        # Verify: print the worst-case expected reward
        best_ev = max(
            (1 / n_classes) * self.config.correct_reward * w
            + ((n_classes - 1) / n_classes) * self.config.wrong_reward
            for w in class_weights.values()
        )
        perfect_ev = sum(
            self.config.correct_reward * w for w in class_weights.values()
        ) / n_classes

        print(f"  [RewardCalibration] K={n_classes}, max_weight={max_w:.2f}")
        print(f"    wrong_reward = {self.config.wrong_reward:+.4f}")
        print(f"    Worst single-class E[r] = {best_ev:+.4f}  (should be <= 0)")
        print(f"    Perfect classifier E[r] = {perfect_ev:+.4f}  (should be >> 0)")

        return self.config.wrong_reward

    def classify_batch_parallel(
        self,
        symbolic_texts: List[str],
        max_workers: int = 8,
    ) -> List[str]:
        """
        Classify a batch of symbolic texts in parallel using ThreadPoolExecutor.

        This is the main speedup: OpenAI API calls are I/O-bound, so we can
        fire many concurrently. With 8 workers, ~8x faster than serial.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        activities = [None] * len(symbolic_texts)

        def _classify_one(idx_text):
            idx, text = idx_text
            # Check cache first
            cache_key = hash(text)
            if cache_key in self._cache:
                return idx, self._cache[cache_key][1]
            activity = self._classify_symbolic(text)
            self._cache[cache_key] = (0.0, activity)
            return idx, activity

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_classify_one, (i, t))
                       for i, t in enumerate(symbolic_texts)]
            for future in as_completed(futures):
                idx, activity = future.result()
                activities[idx] = activity

        return activities

    def _classify_descriptor(self, descriptor_text: str) -> str:
        """
        Classify an activity from a VQ-VAE token descriptor (statistical text).

        Unlike _classify_symbolic which receives neuro-symbolic language,
        this receives a statistical description of VQ-VAE codebook tokens
        (frequencies, transitions, entropy, etc.) and asks the LLM to
        classify the activity from those patterns.
        """
        # Build sensor context section
        if self.sensor_context:
            ctx = self.sensor_context
            channels_str = ", ".join(ctx.get("channel_names", []))
            sensor_section = (
                f"SENSOR SETUP:\n"
                f"  Device: IMU (Inertial Measurement Unit) worn on the body\n"
                f"  Channels: {ctx['num_channels']} ({channels_str})\n"
                f"  Sampling rate: {ctx['sampling_rate']} Hz\n"
                f"  Window duration: {ctx['window_duration_sec']:.2f} seconds\n\n"
            )
        else:
            sensor_section = (
                "SENSOR SETUP:\n"
                "  Device: IMU (Inertial Measurement Unit) worn on the body\n"
                "  Channels: 6 (accelerometer xyz + gyroscope xyz)\n\n"
            )

        options_str = "\n".join(
            f"  {i+1}. {opt}" for i, opt in enumerate(self.activity_options)
        )

        prompt = (
            f"{sensor_section}"
            "A VQ-VAE neural network was trained to learn a codebook of motion primitives "
            "from raw IMU sensor data. Each codebook entry (token) represents a distinct "
            "learned motion pattern. Below is a statistical description of the token "
            "sequence produced for one recording segment.\n\n"
            "TOKEN ANALYSIS:\n"
            f"{descriptor_text}\n\n"
            "Each token (imu_tok_N) represents a learned motion primitive. Patterns like "
            "high self-repetition suggest sustained/static activity, high entropy suggests "
            "varied/dynamic movement, and burst patterns suggest rhythmic/periodic motion.\n\n"
            "Given the above token statistics, the person was performing ONE of these activities:\n"
            f"{options_str}\n\n"
            "Which activity best matches? "
            "Respond with ONLY the activity text (copy exactly from the list), nothing else."
        )

        try:
            response = self.client.responses.create(
                model=self.config.reward_model,
                input=prompt,
                temperature=0.1,
            )
            predicted = response.output_text.strip().lower()
            return self._match_to_option(predicted)
        except Exception as e:
            print(f"[RewardFunction] OpenAI descriptor classification error: {e}")
            return "unknown"

    def classify_descriptors_parallel(
        self,
        descriptor_texts: List[str],
        max_workers: int = 8,
    ) -> List[str]:
        """
        Classify a batch of token descriptors in parallel using ThreadPoolExecutor.
        Same pattern as classify_batch_parallel but uses _classify_descriptor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        activities = [None] * len(descriptor_texts)

        def _classify_one(idx_text):
            idx, text = idx_text
            cache_key = hash(("descriptor", text))
            if cache_key in self._cache:
                return idx, self._cache[cache_key][1]
            activity = self._classify_descriptor(text)
            self._cache[cache_key] = (0.0, activity)
            return idx, activity

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_classify_one, (i, t))
                       for i, t in enumerate(descriptor_texts)]
            for future in as_completed(futures):
                idx, activity = future.result()
                activities[idx] = activity

        return activities

    def _interpret_symbolic(self, symbolic_text: str) -> str:
        """Fallback: free-form interpretation with sensor context."""
        # Build sensor context
        if self.sensor_context:
            ctx = self.sensor_context
            channels_str = ", ".join(ctx.get("channel_names", []))
            context_str = (
                f"The data comes from a {ctx['num_channels']}-channel IMU "
                f"({channels_str}) sampled at {ctx['sampling_rate']}Hz, "
                f"recording {ctx['window_duration_sec']:.1f} seconds of movement.\n\n"
            )
        else:
            context_str = ""

        prompt = (
            "You are interpreting a neuro-symbolic description generated from "
            "IMU (Inertial Measurement Unit) sensor data worn by a person.\n\n"
            f"{context_str}"
            "NEURO-SYMBOLIC DESCRIPTION:\n"
            f"{symbolic_text}\n\n"
            "Based ONLY on the above symbolic description, what physical activity "
            "is the person most likely performing?\n\n"
            "Respond with ONLY a short activity label (1-5 words), nothing else.\n"
            "Examples: walking, running, sitting still, jumping rope, waving hand"
        )

        try:
            response = self.client.responses.create(
                model=self.config.reward_model,
                input=prompt,
                temperature=self.config.reward_temperature,
            )
            return response.output_text.strip().lower()
        except Exception as e:
            print(f"[RewardFunction] OpenAI interpretation error: {e}")
            return "unknown"

    def _compute_match_reward(self, predicted: str, ground_truth: str) -> float:
        """Fallback: semantic similarity via OpenAI (2-step)."""
        prompt = (
            "Compare these two activity descriptions and rate their semantic similarity.\n\n"
            f"PREDICTED: {predicted}\n"
            f"ACTUAL: {ground_truth}\n\n"
            "Rate similarity from 0 to 10:\n"
            "  0 = completely different activities\n"
            "  5 = somewhat related\n"
            "  10 = same activity (even if worded differently)\n\n"
            "Respond with ONLY a number (0-10), nothing else."
        )

        try:
            response = self.client.responses.create(
                model=self.config.reward_model,
                input=prompt,
                temperature=0.1,
            )
            score_str = response.output_text.strip()
            import re
            numbers = re.findall(r'(\d+\.?\d*)', score_str)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score / 10.0))
            return 0.0
        except Exception as e:
            print(f"[RewardFunction] Matching error: {e}")
            return self._simple_match(predicted, ground_truth)

    @staticmethod
    def _simple_match(predicted: str, ground_truth: str) -> float:
        """Fallback string-based matching when API fails."""
        pred = predicted.lower().strip()
        gt = ground_truth.lower().strip()
        if gt in pred or pred in gt:
            return 0.8
        pred_words = set(pred.split())
        gt_words = set(gt.split())
        if pred_words & gt_words:
            overlap = len(pred_words & gt_words) / max(len(pred_words | gt_words), 1)
            return overlap * 0.7
        return 0.0

    @staticmethod
    def _is_parseable(symbolic_text: str) -> bool:
        """Check if the symbolic text has any recognizable structure."""
        import re
        return bool(re.search(r'[A-Z]+\(.*\)', symbolic_text))


# ============================================================================
# Experience Buffer
# ============================================================================

@dataclass
class Experience:
    """A single RL experience tuple."""
    src_ids: torch.Tensor       # (S,) source IMU tokens
    tgt_ids: torch.Tensor       # (T,) generated symbolic tokens
    log_probs: torch.Tensor     # (T,) log probs of generated tokens
    reward: float               # scalar reward from OpenAI
    activity: str               # interpreted activity
    symbolic_text: str          # decoded symbolic output


class ExperienceBuffer:
    """Stores experiences for batch RL updates."""

    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self.buffer: List[Experience] = []

    def add(self, exp: Experience):
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Experience]:
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_all(self) -> List[Experience]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization for the Translator.

    v2 improvements:
      - Running reward normalization (mean/std tracking)
      - LR warmup + cosine decay schedule
      - Entropy coefficient annealing
      - Better advantage estimation
    """

    def __init__(
        self,
        model: TranslatorModel,
        vocab: SymbolicVocab,
        config: RLConfig = RLConfig(),
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.config = config
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

        # LR schedule: warmup + cosine decay
        def lr_lambda(step):
            if step < config.warmup_steps:
                return float(step + 1) / float(max(1, config.warmup_steps))
            progress = float(step - config.warmup_steps) / float(
                max(1, config.total_updates - config.warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.reward_fn = OpenAIRewardFunction(config)
        self.buffer = ExperienceBuffer()

        # Running reward statistics for normalization
        self.reward_history = deque(maxlen=200)
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0

        # Moving average baseline
        self.baseline = 0.0
        self.num_updates = 0

        # Current entropy coefficient (decays over time)
        self.current_entropy_coef = config.entropy_coef

        # Logging
        self.train_log: List[Dict] = []

    def decay_entropy(self):
        """Call at the end of each epoch to decay entropy coefficient."""
        self.current_entropy_coef = max(
            self.config.entropy_coef_min,
            self.current_entropy_coef * self.config.entropy_decay,
        )

    def collect_experience(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        ground_truth: Optional[str] = None,
        temperature: float = 1.0,
    ) -> Experience:
        """
        Generate symbolic output and get reward from OpenAI.

        Args:
            src_ids: (1, S) source IMU tokens
            src_mask: (1, S) attention mask
            ground_truth: optional activity label

        Returns:
            Experience tuple
        """
        self.model.eval()

        # Generate
        generated, log_probs = self.model.generate(
            src_ids.to(self.device),
            src_mask.to(self.device),
            temperature=temperature,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
        )

        # Decode to text
        gen_ids = generated[0].tolist()
        symbolic_text = self.vocab.decode(gen_ids)

        # Get reward from OpenAI
        reward, activity = self.reward_fn.compute_reward(symbolic_text, ground_truth)

        exp = Experience(
            src_ids=src_ids[0].cpu(),
            tgt_ids=generated[0].cpu(),
            log_probs=log_probs[0].cpu(),
            reward=reward,
            activity=activity,
            symbolic_text=symbolic_text,
        )
        self.buffer.add(exp)
        return exp

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using running normalization + baseline.

        v2: uses tracked reward statistics for more stable normalization.
        """
        # Update reward history
        for r in rewards.tolist():
            self.reward_history.append(r)

        # Update running stats
        if len(self.reward_history) > 1:
            hist = list(self.reward_history)
            self.reward_running_mean = sum(hist) / len(hist)
            self.reward_running_var = max(
                sum((r - self.reward_running_mean) ** 2 for r in hist) / len(hist),
                0.01,
            )

        # Update baseline with momentum
        mean_reward = rewards.mean().item()
        self.baseline = (
            self.config.baseline_momentum * self.baseline
            + (1 - self.config.baseline_momentum) * mean_reward
        )

        # Normalize rewards using running statistics
        if self.config.normalize_rewards and len(self.reward_history) > 10:
            rewards_norm = (rewards - self.reward_running_mean) / (
                self.reward_running_var ** 0.5 + 1e-8
            )
            advantages = rewards_norm
        else:
            advantages = rewards - self.baseline

        # Normalize advantages per batch
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def ppo_update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected experiences.

        Returns:
            Dictionary of training metrics.
        """
        if len(self.buffer) < self.config.batch_size:
            return {"error": "not enough experiences"}

        self.model.train()
        experiences = self.buffer.get_all()

        # Prepare batch
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        rewards = rewards * self.config.reward_scale
        advantages = self.compute_advantages(rewards)

        # Pad source and target sequences
        max_src = max(e.src_ids.size(0) for e in experiences)
        max_tgt = max(e.tgt_ids.size(0) for e in experiences)

        B = len(experiences)
        src_batch = torch.zeros(B, max_src, dtype=torch.long)
        tgt_batch = torch.zeros(B, max_tgt, dtype=torch.long)
        src_mask = torch.zeros(B, max_src, dtype=torch.bool)
        tgt_mask = torch.zeros(B, max_tgt, dtype=torch.bool)
        old_log_probs = torch.zeros(B, max_tgt - 1)  # exclude BOS

        for i, exp in enumerate(experiences):
            s_len = exp.src_ids.size(0)
            t_len = exp.tgt_ids.size(0)
            lp_len = exp.log_probs.size(0)

            src_batch[i, :s_len] = exp.src_ids
            tgt_batch[i, :t_len] = exp.tgt_ids
            src_mask[i, :s_len] = True
            tgt_mask[i, :t_len] = True
            old_log_probs[i, :lp_len] = exp.log_probs

        # Move to device
        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)

        # PPO epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0

        for epoch in range(self.config.ppo_epochs):
            # Forward pass: teacher-forced with generated tokens
            tgt_input = tgt_batch[:, :-1]  # shift right
            tgt_target = tgt_batch[:, 1:]  # shift left
            tgt_mask_input = tgt_mask[:, :-1]

            new_log_probs, values, entropy = self.model.get_log_probs_and_values(
                src_batch, tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask_input,
            )

            # Mask to valid positions
            T = new_log_probs.size(1)
            valid_mask = tgt_mask_input[:, :T].float()
            valid_count = valid_mask.sum() + 1e-8

            # Per-sequence log prob (sum over tokens)
            seq_log_probs = (new_log_probs * valid_mask).sum(dim=1)
            seq_old_log_probs = (old_log_probs[:, :T] * valid_mask).sum(dim=1)

            # PPO ratio
            ratio = torch.exp(seq_log_probs - seq_old_log_probs.detach())
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            )

            # Policy loss
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages,
            ).mean()

            # Value loss
            returns = advantages  # simplified — advantages ≈ returns - baseline
            value_pred = (values * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
            value_loss = F.mse_loss(value_pred, returns)

            # Entropy bonus (with annealing)
            entropy_loss = -(entropy * valid_mask).sum() / valid_count

            # Total loss — use current (decayed) entropy coefficient
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.current_entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        # Step LR schedule
        self.scheduler.step()

        n = self.config.ppo_epochs
        metrics = {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy_loss": total_entropy_loss / n,
            "mean_reward": rewards.mean().item(),
            "baseline": self.baseline,
            "buffer_size": len(self.buffer),
            "lr": self.optimizer.param_groups[0]["lr"],
            "entropy_coef": self.current_entropy_coef,
            "reward_running_mean": self.reward_running_mean,
        }
        self.train_log.append(metrics)
        self.num_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        return metrics

    def save_checkpoint(self, tag: str = "latest"):
        """Save training checkpoint."""
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        path = os.path.join(CHECKPOINTS_DIR, f"translator_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "baseline": self.baseline,
            "num_updates": self.num_updates,
            "train_log": self.train_log,
            "current_entropy_coef": self.current_entropy_coef,
            "reward_running_mean": self.reward_running_mean,
            "reward_running_var": self.reward_running_var,
        }, path)
        print(f"[PPOTrainer] Checkpoint saved: {path}")

    def load_checkpoint(self, tag: str = "latest"):
        """Load training checkpoint."""
        path = os.path.join(CHECKPOINTS_DIR, f"translator_{tag}.pt")
        if not os.path.exists(path):
            print(f"[PPOTrainer] No checkpoint found at {path}")
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.baseline = checkpoint.get("baseline", 0.0)
        self.num_updates = checkpoint.get("num_updates", 0)
        self.train_log = checkpoint.get("train_log", [])
        self.current_entropy_coef = checkpoint.get(
            "current_entropy_coef", self.config.entropy_coef
        )
        self.reward_running_mean = checkpoint.get("reward_running_mean", 0.0)
        self.reward_running_var = checkpoint.get("reward_running_var", 1.0)
        print(f"[PPOTrainer] Checkpoint loaded: {path} (update #{self.num_updates})")


# ============================================================================
# Supervised Pretrainer (bootstraps before RL)
# ============================================================================

class SupervisedPretrainer:
    """
    Pretrains the Translator with supervised (IMU→symbolic) pairs
    before RL fine-tuning.
    """

    def __init__(
        self,
        model: TranslatorModel,
        vocab: SymbolicVocab,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def train_step(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> float:
        """
        One supervised training step.

        Args:
            src_ids: (B, S)
            tgt_ids: (B, T) — full target including BOS and EOS
            src_mask: (B, S)
            tgt_mask: (B, T)

        Returns:
            loss value
        """
        self.model.train()

        tgt_input = tgt_ids[:, :-1].to(self.device)
        tgt_target = tgt_ids[:, 1:].to(self.device)
        tgt_mask_input = tgt_mask[:, :-1].to(self.device)

        logits, _ = self.model(
            src_ids.to(self.device),
            tgt_input,
            src_key_padding_mask=src_mask.to(self.device),
            tgt_key_padding_mask=tgt_mask_input,
        )

        # (B, T, V) → (B*T, V) and (B*T,)
        B, T, V = logits.shape
        loss = self.criterion(logits.reshape(B * T, V), tgt_target.reshape(B * T))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
