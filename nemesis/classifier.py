"""
OpenAI Classifier — LLM-based activity classification from VQ-VAE
token descriptors.

Given a statistical description of a VQ-VAE token sequence (frequencies,
entropy, transitions, bursts, etc.), the classifier asks an LLM to pick
the most likely activity from a fixed list.

Also handles reward computation for evaluation metrics and auto-calibration
so that no single-class prediction strategy is profitable.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from nemesis.config import ClassifierConfig, load_api_key


# ============================================================================
# Activity Similarity Map (for partial credit)
# ============================================================================

ACTIVITY_GROUPS = {
    "locomotion": {
        "walking", "walking_upstairs", "walking_downstairs",
        "walking forward on a flat surface at a normal steady pace",
        "walking up a flight of stairs at a steady pace",
        "walking down a flight of stairs at a steady pace",
        "walk",
        "the person is walking around the room at a natural pace",
    },
    "stationary": {
        "sitting", "standing", "laying", "stand", "sit", "lie",
        "sitting still on a chair in a relaxed position",
        "standing upright in place without moving",
        "lying down flat on a surface in a resting position",
        "the person is standing upright without locomotion",
        "the person is sitting still on a chair",
        "the person is lying down on a deckchair",
    },
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
# OpenAI Classifier
# ============================================================================

class OpenAIClassifier:
    """
    Classifies VQ-VAE token descriptor text into one of the known activities
    using an LLM.

    Usage:
        clf = OpenAIClassifier(config)
        clf.set_activity_options(["Stand", "Walk", "Sit", "Lie"])
        clf.set_sensor_context({...})
        activities = clf.classify_batch(descriptor_texts, max_workers=8)
    """

    def __init__(
        self,
        config: ClassifierConfig = ClassifierConfig(),
        activity_options: List[str] = None,
    ):
        self.config = config
        self.client = OpenAI(api_key=load_api_key())
        self.activity_options = activity_options or []
        self.sensor_context: Optional[Dict] = None
        # Cache: hash → predicted activity
        self._cache: Dict[int, str] = {}

    def set_activity_options(self, options: List[str]):
        """Set the valid activity options for classification."""
        self.activity_options = list(set(options))

    def set_sensor_context(self, ctx: Dict):
        """Set IMU sensor context for richer prompts."""
        self.sensor_context = ctx

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, descriptor_text: str) -> str:
        """
        Classify a single token descriptor text into an activity.

        Args:
            descriptor_text: Statistical description of a VQ-VAE token sequence.

        Returns:
            Predicted activity string (from activity_options).
        """
        # Check cache
        cache_key = hash(("descriptor", descriptor_text))
        if cache_key in self._cache:
            return self._cache[cache_key]

        activity = self._call_llm(descriptor_text)
        self._cache[cache_key] = activity
        return activity

    def classify_batch(
        self,
        descriptor_texts: List[str],
        max_workers: int = 8,
    ) -> List[str]:
        """
        Classify a batch of token descriptors in parallel using ThreadPoolExecutor.

        OpenAI API calls are I/O-bound, so parallel threads give ~Nx speedup.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        activities = [None] * len(descriptor_texts)

        def _classify_one(idx_text):
            idx, text = idx_text
            return idx, self.classify(text)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_classify_one, (i, t))
                for i, t in enumerate(descriptor_texts)
            ]
            for future in as_completed(futures):
                idx, activity = future.result()
                activities[idx] = activity

        return activities

    def set_few_shot_examples(self, examples: List[Dict]):
        """Set the few-shot examples for the current query.

        Args:
            examples: List of dicts with keys:
              activity, similarity, top_tokens, confidence
        """
        self._few_shot_examples = examples

    def _call_llm(self, descriptor_text: str) -> str:
        """Send a token descriptor to the LLM for classification."""
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

        # Build few-shot section from memory (grouped by activity)
        few_shot_section = ""
        examples = getattr(self, "_few_shot_examples", None)
        if examples:
            # Group examples by activity, preserving order (top activity first)
            from collections import OrderedDict
            grouped: OrderedDict = OrderedDict()
            for ex in examples:
                act = ex["activity"]
                if act not in grouped:
                    grouped[act] = []
                grouped[act].append(ex)

            lines = [
                "REFERENCE EXAMPLES (from memory — similar token patterns seen before):",
                f"  Retrieved {len(examples)} examples across {len(grouped)} candidate activities.\n",
            ]
            for act_idx, (act, act_examples) in enumerate(grouped.items(), 1):
                avg_sim = sum(e['similarity'] for e in act_examples) / len(act_examples)
                best_sim = max(e['similarity'] for e in act_examples)
                lines.append(
                    f"  Candidate Activity {act_idx}: \"{act}\" "
                    f"({len(act_examples)} matches, best_sim={best_sim*100:.0f}%, "
                    f"avg_sim={avg_sim*100:.0f}%)"
                )
                for j, ex in enumerate(act_examples, 1):
                    sim_pct = ex['similarity'] * 100
                    src = "ground truth" if ex.get('confidence', 0) >= 1.0 else "past inference"
                    lines.append(
                        f"    Entry {j} (sim={sim_pct:.0f}%, {src}): "
                        f"Top tokens: {ex.get('top_tokens', '?')}"
                    )
                lines.append("")
            few_shot_section = "\n".join(lines) + "\n"

        prompt = (
            f"{sensor_section}"
            "A VQ-VAE neural network was trained to learn a codebook of motion primitives "
            "from raw IMU sensor data. Each codebook entry (token) represents a distinct "
            "learned motion pattern. Below is a statistical description of the token "
            "sequence produced for one recording segment.\n\n"
            f"{few_shot_section}"
            "TOKEN ANALYSIS (current sample):\n"
            f"{descriptor_text}\n\n"
            "Each token (imu_tok_N) represents a learned motion primitive. The REFERENCE "
            "EXAMPLES above show what activities were associated with similar token "
            "patterns in the past. Use them to ground your classification.\n\n"
            "Given the above, the person was performing ONE of these activities:\n"
            f"{options_str}\n\n"
            "Which activity best matches? "
            "Respond with ONLY the activity text (copy exactly from the list), nothing else."
        )

        try:
            response = self.client.responses.create(
                model=self.config.model,
                input=prompt,
                temperature=0.1,
            )
            predicted = response.output_text.strip().lower()
            return self._match_to_option(predicted)
        except Exception as e:
            print(f"[Classifier] OpenAI error: {e}")
            return "unknown"

    def _match_to_option(self, predicted: str) -> str:
        """Match LLM response to the closest activity option."""
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

    # ------------------------------------------------------------------
    # Reward / scoring
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        predicted: str,
        ground_truth: str,
        class_weight: float = 1.0,
    ) -> float:
        """
        Compute reward comparing predicted activity to ground truth.

        Only correct predictions are scaled by class_weight (boosting rare
        classes). Wrong predictions get a flat penalty.

        Rewards:
          - Correct:  +correct_reward * class_weight
          - Related:  +partial_reward  (flat)
          - Wrong:    wrong_reward     (flat, NOT class-weighted)
        """
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()

        # Exact or near-exact match
        if pred_lower == gt_lower or pred_lower in gt_lower or gt_lower in pred_lower:
            return self.config.correct_reward * class_weight

        # Word overlap — if most words match, it's correct
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

        # Wrong: flat penalty
        return self.config.wrong_reward

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

        We set wrong_reward = -correct_reward * max(w) / (K-1) * (1+margin)
        making every single-class strategy yield E[r] < 0.

        This is dataset-agnostic: works for any K and any weight distribution.

        Returns:
            The computed wrong_reward (also stored in self.config.wrong_reward).
        """
        max_w = max(class_weights.values())
        wrong_reward = -self.config.correct_reward * max_w / max(n_classes - 1, 1)
        wrong_reward *= (1.0 + margin)
        self.config.wrong_reward = round(wrong_reward, 4)

        # Verify
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
