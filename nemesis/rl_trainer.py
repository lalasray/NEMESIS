"""
RL Trainer — trains the Translator model using reinforcement learning
with OpenAI API as the reward function.

Training loop:
  1. Translator generates neuro-symbolic output from IMU tokens
  2. Output is sent to OpenAI to interpret the activity
  3. OpenAI rates the quality / clarity of the symbolic output (reward)
  4. Policy gradient (REINFORCE or PPO) updates Translator weights

Supports both:
  - REINFORCE with baseline (simple, less stable)
  - PPO (clipped objective, recommended)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

from nemesis.config import RLConfig, load_api_key, CHECKPOINTS_DIR
from nemesis.translator import TranslatorModel
from nemesis.neuro_symbolic import SymbolicVocab


# ============================================================================
# Reward Function via OpenAI
# ============================================================================

class OpenAIRewardFunction:
    """
    Uses OpenAI API to score the quality of neuro-symbolic translations.

    The reward is based on:
      1. Clarity: Is the symbolic output interpretable?
      2. Specificity: Does it describe a recognizable activity?
      3. Coherence: Are the statements internally consistent?
    """

    def __init__(self, config: RLConfig = RLConfig()):
        self.config = config
        self.client = OpenAI(api_key=load_api_key())

    def compute_reward(
        self,
        symbolic_text: str,
        ground_truth: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Score a neuro-symbolic output.

        Args:
            symbolic_text: The generated neuro-symbolic description
            ground_truth: Optional ground truth activity label

        Returns:
            reward: float in [0, 1]
            activity: The interpreted activity description
        """
        prompt = self._build_reward_prompt(symbolic_text, ground_truth)

        try:
            response = self.client.responses.create(
                model=self.config.reward_model,
                input=prompt,
                temperature=self.config.reward_temperature,
            )
            result = response.output_text.strip()
            return self._parse_reward_response(result)
        except Exception as e:
            print(f"[RewardFunction] OpenAI API error: {e}")
            return 0.0, "unknown"

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
            time.sleep(0.1)  # Rate limiting

        return results

    def _build_reward_prompt(self, symbolic_text: str, ground_truth: Optional[str]) -> str:
        prompt = (
            "You are evaluating a neuro-symbolic translation of IMU sensor data.\n\n"
            "NEURO-SYMBOLIC OUTPUT:\n"
            f"{symbolic_text}\n\n"
        )
        if ground_truth:
            prompt += f"GROUND TRUTH ACTIVITY: {ground_truth}\n\n"

        prompt += (
            "Rate this output on a scale of 0 to 10 based on:\n"
            "1. CLARITY: Are the statements clear and well-formed?\n"
            "2. SPECIFICITY: Do they describe a recognizable human activity?\n"
            "3. COHERENCE: Are the statements internally consistent?\n"
        )
        if ground_truth:
            prompt += "4. ACCURACY: Does the description match the ground truth?\n"

        prompt += (
            "\nRespond in exactly this JSON format:\n"
            '{"score": <0-10>, "activity": "<what the person is doing>", '
            '"reasoning": "<brief explanation>"}\n'
        )
        return prompt

    def _parse_reward_response(self, response: str) -> Tuple[float, str]:
        """Parse the structured JSON reward response."""
        try:
            # Try to extract JSON from response
            # Handle markdown code blocks
            if "```" in response:
                json_str = response.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()
            else:
                json_str = response

            data = json.loads(json_str)
            score = float(data.get("score", 0))
            activity = data.get("activity", "unknown")
            # Normalize to [0, 1]
            reward = max(0.0, min(1.0, score / 10.0))
            return reward, activity
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract a number
            import re
            numbers = re.findall(r'(\d+\.?\d*)', response)
            if numbers:
                score = float(numbers[0])
                reward = max(0.0, min(1.0, score / 10.0))
                return reward, "unknown"
            return 0.0, "unknown"


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

    Workflow:
      1. Collect experiences (generate → get reward)
      2. Compute advantages
      3. PPO update (multiple epochs over the batch)
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

        self.optimizer = AdamW(model.parameters(), lr=config.lr)
        self.reward_fn = OpenAIRewardFunction(config)
        self.buffer = ExperienceBuffer()

        # Moving average baseline
        self.baseline = 0.0
        self.num_updates = 0

        # Logging
        self.train_log: List[Dict] = []

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
        Compute advantages using a simple moving average baseline.

        Args:
            rewards: (B,) batch of rewards

        Returns:
            advantages: (B,) normalized advantages
        """
        # Update baseline
        mean_reward = rewards.mean().item()
        self.baseline = (
            self.config.baseline_momentum * self.baseline
            + (1 - self.config.baseline_momentum) * mean_reward
        )

        advantages = rewards - self.baseline
        # Normalize
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

            # Entropy bonus
            entropy_loss = -(entropy * valid_mask).sum() / valid_count

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        n = self.config.ppo_epochs
        metrics = {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy_loss": total_entropy_loss / n,
            "mean_reward": rewards.mean().item(),
            "baseline": self.baseline,
            "buffer_size": len(self.buffer),
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
            "config": self.config,
            "baseline": self.baseline,
            "num_updates": self.num_updates,
            "train_log": self.train_log,
        }, path)
        print(f"[PPOTrainer] Checkpoint saved: {path}")

    def load_checkpoint(self, tag: str = "latest"):
        """Load training checkpoint."""
        path = os.path.join(CHECKPOINTS_DIR, f"translator_{tag}.pt")
        if not os.path.exists(path):
            print(f"[PPOTrainer] No checkpoint found at {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.baseline = checkpoint.get("baseline", 0.0)
        self.num_updates = checkpoint.get("num_updates", 0)
        self.train_log = checkpoint.get("train_log", [])
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
