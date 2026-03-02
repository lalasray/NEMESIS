"""
NEMESIS Pipeline — orchestrates the full IMU → Activity pipeline.

                  ┌─────────────────┐
  IMU stream ──►  │  IMU Tokenizer  │ ──► token IDs
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ Memory Recall   │ ──► context from past sessions
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   Translator    │ ──► neuro-symbolic text
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ Knowledge Cache │ ──► known? → return cached activity
                  └────────┬────────┘
                           │ (cache miss)
                  ┌────────▼────────┐
                  │   OpenAI API    │ ──► activity description + reward
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Memory Store   │ ──► persist for future sessions
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   RL Update     │ ──► improve Translator
                  └─────────────────┘
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

from nemesis.config import (
    IMUTokenizerConfig, TranslatorConfig, NeuroSymbolicConfig,
    RLConfig, MemoryConfig, load_api_key, CHECKPOINTS_DIR,
)
from nemesis.imu_tokenizer import BinningTokenizer, VQVAE_Tokenizer, generate_synthetic_imu
from nemesis.neuro_symbolic import SymbolicVocab, SymbolicProgram, get_template_program
from nemesis.translator import TranslatorModel
from nemesis.rl_trainer import PPOTrainer, SupervisedPretrainer
from nemesis.memory import MemoryManager


# ============================================================================
# Translation Result
# ============================================================================

@dataclass
class TranslationResult:
    """Complete result of one translation pass."""
    symbolic_text: str
    activity: str
    confidence: float
    from_cache: bool
    imu_tokens: List[int]
    memory_context: str


# ============================================================================
# NEMESIS Pipeline
# ============================================================================

class NemesisPipeline:
    """
    Full NEMESIS pipeline — from raw IMU data to activity recognition
    with RL training and persistent memory.
    """

    def __init__(
        self,
        imu_config: IMUTokenizerConfig = IMUTokenizerConfig(),
        translator_config: TranslatorConfig = None,
        ns_config: NeuroSymbolicConfig = NeuroSymbolicConfig(),
        rl_config: RLConfig = RLConfig(),
        memory_config: MemoryConfig = MemoryConfig(),
        use_vqvae: bool = False,
        device: str = "cpu",
    ):
        # --- Symbolic vocabulary (needed to set translator vocab size) ---
        self.vocab = SymbolicVocab(ns_config)

        # --- Translator config ---
        if translator_config is None:
            translator_config = TranslatorConfig(
                src_vocab_size=imu_config.codebook_size + imu_config.num_special_tokens,
                tgt_vocab_size=self.vocab.size,
            )

        # --- Components ---
        self.device = device
        if use_vqvae:
            self.tokenizer = VQVAE_Tokenizer(imu_config)
        else:
            self.tokenizer = BinningTokenizer(imu_config)

        self.translator = TranslatorModel(translator_config)
        self.rl_trainer = PPOTrainer(self.translator, self.vocab, rl_config, device)
        self.memory = MemoryManager(memory_config)
        self.openai_client = OpenAI(api_key=load_api_key())

        # Config refs
        self.imu_config = imu_config
        self.translator_config = translator_config
        self.rl_config = rl_config

        # Try to load existing checkpoint
        self._try_load_checkpoint()

        print(f"[NEMESIS] Pipeline initialized")
        print(f"  IMU Tokenizer: {'VQ-VAE' if use_vqvae else 'Binning'}")
        print(f"  Translator: {sum(p.numel() for p in self.translator.parameters()):,} params")
        print(f"  Symbolic vocab: {self.vocab.size} tokens")
        print(f"  Memory: {self.memory.get_stats()}")
        print(f"  Device: {device}")

    def start_session(self, notes: str = ""):
        """Start a new session — enables cross-session memory tracking."""
        self.memory.start_session(notes)

    def end_session(self):
        """End the current session and persist everything."""
        self.memory.end_session()
        self.rl_trainer.save_checkpoint()

    # ---------------------------------------------------------------
    # Main entry point: translate IMU → activity
    # ---------------------------------------------------------------

    def translate(
        self,
        imu_data: np.ndarray,
        ground_truth: Optional[str] = None,
        use_memory: bool = True,
        train: bool = True,
        temperature: float = 1.0,
    ) -> TranslationResult:
        """
        Full pipeline: IMU data → activity description.

        Args:
            imu_data: (T, C) raw IMU sensor data
            ground_truth: optional activity label (for RL reward)
            use_memory: whether to use memory for context/caching
            train: whether to collect RL experience
            temperature: generation temperature

        Returns:
            TranslationResult with activity and metadata
        """
        # Step 1: Tokenize IMU
        if isinstance(self.tokenizer, BinningTokenizer):
            imu_tokens = self.tokenizer.tokenize(imu_data)
            src_ids = torch.tensor([imu_tokens], dtype=torch.long)
            src_mask = torch.ones(1, len(imu_tokens), dtype=torch.bool)
        else:
            imu_tokens = self.tokenizer.tokenize(imu_data)
            src_ids = torch.tensor([imu_tokens], dtype=torch.long)
            src_mask = torch.ones(1, len(imu_tokens), dtype=torch.bool)

        # Step 2: Generate neuro-symbolic output
        self.translator.eval()
        generated, log_probs = self.translator.generate(
            src_ids.to(self.device),
            src_mask.to(self.device),
            temperature=temperature,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
            sample=(temperature > 0),
        )
        symbolic_text = self.vocab.decode(generated[0].tolist())

        # Step 3: Check knowledge cache
        memory_context = ""
        if use_memory:
            cached = self.memory.fast_lookup(symbolic_text)
            if cached:
                result = TranslationResult(
                    symbolic_text=symbolic_text,
                    activity=cached["activity"],
                    confidence=cached["confidence"],
                    from_cache=True,
                    imu_tokens=imu_tokens,
                    memory_context="[from knowledge cache]",
                )
                self.memory._session_stats["translations"] += 1
                return result

            # Get memory context for prompt augmentation
            # Use a simple embedding (mean of token IDs) for similarity search
            embedding = self._compute_embedding(imu_tokens)
            memory_context = self.memory.get_context_for_prompt(embedding)

        # Step 4: Ask OpenAI to interpret
        activity, confidence = self._interpret_with_openai(symbolic_text, memory_context)

        # Step 5: Store in memory
        if use_memory:
            embedding = self._compute_embedding(imu_tokens)
            self.memory.store_translation(
                imu_tokens=imu_tokens,
                symbolic_text=symbolic_text,
                activity=activity,
                confidence=confidence,
                embedding=embedding,
            )

        # Step 6: Collect RL experience
        if train:
            from nemesis.rl_trainer import Experience
            exp = Experience(
                src_ids=src_ids[0].cpu(),
                tgt_ids=generated[0].cpu(),
                log_probs=log_probs[0].cpu(),
                reward=confidence,
                activity=activity,
                symbolic_text=symbolic_text,
            )
            self.rl_trainer.buffer.add(exp)

        return TranslationResult(
            symbolic_text=symbolic_text,
            activity=activity,
            confidence=confidence,
            from_cache=False,
            imu_tokens=imu_tokens,
            memory_context=memory_context,
        )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def supervised_pretrain(
        self,
        activities: List[str] = None,
        num_epochs: int = 50,
        samples_per_activity: int = 10,
    ):
        """
        Pretrain the Translator with synthetic IMU → symbolic pairs.

        Args:
            activities: list of activity names to train on
            num_epochs: number of training epochs
            samples_per_activity: samples to generate per activity
        """
        if activities is None:
            activities = ["walking", "running", "sitting", "jumping", "waving"]

        pretrainer = SupervisedPretrainer(
            self.translator, self.vocab, lr=1e-3, device=self.device
        )

        print(f"[Pretrain] Starting supervised pretraining: {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for activity in activities:
                for _ in range(samples_per_activity):
                    # Generate synthetic IMU
                    imu_data = generate_synthetic_imu(activity, duration_sec=3.0)

                    # Tokenize IMU
                    if isinstance(self.tokenizer, BinningTokenizer):
                        imu_tokens = self.tokenizer.tokenize(imu_data)
                    else:
                        imu_tokens = self.tokenizer.tokenize(imu_data)

                    # Get template symbolic output
                    program = get_template_program(activity)
                    sym_ids = self.vocab.encode(program.to_string())

                    # Prepare tensors
                    src = torch.tensor([imu_tokens], dtype=torch.long)
                    tgt = torch.tensor([sym_ids], dtype=torch.long)
                    src_mask = torch.ones(1, len(imu_tokens), dtype=torch.bool)
                    tgt_mask = torch.ones(1, len(sym_ids), dtype=torch.bool)

                    loss = pretrainer.train_step(src, tgt, src_mask, tgt_mask)
                    epoch_loss += loss
                    num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs} — loss: {avg_loss:.4f}")

        print("[Pretrain] Done!")
        self.rl_trainer.save_checkpoint("pretrained")

    def rl_train_step(self) -> Optional[Dict]:
        """
        Perform one PPO update if enough experience has been collected.

        Returns:
            Training metrics or None if not enough data.
        """
        if len(self.rl_trainer.buffer) < self.rl_config.batch_size:
            return None

        metrics = self.rl_trainer.ppo_update()
        print(f"[RL] Update #{self.rl_trainer.num_updates}: "
              f"loss={metrics['loss']:.4f}, reward={metrics['mean_reward']:.4f}")
        return metrics

    # ---------------------------------------------------------------
    # OpenAI Integration
    # ---------------------------------------------------------------

    def _interpret_with_openai(
        self,
        symbolic_text: str,
        memory_context: str = "",
    ) -> Tuple[str, float]:
        """
        Send symbolic output to OpenAI for interpretation and scoring.

        Returns:
            (activity_description, confidence_score)
        """
        prompt = memory_context + (
            "You are NEMESIS, an AI that interprets neuro-symbolic descriptions "
            "of IMU sensor data to determine what physical activity a person is performing.\n\n"
            "NEURO-SYMBOLIC INPUT:\n"
            f"{symbolic_text}\n\n"
            "Respond in this exact JSON format:\n"
            '{"activity": "<concise activity description>", '
            '"confidence": <0.0-1.0>, '
            '"details": "<brief explanation>"}\n'
        )

        try:
            response = self.openai_client.responses.create(
                model=self.rl_config.reward_model,
                input=prompt,
                temperature=0.3,
            )
            result = response.output_text.strip()
            return self._parse_openai_response(result)
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return "unknown activity", 0.0

    def _parse_openai_response(self, response: str) -> Tuple[str, float]:
        """Parse OpenAI JSON response."""
        import json
        try:
            # Handle markdown code blocks
            if "```" in response:
                json_str = response.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()
            else:
                json_str = response

            data = json.loads(json_str)
            activity = data.get("activity", "unknown")
            confidence = float(data.get("confidence", 0.0))
            return activity, confidence
        except (json.JSONDecodeError, ValueError):
            return response[:100], 0.3  # fallback

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _compute_embedding(self, imu_tokens: List[int]) -> np.ndarray:
        """
        Compute a simple embedding for memory storage/retrieval.
        Uses token frequency histogram.
        """
        # Histogram of token IDs (excluding special tokens)
        max_id = self.imu_config.codebook_size + self.imu_config.num_special_tokens
        hist = np.zeros(min(max_id, 256), dtype=np.float32)
        for t in imu_tokens:
            if t >= 4 and t < len(hist) + 4:  # Skip special tokens
                hist[t - 4] += 1
        # Normalize
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist

    def _try_load_checkpoint(self):
        """Try to load the latest checkpoint."""
        try:
            self.rl_trainer.load_checkpoint("latest")
        except Exception:
            try:
                self.rl_trainer.load_checkpoint("pretrained")
            except Exception:
                pass  # No checkpoint available

    def get_stats(self) -> Dict:
        """Get full pipeline statistics."""
        return {
            "memory": self.memory.get_stats(),
            "rl_updates": self.rl_trainer.num_updates,
            "rl_baseline": self.rl_trainer.baseline,
            "buffer_size": len(self.rl_trainer.buffer),
        }


# ============================================================================
# Quick Demo Function
# ============================================================================

def demo():
    """Run a quick demonstration of the NEMESIS pipeline."""
    print("=" * 60)
    print("  NEMESIS — Neural Memory-augmented Symbolic Interface")
    print("=" * 60)

    # Initialize
    pipeline = NemesisPipeline(device="cpu")
    pipeline.start_session("demo session")

    # Step 1: Supervised pretraining
    print("\n--- Phase 1: Supervised Pretraining ---")
    pipeline.supervised_pretrain(num_epochs=20, samples_per_activity=5)

    # Step 2: Translate some IMU data
    print("\n--- Phase 2: Translation ---")
    activities = ["walking", "running", "sitting", "jumping", "waving"]

    for activity in activities:
        print(f"\n[Testing: {activity}]")
        imu_data = generate_synthetic_imu(activity, duration_sec=3.0)
        result = pipeline.translate(imu_data, ground_truth=activity)

        print(f"  Symbolic: {result.symbolic_text[:80]}...")
        print(f"  Activity: {result.activity}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  From cache: {result.from_cache}")

    # Step 3: RL update
    print("\n--- Phase 3: RL Update ---")
    metrics = pipeline.rl_train_step()
    if metrics:
        print(f"  Metrics: {metrics}")

    # Step 4: Test cache hit (same activity again)
    print("\n--- Phase 4: Memory Recall ---")
    imu_data = generate_synthetic_imu("walking", duration_sec=3.0)
    result = pipeline.translate(imu_data)
    print(f"  Activity: {result.activity}")
    print(f"  From cache: {result.from_cache}")

    # Stats
    print(f"\n--- Stats ---")
    print(pipeline.get_stats())

    pipeline.end_session()
    print("\nDone!")


if __name__ == "__main__":
    demo()
