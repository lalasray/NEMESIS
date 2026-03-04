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
        self.use_vqvae = use_vqvae
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
        tok_name = 'VQ-VAE' if use_vqvae else 'Binning'
        if use_vqvae:
            tok_name += ' (NEEDS PRE-TRAINING)' if not self.tokenizer.is_trained else ' (trained)'
        print(f"  IMU Tokenizer: {tok_name}")
        print(f"  Translator: {sum(p.numel() for p in self.translator.parameters()):,} params")
        print(f"  Symbolic vocab: {self.vocab.size} tokens")
        print(f"  Memory: {self.memory.get_stats()}")
        print(f"  Device: {device}")

    def set_activity_options(self, options: List[str]):
        """
        Set the valid activity descriptions for classification-style reward.
        Must be called before training for classify_reward mode.
        """
        self.rl_trainer.reward_fn.set_activity_options(options)
        print(f"[NEMESIS] Activity options set: {len(options)} classes")

    def set_sensor_context(
        self,
        num_channels: int = 6,
        channel_names: List[str] = None,
        sampling_rate: int = 50,
        window_duration_sec: float = 2.56,
    ):
        """
        Set IMU sensor context that gets included in OpenAI prompts.
        This helps OpenAI understand what the symbolic text represents.
        """
        if channel_names is None:
            channel_names = [f"ch_{i}" for i in range(num_channels)]
        self._sensor_context = {
            "num_channels": num_channels,
            "channel_names": channel_names,
            "sampling_rate": sampling_rate,
            "window_duration_sec": window_duration_sec,
        }
        # Propagate to reward function
        self.rl_trainer.reward_fn.set_sensor_context(self._sensor_context)
        print(f"[NEMESIS] Sensor context set: {num_channels}ch @ {sampling_rate}Hz, "
              f"{window_duration_sec:.2f}s windows")

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

        This MUST be called before RL training when use_vqvae=True.
        It learns a codebook of motion primitives by reconstructing
        IMU signals — no labels needed.

        Uses the reference architecture:
          - Channels-first Conv1d encoder with residual blocks
          - VectorQuantizerEMA with decay=0.99
          - Reconstruction loss normalised by data variance
          - Perplexity tracking for codebook health

        Args:
            imu_data_list: List of (T, C) raw IMU sequences from the dataset
            num_epochs: VQ-VAE training epochs
            batch_size: Batch size (reference uses 1024, reduced for CPU)
            lr: Learning rate
        """
        if not self.use_vqvae:
            print("[NEMESIS] Skipping VQ-VAE pre-training (using BinningTokenizer)")
            return

        if not isinstance(self.tokenizer, VQVAE_Tokenizer):
            raise TypeError("Tokenizer is not a VQ-VAE — can't pre-train")

        print("\n--- VQ-VAE Pre-training (unsupervised) ---")
        print(f"  Learning motion primitives from {len(imu_data_list)} IMU sequences")
        print(f"  This is UNSUPERVISED — no labels needed, just reconstruction")

        metrics = self.tokenizer.train_vqvae(
            imu_data_list=imu_data_list,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            device=self.device,
            patience=patience,
        )

        # Save the VQ-VAE weights
        vqvae_path = os.path.join(CHECKPOINTS_DIR, "vqvae_pretrained.pt")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        self.tokenizer.save_pretrained(vqvae_path)

        return metrics

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
        **kwargs,
    ) -> TranslationResult:
        """
        Full pipeline: IMU data → activity description.

        Args:
            imu_data: (T, C) raw IMU sensor data
            ground_truth: optional activity label (for RL reward)
            use_memory: whether to use memory for context/caching
            train: whether to collect RL experience
            temperature: generation temperature
            **kwargs: passed through (e.g. class_weight for reward scaling)

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

        # Step 4: Get activity prediction
        # In classify mode: single API call picks from known activities
        # In interpret mode: free-form interpretation
        if (self.rl_config.classify_reward
                and self.rl_trainer.reward_fn.activity_options):
            activity = self.rl_trainer.reward_fn._classify_symbolic(symbolic_text)
        else:
            activity = self._interpret_with_openai(symbolic_text, memory_context)

        # Step 5: Compute reward (with optional class weighting)
        if ground_truth is not None:
            class_weight = kwargs.get('class_weight', 1.0)
            if self.rl_config.classify_reward:
                confidence = self.rl_trainer.reward_fn._compute_classification_reward(
                    activity, ground_truth, class_weight=class_weight
                )
            else:
                confidence = self.rl_trainer.reward_fn._compute_match_reward(
                    activity, ground_truth
                )
        else:
            confidence = 0.1 if self.rl_trainer.reward_fn._is_parseable(symbolic_text) else 0.0

        # Step 6: Store in memory
        if use_memory:
            embedding = self._compute_embedding(imu_tokens)
            self.memory.store_translation(
                imu_tokens=imu_tokens,
                symbolic_text=symbolic_text,
                activity=activity,
                confidence=confidence,
                embedding=embedding,
            )

        # Step 7: Collect RL experience
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

    def translate_batch(
        self,
        imu_batch: List[np.ndarray],
        ground_truths: Optional[List[str]] = None,
        train: bool = True,
        temperature: float = 1.0,
        class_weights: Optional[List[float]] = None,
        max_workers: int = 8,
    ) -> List[TranslationResult]:
        """
        Batched pipeline: processes multiple IMU segments with parallel OpenAI calls.

        Steps:
          1. Tokenize all samples (serial, fast)
          2. Generate symbolic text for all (serial, local model)
          3. Classify all with OpenAI in PARALLEL (ThreadPoolExecutor)
          4. Compute rewards and collect RL experience

        This is ~8x faster than serial translate() for I/O bound API calls.
        """
        B = len(imu_batch)
        if ground_truths is None:
            ground_truths = [None] * B
        if class_weights is None:
            class_weights = [1.0] * B

        # --- Step 1+2: Tokenize and generate symbolic (serial, fast) ---
        all_imu_tokens = []
        all_src_ids = []
        all_generated = []
        all_log_probs = []
        all_symbolic = []

        self.translator.eval()
        for imu_data in imu_batch:
            imu_tokens = self.tokenizer.tokenize(imu_data)
            src_ids = torch.tensor([imu_tokens], dtype=torch.long)
            src_mask = torch.ones(1, len(imu_tokens), dtype=torch.bool)

            generated, log_probs = self.translator.generate(
                src_ids.to(self.device),
                src_mask.to(self.device),
                temperature=temperature,
                bos_id=self.vocab.bos_id,
                eos_id=self.vocab.eos_id,
                sample=(temperature > 0),
            )
            symbolic_text = self.vocab.decode(generated[0].tolist())

            all_imu_tokens.append(imu_tokens)
            all_src_ids.append(src_ids)
            all_generated.append(generated)
            all_log_probs.append(log_probs)
            all_symbolic.append(symbolic_text)

        # --- Step 3: Parallel OpenAI classification ---
        reward_fn = self.rl_trainer.reward_fn
        if self.rl_config.classify_reward and reward_fn.activity_options:
            activities = reward_fn.classify_batch_parallel(all_symbolic, max_workers=max_workers)
        else:
            # Fallback: serial interpretation
            activities = [self._interpret_with_openai(s, "") for s in all_symbolic]

        # --- Step 4: Compute rewards and collect experience ---
        results = []
        for i in range(B):
            gt = ground_truths[i]
            cw = class_weights[i]
            activity = activities[i]
            symbolic_text = all_symbolic[i]
            imu_tokens = all_imu_tokens[i]

            if gt is not None:
                if self.rl_config.classify_reward:
                    confidence = reward_fn._compute_classification_reward(
                        activity, gt, class_weight=cw)
                else:
                    confidence = reward_fn._compute_match_reward(activity, gt)
            else:
                confidence = 0.1 if reward_fn._is_parseable(symbolic_text) else 0.0

            if train:
                from nemesis.rl_trainer import Experience
                exp = Experience(
                    src_ids=all_src_ids[i][0].cpu(),
                    tgt_ids=all_generated[i][0].cpu(),
                    log_probs=all_log_probs[i][0].cpu(),
                    reward=confidence,
                    activity=activity,
                    symbolic_text=symbolic_text,
                )
                self.rl_trainer.buffer.add(exp)

            results.append(TranslationResult(
                symbolic_text=symbolic_text,
                activity=activity,
                confidence=confidence,
                from_cache=False,
                imu_tokens=imu_tokens,
                memory_context="",
            ))

        return results

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def warm_start(
        self,
        num_epochs: int = 30,
        samples_per_epoch: int = 50,
    ):
        """
        Warm-start the Translator to produce syntactically valid symbolic
        tokens (but NOT semantically correct ones).

        This does NOT teach the model what the "correct" symbolic output is
        — we don't have ground truth for that. It only teaches:
          - Generate tokens from the symbolic vocabulary
          - Produce PREDICATE(key=value) structure
          - End with EOS

        The actual semantic meaning is learned via RL (OpenAI reward).

        Uses randomly shuffled templates as diverse structural targets.
        """
        import random
        from nemesis.neuro_symbolic import (
            SymbolicStatement, SymbolicProgram,
            ACTIVITY_TEMPLATES,
        )

        pretrainer = SupervisedPretrainer(
            self.translator, self.vocab, lr=1e-3, device=self.device
        )

        # Collect ALL template statements (from every activity)
        all_statements = []
        for stmts in ACTIVITY_TEMPLATES.values():
            all_statements.extend(stmts)

        activities = list(ACTIVITY_TEMPLATES.keys())

        print(f"[WarmStart] Teaching syntactic structure: {num_epochs} epochs")
        print(f"  NOTE: This only teaches token structure, not semantic meaning.")
        print(f"  Semantic learning happens during RL with OpenAI reward.")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for _ in range(samples_per_epoch):
                # Pick a random activity for IMU generation
                activity = random.choice(activities)
                imu_data = generate_synthetic_imu(activity, duration_sec=3.0)
                imu_tokens = self.tokenizer.tokenize(imu_data)

                # Create a RANDOM symbolic program (shuffled statements)
                # This teaches structure, not correct mapping
                num_stmts = random.randint(2, 5)
                stmts = random.sample(all_statements, min(num_stmts, len(all_statements)))
                program = SymbolicProgram(statements=stmts)
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

        print("[WarmStart] Done! Model can now produce structured symbolic tokens.")
        print("  Run rl_train_step() to teach it the CORRECT symbolic mappings.")
        self.rl_trainer.save_checkpoint("warmstart")

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
    ) -> str:
        """
        Send symbolic output to OpenAI for interpretation with sensor context.

        OpenAI NEVER sees the ground truth. It only interprets the
        symbolic text and predicts what activity it describes.
        """
        # Build sensor context
        ctx = getattr(self, '_sensor_context', None)
        if ctx:
            channels_str = ", ".join(ctx.get("channel_names", []))
            sensor_intro = (
                f"A person is wearing an IMU sensor ({ctx['num_channels']} channels: "
                f"{channels_str}) sampled at {ctx['sampling_rate']}Hz. "
                f"The following neuro-symbolic description was generated from "
                f"{ctx['window_duration_sec']:.1f} seconds of their sensor data.\n\n"
            )
        else:
            sensor_intro = ""

        prompt = memory_context + sensor_intro + (
            "You are NEMESIS, an AI that interprets neuro-symbolic descriptions "
            "of IMU sensor data to determine what physical activity a person is performing.\n\n"
            "NEURO-SYMBOLIC DESCRIPTION:\n"
            f"{symbolic_text}\n\n"
            "Based ONLY on the symbolic description above, what physical activity "
            "is the person most likely performing?\n\n"
            "Respond with ONLY a short activity label (1-5 words), nothing else.\n"
            "Examples: walking, running, sitting still, jumping rope, waving hand"
        )

        try:
            response = self.openai_client.responses.create(
                model=self.rl_config.reward_model,
                input=prompt,
                temperature=0.3,
            )
            return response.output_text.strip().lower()
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return "unknown"

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

    # Step 1: Warm start (teaches syntax, NOT semantics)
    print("\n--- Phase 1: Warm Start (syntactic structure only) ---")
    pipeline.warm_start(num_epochs=20, samples_per_epoch=20)

    # Step 2: Translate some IMU data
    # Ground truth is the ACTIVITY LABEL, not the symbolic output
    print("\n--- Phase 2: Translation + RL Collection ---")
    print("  (OpenAI interprets symbolic output → compared to ground truth label)")
    activities = ["walking", "running", "sitting", "jumping", "waving"]

    for activity in activities:
        print(f"\n[Testing: {activity}]")
        imu_data = generate_synthetic_imu(activity, duration_sec=3.0)

        # ground_truth = activity label (the only supervision we have)
        result = pipeline.translate(imu_data, ground_truth=activity)

        print(f"  Symbolic (latent):  {result.symbolic_text[:80]}...")
        print(f"  OpenAI predicted:   {result.activity}")
        print(f"  Reward (match GT):  {result.confidence:.2f}")
        print(f"  From cache:         {result.from_cache}")

    # Step 3: RL update — uses accumulated rewards to improve Translator
    print("\n--- Phase 3: RL Update ---")
    print("  Updating Translator so its symbolic outputs lead OpenAI to correct activities")
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
