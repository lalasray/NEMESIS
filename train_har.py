
"""Train NEMESIS on a real HAR dataset (UCI HAR, WISDM, or Opportunity).

Usage:
    # Train on UCI HAR (default)
    python train_har.py

    # Train on UCI HAR with custom settings
    python train_har.py --dataset uci_har --rl-epochs 5 --warmup-epochs 30

    # Train on WISDM
    python train_har.py --dataset wisdm

    # Train on Opportunity (variable-length, 6-axis IMU)
    python train_har.py --dataset opportunity --use-vqvae

    # Evaluate only (skip training)
    python train_har.py --eval-only --checkpoint latest

Training flow:
    1. Download + load the HAR dataset
    2. Warm-start: teach Translator to produce valid symbolic syntax
    3. RL training loop:
       - For each IMU sample + ground truth activity label:
         a) Tokenize IMU → Translator → symbolic text
         b) OpenAI interprets symbolic text → predicted activity
         c) Reward = semantic match(predicted, ground truth)
         d) PPO update on accumulated batch
    4. Evaluate on test set (no training, measure accuracy)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch

from nemesis.config import (
    IMUTokenizerConfig, TranslatorConfig, RLConfig, MemoryConfig,
    CHECKPOINTS_DIR, PROJECT_ROOT,
)
from nemesis.datasets import (
    load_uci_har, load_wisdm, load_opportunity, print_dataset_info, HARDataset,
)
from nemesis.imu_tokenizer import BinningTokenizer, VQVAE_Tokenizer
from nemesis.neuro_symbolic import SymbolicVocab
from nemesis.translator import TranslatorModel
from nemesis.rl_trainer import PPOTrainer, SupervisedPretrainer, OpenAIRewardFunction
from nemesis.memory import MemoryManager
from nemesis.pipeline import NemesisPipeline


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    pipeline: NemesisPipeline,
    dataset: HARDataset,
    max_samples: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate the pipeline on a test set.

    For each sample:
      1. Translator generates symbolic text
      2. OpenAI classifies it (picks from known activities)
      3. Compare classification to ground truth

    Returns:
        Dictionary with accuracy, per-class accuracy, and confusion info
    """
    reward_fn = pipeline.rl_trainer.reward_fn

    n = min(max_samples, len(dataset))
    dataset = dataset.shuffle(seed=99).subset(n)

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    results = []

    print(f"\n[Eval] Evaluating on {n} samples from {dataset.dataset_name}/{dataset.split}...")

    for i in range(n):
        imu_data, description, label = dataset.get_sample(i)

        # Translate (no training, no memory writes)
        result = pipeline.translate(
            imu_data,
            ground_truth=description,
            use_memory=False,
            train=False,
            temperature=0.3,  # lower temperature for eval (more deterministic)
        )

        # Check if prediction matches — use classification reward threshold
        is_correct = result.confidence >= 0.8  # stricter threshold for classify mode
        if is_correct:
            correct += 1
            per_class_correct[label] += 1
        per_class_total[label] += 1
        total += 1

        results.append({
            "label": label,
            "ground_truth": description,
            "predicted": result.activity,
            "reward": result.confidence,
            "symbolic": result.symbolic_text[:100],
            "correct": is_correct,
        })

        if verbose and (i + 1) % 10 == 0:
            acc_so_far = correct / total
            print(f"  [{i+1}/{n}] Running accuracy: {acc_so_far:.1%}")

        # Rate limit for OpenAI
        time.sleep(0.1)

    # Summary
    accuracy = correct / max(total, 1)
    metrics = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_class": {},
    }

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Overall accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"\n  Per-class accuracy:")
    for label in sorted(per_class_total.keys()):
        cls_correct = per_class_correct[label]
        cls_total = per_class_total[label]
        cls_acc = cls_correct / max(cls_total, 1)
        metrics["per_class"][label] = {
            "accuracy": cls_acc,
            "correct": cls_correct,
            "total": cls_total,
        }
        print(f"    {label:25s}: {cls_acc:.1%} ({cls_correct}/{cls_total})")

    # Show some examples
    print(f"\n  Sample predictions:")
    for r in results[:10]:
        icon = "✓" if r["correct"] else "✗"
        print(f"    {icon} GT: {r['label']:20s} → Predicted: {r['predicted'][:30]:30s} "
              f"(reward={r['reward']:.2f})")

    return metrics


# ============================================================================
# RL Training Loop
# ============================================================================

def rl_train_epoch(
    pipeline: NemesisPipeline,
    dataset: HARDataset,
    epoch: int,
    max_samples: int = 200,
    batch_size: int = 16,
) -> Dict:
    """
    One epoch of RL training on the dataset.

    v2: Uses classification-style reward (single API call per sample),
    entropy decay, and better logging.
    """
    n = min(max_samples, len(dataset))
    dataset_shuffled = dataset.shuffle(seed=epoch)

    total_reward = 0.0
    num_updates = 0
    num_samples = 0
    rewards_list = []
    correct_count = 0

    print(f"\n[RL Epoch {epoch+1}] Training on {n} samples "
          f"(entropy_coef={pipeline.rl_trainer.current_entropy_coef:.4f}, "
          f"lr={pipeline.rl_trainer.optimizer.param_groups[0]['lr']:.6f})...")

    for i in range(n):
        imu_data, description, label = dataset_shuffled.get_sample(i)

        # Translate with training enabled
        result = pipeline.translate(
            imu_data,
            ground_truth=description,
            use_memory=True,
            train=True,
            temperature=max(0.5, 1.0 - epoch * 0.1),  # decay temperature over epochs
        )

        total_reward += result.confidence
        rewards_list.append(result.confidence)
        if result.confidence >= 0.8:
            correct_count += 1
        num_samples += 1

        # PPO update when buffer is full
        if len(pipeline.rl_trainer.buffer) >= batch_size:
            metrics = pipeline.rl_train_step()
            if metrics:
                num_updates += 1
                if num_updates % 3 == 0:
                    print(f"  [Sample {i+1}/{n}] PPO #{num_updates}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.3f}, "
                          f"lr={metrics.get('lr', 0):.6f}")

        # Rate limit for OpenAI API (lighter since single call now)
        time.sleep(0.08)

    # Final PPO update with remaining buffer
    if len(pipeline.rl_trainer.buffer) >= 4:
        pipeline.rl_trainer.config.batch_size = len(pipeline.rl_trainer.buffer)
        metrics = pipeline.rl_train_step()
        if metrics:
            num_updates += 1
        pipeline.rl_trainer.config.batch_size = batch_size

    # Decay entropy coefficient after each epoch
    pipeline.rl_trainer.decay_entropy()

    avg_reward = total_reward / max(num_samples, 1)
    train_acc = correct_count / max(num_samples, 1)
    epoch_metrics = {
        "epoch": epoch + 1,
        "avg_reward": avg_reward,
        "train_accuracy": train_acc,
        "total_samples": num_samples,
        "ppo_updates": num_updates,
        "reward_std": float(np.std(rewards_list)) if rewards_list else 0,
        "entropy_coef": pipeline.rl_trainer.current_entropy_coef,
    }

    print(f"  [Epoch {epoch+1} done] avg_reward={avg_reward:.3f}, "
          f"train_acc={train_acc:.1%}, "
          f"ppo_updates={num_updates}, samples={num_samples}")

    return epoch_metrics


# ============================================================================
# Main Training Flow
# ============================================================================

def train(args):
    """Full training pipeline."""

    # ----- Load dataset -----
    print("\n" + "=" * 60)
    print("  NEMESIS — Training on HAR Dataset")
    print("=" * 60)

    if args.dataset == "uci_har":
        train_data = load_uci_har(split="train", description_style="standard")
        test_data = load_uci_har(split="test", description_style="standard")
    elif args.dataset == "wisdm":
        train_data = load_wisdm(split="train")
        test_data = load_wisdm(split="test")
    elif args.dataset == "opportunity":
        train_data = load_opportunity(split="train", description_style="rich")
        test_data = load_opportunity(split="test", description_style="rich")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print_dataset_info(train_data)
    print_dataset_info(test_data)

    # ----- Configure for dataset -----
    num_channels = train_data.num_channels
    imu_config = IMUTokenizerConfig(
        num_channels=num_channels,
        window_size=25,
        window_overlap=5,
        sampling_rate=train_data.sampling_rate,
    )

    # Estimate total PPO updates for LR schedule
    samples_per_epoch = args.train_samples_per_epoch
    updates_per_epoch = max(1, samples_per_epoch // args.batch_size)
    total_updates = updates_per_epoch * args.rl_epochs + 5  # +5 for warmup

    rl_config = RLConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        reward_model=args.reward_model,
        total_updates=total_updates,
        warmup_steps=5,
        classify_reward=True,  # Single API call classification
        normalize_rewards=True,
        entropy_coef=0.05,
        entropy_decay=0.85,
    )

    # ----- Initialize pipeline -----
    pipeline = NemesisPipeline(
        imu_config=imu_config,
        rl_config=rl_config,
        device=args.device,
        use_vqvae=args.use_vqvae,
    )

    # Set activity options for classification-style reward
    activity_descriptions = list(set(train_data.descriptions))
    pipeline.set_activity_options(activity_descriptions)

    # Set sensor context so OpenAI prompts include physical meaning
    channel_names = train_data.channels

    if train_data.is_variable_length:
        length_stats = train_data.get_length_stats()
        mean_window = length_stats['mean']
        window_duration = mean_window / train_data.sampling_rate
        window_dur_label = f"{window_duration:.1f}s (variable)"
    else:
        window_samples = train_data.X.shape[1]
        window_duration = window_samples / train_data.sampling_rate
        window_dur_label = f"{window_duration:.2f}s"

    pipeline.set_sensor_context(
        num_channels=num_channels,
        channel_names=channel_names,
        sampling_rate=train_data.sampling_rate,
        window_duration_sec=window_duration,
    )
    print(f"[Context] {len(channel_names)} channels, "
          f"{train_data.sampling_rate}Hz, {window_dur_label} windows")

    pipeline.start_session(f"training_{args.dataset}")

    # ----- Load checkpoint if specified -----
    if args.checkpoint:
        pipeline.rl_trainer.load_checkpoint(args.checkpoint)

    # ----- Eval only mode -----
    if args.eval_only:
        print("\n--- Evaluation Only ---")
        eval_metrics = evaluate(pipeline, test_data, max_samples=args.eval_samples)
        save_metrics({"eval": eval_metrics}, args)
        pipeline.end_session()
        return

    # ----- Phase 0: VQ-VAE pre-training (if enabled) -----
    if args.use_vqvae:
        vqvae_ckpt = os.path.join(CHECKPOINTS_DIR, "vqvae_pretrained.pt")
        if os.path.isfile(vqvae_ckpt) and not args.retrain_vqvae:
            print(f"\n--- Loading pre-trained VQ-VAE from {vqvae_ckpt} ---")
            pipeline.tokenizer = VQVAE_Tokenizer.load_pretrained(
                vqvae_ckpt, device=args.device
            )
            print("  VQ-VAE loaded successfully.")
        else:
            print("\n--- Phase 0: VQ-VAE Pre-training ---")
            # Collect raw IMU arrays for unsupervised training
            raw_imu_list = [train_data.X[i] for i in range(len(train_data))]
            pipeline.pretrain_tokenizer(
                raw_imu_list,
                num_epochs=args.vqvae_epochs,
            )

    # ----- Phase 1: Warm start -----
    if not args.skip_warmup:
        print("\n--- Phase 1: Warm Start (syntax only) ---")
        pipeline.warm_start(
            num_epochs=args.warmup_epochs,
            samples_per_epoch=args.warmup_samples,
        )

    # ----- Phase 2: RL training -----
    print("\n--- Phase 2: RL Training ---")
    all_epoch_metrics = []

    for epoch in range(args.rl_epochs):
        epoch_metrics = rl_train_epoch(
            pipeline,
            train_data,
            epoch=epoch,
            max_samples=args.train_samples_per_epoch,
            batch_size=args.batch_size,
        )
        all_epoch_metrics.append(epoch_metrics)

        # Save checkpoint after each epoch
        pipeline.rl_trainer.save_checkpoint(f"epoch_{epoch+1}")
        pipeline.rl_trainer.save_checkpoint("latest")

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0:
            print(f"\n--- Evaluation after epoch {epoch+1} ---")
            eval_metrics = evaluate(
                pipeline, test_data,
                max_samples=args.eval_samples,
                verbose=True,
            )
            all_epoch_metrics[-1]["eval"] = eval_metrics

    # ----- Phase 3: Final evaluation -----
    print("\n--- Final Evaluation ---")
    final_metrics = evaluate(pipeline, test_data, max_samples=args.eval_samples)

    # ----- Save results -----
    results = {
        "args": vars(args),
        "epoch_metrics": all_epoch_metrics,
        "final_eval": final_metrics,
        "memory_stats": pipeline.get_stats(),
    }
    save_metrics(results, args)

    pipeline.end_session()
    print("\nTraining complete!")


def save_metrics(results: Dict, args):
    """Save training metrics to JSON."""
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(PROJECT_ROOT, "results", f"{args.dataset}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Results] Saved to {path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NEMESIS on a public HAR dataset"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="uci_har",
                        choices=["uci_har", "wisdm", "opportunity"],
                        help="Which dataset to use")

    # Training
    parser.add_argument("--warmup-epochs", type=int, default=30,
                        help="Warm-start epochs (syntax learning)")
    parser.add_argument("--warmup-samples", type=int, default=50,
                        help="Samples per warm-start epoch")
    parser.add_argument("--rl-epochs", type=int, default=8,
                        help="Number of RL training epochs")
    parser.add_argument("--train-samples-per-epoch", type=int, default=200,
                        help="Training samples per RL epoch")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PPO batch size")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO mini-epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")

    # Evaluation
    parser.add_argument("--eval-samples", type=int, default=60,
                        help="Number of test samples for evaluation")
    parser.add_argument("--eval-every", type=int, default=2,
                        help="Evaluate every N RL epochs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate")

    # Model
    parser.add_argument("--reward-model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model for reward computation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint tag (e.g. 'latest', 'epoch_3')")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip warm-start phase")

    # VQ-VAE tokenizer
    parser.add_argument("--use-vqvae", action="store_true",
                        help="Use VQ-VAE tokenizer instead of binning")
    parser.add_argument("--vqvae-epochs", type=int, default=50,
                        help="VQ-VAE pre-training epochs")
    parser.add_argument("--retrain-vqvae", action="store_true",
                        help="Force re-train VQ-VAE even if checkpoint exists")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
