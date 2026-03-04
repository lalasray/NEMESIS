
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

def _compute_macro_f1(per_class_correct: Dict, per_class_total: Dict,
                      per_class_predicted: Dict) -> float:
    """Compute macro-averaged F1 score across all classes."""
    f1_scores = []
    all_labels = set(per_class_total.keys()) | set(per_class_predicted.keys())
    for label in all_labels:
        tp = per_class_correct.get(label, 0)
        support = per_class_total.get(label, 0)       # true count
        predicted = per_class_predicted.get(label, 0)  # predicted count
        precision = tp / max(predicted, 1)
        recall = tp / max(support, 1)
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def evaluate(
    pipeline: NemesisPipeline,
    dataset: HARDataset,
    max_samples: int = 0,
    verbose: bool = True,
    api_workers: int = 8,
    descriptor_mode: bool = False,
) -> Dict:
    """
    Evaluate the pipeline on a test set with parallel OpenAI calls.

    Primary metric: **macro F1** (handles class imbalance).

    Args:
        max_samples: 0 = use entire dataset
        api_workers: number of parallel OpenAI API threads
        descriptor_mode: if True, use VQ-VAE token descriptors instead of Translator
    """
    reward_fn = pipeline.rl_trainer.reward_fn

    n = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    dataset = dataset.shuffle(seed=99).subset(n)

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_predicted = defaultdict(int)
    results = []

    eval_batch_size = max(api_workers, 16)  # process in batches

    mode_label = "descriptor" if descriptor_mode else "translator"
    print(f"\n[Eval] Evaluating on {n} samples (mode={mode_label}, workers={api_workers})...")

    for batch_start in range(0, n, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, n)

        imu_batch = []
        gt_batch = []
        labels_batch = []
        for i in range(batch_start, batch_end):
            imu_data, description, label = dataset.get_sample(i)
            imu_batch.append(imu_data)
            gt_batch.append(description)
            labels_batch.append(label)

        if descriptor_mode:
            # VQ-VAE tokens → statistical description → LLM classify
            batch_results = pipeline.describe_and_classify_batch(
                imu_batch,
                ground_truths=gt_batch,
                max_workers=api_workers,
            )
        else:
            # Translator → symbolic text → LLM classify
            batch_results = pipeline.translate_batch(
                imu_batch,
                ground_truths=gt_batch,
                train=False,
                temperature=0.3,
                max_workers=api_workers,
            )

        for j, result in enumerate(batch_results):
            label = labels_batch[j]
            is_correct = result.confidence >= 0.8
            if is_correct:
                correct += 1
                per_class_correct[label] += 1
            per_class_total[label] += 1

            pred_label = _description_to_label(result.activity, dataset)
            per_class_predicted[pred_label] += 1
            total += 1

            results.append({
                "label": label,
                "ground_truth": gt_batch[j],
                "predicted": result.activity,
                "pred_label": pred_label,
                "reward": result.confidence,
                "symbolic": result.symbolic_text[:100],
                "correct": is_correct,
            })

        if verbose and batch_end % (eval_batch_size * 4) < eval_batch_size:
            running_f1 = _compute_macro_f1(
                per_class_correct, per_class_total, per_class_predicted)
            print(f"  [{batch_end}/{n}] Running macro-F1: {running_f1:.3f}")

    # Summary
    accuracy = correct / max(total, 1)
    macro_f1 = _compute_macro_f1(per_class_correct, per_class_total,
                                 per_class_predicted)
    metrics = {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_class": {},
    }

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Macro F1:        {macro_f1:.3f}")
    print(f"  Accuracy:        {accuracy:.1%} ({correct}/{total})")
    print(f"\n  Per-class breakdown:")
    all_labels = sorted(set(per_class_total.keys()) | set(per_class_predicted.keys()))
    for label in all_labels:
        tp = per_class_correct.get(label, 0)
        sup = per_class_total.get(label, 0)
        pred = per_class_predicted.get(label, 0)
        prec = tp / max(pred, 1)
        rec = tp / max(sup, 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics["per_class"][label] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": sup, "predicted": pred, "tp": tp,
        }
        print(f"    {label:15s}  P={prec:.2f}  R={rec:.2f}  F1={f1:.3f}  "
              f"(TP={tp}, support={sup}, predicted={pred})")

    # Show some examples
    print(f"\n  Sample predictions:")
    for r in results[:10]:
        icon = "✓" if r["correct"] else "✗"
        print(f"    {icon} GT: {r['label']:15s} → Pred: {r['pred_label']:15s} "
              f"(reward={r['reward']:.2f})")

    return metrics


def _description_to_label(description: str, dataset: HARDataset) -> str:
    """Map a predicted activity description back to the short label."""
    desc_lower = description.lower().strip()
    # Try exact match from the dataset's description→label mapping
    for i in range(min(200, len(dataset))):
        if dataset.descriptions[i].lower().strip() == desc_lower:
            return dataset.labels[i]
    # Fallback: keyword matching
    for label in sorted(set(dataset.labels)):
        if label.lower() in desc_lower:
            return label
    return "UNKNOWN"


# ============================================================================
# RL Training Loop
# ============================================================================

def rl_train_epoch(
    pipeline: NemesisPipeline,
    dataset: HARDataset,
    epoch: int,
    batch_size: int = 16,
    class_weights: Optional[Dict] = None,
    train_fraction: float = 1.0,
    api_workers: int = 8,
) -> Dict:
    """
    One epoch of RL training with **parallel OpenAI API calls**.

    Uses class-balanced sampling + batched translate for speed:
      1. Collect a micro-batch of samples
      2. Tokenize + generate symbolic text (serial, fast)
      3. Fire all OpenAI classify calls in parallel (ThreadPoolExecutor)
      4. Collect rewards, add to PPO buffer, do PPO update
    """
    from collections import Counter
    class_counts = Counter(dataset.y.tolist())
    max_class_size = max(class_counts.values())
    n_per_class = max(1, int(max_class_size * train_fraction))
    balanced_indices = dataset.class_balanced_indices(n_per_class=n_per_class, seed=epoch)
    n = len(balanced_indices)

    total_reward = 0.0
    num_updates = 0
    num_samples = 0
    rewards_list = []
    correct_count = 0

    # Process in micro-batches of batch_size for parallel API calls
    api_batch_size = batch_size  # matches PPO batch size

    print(f"\n[RL Epoch {epoch+1}] Training on {n} class-balanced samples "
          f"(workers={api_workers}, "
          f"entropy={pipeline.rl_trainer.current_entropy_coef:.4f}, "
          f"lr={pipeline.rl_trainer.optimizer.param_groups[0]['lr']:.6f})...")

    for batch_start in range(0, n, api_batch_size):
        batch_end = min(batch_start + api_batch_size, n)
        batch_indices = balanced_indices[batch_start:batch_end]

        # Gather batch data
        imu_batch = []
        gt_batch = []
        cw_batch = []
        for idx in batch_indices:
            imu_data, description, label = dataset.get_sample(idx)
            label_int = int(dataset.y[idx])
            cw = class_weights.get(label_int, 1.0) if class_weights else 1.0
            imu_batch.append(imu_data)
            gt_batch.append(description)
            cw_batch.append(cw)

        # Batched translate with parallel API calls
        results = pipeline.translate_batch(
            imu_batch,
            ground_truths=gt_batch,
            train=True,
            temperature=max(0.5, 1.0 - epoch * 0.05),
            class_weights=cw_batch,
            max_workers=api_workers,
        )

        for r in results:
            total_reward += r.confidence
            rewards_list.append(r.confidence)
            if r.confidence >= 0.8:
                correct_count += 1
            num_samples += 1

        # PPO update after each micro-batch
        if len(pipeline.rl_trainer.buffer) >= batch_size:
            metrics = pipeline.rl_train_step()
            if metrics:
                num_updates += 1
                if num_updates % max(1, (n // batch_size) // 8) == 0:
                    print(f"  [Sample {batch_end}/{n}] PPO #{num_updates}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.3f}, "
                          f"lr={metrics.get('lr', 0):.6f}")

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

    # Estimate total PPO updates for LR schedule (use full dataset)
    samples_per_epoch = len(train_data)
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
        entropy_coef=0.08,      # Higher initial exploration
        entropy_decay=0.95,     # Slow decay to prevent mode collapse
        entropy_coef_min=0.02,  # Higher floor keeps some exploration always
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
        eval_metrics = evaluate(pipeline, test_data, max_samples=args.eval_samples,
                                  api_workers=args.workers,
                                  descriptor_mode=args.descriptor_mode)
        save_metrics({"eval": eval_metrics}, args)
        pipeline.end_session()
        return

    # ----- Descriptor mode: zero-shot (no training needed) -----
    if args.descriptor_mode:
        print("\n" + "=" * 60)
        print("  DESCRIPTOR MODE — zero-shot VQ-VAE token classification")
        print("  No Translator, no RL training.")
        print("  VQ-VAE tokens → statistical description → LLM classify")
        print("=" * 60)

        # Need VQ-VAE
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
                raw_imu_list = [train_data.X[i] for i in range(len(train_data))]
                pipeline.pretrain_tokenizer(
                    raw_imu_list,
                    num_epochs=args.vqvae_epochs,
                    patience=args.vqvae_patience,
                )

        # Calibrate rewards for fair evaluation
        class_weights = train_data.get_class_weights()
        n_classes = len(set(train_data.y.tolist()))
        pipeline.rl_trainer.reward_fn.calibrate_rewards(
            n_classes=n_classes,
            class_weights=class_weights,
        )

        # Show a sample description for the first test sample
        print("\n--- Sample Token Description ---")
        sample_imu, sample_desc, sample_label = test_data.get_sample(0)
        sample_tokens = pipeline.tokenizer.tokenize(sample_imu)
        from nemesis.token_descriptor import TokenDescriptor
        desc = TokenDescriptor(codebook_size=imu_config.codebook_size)
        print(desc.describe(sample_tokens))
        print(f"\n(Ground truth: {sample_label} — {sample_desc})")

        # Evaluate
        print("\n--- Descriptor Mode Evaluation ---")
        eval_metrics = evaluate(
            pipeline, test_data,
            max_samples=args.eval_samples,
            verbose=True,
            api_workers=args.workers,
            descriptor_mode=True,
        )

        results = {
            "args": vars(args),
            "mode": "descriptor",
            "eval": eval_metrics,
        }
        save_metrics(results, args)
        pipeline.end_session()
        print("\nDescriptor mode evaluation complete!")
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
                patience=args.vqvae_patience,
            )

    # ----- Phase 1: Warm start -----
    if not args.skip_warmup:
        print("\n--- Phase 1: Warm Start (syntax only) ---")
        pipeline.warm_start(
            num_epochs=args.warmup_epochs,
            samples_per_epoch=args.warmup_samples,
        )

    # ----- Phase 2: RL training (with early stopping on macro F1) -----
    print("\n--- Phase 2: RL Training ---")
    print(f"  Full dataset per epoch: {len(train_data)} samples")
    print(f"  Max epochs: {args.rl_epochs}, eval every {args.eval_every}, "
          f"patience: {args.patience}")

    # Class weights for balanced reward scaling
    class_weights = train_data.get_class_weights()
    print(f"  Class weights (inverse frequency, capped):")
    for cls_int, weight in sorted(class_weights.items()):
        # Map int label to name
        matching = [train_data.labels[i] for i in range(len(train_data)) if train_data.y[i] == cls_int]
        name = matching[0] if matching else f"class_{cls_int}"
        print(f"    {name:15s}: {weight:.2f}")

    # Auto-calibrate wrong_reward so no single-class strategy is profitable
    n_classes = len(set(train_data.y.tolist()))
    pipeline.rl_trainer.reward_fn.calibrate_rewards(
        n_classes=n_classes,
        class_weights=class_weights,
    )

    all_epoch_metrics = []
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(args.rl_epochs):
        epoch_metrics = rl_train_epoch(
            pipeline,
            train_data,
            epoch=epoch,
            batch_size=args.batch_size,
            class_weights=class_weights,
            train_fraction=args.train_fraction,
            api_workers=args.workers,
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
                api_workers=args.workers,
            )
            all_epoch_metrics[-1]["eval"] = eval_metrics
            current_f1 = eval_metrics["macro_f1"]

            # Early stopping check
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                pipeline.rl_trainer.save_checkpoint("best")
                print(f"  [EarlyStop] New best macro-F1: {best_f1:.3f} — saved 'best' checkpoint")
            else:
                patience_counter += 1
                print(f"  [EarlyStop] No improvement ({current_f1:.3f} <= {best_f1:.3f}), "
                      f"patience {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"\n  *** Early stopping triggered after epoch {epoch+1} ***")
                print(f"  *** Best macro-F1: {best_f1:.3f} ***")
                break

    # ----- Phase 3: Final evaluation (reload best if available) -----
    best_ckpt = os.path.join(CHECKPOINTS_DIR, "translator_best.pt")
    if os.path.isfile(best_ckpt):
        print(f"\n--- Loading best checkpoint (macro-F1={best_f1:.3f}) ---")
        pipeline.rl_trainer.load_checkpoint("best")
    print("\n--- Final Evaluation ---")
    final_metrics = evaluate(pipeline, test_data, max_samples=args.eval_samples,
                              api_workers=args.workers)

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
    parser.add_argument("--rl-epochs", type=int, default=30,
                        help="Max RL training epochs (early stopping may end sooner)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PPO batch size")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO mini-epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience (eval rounds without F1 improvement)")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of dataset per RL epoch (0.1 = 10%%, class-balanced)")

    # Evaluation
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="Eval samples (0 = use entire test set)")
    parser.add_argument("--eval-every", type=int, default=1,
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
    parser.add_argument("--vqvae-epochs", type=int, default=200,
                        help="Max VQ-VAE pre-training epochs (early stopping included)")
    parser.add_argument("--vqvae-patience", type=int, default=15,
                        help="VQ-VAE early stopping patience (epochs without recon improvement)")
    parser.add_argument("--retrain-vqvae", action="store_true",
                        help="Force re-train VQ-VAE even if checkpoint exists")

    # Descriptor mode (bypass Translator entirely)
    parser.add_argument("--descriptor-mode", action="store_true",
                        help="Use VQ-VAE token descriptors instead of Translator. "
                             "Zero-shot: VQ-VAE tokens → statistical text → LLM classify. "
                             "No RL training needed.")

    # Parallelism
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers for OpenAI calls")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
