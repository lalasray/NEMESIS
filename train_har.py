"""Train NEMESIS on a real HAR dataset (UCI HAR or Opportunity).

Usage:
    # Opportunity with VQ-VAE (recommended)
    python train_har.py --dataset opportunity --workers 8

    # UCI HAR
    python train_har.py --dataset uci_har --workers 8

    # Quick test (100 samples)
    python train_har.py --dataset opportunity --eval-samples 100 --workers 8

Pipeline:
    1. Download + load the HAR dataset
    2. Pre-train VQ-VAE on raw IMU data (unsupervised)
    3. Tokenize test samples → statistical descriptions → LLM classify
    4. Report macro F1, per-class precision/recall/F1
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from nemesis.config import (
    IMUTokenizerConfig, ClassifierConfig, MemoryConfig,
    CHECKPOINTS_DIR, PROJECT_ROOT, MEMORY_DIR,
)
from nemesis.datasets import (
    load_uci_har, load_opportunity, print_dataset_info, HARDataset,
)
from nemesis.imu_tokenizer import VQVAE_Tokenizer
from nemesis.token_descriptor import TokenDescriptor
from nemesis.pipeline import NemesisPipeline


# ============================================================================
# Evaluation
# ============================================================================

def _compute_macro_f1(
    per_class_correct: Dict,
    per_class_total: Dict,
    per_class_predicted: Dict,
) -> float:
    """Compute macro-averaged F1 score across all classes."""
    f1_scores = []
    all_labels = set(per_class_total.keys()) | set(per_class_predicted.keys())
    for label in all_labels:
        tp = per_class_correct.get(label, 0)
        support = per_class_total.get(label, 0)
        predicted = per_class_predicted.get(label, 0)
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
) -> Dict:
    """
    Evaluate the pipeline on a test set with parallel OpenAI calls.

    Primary metric: **macro F1** (handles class imbalance).

    Args:
        max_samples: 0 = use entire dataset
        api_workers: number of parallel OpenAI API threads
    """
    n = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    dataset = dataset.shuffle(seed=99).subset(n)

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_predicted = defaultdict(int)
    results = []

    eval_batch_size = max(api_workers, 16)

    print(f"\n[Eval] Evaluating on {n} samples (workers={api_workers})...")

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

        batch_results = pipeline.classify_batch(
            imu_batch,
            ground_truths=gt_batch,
            max_workers=api_workers,
        )

        for j, result in enumerate(batch_results):
            label = labels_batch[j]
            is_correct = result.reward >= 0.8
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
                "reward": result.reward,
                "descriptor": result.descriptor_text[:100],
                "correct": is_correct,
            })

        if verbose and batch_end % (eval_batch_size * 4) < eval_batch_size:
            running_f1 = _compute_macro_f1(
                per_class_correct, per_class_total, per_class_predicted)
            print(f"  [{batch_end}/{n}] Running macro-F1: {running_f1:.3f}")

    # Summary
    accuracy = correct / max(total, 1)
    macro_f1 = _compute_macro_f1(per_class_correct, per_class_total, per_class_predicted)
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

    # Sample predictions
    print(f"\n  Sample predictions:")
    for r in results[:10]:
        icon = "✓" if r["correct"] else "✗"
        print(f"    {icon} GT: {r['label']:15s} → Pred: {r['pred_label']:15s} "
              f"(reward={r['reward']:.2f})")

    return metrics


def _description_to_label(description: str, dataset: HARDataset) -> str:
    """Map a predicted activity description back to the short label."""
    desc_lower = description.lower().strip()
    for i in range(min(200, len(dataset))):
        if dataset.descriptions[i].lower().strip() == desc_lower:
            return dataset.labels[i]
    for label in sorted(set(dataset.labels)):
        if label.lower() in desc_lower:
            return label
    return "UNKNOWN"


# ============================================================================
# Main
# ============================================================================

def run(args):
    """Main training + evaluation flow."""

    print("\n" + "=" * 60)
    print("  NEMESIS — Few-Shot Memory Mode")
    print("  VQ-VAE tokens → memory query → few-shot LLM classify")
    print("=" * 60)

    # ----- Load dataset -----
    if args.dataset == "uci_har":
        train_data = load_uci_har(split="train", description_style="standard")
        test_data = load_uci_har(split="test", description_style="standard")
    elif args.dataset == "opportunity":
        train_data = load_opportunity(split="train", description_style="rich")
        test_data = load_opportunity(split="test", description_style="rich")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print_dataset_info(train_data)
    print_dataset_info(test_data)

    # ----- Configure -----
    num_channels = train_data.num_channels
    imu_config = IMUTokenizerConfig(
        num_channels=num_channels,
        window_size=25,
        window_overlap=5,
        sampling_rate=train_data.sampling_rate,
    )

    classifier_config = ClassifierConfig(
        model=args.model,
        correct_reward=1.0,
        wrong_reward=-1.0,
        partial_reward=0.3,
    )

    memory_config = MemoryConfig(
        db_path=os.path.join(MEMORY_DIR, f"{args.dataset}_memory.db"),
        codebook_size=imu_config.codebook_size,
        top_k=args.top_k,
        store_threshold=0.6,
        promote_threshold=0.85,
    )

    # ----- Initialise pipeline -----
    pipeline = NemesisPipeline(
        imu_config=imu_config,
        classifier_config=classifier_config,
        memory_config=memory_config,
        device=args.device,
    )

    # Activity options
    activity_descriptions = list(set(train_data.descriptions))
    pipeline.set_activity_options(activity_descriptions)

    # Sensor context
    channel_names = train_data.channels
    if train_data.is_variable_length:
        length_stats = train_data.get_length_stats()
        window_duration = length_stats['mean'] / train_data.sampling_rate
    else:
        window_samples = train_data.X.shape[1]
        window_duration = window_samples / train_data.sampling_rate

    # Determine IMU position from dataset
    imu_position = ""
    if args.dataset == "opportunity":
        imu_position = "BACK"
    elif args.dataset == "uci_har":
        imu_position = "WAIST"

    pipeline.set_sensor_context(
        num_channels=num_channels,
        channel_names=channel_names,
        sampling_rate=train_data.sampling_rate,
        window_duration_sec=window_duration,
        dataset=args.dataset,
        imu_position=imu_position,
    )

    # ----- VQ-VAE pre-training -----
    vqvae_ckpt = os.path.join(CHECKPOINTS_DIR, "vqvae_pretrained.pt")
    if os.path.isfile(vqvae_ckpt) and not args.retrain_vqvae:
        print(f"\n--- Loading pre-trained VQ-VAE from {vqvae_ckpt} ---")
        pipeline.tokenizer = VQVAE_Tokenizer.load_pretrained(
            vqvae_ckpt, device=args.device
        )
        print("  VQ-VAE loaded successfully.")
    else:
        print("\n--- VQ-VAE Pre-training ---")
        raw_imu_list = [train_data.X[i] for i in range(len(train_data))]
        pipeline.pretrain_tokenizer(
            raw_imu_list,
            num_epochs=args.vqvae_epochs,
            patience=args.vqvae_patience,
        )

    # ----- Calibrate rewards -----
    class_weights = train_data.get_class_weights()
    n_classes = len(set(train_data.y.tolist()))
    pipeline.classifier.calibrate_rewards(
        n_classes=n_classes,
        class_weights=class_weights,
    )

    # ----- Bootstrap memory -----
    if args.clear_memory:
        print("\n--- Clearing memory ---")
        pipeline.memory.clear()

    if pipeline.memory.count() == 0:
        print("\n--- Bootstrapping memory from training data ---")
        print(f"  Tokenizing {len(train_data)} training samples...")
        tokens_list = []
        train_labels = []
        for i in range(len(train_data)):
            imu_data, desc, label = train_data.get_sample(i)
            tokens = pipeline.tokenizer.tokenize(imu_data)
            tokens_list.append(tokens)
            train_labels.append(desc)  # use description (full activity name)
        pipeline.bootstrap_memory(tokens_list, train_labels)
    else:
        print(f"\n--- Memory already populated: {pipeline.memory.count()} entries ---")

    # ----- Show sample description -----
    print("\n--- Sample Token Description ---")
    sample_imu, sample_desc, sample_label = test_data.get_sample(0)
    sample_tokens = pipeline.tokenizer.tokenize(sample_imu)
    print(pipeline.descriptor.describe(sample_tokens))
    print(f"\n(Ground truth: {sample_label} — {sample_desc})")

    # ----- Evaluate -----
    print("\n--- Evaluation ---")
    eval_metrics = evaluate(
        pipeline, test_data,
        max_samples=args.eval_samples,
        verbose=True,
        api_workers=args.workers,
    )

    # ----- Promote and show memory stats -----
    pipeline.memory.promote_short_term()
    pipeline.memory._print_stats()

    # ----- Save results -----
    results = {
        "args": vars(args),
        "mode": "few_shot_memory",
        "eval": eval_metrics,
        "memory_entries": pipeline.memory.count(),
    }
    save_results(results, args)
    print("\nDone!")


def save_results(results: Dict, args):
    """Save evaluation results to JSON."""
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
        description="NEMESIS — Few-shot memory mode for HAR"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="opportunity",
                        choices=["uci_har", "opportunity"],
                        help="Which dataset to use")

    # Evaluation
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="Eval samples (0 = full test set)")

    # Model
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model for classification")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")

    # VQ-VAE
    parser.add_argument("--vqvae-epochs", type=int, default=200,
                        help="Max VQ-VAE pre-training epochs")
    parser.add_argument("--vqvae-patience", type=int, default=15,
                        help="VQ-VAE early stopping patience")
    parser.add_argument("--retrain-vqvae", action="store_true",
                        help="Force re-train VQ-VAE even if checkpoint exists")

    # Memory
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of few-shot neighbours to retrieve")
    parser.add_argument("--clear-memory", action="store_true",
                        help="Clear memory DB before bootstrapping")

    # Parallelism
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel OpenAI API threads")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
