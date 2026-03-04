#!/usr/bin/env python3
"""
train_har.py — Multi-dataset IMU activity recognition with NEMESIS.

Supports:
  • Training / evaluation on individual or merged datasets
  • Leave-One-Dataset-Out (LODO) cross-validation
  • Automatic download, resampling, and channel padding

Usage examples:
  # Single dataset
  python train_har.py --dataset opportunity

  # All 8 datasets, merged evaluation
  python train_har.py --dataset all

  # Leave-one-dataset-out
  python train_har.py --dataset all --lodo

  # Specific datasets
  python train_har.py --dataset opportunity pamap2 mhealth --target-rate 30
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── NEMESIS ──────────────────────────────────────────────────────────────────
from nemesis.config import (
    IMUTokenizerConfig,
    ClassifierConfig,
    MemoryConfig,
    LearnerConfig,
    CHECKPOINTS_DIR,
    MEMORY_DIR,
)
from nemesis.pipeline import NemesisPipeline
from nemesis.datasets import (
    DATASET_LOADERS,
    HARDataset,
    merge_datasets,
    resample_dataset,
    print_dataset_info,
)


# ============================================================================
# Helpers
# ============================================================================

def load_single_dataset(name: str) -> dict:
    """Load train+test for a single dataset. Returns dict or raises."""
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_LOADERS.keys())}"
        )
    entry = DATASET_LOADERS[name]
    print(f"\n{'='*60}")
    print(f"  Loading: {name}")
    print(f"{'='*60}")
    train_ds = entry["load_train"]()
    test_ds = entry["load_test"]()
    return {
        "name": name,
        "train": train_ds,
        "test": test_ds,
        "imu_position": entry["imu_position"],
        "sampling_rate": entry["sampling_rate"],
    }


def resample_if_needed(ds: HARDataset, target_rate: int) -> HARDataset:
    """Resample dataset to target_rate if rates differ."""
    if ds.sampling_rate == target_rate:
        return ds
    print(f"  Resampling {ds.dataset_name}/{ds.split} "
          f"{ds.sampling_rate}Hz → {target_rate}Hz ...")
    return resample_dataset(ds, target_rate)


def pad_channels(ds: HARDataset, target_channels: int = 6) -> HARDataset:
    """Zero-pad channels if < target_channels."""
    if ds.num_channels >= target_channels:
        return ds
    pad_width = target_channels - ds.num_channels
    if ds.is_variable_length:
        new_X = [
            np.concatenate([x, np.zeros((x.shape[0], pad_width), dtype=np.float32)], axis=1)
            for x in ds.X
        ]
    else:
        new_X = np.concatenate(
            [ds.X, np.zeros((*ds.X.shape[:-1], pad_width), dtype=np.float32)],
            axis=-1,
        )
    ch_names = list(ds.channels) + [f"pad_{i}" for i in range(pad_width)]
    return HARDataset(
        X=new_X,
        y=ds.y,
        descriptions=ds.descriptions,
        labels=ds.labels,
        dataset_name=ds.dataset_name,
        split=ds.split,
        num_classes=ds.num_classes,
        sampling_rate=ds.sampling_rate,
        channels=ch_names,
    )


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_pipeline(
    pipeline: NemesisPipeline,
    test_ds: HARDataset,
    batch_size: int = 32,
    max_workers: int = 8,
    max_samples: int = 200,
    tag: str = "",
) -> dict:
    """
    Evaluate the pipeline on a test dataset.
    Returns dict with accuracy, macro_f1, classification_report.
    """
    N = len(test_ds)
    if N > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(N, max_samples, replace=False)
    else:
        indices = np.arange(N)

    all_preds = []
    all_gts = []
    all_labels = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        imu_batch = [test_ds[int(i)][0] for i in batch_idx]
        gt_batch = [test_ds.labels[int(i)] for i in batch_idx]

        results = pipeline.classify_batch(
            imu_batch=imu_batch,
            ground_truths=gt_batch,
            max_workers=max_workers,
        )

        for r, gt in zip(results, gt_batch):
            all_preds.append(r.activity)
            all_gts.append(gt)

        done = min(start + batch_size, len(indices))
        print(f"  [{tag}] Evaluated {done}/{len(indices)}", end="\r")

    print()

    # Map to consistent label set from test_ds
    label_set = sorted(set(all_gts))
    accuracy = accuracy_score(all_gts, all_preds)
    macro_f1 = f1_score(all_gts, all_preds, average="macro", zero_division=0)

    pred_counter = Counter(all_preds)
    gt_counter = Counter(all_gts)

    result = {
        "tag": tag,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_samples": len(all_gts),
        "n_classes_gt": len(label_set),
        "n_classes_pred": len(set(all_preds)),
    }

    # Print summary
    print(f"\n  --- {tag} Results ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Samples:   {len(all_gts)}")
    print(f"  GT classes: {len(label_set)}, Pred classes: {len(set(all_preds))}")

    # Per-class report
    try:
        report = classification_report(
            all_gts, all_preds, zero_division=0, output_dict=True
        )
        result["per_class"] = report
        print(classification_report(all_gts, all_preds, zero_division=0))
    except Exception:
        pass

    return result


# ============================================================================
# Single-dataset or merged evaluation
# ============================================================================

def run_standard(args, datasets: dict) -> list:
    """
    Standard evaluation: train VQ-VAE on all training data,
    bootstrap memory, then evaluate on each test set + combined.
    """
    results = []
    target_rate = args.target_rate

    # Collect all training IMU for VQ-VAE pre-training
    all_train_imu = []
    all_train_tokens_data = []  # (dataset_name, train_ds, imu_position)

    for name, ds_info in datasets.items():
        train_ds = resample_if_needed(ds_info["train"], target_rate)
        train_ds = pad_channels(train_ds, 6)
        ds_info["train_processed"] = train_ds
        test_ds = resample_if_needed(ds_info["test"], target_rate)
        test_ds = pad_channels(test_ds, 6)
        ds_info["test_processed"] = test_ds

        # Gather IMU for VQ-VAE training
        for i in range(len(train_ds)):
            imu, _, _ = train_ds[i]
            all_train_imu.append(imu)

        all_train_tokens_data.append((name, train_ds, ds_info["imu_position"]))

    # ── VQ-VAE Pre-training ──────────────────────────────────────────────
    imu_config = IMUTokenizerConfig(
        num_channels=6,
        sampling_rate=target_rate,
    )
    pipeline = NemesisPipeline(
        imu_config=imu_config,
        device="cuda" if args.gpu else "cpu",
    )

    ckpt_tag = "_".join(sorted(datasets.keys()))
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"vqvae_{ckpt_tag}_{target_rate}Hz.pt")

    if os.path.exists(ckpt_path) and not args.retrain:
        print(f"\n[VQ-VAE] Loading checkpoint: {ckpt_path}")
        pipeline.tokenizer.load_pretrained(ckpt_path)
    else:
        print(f"\n[VQ-VAE] Training on {len(all_train_imu)} samples from "
              f"{list(datasets.keys())}...")
        pipeline.pretrain_tokenizer(
            imu_data_list=all_train_imu,
            num_epochs=args.vqvae_epochs,
            batch_size=256,
            patience=15,
        )
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        pipeline.tokenizer.save_pretrained(ckpt_path)

    # ── Bootstrap Memory ─────────────────────────────────────────────────
    pipeline.memory.clear()
    for name, train_ds, imu_pos in all_train_tokens_data:
        print(f"\n[Memory] Bootstrapping from {name} ({len(train_ds)} samples)...")
        pipeline.set_sensor_context(
            num_channels=6,
            channel_names=train_ds.channels,
            sampling_rate=target_rate,
            dataset=name,
            imu_position=imu_pos,
        )
        # Tokenize
        tokens_list = []
        labels_list = []
        max_bootstrap = min(len(train_ds), args.max_bootstrap)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_ds), max_bootstrap, replace=False) if len(train_ds) > max_bootstrap else np.arange(len(train_ds))
        for i in idx:
            imu, _, _ = train_ds[int(i)]
            tokens = pipeline.tokenizer.tokenize(imu)
            tokens_list.append(tokens)
            labels_list.append(train_ds.labels[int(i)])

        pipeline.bootstrap_memory(tokens_list, labels_list)

    print(f"\n[Memory] Total entries: {pipeline.memory.count()}")

    # ── Memory Learning (prototype refinement + prompt tuning) ───────────
    if args.learn_epochs > 0:
        # Prepare tokenized training data for learning
        learn_tokens, learn_descs, learn_gts = [], [], []
        for name, train_ds, imu_pos in all_train_tokens_data:
            pipeline.set_sensor_context(
                num_channels=6, channel_names=train_ds.channels,
                sampling_rate=target_rate, dataset=name, imu_position=imu_pos,
            )
            max_learn = min(len(train_ds), args.max_bootstrap)
            rng = np.random.RandomState(123)
            idx = rng.choice(len(train_ds), max_learn, replace=False) if len(train_ds) > max_learn else np.arange(len(train_ds))
            for i in idx:
                imu, desc, label = train_ds[int(i)]
                tokens = pipeline.tokenizer.tokenize(imu)
                learn_tokens.append(tokens)
                learn_descs.append(pipeline.descriptor.describe(tokens))
                learn_gts.append(label)

        pipeline.set_activity_options(sorted(set(learn_gts)))
        pipeline.learn_loop(
            tokens_list=learn_tokens,
            descriptions=learn_descs,
            ground_truths=learn_gts,
            num_epochs=args.learn_epochs,
            patience=args.learn_patience,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
        )

    # ── Evaluate per-dataset ─────────────────────────────────────────────
    for name, ds_info in datasets.items():
        test_ds = ds_info["test_processed"]
        train_ds = ds_info["train_processed"]
        pipeline.set_sensor_context(
            num_channels=6,
            channel_names=test_ds.channels,
            sampling_rate=target_rate,
            dataset=name,
            imu_position=ds_info["imu_position"],
        )
        pipeline.set_activity_options(sorted(set(test_ds.labels)))

        tag = f"{name} ({target_rate}Hz)"
        r = evaluate_pipeline(
            pipeline, test_ds,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_samples=args.max_eval_samples,
            tag=tag,
        )
        r["dataset"] = name
        results.append(r)

    return results


# ============================================================================
# Leave-One-Dataset-Out (LODO) Evaluation
# ============================================================================

def run_lodo(args, datasets: dict) -> list:
    """
    For each dataset D, train on ALL other datasets, evaluate on D.
    Tests how well knowledge transfers between domains.
    """
    all_names = sorted(datasets.keys())
    results = []
    target_rate = args.target_rate

    for held_out in all_names:
        print(f"\n{'#'*70}")
        print(f"  LODO: Held-out = {held_out}")
        print(f"  Training on: {[n for n in all_names if n != held_out]}")
        print(f"{'#'*70}")

        # Collect training data from all OTHER datasets
        train_names = [n for n in all_names if n != held_out]
        all_train_imu = []
        all_train_tokens_data = []

        for name in train_names:
            ds_info = datasets[name]
            train_ds = resample_if_needed(ds_info["train"], target_rate)
            train_ds = pad_channels(train_ds, 6)
            for i in range(len(train_ds)):
                imu, _, _ = train_ds[i]
                all_train_imu.append(imu)
            all_train_tokens_data.append((name, train_ds, ds_info["imu_position"]))

        # Prepare held-out test
        held_info = datasets[held_out]
        held_test = resample_if_needed(held_info["test"], target_rate)
        held_test = pad_channels(held_test, 6)

        # Fresh pipeline
        imu_config = IMUTokenizerConfig(num_channels=6, sampling_rate=target_rate)
        pipeline = NemesisPipeline(
            imu_config=imu_config,
            device="cuda" if args.gpu else "cpu",
        )

        # VQ-VAE
        ckpt_tag = "lodo_" + "_".join(train_names)
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"vqvae_{ckpt_tag}_{target_rate}Hz.pt")

        if os.path.exists(ckpt_path) and not args.retrain:
            pipeline.tokenizer.load_pretrained(ckpt_path)
        else:
            print(f"[VQ-VAE] Training on {len(all_train_imu)} samples...")
            pipeline.pretrain_tokenizer(
                imu_data_list=all_train_imu,
                num_epochs=args.vqvae_epochs,
                batch_size=256,
                patience=15,
            )
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            pipeline.tokenizer.save_pretrained(ckpt_path)

        # Bootstrap memory from training datasets only
        pipeline.memory.clear()
        for name, train_ds, imu_pos in all_train_tokens_data:
            pipeline.set_sensor_context(
                num_channels=6,
                channel_names=train_ds.channels,
                sampling_rate=target_rate,
                dataset=name,
                imu_position=imu_pos,
            )
            tokens_list, labels_list = [], []
            max_bootstrap = min(len(train_ds), args.max_bootstrap)
            rng = np.random.RandomState(42)
            idx = rng.choice(len(train_ds), max_bootstrap, replace=False) if len(train_ds) > max_bootstrap else np.arange(len(train_ds))
            for i in idx:
                imu, _, _ = train_ds[int(i)]
                tokens = pipeline.tokenizer.tokenize(imu)
                tokens_list.append(tokens)
                labels_list.append(train_ds.labels[int(i)])
            pipeline.bootstrap_memory(tokens_list, labels_list)

        print(f"[Memory] Total entries after bootstrap: {pipeline.memory.count()}")

        # ── Memory Learning ──────────────────────────────────────────────
        if args.learn_epochs > 0:
            learn_tokens, learn_descs, learn_gts = [], [], []
            for name, train_ds, imu_pos in all_train_tokens_data:
                pipeline.set_sensor_context(
                    num_channels=6, channel_names=train_ds.channels,
                    sampling_rate=target_rate, dataset=name, imu_position=imu_pos,
                )
                max_learn = min(len(train_ds), args.max_bootstrap)
                rng = np.random.RandomState(123)
                idx = rng.choice(len(train_ds), max_learn, replace=False) if len(train_ds) > max_learn else np.arange(len(train_ds))
                for i in idx:
                    imu, desc, label = train_ds[int(i)]
                    tokens = pipeline.tokenizer.tokenize(imu)
                    learn_tokens.append(tokens)
                    learn_descs.append(pipeline.descriptor.describe(tokens))
                    learn_gts.append(label)

            pipeline.set_activity_options(sorted(set(learn_gts)))
            pipeline.learn_loop(
                tokens_list=learn_tokens,
                descriptions=learn_descs,
                ground_truths=learn_gts,
                num_epochs=args.learn_epochs,
                patience=args.learn_patience,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
            )

        # Evaluate on held-out
        pipeline.set_sensor_context(
            num_channels=6,
            channel_names=held_test.channels,
            sampling_rate=target_rate,
            dataset=held_out,
            imu_position=held_info["imu_position"],
        )
        pipeline.set_activity_options(sorted(set(held_test.labels)))

        tag = f"LODO held={held_out}"
        r = evaluate_pipeline(
            pipeline, held_test,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_samples=args.max_eval_samples,
            tag=tag,
        )
        r["held_out"] = held_out
        r["train_datasets"] = train_names
        results.append(r)

        print(f"  >>> LODO {held_out}: Macro F1 = {r['macro_f1']:.4f}, "
              f"Accuracy = {r['accuracy']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("  LODO Summary")
    print(f"{'='*70}")
    f1s = []
    for r in results:
        f1s.append(r["macro_f1"])
        print(f"  {r['held_out']:15s}  F1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}")
    print(f"  {'Mean':15s}  F1={np.mean(f1s):.4f}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NEMESIS multi-dataset HAR training & evaluation"
    )
    parser.add_argument(
        "--dataset", nargs="+", default=["opportunity"],
        help="Datasets to use. 'all' for all 8. Or list: opportunity pamap2 ...",
    )
    parser.add_argument(
        "--lodo", action="store_true",
        help="Leave-One-Dataset-Out cross-validation",
    )
    parser.add_argument(
        "--target-rate", type=int, default=30,
        help="Common resampling rate (Hz) for merging datasets (default: 30)",
    )
    parser.add_argument(
        "--vqvae-epochs", type=int, default=1000,
        help="VQ-VAE pre-training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Parallel LLM threads (default: 8)",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=200,
        help="Max samples per dataset for evaluation (default: 200)",
    )
    parser.add_argument(
        "--max-bootstrap", type=int, default=500,
        help="Max samples per dataset for memory bootstrap (default: 500)",
    )
    parser.add_argument(
        "--learn-epochs", type=int, default=100,
        help="Memory learning epochs (prototype refinement + prompt tuning, default: 100, 0 to skip)",
    )
    parser.add_argument(
        "--learn-patience", type=int, default=10,
        help="Early stopping patience for memory learning (default: 10)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force VQ-VAE retraining even if checkpoint exists",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA GPU",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Resolve 'all'
    requested = args.dataset
    if "all" in requested:
        requested = list(DATASET_LOADERS.keys())

    print(f"\n{'='*70}")
    print(f"  NEMESIS Multi-Dataset HAR")
    print(f"{'='*70}")
    print(f"  Datasets:     {requested}")
    print(f"  LODO:         {args.lodo}")
    print(f"  Target rate:  {args.target_rate} Hz")
    print(f"  VQ-VAE epochs:{args.vqvae_epochs}")
    print(f"  Learn epochs: {args.learn_epochs} (patience={args.learn_patience})")
    print(f"  Max eval:     {args.max_eval_samples} per dataset")
    print(f"  Output:       {args.output}")
    print()

    # ── Load all requested datasets ──────────────────────────────────────
    datasets = {}
    failed = []
    for name in requested:
        try:
            ds_info = load_single_dataset(name)
            datasets[name] = ds_info
            print_dataset_info(ds_info["train"])
            print_dataset_info(ds_info["test"])
        except Exception as e:
            print(f"\n[ERROR] Failed to load '{name}': {e}")
            traceback.print_exc()
            failed.append(name)

    if not datasets:
        print("[FATAL] No datasets loaded successfully. Exiting.")
        sys.exit(1)

    if failed:
        print(f"\n[WARNING] Failed datasets: {failed}")
        print(f"[WARNING] Continuing with {list(datasets.keys())}")

    print(f"\n  Successfully loaded {len(datasets)} datasets: "
          f"{list(datasets.keys())}")

    # ── Run evaluation ───────────────────────────────────────────────────
    t0 = time.time()

    if args.lodo and len(datasets) >= 2:
        results = run_lodo(args, datasets)
    else:
        if args.lodo and len(datasets) < 2:
            print("[WARNING] LODO requires ≥2 datasets. Running standard eval.")
        results = run_standard(args, datasets)

    elapsed = time.time() - t0

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "datasets_loaded": list(datasets.keys()),
        "datasets_failed": failed,
        "mode": "lodo" if args.lodo and len(datasets) >= 2 else "standard",
        "target_rate": args.target_rate,
        "elapsed_seconds": elapsed,
        "results": [],
    }
    for r in results:
        entry = {k: v for k, v in r.items() if k != "per_class"}
        output["results"].append(entry)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Done! Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*70}")

    # Print final summary table
    print(f"\n  {'Dataset':<20s} {'Macro F1':<12s} {'Accuracy':<12s} {'Samples':<10s}")
    print(f"  {'─'*54}")
    for r in results:
        tag = r.get("held_out", r.get("dataset", r.get("tag", "?")))
        print(f"  {tag:<20s} {r['macro_f1']:<12.4f} {r['accuracy']:<12.4f} "
              f"{r['n_samples']:<10d}")

    if len(results) > 1:
        avg_f1 = np.mean([r["macro_f1"] for r in results])
        avg_acc = np.mean([r["accuracy"] for r in results])
        print(f"  {'─'*54}")
        print(f"  {'MEAN':<20s} {avg_f1:<12.4f} {avg_acc:<12.4f}")


if __name__ == "__main__":
    main()
