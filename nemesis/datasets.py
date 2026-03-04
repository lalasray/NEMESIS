"""
Dataset Loaders — download and prepare public IMU HAR datasets.

Supported datasets:
  1. UCI HAR (Human Activity Recognition Using Smartphones)
     - 6 activities, 30 subjects, accel + gyro at 50 Hz
     - 128-sample fixed windows, 6 channels (body_acc + body_gyro)

  2. WISDM (Wireless Sensor Data Mining)
     - 6 activities, 36 subjects, accelerometer at 20 Hz

Label classes are converted to rich activity descriptions so that
the OpenAI reward function can do meaningful semantic comparison.
"""

import os
import zipfile
import urllib.request
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import resample_poly
from math import gcd

from nemesis.config import PROJECT_ROOT


# ============================================================================
# Dataset directory
# ============================================================================
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")


# ============================================================================
# Label → Activity Description Mapping
# ============================================================================

# ----------- UCI HAR -----------
UCI_HAR_LABEL_MAP: Dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

UCI_HAR_DESCRIPTIONS: Dict[int, str] = {
    1: "walking forward on a flat surface at a normal steady pace",
    2: "walking up a flight of stairs at a steady pace",
    3: "walking down a flight of stairs at a steady pace",
    4: "sitting still on a chair in a relaxed position",
    5: "standing upright in place without moving",
    6: "lying down flat on a surface in a resting position",
}

# Extra-rich descriptions for few-shot prompting
UCI_HAR_RICH_DESCRIPTIONS: Dict[int, str] = {
    1: (
        "The person is walking forward at a regular pace on a flat surface. "
        "The gait is rhythmic with alternating leg swings, moderate arm movement, "
        "and a stable upright posture."
    ),
    2: (
        "The person is ascending stairs, stepping up repeatedly. The motion involves "
        "lifting each leg higher than in normal walking, with increased vertical "
        "acceleration and a slight forward lean."
    ),
    3: (
        "The person is descending stairs, stepping down carefully. The motion involves "
        "controlled lowering of each foot, with deceleration impacts and a slight "
        "backward lean for balance."
    ),
    4: (
        "The person is sitting still in a chair. There is minimal body movement, "
        "very low acceleration variation, and a stable bent-hip posture."
    ),
    5: (
        "The person is standing upright without walking. Small postural sway may be "
        "present but no locomotion. Acceleration is near gravitational baseline."
    ),
    6: (
        "The person is lying down flat on a surface. The body is horizontal with "
        "minimal movement. Gravity axis is shifted compared to upright postures."
    ),
}

# ----------- WISDM -----------
WISDM_LABEL_MAP: Dict[str, str] = {
    "Walking": "WALKING",
    "Jogging": "JOGGING",
    "Upstairs": "WALKING_UPSTAIRS",
    "Downstairs": "WALKING_DOWNSTAIRS",
    "Sitting": "SITTING",
    "Standing": "STANDING",
}

WISDM_DESCRIPTIONS: Dict[str, str] = {
    "Walking": "walking forward on a flat surface at a normal steady pace",
    "Jogging": "jogging at a brisk pace with bouncing arm and leg movements",
    "Upstairs": "walking up a flight of stairs at a steady pace",
    "Downstairs": "walking down a flight of stairs at a steady pace",
    "Sitting": "sitting still on a chair in a relaxed position",
    "Standing": "standing upright in place without moving",
}

# ----------- Opportunity -----------
# Locomotion labels in the .dat files (column index 243, 0-based)
OPP_LOCOMOTION_MAP: Dict[int, str] = {
    1: "STAND",
    2: "WALK",
    4: "SIT",
    5: "LIE",
}

OPP_DESCRIPTIONS: Dict[int, str] = {
    1: "standing upright in place without significant body movement",
    2: "walking around the room at a natural pace",
    4: "sitting on a chair in a relaxed position",
    5: "lying down on a deckchair in a resting position",
}

OPP_RICH_DESCRIPTIONS: Dict[int, str] = {
    1: (
        "The person is standing upright without locomotion. Small postural sway "
        "may be present. The IMU on the back registers near-gravitational "
        "baseline with minimal dynamic acceleration."
    ),
    2: (
        "The person is walking around the room at a natural pace. The gait is "
        "rhythmic with alternating leg swings. The back-mounted IMU shows "
        "periodic vertical and lateral oscillations from the walking cycle."
    ),
    4: (
        "The person is sitting still on a chair. The trunk is relatively stable "
        "with bent hips. Very low acceleration variation, gravity axis shifted "
        "compared to standing."
    ),
    5: (
        "The person is lying down on a deckchair in a resting position. The body "
        "is reclined/horizontal. Gravity axis is substantially shifted compared "
        "to upright postures, with minimal dynamic movement."
    ),
}

# BACK IMU columns in .dat files (0-indexed):
# Columns 37-39: InertialMeasurementUnit BACK - acc (x, y, z)
# Columns 40-42: InertialMeasurementUnit BACK - gyro (x, y, z)
OPP_BACK_IMU_COLS = [37, 38, 39, 40, 41, 42]
OPP_LOCOMOTION_COL = 243  # 0-indexed
OPP_SAMPLING_RATE = 30  # Hz
OPP_CHANNEL_NAMES = [
    "back_acc_x", "back_acc_y", "back_acc_z",
    "back_gyro_x", "back_gyro_y", "back_gyro_z",
]


# ============================================================================
# IMU Resampling
# ============================================================================

def resample_imu(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample a single IMU segment from orig_rate to target_rate Hz.

    Uses polyphase rational resampling (scipy.signal.resample_poly)
    which avoids aliasing and handles arbitrary rate ratios.

    Args:
        data:        (T, C) array of IMU channels
        orig_rate:   source sampling rate in Hz
        target_rate: desired sampling rate in Hz

    Returns:
        (T', C) resampled array where T' ≈ T * target_rate / orig_rate
    """
    if orig_rate == target_rate:
        return data
    g = gcd(orig_rate, target_rate)
    up = target_rate // g
    down = orig_rate // g
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def resample_dataset(dataset: "HARDataset", target_rate: int) -> "HARDataset":
    """
    Resample all IMU data in a HARDataset to target_rate Hz.

    Works for both fixed-length and variable-length datasets.
    Updates sampling_rate metadata accordingly.
    """
    if dataset.sampling_rate == target_rate:
        return dataset

    orig_rate = dataset.sampling_rate
    print(f"  [Resample] {dataset.dataset_name}/{dataset.split}: "
          f"{orig_rate}Hz → {target_rate}Hz")

    if dataset.is_variable_length:
        X_new = [
            resample_imu(seg, orig_rate, target_rate)
            for seg in dataset.X
        ]
    else:
        # Fixed-length: resample each sample
        X_new = np.stack([
            resample_imu(dataset.X[i], orig_rate, target_rate)
            for i in range(len(dataset))
        ]).astype(np.float32)

    return HARDataset(
        X=X_new,
        y=dataset.y,
        descriptions=dataset.descriptions,
        labels=dataset.labels,
        dataset_name=dataset.dataset_name,
        split=dataset.split,
        num_classes=dataset.num_classes,
        sampling_rate=target_rate,
        channels=dataset.channels,
    )


# ============================================================================
# Multi-Dataset Merging
# ============================================================================

def merge_datasets(datasets: List["HARDataset"], target_rate: int = 30) -> "HARDataset":
    """
    Merge multiple HARDatasets into a single unified dataset.

    All datasets are resampled to target_rate Hz first.  Labels are
    prefixed with the source dataset name so they remain distinguishable
    (e.g. "WALK" → "opportunity:WALK", "WALKING" → "uci_har:WALKING").
    Integer labels (y) are remapped to a contiguous global space.

    Channels are unified to 6 (acc_xyz + gyro_xyz) — datasets with fewer
    channels are zero-padded.

    Args:
        datasets:    list of HARDataset objects (any split)
        target_rate: common sampling rate in Hz

    Returns:
        Merged HARDataset (always variable-length since different datasets
        may have different segment lengths).
    """
    print(f"\n--- Merging {len(datasets)} datasets at {target_rate}Hz ---")

    all_X: List[np.ndarray] = []
    all_descriptions: List[str] = []
    all_labels: List[str] = []
    all_dataset_tags: List[str] = []  # track origin
    target_channels = 6

    for ds in datasets:
        # Resample if needed
        ds_r = resample_dataset(ds, target_rate)

        for i in range(len(ds_r)):
            seg = ds_r.X[i] if ds_r.is_variable_length else ds_r.X[i]

            # Pad to 6 channels if fewer
            if seg.shape[-1] < target_channels:
                pad = np.zeros(
                    (seg.shape[0], target_channels - seg.shape[-1]),
                    dtype=np.float32,
                )
                seg = np.concatenate([seg, pad], axis=-1)
            elif seg.shape[-1] > target_channels:
                seg = seg[:, :target_channels]

            all_X.append(seg)
            all_descriptions.append(ds_r.descriptions[i])
            # Prefix label with dataset name for disambiguation
            all_labels.append(f"{ds_r.dataset_name}:{ds_r.labels[i]}")
            all_dataset_tags.append(ds_r.dataset_name)

    # Build contiguous integer labels
    unique_labels = sorted(set(all_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in all_labels], dtype=int)

    # Merge dataset names
    ds_names = "+".join(sorted(set(d.dataset_name for d in datasets)))
    split = datasets[0].split if all(d.split == datasets[0].split for d in datasets) else "mixed"

    merged = HARDataset(
        X=all_X,
        y=y,
        descriptions=all_descriptions,
        labels=all_labels,
        dataset_name=ds_names,
        split=split,
        num_classes=len(unique_labels),
        sampling_rate=target_rate,
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )

    print(f"  Merged: {len(merged)} samples, {merged.num_classes} classes, "
          f"{target_rate}Hz, 6ch")
    for tag in sorted(set(all_dataset_tags)):
        count = all_dataset_tags.count(tag)
        print(f"    {tag}: {count} samples")

    return merged


# ============================================================================
# Data Container
# ============================================================================

@dataclass
class HARDataset:
    """
    Container for a loaded HAR dataset split.

    Supports both fixed-length (X is ndarray of shape (N,T,C)) and
    variable-length (X is a list of (T_i, C) arrays) IMU data.
    """
    # IMU data:
    #   Fixed-length:    np.ndarray of shape (N, T, C)
    #   Variable-length: List[np.ndarray] where each is (T_i, C)
    X: any  # np.ndarray or List[np.ndarray]
    # Integer class labels: (N,)
    y: np.ndarray
    # Activity descriptions: list of N strings
    descriptions: List[str]
    # Short labels: list of N strings (e.g. "WALKING")
    labels: List[str]
    # Metadata
    dataset_name: str
    split: str  # "train" or "test"
    num_classes: int
    sampling_rate: int
    channels: List[str]

    @property
    def is_variable_length(self) -> bool:
        return isinstance(self.X, list)

    @property
    def num_channels(self) -> int:
        if self.is_variable_length:
            return self.X[0].shape[-1] if len(self.X) > 0 else 0
        return self.X.shape[-1]

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        if self.is_variable_length:
            lengths = [x.shape[0] for x in self.X]
            return (
                f"HARDataset({self.dataset_name}/{self.split}, "
                f"samples={len(self)}, classes={self.num_classes}, "
                f"var_length=[{min(lengths)}-{max(lengths)}], "
                f"channels={self.num_channels})"
            )
        return (
            f"HARDataset({self.dataset_name}/{self.split}, "
            f"samples={len(self)}, classes={self.num_classes}, "
            f"shape={self.X.shape})"
        )

    def get_sample(self, idx: int) -> Tuple[np.ndarray, str, str]:
        """Get a single (imu_data, description, label) tuple."""
        return self.X[idx], self.descriptions[idx], self.labels[idx]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, str]:
        """Get a single (imu_data, description, label) tuple."""
        return self.X[idx], self.descriptions[idx], self.labels[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        """Get count of samples per class."""
        from collections import Counter
        return dict(Counter(self.labels))

    def get_length_stats(self) -> Dict:
        """Get statistics about sequence lengths (for variable-length datasets)."""
        if not self.is_variable_length:
            T = self.X.shape[1]
            return {"min": T, "max": T, "mean": T, "fixed": True}
        lengths = [x.shape[0] for x in self.X]
        return {
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "fixed": False,
        }

    def shuffle(self, seed: int = 42) -> "HARDataset":
        """Return a shuffled copy."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self))
        if self.is_variable_length:
            X_shuffled = [self.X[i] for i in indices]
        else:
            X_shuffled = self.X[indices]
        return HARDataset(
            X=X_shuffled,
            y=self.y[indices],
            descriptions=[self.descriptions[i] for i in indices],
            labels=[self.labels[i] for i in indices],
            dataset_name=self.dataset_name,
            split=self.split,
            num_classes=self.num_classes,
            sampling_rate=self.sampling_rate,
            channels=self.channels,
        )

    def subset(self, n: int) -> "HARDataset":
        """Return first n samples."""
        if self.is_variable_length:
            X_sub = self.X[:n]
        else:
            X_sub = self.X[:n]
        return HARDataset(
            X=X_sub,
            y=self.y[:n],
            descriptions=self.descriptions[:n],
            labels=self.labels[:n],
            dataset_name=self.dataset_name,
            split=self.split,
            num_classes=self.num_classes,
            sampling_rate=self.sampling_rate,
            channels=self.channels,
        )

    def class_balanced_indices(self, n_per_class: int, seed: int = 42) -> List[int]:
        """
        Return indices that give equal representation per class.

        Oversamples minority classes and undersamples majority classes
        so each class has exactly n_per_class samples.

        Args:
            n_per_class: samples per class. 0 = use size of largest class.
            seed: random seed

        Returns:
            list of indices (may contain repeats for minority classes)
        """
        rng = np.random.RandomState(seed)
        # Group indices by class
        class_indices: Dict[int, List[int]] = {}
        for i, label_int in enumerate(self.y):
            class_indices.setdefault(int(label_int), []).append(i)

        if n_per_class <= 0:
            n_per_class = max(len(v) for v in class_indices.values())

        balanced = []
        for cls, indices in class_indices.items():
            if len(indices) >= n_per_class:
                # Undersample
                balanced.extend(rng.choice(indices, size=n_per_class, replace=False).tolist())
            else:
                # Oversample: take all + randomly repeat
                balanced.extend(indices)
                extra = n_per_class - len(indices)
                balanced.extend(rng.choice(indices, size=extra, replace=True).tolist())

        rng.shuffle(balanced)
        return balanced

    def get_class_weights(self, max_weight: float = 3.0) -> Dict[int, float]:
        """
        Compute inverse-frequency class weights, capped to prevent extreme ratios.

        Args:
            max_weight: Maximum allowed weight. Prevents extreme values for very
                        rare classes (e.g. LIE with only 29 samples → 22.4×
                        uncapped) which create perverse RL incentives.

        Returns:
            Dict mapping integer class label → weight (higher for rare classes).
            Weights are normalised so the mean weight ≈ 1.0, then capped.
        """
        from collections import Counter
        counts = Counter(self.y.tolist())
        total = sum(counts.values())
        n_classes = len(counts)
        weights = {}
        for cls, count in counts.items():
            # inverse frequency, normalised
            raw = total / (n_classes * count)
            weights[cls] = min(raw, max_weight)
        return weights


# ============================================================================
# UCI HAR Dataset Loader
# ============================================================================

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


def download_uci_har(data_dir: str = None) -> str:
    """
    Download and extract UCI HAR dataset.

    Returns path to extracted 'UCI HAR Dataset' directory.
    """
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "uci_har")
    os.makedirs(data_dir, exist_ok=True)

    extracted_dir = os.path.join(data_dir, "UCI HAR Dataset")
    if os.path.isdir(extracted_dir):
        print(f"[Dataset] UCI HAR already exists at {extracted_dir}")
        return extracted_dir

    zip_path = os.path.join(data_dir, "uci_har.zip")

    if not os.path.exists(zip_path):
        print(f"[Dataset] Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(UCI_HAR_URL, zip_path)
        print(f"[Dataset] Downloaded to {zip_path}")

    print(f"[Dataset] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    print(f"[Dataset] Extracted to {extracted_dir}")

    return extracted_dir


def _load_uci_har_inertial(
    base_dir: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw inertial signals from UCI HAR.

    Returns:
        X: (N, 128, 6) — body_acc(xyz) + body_gyro(xyz)
        y: (N,) integer labels 1-6
    """
    split_dir = os.path.join(base_dir, split)
    inertial_dir = os.path.join(split_dir, "Inertial Signals")

    # Load 6 channels: body accelerometer + body gyroscope
    channel_files = [
        "body_acc_x_{}.txt", "body_acc_y_{}.txt", "body_acc_z_{}.txt",
        "body_gyro_x_{}.txt", "body_gyro_y_{}.txt", "body_gyro_z_{}.txt",
    ]

    channels = []
    for fname_template in channel_files:
        fname = fname_template.format(split)
        fpath = os.path.join(inertial_dir, fname)
        data = np.loadtxt(fpath)  # (N, 128)
        channels.append(data)

    # Stack: (N, 128, 6)
    X = np.stack(channels, axis=-1).astype(np.float32)

    # Load labels
    y_path = os.path.join(split_dir, f"y_{split}.txt")
    y = np.loadtxt(y_path, dtype=int)

    return X, y


def load_uci_har(
    split: str = "train",
    description_style: str = "standard",
    data_dir: str = None,
) -> HARDataset:
    """
    Load UCI HAR dataset.

    Args:
        split: "train" or "test"
        description_style: "standard" (concise) or "rich" (detailed)
        data_dir: override dataset directory

    Returns:
        HARDataset ready for NEMESIS training
    """
    base_dir = download_uci_har(data_dir)
    X, y = _load_uci_har_inertial(base_dir, split)

    desc_map = UCI_HAR_RICH_DESCRIPTIONS if description_style == "rich" else UCI_HAR_DESCRIPTIONS

    descriptions = [desc_map[label] for label in y]
    labels = [UCI_HAR_LABEL_MAP[label] for label in y]

    return HARDataset(
        X=X,
        y=y,
        descriptions=descriptions,
        labels=labels,
        dataset_name="UCI_HAR",
        split=split,
        num_classes=6,
        sampling_rate=50,
        channels=["body_acc_x", "body_acc_y", "body_acc_z",
                  "body_gyro_x", "body_gyro_y", "body_gyro_z"],
    )


# ============================================================================
# WISDM Dataset Loader
# ============================================================================

WISDM_URL = "https://raw.githubusercontent.com/jfuerth/hadoop-streaming/master/src/test/resources/WISDM_ar_v1.1_raw.txt"


def download_wisdm(data_dir: str = None) -> str:
    """Download WISDM dataset. Returns path to raw data file."""
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "wisdm")
    os.makedirs(data_dir, exist_ok=True)

    raw_path = os.path.join(data_dir, "WISDM_ar_v1.1_raw.txt")
    if os.path.exists(raw_path):
        print(f"[Dataset] WISDM already exists at {raw_path}")
        return raw_path

    print(f"[Dataset] Downloading WISDM dataset...")
    try:
        urllib.request.urlretrieve(WISDM_URL, raw_path)
        print(f"[Dataset] Downloaded to {raw_path}")
    except Exception as e:
        print(f"[Dataset] WISDM download failed: {e}")
        print(f"[Dataset] Please download manually from https://www.cis.fordham.edu/wisdm/dataset.php")
        raise

    return raw_path


def load_wisdm(
    window_size: int = 128,
    overlap: int = 64,
    test_ratio: float = 0.2,
    split: str = "train",
    data_dir: str = None,
) -> HARDataset:
    """
    Load WISDM dataset, window it, and split into train/test.

    Args:
        window_size: samples per window
        overlap: overlapping samples
        test_ratio: fraction for test set
        split: "train" or "test"
        data_dir: override dataset directory

    Returns:
        HARDataset
    """
    raw_path = download_wisdm(data_dir)

    # Parse the raw CSV-like format: user,activity,timestamp,x,y,z;
    activities_data = {}  # activity → list of (x, y, z)
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                continue
            try:
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].rstrip(";"))
                if activity not in activities_data:
                    activities_data[activity] = []
                activities_data[activity].append([x, y, z])
            except (ValueError, IndexError):
                continue

    # Window each activity
    all_X = []
    all_labels = []
    step = window_size - overlap

    for activity, samples in activities_data.items():
        if activity not in WISDM_DESCRIPTIONS:
            continue
        data = np.array(samples, dtype=np.float32)
        n_windows = max(0, (len(data) - window_size) // step)
        for i in range(n_windows):
            start = i * step
            window = data[start:start + window_size]
            if len(window) == window_size:
                all_X.append(window)
                all_labels.append(activity)

    X = np.array(all_X, dtype=np.float32)  # (N, window_size, 3)

    # Pad to 6 channels (acc only → add zero gyro columns)
    zeros = np.zeros((X.shape[0], window_size, 3), dtype=np.float32)
    X = np.concatenate([X, zeros], axis=-1)  # (N, window_size, 6)

    # Shuffle and split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    X = X[indices]
    all_labels = [all_labels[i] for i in indices]

    split_idx = int(len(X) * (1 - test_ratio))
    if split == "train":
        X = X[:split_idx]
        all_labels = all_labels[:split_idx]
    else:
        X = X[split_idx:]
        all_labels = all_labels[split_idx:]

    # Build integer labels and descriptions
    label_to_int = {name: i for i, name in enumerate(sorted(WISDM_DESCRIPTIONS.keys()))}
    y = np.array([label_to_int[l] for l in all_labels], dtype=int)
    descriptions = [WISDM_DESCRIPTIONS[l] for l in all_labels]
    short_labels = [WISDM_LABEL_MAP[l] for l in all_labels]

    return HARDataset(
        X=X,
        y=y,
        descriptions=descriptions,
        labels=short_labels,
        dataset_name="WISDM",
        split=split,
        num_classes=len(WISDM_DESCRIPTIONS),
        sampling_rate=20,
        channels=["acc_x", "acc_y", "acc_z", "gyro_x_pad", "gyro_y_pad", "gyro_z_pad"],
    )


# ============================================================================
# Opportunity Dataset Loader
# ============================================================================

OPP_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"

# Subject-run files used for each split (standard Opportunity challenge split)
_OPP_TRAIN_FILES = [
    "S1-ADL1.dat", "S1-ADL2.dat", "S1-ADL3.dat", "S1-ADL4.dat", "S1-ADL5.dat",
    "S1-Drill.dat",
    "S2-ADL1.dat", "S2-ADL2.dat", "S2-ADL3.dat", "S2-ADL4.dat", "S2-ADL5.dat",
    "S2-Drill.dat",
    "S3-ADL1.dat", "S3-ADL2.dat", "S3-ADL3.dat", "S3-ADL4.dat", "S3-ADL5.dat",
    "S3-Drill.dat",
]
_OPP_TEST_FILES = [
    "S4-ADL1.dat", "S4-ADL2.dat", "S4-ADL3.dat", "S4-ADL4.dat", "S4-ADL5.dat",
    "S4-Drill.dat",
]


def download_opportunity(data_dir: str = None) -> str:
    """
    Download and extract the Opportunity Activity Recognition dataset.

    Returns path to directory containing .dat files.
    """
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "opportunity")
    os.makedirs(data_dir, exist_ok=True)

    # The zip contains  OpportunityUCIDataset/  with a dataset/ subfolder
    dataset_dir = os.path.join(data_dir, "OpportunityUCIDataset", "dataset")
    if os.path.isdir(dataset_dir):
        # Verify at least one .dat file exists
        if any(f.endswith(".dat") for f in os.listdir(dataset_dir)):
            print(f"[Dataset] Opportunity already exists at {dataset_dir}")
            return dataset_dir

    zip_path = os.path.join(data_dir, "opportunity.zip")

    if not os.path.exists(zip_path):
        print(f"[Dataset] Downloading Opportunity dataset (~350 MB)...")
        urllib.request.urlretrieve(OPP_URL, zip_path)
        print(f"[Dataset] Downloaded to {zip_path}")

    print(f"[Dataset] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    print(f"[Dataset] Extracted to {data_dir}")

    if not os.path.isdir(dataset_dir):
        # Try to find the .dat files wherever they were extracted
        for root, dirs, files in os.walk(data_dir):
            if any(f.endswith(".dat") for f in files):
                dataset_dir = root
                break

    return dataset_dir


def _parse_opportunity_dat(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a single Opportunity .dat file.

    Returns:
        imu: (T, 6)  — BACK IMU [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        labels: (T,)  — locomotion label per timestep
    """
    # .dat files are whitespace-separated with NaN for missing values
    raw = np.genfromtxt(file_path, dtype=np.float64)  # (T, ~250 cols)

    imu = raw[:, OPP_BACK_IMU_COLS].astype(np.float32)  # (T, 6)
    labels = raw[:, OPP_LOCOMOTION_COL].astype(np.float32)  # (T,)

    return imu, labels


def _interpolate_nans(data: np.ndarray) -> np.ndarray:
    """Interpolate NaN values channel-wise using linear interpolation."""
    result = data.copy()
    for ch in range(result.shape[1]):
        col = result[:, ch]
        nans = np.isnan(col)
        if nans.all():
            result[:, ch] = 0.0
            continue
        if nans.any():
            valid = ~nans
            indices = np.arange(len(col))
            col[nans] = np.interp(indices[nans], indices[valid], col[valid])
            result[:, ch] = col
    return result


def _extract_segments(
    imu: np.ndarray,
    labels: np.ndarray,
    min_length: int = 30,
) -> List[Tuple[np.ndarray, int]]:
    """
    Extract contiguous same-label segments from a recording.

    Args:
        imu: (T, 6) — IMU data
        labels: (T,) — per-timestep locomotion labels
        min_length: minimum segment length in samples (default 30 = 1s at 30Hz)

    Returns:
        list of (segment_imu, label) pairs, where segment_imu is (T_i, 6)
    """
    valid_labels = set(OPP_LOCOMOTION_MAP.keys())
    segments = []

    # Find change points
    i = 0
    while i < len(labels):
        lbl = int(labels[i]) if not np.isnan(labels[i]) else 0
        if lbl not in valid_labels:
            i += 1
            continue

        # Start of a new segment
        j = i + 1
        while j < len(labels):
            next_lbl = int(labels[j]) if not np.isnan(labels[j]) else 0
            if next_lbl != lbl:
                break
            j += 1

        seg_len = j - i
        if seg_len >= min_length:
            seg_imu = imu[i:j]
            # Skip if the segment is entirely NaN
            if not np.isnan(seg_imu).all():
                seg_imu = _interpolate_nans(seg_imu)
                segments.append((seg_imu, lbl))

        i = j

    return segments


def load_opportunity(
    split: str = "train",
    description_style: str = "rich",
    min_segment_length: int = 30,
    max_segment_length: int = 3000,
    data_dir: str = None,
) -> HARDataset:
    """
    Load the Opportunity dataset with variable-length locomotion segments.

    Each sample is a contiguous segment of the same locomotion activity,
    extracted from the BACK-mounted IMU (6 channels: acc + gyro, 30 Hz).

    Args:
        split: "train" or "test"
        description_style: "standard" or "rich"
        min_segment_length: discard segments shorter than this (samples)
        max_segment_length: split longer segments at this length (samples)
        data_dir: override dataset directory

    Returns:
        HARDataset with variable-length X (list of (T_i, 6) arrays)
    """
    dataset_dir = download_opportunity(data_dir)

    files = _OPP_TRAIN_FILES if split == "train" else _OPP_TEST_FILES
    desc_map = OPP_RICH_DESCRIPTIONS if description_style == "rich" else OPP_DESCRIPTIONS

    all_segments: List[np.ndarray] = []
    all_labels_int: List[int] = []

    for fname in files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"[Dataset] Warning: {fname} not found, skipping")
            continue

        print(f"[Dataset] Parsing {fname}...")
        imu, labels = _parse_opportunity_dat(fpath)
        segments = _extract_segments(imu, labels, min_length=min_segment_length)

        for seg_imu, lbl in segments:
            # Split long segments into chunks of max_segment_length
            if len(seg_imu) > max_segment_length:
                for start in range(0, len(seg_imu) - min_segment_length + 1, max_segment_length):
                    chunk = seg_imu[start:start + max_segment_length]
                    if len(chunk) >= min_segment_length:
                        all_segments.append(chunk)
                        all_labels_int.append(lbl)
            else:
                all_segments.append(seg_imu)
                all_labels_int.append(lbl)

    if len(all_segments) == 0:
        raise RuntimeError(f"No valid segments found for split='{split}'")

    # Remap labels to contiguous 0-based integers
    unique_labels = sorted(set(all_labels_int))
    label_remap = {orig: new_idx for new_idx, orig in enumerate(unique_labels)}

    y = np.array([label_remap[l] for l in all_labels_int], dtype=int)
    descriptions = [desc_map[l] for l in all_labels_int]
    short_labels = [OPP_LOCOMOTION_MAP[l] for l in all_labels_int]

    print(f"[Dataset] Opportunity/{split}: {len(all_segments)} segments, "
          f"{len(unique_labels)} classes")
    lengths = [s.shape[0] for s in all_segments]
    print(f"[Dataset] Segment lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    return HARDataset(
        X=all_segments,  # list of (T_i, 6) arrays — variable length
        y=y,
        descriptions=descriptions,
        labels=short_labels,
        dataset_name="Opportunity",
        split=split,
        num_classes=len(unique_labels),
        sampling_rate=OPP_SAMPLING_RATE,
        channels=OPP_CHANNEL_NAMES,
    )


# ============================================================================
# Generic loader from numpy files
# ============================================================================

def load_from_numpy(
    X_path: str,
    y_path: str,
    label_descriptions: Dict[int, str],
    label_names: Optional[Dict[int, str]] = None,
    split: str = "train",
    sampling_rate: int = 50,
    channels: Optional[List[str]] = None,
) -> HARDataset:
    """
    Load a dataset from numpy files.

    Args:
        X_path: path to .npy file with shape (N, T, C)
        y_path: path to .npy file with shape (N,) integer labels
        label_descriptions: mapping from int label → activity description
        label_names: mapping from int label → short name (optional)
        split: "train" or "test"
        sampling_rate: Hz
        channels: channel names

    Returns:
        HARDataset
    """
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(int)

    descriptions = [label_descriptions.get(label, f"activity {label}") for label in y]
    if label_names:
        labels = [label_names.get(label, f"CLASS_{label}") for label in y]
    else:
        labels = [f"CLASS_{label}" for label in y]

    if channels is None:
        channels = [f"ch_{i}" for i in range(X.shape[-1])]

    return HARDataset(
        X=X,
        y=y,
        descriptions=descriptions,
        labels=labels,
        dataset_name="custom",
        split=split,
        num_classes=len(set(y)),
        sampling_rate=sampling_rate,
        channels=channels,
    )


# ============================================================================
# PAMAP2 Dataset Loader (Physical Activity Monitoring)
# ============================================================================
# 100Hz, 9 subjects, chest IMU acc+gyro = 6ch, 12+ activities
# Columns: timestamp, activityID, HR, [hand IMU 17cols], [chest IMU 17cols], [ankle IMU 17cols]
# Chest acc(6g): cols 21-23, chest gyro: cols 24-26

PAMAP2_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

PAMAP2_ACTIVITY_MAP = {
    1: "LYING", 2: "SITTING", 3: "STANDING", 4: "WALKING",
    5: "RUNNING", 6: "CYCLING", 7: "NORDIC_WALKING",
    12: "ASCENDING_STAIRS", 13: "DESCENDING_STAIRS",
    16: "VACUUM_CLEANING", 17: "IRONING", 24: "ROPE_JUMPING",
}

PAMAP2_DESCRIPTIONS = {
    1: "The person is lying down in a resting position with minimal body movement.",
    2: "The person is sitting still in a relaxed posture, trunk stable.",
    3: "The person is standing upright without locomotion, small postural sway.",
    4: "The person is walking at a normal pace with rhythmic gait pattern.",
    5: "The person is running with vigorous bouncing, high-frequency impacts.",
    6: "The person is cycling with periodic leg rotation and relatively stable upper body.",
    7: "The person is Nordic walking with exaggerated arm swings and walking poles.",
    12: "The person is ascending stairs, stepping upward with increased vertical acceleration.",
    13: "The person is descending stairs, stepping downward with controlled deceleration.",
    16: "The person is vacuum cleaning with irregular arm/torso movements.",
    17: "The person is ironing with repetitive arm sliding on a flat surface.",
    24: "The person is jumping rope with rhythmic vertical bouncing.",
}

PAMAP2_CHEST_ACC_COLS = [24, 25, 26]   # chest acc 6g (x,y,z)
PAMAP2_CHEST_GYRO_COLS = [27, 28, 29]  # chest gyro (x,y,z)
PAMAP2_ACTIVITY_COL = 1
PAMAP2_TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
PAMAP2_TEST_SUBJECTS = [8, 9]


def download_pamap2(data_dir: str = None) -> str:
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "pamap2")
    os.makedirs(data_dir, exist_ok=True)

    # Look for Protocol dir specifically
    protocol_path = None
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "Protocol" and any(
            f.startswith("subject10") and f.endswith(".dat") for f in files
        ):
            print(f"[Dataset] PAMAP2 already exists at {root}")
            return root
        # Fallback: any dir with .dat files
        if protocol_path is None and any(
            f.startswith("subject10") and f.endswith(".dat") for f in files
        ):
            protocol_path = root
    if protocol_path is not None:
        print(f"[Dataset] PAMAP2 already exists at {protocol_path}")
        return protocol_path

    zip_path = os.path.join(data_dir, "pamap2.zip")
    if not os.path.exists(zip_path):
        print("[Dataset] Downloading PAMAP2 dataset (~600 MB)...")
        urllib.request.urlretrieve(PAMAP2_URL, zip_path)

    print("[Dataset] Extracting PAMAP2...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # The outer zip may contain an inner PAMAP2_Dataset.zip
    inner_zip = os.path.join(data_dir, "PAMAP2_Dataset.zip")
    if os.path.exists(inner_zip):
        print("[Dataset] Extracting inner PAMAP2_Dataset.zip...")
        with zipfile.ZipFile(inner_zip, "r") as z2:
            z2.extractall(data_dir)

    protocol_path = None
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "Protocol" and any(
            f.startswith("subject10") and f.endswith(".dat") for f in files
        ):
            return root
        if protocol_path is None and any(
            f.startswith("subject10") and f.endswith(".dat") for f in files
        ):
            protocol_path = root
    if protocol_path is not None:
        return protocol_path
    raise RuntimeError("PAMAP2: could not find subject .dat files after extraction")


def load_pamap2(
    split: str = "train",
    window_size: int = 256,
    overlap: int = 128,
    data_dir: str = None,
) -> HARDataset:
    protocol_dir = download_pamap2(data_dir)
    subjects = PAMAP2_TRAIN_SUBJECTS if split == "train" else PAMAP2_TEST_SUBJECTS
    imu_cols = PAMAP2_CHEST_ACC_COLS + PAMAP2_CHEST_GYRO_COLS

    all_X, all_y, all_desc, all_labels = [], [], [], []
    step = window_size - overlap

    for subj in subjects:
        fpath = os.path.join(protocol_dir, f"subject10{subj}.dat")
        if not os.path.exists(fpath):
            print(f"[Dataset] PAMAP2: subject10{subj}.dat not found, skipping")
            continue
        # Use pandas for fast reading of space-separated with NaN
        try:
            import pandas as pd
            df = pd.read_csv(fpath, sep=r"\s+", header=None, dtype=np.float64,
                             na_values=["NaN", "nan"])
            raw = df.values
        except ImportError:
            raw = np.genfromtxt(fpath, dtype=np.float64)
        activities = raw[:, PAMAP2_ACTIVITY_COL].astype(int)
        imu = raw[:, imu_cols].astype(np.float32)
        # Replace NaN
        for ch in range(imu.shape[1]):
            nans = np.isnan(imu[:, ch])
            if nans.any():
                valid = ~nans
                if valid.any():
                    imu[nans, ch] = np.interp(
                        np.where(nans)[0], np.where(valid)[0], imu[valid, ch]
                    )
                else:
                    imu[:, ch] = 0.0

        for lbl in PAMAP2_ACTIVITY_MAP:
            mask = activities == lbl
            indices = np.where(mask)[0]
            if len(indices) < window_size:
                continue
            # Find contiguous runs
            splits = np.where(np.diff(indices) > 1)[0] + 1
            for run in np.split(indices, splits):
                if len(run) < window_size:
                    continue
                seg = imu[run]
                for start in range(0, len(seg) - window_size + 1, step):
                    window = seg[start:start + window_size]
                    all_X.append(window)
                    all_y.append(lbl)
                    all_desc.append(PAMAP2_DESCRIPTIONS[lbl])
                    all_labels.append(PAMAP2_ACTIVITY_MAP[lbl])

    if not all_X:
        raise RuntimeError(f"PAMAP2: no valid windows for split={split}")

    label_set = sorted(set(all_y))
    label_remap = {v: i for i, v in enumerate(label_set)}
    y = np.array([label_remap[l] for l in all_y], dtype=int)

    print(f"[Dataset] PAMAP2/{split}: {len(all_X)} windows, {len(label_set)} classes")
    return HARDataset(
        X=np.stack(all_X).astype(np.float32),
        y=y, descriptions=all_desc, labels=all_labels,
        dataset_name="PAMAP2", split=split,
        num_classes=len(label_set), sampling_rate=100,
        channels=["chest_acc_x", "chest_acc_y", "chest_acc_z",
                  "chest_gyro_x", "chest_gyro_y", "chest_gyro_z"],
    )


# ============================================================================
# MHEALTH Dataset Loader
# ============================================================================
# 50Hz, 10 subjects, right-arm acc+gyro = 6ch, 12 activities
# Columns: chest_acc(3), ECG(2), l_ankle_acc(3), l_ankle_gyro(3), l_ankle_mag(3),
#           r_arm_acc(3), r_arm_gyro(3), r_arm_mag(3), label

MHEALTH_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"

MHEALTH_ACTIVITY_MAP = {
    1: "STANDING", 2: "SITTING", 3: "LYING", 4: "WALKING",
    5: "CLIMBING_STAIRS", 6: "WAIST_BENDS", 7: "FRONTAL_ARM_ELEVATION",
    8: "KNEE_BENDS", 9: "CYCLING", 10: "JOGGING", 11: "RUNNING",
    12: "JUMP_FRONT_BACK",
}

MHEALTH_DESCRIPTIONS = {
    1: "The person is standing still with minimal body movement.",
    2: "The person is sitting in a relaxed position on a chair.",
    3: "The person is lying down in a resting position.",
    4: "The person is walking at a normal steady pace.",
    5: "The person is climbing stairs, stepping upward repeatedly.",
    6: "The person is doing waist bends, flexing the torso forward and back.",
    7: "The person is raising their arms in front (frontal elevation) repeatedly.",
    8: "The person is bending their knees (squatting) repeatedly.",
    9: "The person is cycling on an exercise bike with periodic leg rotation.",
    10: "The person is jogging at a moderate pace with bouncing movements.",
    11: "The person is running at a vigorous pace with high-frequency impacts.",
    12: "The person is jumping forward and backward in a rhythmic pattern.",
}

MHEALTH_ARM_ACC_COLS = [14, 15, 16]   # right-arm acc
MHEALTH_ARM_GYRO_COLS = [17, 18, 19]  # right-arm gyro
MHEALTH_LABEL_COL = 23
MHEALTH_TRAIN_SUBJECTS = list(range(1, 9))   # 1-8
MHEALTH_TEST_SUBJECTS = [9, 10]


def download_mhealth(data_dir: str = None) -> str:
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "mhealth")
    os.makedirs(data_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        if any(f.startswith("mHealth_subject") and f.endswith(".log") for f in files):
            print(f"[Dataset] MHEALTH already exists at {root}")
            return root

    zip_path = os.path.join(data_dir, "mhealth.zip")
    if not os.path.exists(zip_path):
        print("[Dataset] Downloading MHEALTH dataset...")
        urllib.request.urlretrieve(MHEALTH_URL, zip_path)

    print("[Dataset] Extracting MHEALTH...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    for root, dirs, files in os.walk(data_dir):
        if any(f.startswith("mHealth_subject") and f.endswith(".log") for f in files):
            return root
    raise RuntimeError("MHEALTH: .log files not found after extraction")


def load_mhealth(
    split: str = "train",
    window_size: int = 128,
    overlap: int = 64,
    data_dir: str = None,
) -> HARDataset:
    log_dir = download_mhealth(data_dir)
    subjects = MHEALTH_TRAIN_SUBJECTS if split == "train" else MHEALTH_TEST_SUBJECTS
    imu_cols = MHEALTH_ARM_ACC_COLS + MHEALTH_ARM_GYRO_COLS

    all_X, all_y, all_desc, all_labels = [], [], [], []
    step = window_size - overlap

    for subj in subjects:
        fpath = os.path.join(log_dir, f"mHealth_subject{subj}.log")
        if not os.path.exists(fpath):
            continue
        raw = np.loadtxt(fpath)
        labels = raw[:, MHEALTH_LABEL_COL].astype(int)
        imu = raw[:, imu_cols].astype(np.float32)

        for lbl in MHEALTH_ACTIVITY_MAP:
            mask = labels == lbl
            indices = np.where(mask)[0]
            if len(indices) < window_size:
                continue
            splits = np.where(np.diff(indices) > 1)[0] + 1
            for run in np.split(indices, splits):
                if len(run) < window_size:
                    continue
                seg = imu[run]
                for start in range(0, len(seg) - window_size + 1, step):
                    w = seg[start:start + window_size]
                    all_X.append(w)
                    all_y.append(lbl)
                    all_desc.append(MHEALTH_DESCRIPTIONS[lbl])
                    all_labels.append(MHEALTH_ACTIVITY_MAP[lbl])

    if not all_X:
        raise RuntimeError(f"MHEALTH: no valid windows for split={split}")

    label_set = sorted(set(all_y))
    label_remap = {v: i for i, v in enumerate(label_set)}
    y = np.array([label_remap[l] for l in all_y], dtype=int)

    print(f"[Dataset] MHEALTH/{split}: {len(all_X)} windows, {len(label_set)} classes")
    return HARDataset(
        X=np.stack(all_X).astype(np.float32), y=y,
        descriptions=all_desc, labels=all_labels,
        dataset_name="MHEALTH", split=split,
        num_classes=len(label_set), sampling_rate=50,
        channels=["rarm_acc_x", "rarm_acc_y", "rarm_acc_z",
                  "rarm_gyro_x", "rarm_gyro_y", "rarm_gyro_z"],
    )


# ============================================================================
# Daphnet Freezing of Gait Dataset
# ============================================================================
# 64Hz, 10 subjects, trunk acc only (3ch, pad gyro → 6ch), 2 classes
# Columns: timestamp, ankle(3), upper_leg(3), trunk(3), annotation

DAPHNET_URL = "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"

DAPHNET_ACTIVITY_MAP = {1: "NORMAL_GAIT", 2: "FREEZE_OF_GAIT"}
DAPHNET_DESCRIPTIONS = {
    1: "The person is walking normally with regular gait pattern, no freezing episodes.",
    2: "The person is experiencing a freezing of gait episode, unable to move forward despite the intention to walk.",
}
DAPHNET_TRUNK_COLS = [7, 8, 9]  # trunk horizontal, vertical, lateral
DAPHNET_ANNOTATION_COL = 10
DAPHNET_TRAIN_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]
DAPHNET_TEST_SUBJECTS = ["S08", "S09", "S10"]


def download_daphnet(data_dir: str = None) -> str:
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "daphnet")
    os.makedirs(data_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        if any(f.endswith(".txt") and f.startswith("S") for f in files):
            print(f"[Dataset] Daphnet already exists at {root}")
            return root

    zip_path = os.path.join(data_dir, "daphnet.zip")
    if not os.path.exists(zip_path):
        print("[Dataset] Downloading Daphnet dataset...")
        urllib.request.urlretrieve(DAPHNET_URL, zip_path)

    print("[Dataset] Extracting Daphnet...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    for root, dirs, files in os.walk(data_dir):
        if any(f.endswith(".txt") and f.startswith("S") for f in files):
            return root
    raise RuntimeError("Daphnet: .txt files not found after extraction")


def load_daphnet(
    split: str = "train",
    window_size: int = 192,
    overlap: int = 96,
    data_dir: str = None,
) -> HARDataset:
    dataset_dir = download_daphnet(data_dir)
    subjects = DAPHNET_TRAIN_SUBJECTS if split == "train" else DAPHNET_TEST_SUBJECTS

    all_X, all_y, all_desc, all_labels = [], [], [], []
    step = window_size - overlap

    for subj in subjects:
        # Each subject may have R01, R02 runs
        for run in ["R01", "R02"]:
            fpath = os.path.join(dataset_dir, f"{subj}{run}.txt")
            if not os.path.exists(fpath):
                continue
            try:
                raw = np.loadtxt(fpath, dtype=np.float64)
            except Exception:
                continue
            if raw.shape[1] < 11:
                continue
            trunk = raw[:, DAPHNET_TRUNK_COLS].astype(np.float32)
            # Pad with zeros for gyro (only has accelerometer)
            gyro_pad = np.zeros_like(trunk)
            imu = np.concatenate([trunk, gyro_pad], axis=-1)  # (T, 6)
            annotations = raw[:, DAPHNET_ANNOTATION_COL].astype(int)

            for lbl in DAPHNET_ACTIVITY_MAP:
                mask = annotations == lbl
                indices = np.where(mask)[0]
                if len(indices) < window_size:
                    continue
                splits_arr = np.where(np.diff(indices) > 1)[0] + 1
                for run_idx in np.split(indices, splits_arr):
                    if len(run_idx) < window_size:
                        continue
                    seg = imu[run_idx]
                    for start in range(0, len(seg) - window_size + 1, step):
                        w = seg[start:start + window_size]
                        all_X.append(w)
                        all_y.append(lbl)
                        all_desc.append(DAPHNET_DESCRIPTIONS[lbl])
                        all_labels.append(DAPHNET_ACTIVITY_MAP[lbl])

    if not all_X:
        raise RuntimeError(f"Daphnet: no valid windows for split={split}")

    label_set = sorted(set(all_y))
    label_remap = {v: i for i, v in enumerate(label_set)}
    y = np.array([label_remap[l] for l in all_y], dtype=int)

    print(f"[Dataset] Daphnet/{split}: {len(all_X)} windows, {len(label_set)} classes")
    return HARDataset(
        X=np.stack(all_X).astype(np.float32), y=y,
        descriptions=all_desc, labels=all_labels,
        dataset_name="Daphnet", split=split,
        num_classes=len(label_set), sampling_rate=64,
        channels=["trunk_acc_h", "trunk_acc_v", "trunk_acc_l",
                  "gyro_pad_x", "gyro_pad_y", "gyro_pad_z"],
    )


# ============================================================================
# DSA — Daily and Sports Activities Dataset
# ============================================================================
# 25Hz, 8 subjects, torso acc+gyro = 6ch, 19 activities
# Structure: a{01-19}/p{1-8}/s{01-60}.txt  (125 rows × 45 cols)
# 5 sensors × 9 axes: torso(0-8), R arm(9-17), L arm(18-26), R leg(27-35), L leg(36-44)

DSA_URL = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"

DSA_ACTIVITY_MAP = {
    1: "SITTING", 2: "STANDING", 3: "LYING_BACK", 4: "LYING_RIGHT",
    5: "ASCENDING_STAIRS", 6: "DESCENDING_STAIRS",
    7: "STANDING_IN_ELEVATOR", 8: "MOVING_IN_ELEVATOR",
    9: "WALKING_PARKING", 10: "WALKING_TREADMILL_FLAT",
    11: "WALKING_TREADMILL_INCLINE", 12: "RUNNING_TREADMILL",
    13: "STEPPER", 14: "CROSS_TRAINER", 15: "CYCLING_HORIZONTAL",
    16: "CYCLING_VERTICAL", 17: "ROWING", 18: "JUMPING", 19: "BASKETBALL",
}

DSA_DESCRIPTIONS = {
    1: "The person is sitting still in a chair with minimal body movement.",
    2: "The person is standing upright without locomotion.",
    3: "The person is lying on their back in a resting position.",
    4: "The person is lying on their right side.",
    5: "The person is ascending stairs, stepping upward.",
    6: "The person is descending stairs, stepping downward.",
    7: "The person is standing still inside an elevator.",
    8: "The person is in a moving elevator (standing while it moves).",
    9: "The person is walking in a parking lot at a normal pace.",
    10: "The person is walking on a flat treadmill at about 4 km/h.",
    11: "The person is walking on an inclined treadmill at about 4 km/h.",
    12: "The person is running on a treadmill at about 8 km/h.",
    13: "The person is exercising on a stepper machine with rhythmic stepping.",
    14: "The person is exercising on a cross-trainer with elliptical arm/leg motion.",
    15: "The person is cycling on a horizontal exercise bike.",
    16: "The person is cycling on a vertical exercise bike.",
    17: "The person is rowing on a rowing machine with periodic pull strokes.",
    18: "The person is jumping repeatedly in place.",
    19: "The person is playing basketball with varied movements.",
}

DSA_TORSO_ACC_COLS = [0, 1, 2]
DSA_TORSO_GYRO_COLS = [3, 4, 5]
DSA_TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
DSA_TEST_SUBJECTS = [7, 8]


def download_dsa(data_dir: str = None) -> str:
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "dsa")
    os.makedirs(data_dir, exist_ok=True)

    # Look for the data/ dir structure
    for root, dirs, files in os.walk(data_dir):
        if "a01" in dirs or "a19" in [d for d in dirs]:
            print(f"[Dataset] DSA already exists at {root}")
            return root

    zip_path = os.path.join(data_dir, "dsa.zip")
    if not os.path.exists(zip_path):
        print("[Dataset] Downloading DSA dataset...")
        urllib.request.urlretrieve(DSA_URL, zip_path)

    print("[Dataset] Extracting DSA...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    for root, dirs, files in os.walk(data_dir):
        if "a01" in dirs:
            return root
    raise RuntimeError("DSA: activity folders not found after extraction")


def load_dsa(
    split: str = "train",
    data_dir: str = None,
) -> HARDataset:
    base_dir = download_dsa(data_dir)
    subjects = DSA_TRAIN_SUBJECTS if split == "train" else DSA_TEST_SUBJECTS
    imu_cols = DSA_TORSO_ACC_COLS + DSA_TORSO_GYRO_COLS

    all_X, all_y, all_desc, all_labels = [], [], [], []

    for act_id in DSA_ACTIVITY_MAP:
        act_dir = os.path.join(base_dir, f"a{act_id:02d}")
        if not os.path.isdir(act_dir):
            continue
        for subj in subjects:
            subj_dir = os.path.join(act_dir, f"p{subj}")
            if not os.path.isdir(subj_dir):
                continue
            for seg_id in range(1, 61):
                fpath = os.path.join(subj_dir, f"s{seg_id:02d}.txt")
                if not os.path.exists(fpath):
                    continue
                try:
                    raw = np.loadtxt(fpath, delimiter=",")
                except Exception:
                    continue
                if raw.shape[0] < 50 or raw.shape[1] < 6:
                    continue
                imu = raw[:, imu_cols].astype(np.float32)
                all_X.append(imu)
                all_y.append(act_id)
                all_desc.append(DSA_DESCRIPTIONS[act_id])
                all_labels.append(DSA_ACTIVITY_MAP[act_id])

    if not all_X:
        raise RuntimeError(f"DSA: no valid segments for split={split}")

    label_set = sorted(set(all_y))
    label_remap = {v: i for i, v in enumerate(label_set)}
    y = np.array([label_remap[l] for l in all_y], dtype=int)

    # DSA is fixed-length (125 samples per segment at 25Hz = 5s)
    X = np.stack(all_X).astype(np.float32)
    print(f"[Dataset] DSA/{split}: {len(X)} segments, {len(label_set)} classes")
    return HARDataset(
        X=X, y=y, descriptions=all_desc, labels=all_labels,
        dataset_name="DSA", split=split,
        num_classes=len(label_set), sampling_rate=25,
        channels=["torso_acc_x", "torso_acc_y", "torso_acc_z",
                  "torso_gyro_x", "torso_gyro_y", "torso_gyro_z"],
    )


# ============================================================================
# HAPT — Smartphone-Based Recognition of Human Activities and
#         Postural Transitions
# ============================================================================
# 50Hz, 30 subjects, smartphone acc+gyro = 6ch, 12 activities
# Raw data: acc_expNN_userNN.txt, gyro_expNN_userNN.txt

HAPT_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"

HAPT_ACTIVITY_MAP = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
    7: "STAND_TO_SIT", 8: "SIT_TO_STAND", 9: "SIT_TO_LIE",
    10: "LIE_TO_SIT", 11: "STAND_TO_LIE", 12: "LIE_TO_STAND",
}

HAPT_DESCRIPTIONS = {
    1: "The person is walking forward at a normal pace on a flat surface.",
    2: "The person is ascending stairs.",
    3: "The person is descending stairs.",
    4: "The person is sitting still on a chair.",
    5: "The person is standing upright without moving.",
    6: "The person is lying down on a flat surface.",
    7: "The person is transitioning from standing to sitting.",
    8: "The person is transitioning from sitting to standing.",
    9: "The person is transitioning from sitting to lying down.",
    10: "The person is transitioning from lying to sitting.",
    11: "The person is transitioning from standing to lying down.",
    12: "The person is transitioning from lying to standing.",
}

HAPT_TRAIN_USERS = list(range(1, 22))   # users 1-21
HAPT_TEST_USERS  = list(range(22, 31))  # users 22-30


def download_hapt(data_dir: str = None) -> str:
    if data_dir is None:
        data_dir = os.path.join(DATASETS_DIR, "hapt")
    os.makedirs(data_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        if "labels.txt" in files and any(f.startswith("acc_") for f in files):
            print(f"[Dataset] HAPT already exists at {root}")
            return root

    zip_path = os.path.join(data_dir, "hapt.zip")
    if not os.path.exists(zip_path):
        print("[Dataset] Downloading HAPT dataset...")
        urllib.request.urlretrieve(HAPT_URL, zip_path)

    print("[Dataset] Extracting HAPT...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    for root, dirs, files in os.walk(data_dir):
        if "labels.txt" in files and any(f.startswith("acc_") for f in files):
            return root

    # Try looking in RawData subfolder
    for root, dirs, files in os.walk(data_dir):
        if any(f.startswith("acc_exp") for f in files):
            return root

    raise RuntimeError("HAPT: raw data files not found after extraction")


def load_hapt(
    split: str = "train",
    window_size: int = 128,
    overlap: int = 64,
    data_dir: str = None,
) -> HARDataset:
    raw_dir = download_hapt(data_dir)

    # Find labels.txt (may be in parent dir)
    labels_path = os.path.join(raw_dir, "labels.txt")
    if not os.path.exists(labels_path):
        parent = os.path.dirname(raw_dir)
        labels_path = os.path.join(parent, "labels.txt")
    if not os.path.exists(labels_path):
        # Search for it
        for root, dirs, files in os.walk(os.path.dirname(raw_dir)):
            if "labels.txt" in files:
                labels_path = os.path.join(root, "labels.txt")
                break

    # Parse labels: exp_id, user_id, activity_id, start_sample, end_sample
    label_entries = np.loadtxt(labels_path, dtype=int)
    target_users = set(HAPT_TRAIN_USERS if split == "train" else HAPT_TEST_USERS)

    all_X, all_y, all_desc, all_labels = [], [], [], []

    # Group by (exp_id, user_id)
    exp_user_pairs = set()
    for row in label_entries:
        exp_id, user_id = int(row[0]), int(row[1])
        if user_id in target_users:
            exp_user_pairs.add((exp_id, user_id))

    for exp_id, user_id in sorted(exp_user_pairs):
        acc_path = os.path.join(raw_dir, f"acc_exp{exp_id:02d}_user{user_id:02d}.txt")
        gyro_path = os.path.join(raw_dir, f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt")
        if not (os.path.exists(acc_path) and os.path.exists(gyro_path)):
            continue

        acc = np.loadtxt(acc_path).astype(np.float32)   # (T, 3)
        gyro = np.loadtxt(gyro_path).astype(np.float32)  # (T, 3)
        T = min(len(acc), len(gyro))
        imu = np.concatenate([acc[:T], gyro[:T]], axis=1)  # (T, 6)

        # Extract labeled segments
        for row in label_entries:
            if int(row[0]) != exp_id or int(row[1]) != user_id:
                continue
            act_id, start, end = int(row[2]), int(row[3]) - 1, int(row[4])
            if act_id not in HAPT_ACTIVITY_MAP:
                continue
            seg = imu[start:end]
            if len(seg) < window_size:
                continue

            step = window_size - overlap
            for ws in range(0, len(seg) - window_size + 1, step):
                w = seg[ws:ws + window_size]
                all_X.append(w)
                all_y.append(act_id)
                all_desc.append(HAPT_DESCRIPTIONS[act_id])
                all_labels.append(HAPT_ACTIVITY_MAP[act_id])

    if not all_X:
        raise RuntimeError(f"HAPT: no valid windows for split={split}")

    label_set = sorted(set(all_y))
    label_remap = {v: i for i, v in enumerate(label_set)}
    y = np.array([label_remap[l] for l in all_y], dtype=int)

    print(f"[Dataset] HAPT/{split}: {len(all_X)} windows, {len(label_set)} classes")
    return HARDataset(
        X=np.stack(all_X).astype(np.float32), y=y,
        descriptions=all_desc, labels=all_labels,
        dataset_name="HAPT", split=split,
        num_classes=len(label_set), sampling_rate=50,
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
    )


# ============================================================================
# Quick summary
# ============================================================================

def print_dataset_info(dataset: HARDataset):
    """Print a summary of a loaded dataset."""
    print(f"\n{'='*50}")
    print(f"  {dataset.dataset_name} / {dataset.split}")
    print(f"{'='*50}")
    print(f"  Samples:       {len(dataset)}")
    if dataset.is_variable_length:
        stats = dataset.get_length_stats()
        print(f"  Channels:      {dataset.num_channels}")
        print(f"  Length (min):   {stats['min']}")
        print(f"  Length (max):   {stats['max']}")
        print(f"  Length (mean):  {stats['mean']:.1f}")
        print(f"  Length (med):   {stats['median']:.1f}")
    else:
        print(f"  Shape:         {dataset.X.shape}")
    print(f"  Classes:       {dataset.num_classes}")
    print(f"  Sampling rate: {dataset.sampling_rate} Hz")
    print(f"  Channels:      {dataset.channels}")
    print(f"\n  Class distribution:")
    for label, count in sorted(dataset.get_class_distribution().items()):
        print(f"    {label:25s}: {count:5d}")
    print(f"\n  Sample descriptions:")
    seen = set()
    for desc, label in zip(dataset.descriptions, dataset.labels):
        if label not in seen:
            seen.add(label)
            print(f"    {label:25s} → \"{desc}\"")
    print()


# ============================================================================
# DATASET_LOADERS — central registry used by train_har.py
# ============================================================================
# Each entry maps a short name → dict with:
#   load_train:    callable() -> HARDataset  (training split)
#   load_test:     callable() -> HARDataset  (test split)
#   imu_position:  str  (body placement for LLM context)
#   sampling_rate: int  (native Hz)

DATASET_LOADERS = {
    "opportunity": {
        "load_train": lambda: load_opportunity(split="train"),
        "load_test":  lambda: load_opportunity(split="test"),
        "imu_position": "BACK",
        "sampling_rate": 30,
    },
    "uci_har": {
        "load_train": lambda: load_uci_har(split="train"),
        "load_test":  lambda: load_uci_har(split="test"),
        "imu_position": "WAIST",
        "sampling_rate": 50,
    },
    "wisdm": {
        "load_train": lambda: load_wisdm(split="train"),
        "load_test":  lambda: load_wisdm(split="test"),
        "imu_position": "POCKET",
        "sampling_rate": 20,
    },
    "pamap2": {
        "load_train": lambda: load_pamap2(split="train"),
        "load_test":  lambda: load_pamap2(split="test"),
        "imu_position": "CHEST",
        "sampling_rate": 100,
    },
    "mhealth": {
        "load_train": lambda: load_mhealth(split="train"),
        "load_test":  lambda: load_mhealth(split="test"),
        "imu_position": "RIGHT_ARM",
        "sampling_rate": 50,
    },
    "daphnet": {
        "load_train": lambda: load_daphnet(split="train"),
        "load_test":  lambda: load_daphnet(split="test"),
        "imu_position": "TRUNK",
        "sampling_rate": 64,
    },
    "dsa": {
        "load_train": lambda: load_dsa(split="train"),
        "load_test":  lambda: load_dsa(split="test"),
        "imu_position": "TORSO",
        "sampling_rate": 25,
    },
    "hapt": {
        "load_train": lambda: load_hapt(split="train"),
        "load_test":  lambda: load_hapt(split="test"),
        "imu_position": "WAIST",
        "sampling_rate": 50,
    },
}
