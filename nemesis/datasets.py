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

WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_v1.1_raw.txt"


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
