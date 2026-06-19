"""
Download UEA MTS datasets and convert to PDGNetV2-compatible .npy format.

Output structure:
    data/{name}/X_train.npy, y_train.npy, X_test.npy, y_test.npy
"""

import os
import numpy as np
from sktime.datasets import load_UCR_UEA_dataset

# ── Target datasets ─────────────────────────────────────────────────
DATASETS = [
    "NATOPS",                  # 24 sensors, 51 timesteps, 6 classes, 180/180
    "ERing",                   # 65 sensors, 65 timesteps, 4 classes, 30/270
    "SelfRegulationSCP2",      # 7 sensors, 256 timesteps, 2 classes, 200/180
    "HandMovementDirection",   # 10 sensors, 400 timesteps, 4 classes, 160/74
    "BasicMotions",            # 6 sensors, 100 timesteps, 4 classes, 40/40
    "Epilepsy",                # 3 sensors, 206 timesteps, 4 classes, 137/138
    "UWaveGestureLibrary",     # 3 sensors, 315 timesteps, 8 classes, 120/320
    "Cricket",                 # 6 sensors, 1197 timesteps, 12 classes, 108/72
    "Libras",                  # 2 sensors, 45 timesteps, 15 classes, 180/180
    "AtrialFibrillation",      # 2 sensors, 640 timesteps, 3 classes, 15/15
]

OUTPUT_DIR = "data"


def convert_to_npy(name):
    """Download one UEA dataset and save as .npy files."""
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")

    try:
        # Load train/test splits
        X_train, y_train = load_UCR_UEA_dataset(name=name, split="train", return_type="numpy3d")
        X_test, y_test = load_UCR_UEA_dataset(name=name, split="test", return_type="numpy3d")
    except Exception as e:
        print(f"  [FAIL] Failed to load: {e}")
        return None

    # sktime returns y as string labels for some datasets; encode to int
    if y_train.dtype.kind in ("U", "S", "O"):
        classes = sorted(set(y_train) | set(y_test))
        class_map = {c: i for i, c in enumerate(classes)}
        y_train = np.array([class_map[y] for y in y_train])
        y_test = np.array([class_map[y] for y in y_test])

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}  classes: {np.unique(y_train)}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}  classes: {np.unique(y_test)}")
    print(f"  dtype: {X_train.dtype}, NaN: {np.any(np.isnan(X_train))}")

    # Check for NaN
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print(f"  [WARN] NaN values detected!")

    # Save
    np.save(os.path.join(out_path, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(out_path, "y_train.npy"), y_train)
    np.save(os.path.join(out_path, "X_test.npy"), X_test.astype(np.float32))
    np.save(os.path.join(out_path, "y_test.npy"), y_test)
    print(f"  [OK] Saved to {out_path}")

    return {
        "name": name,
        "sensors": X_train.shape[1],
        "timesteps": X_train.shape[2],
        "train_n": len(y_train),
        "test_n": len(y_test),
        "classes": len(np.unique(y_train)),
        "has_nan": bool(np.any(np.isnan(X_train))),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = []
    for name in DATASETS:
        info = convert_to_npy(name)
        if info:
            summary.append(info)

    # ── Print summary ────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY: Downloaded UEA Datasets")
    print(f"{'='*80}")
    print(f"  {'Name':30s} | {'N':>4s} | {'L':>5s} | {'Train':>5s} | {'Test':>5s} | {'Classes':>7s} | NaN")
    print(f"  {'-'*78}")
    for s in summary:
        nan_flag = "WARN" if s["has_nan"] else "OK"
        print(f"  {s['name']:30s} | {s['sensors']:4d} | {s['timesteps']:5d} | {s['train_n']:5d} | {s['test_n']:5d} | {s['classes']:7d} | {nan_flag}")
    print(f"\n  Total: {len(summary)} datasets downloaded to data/*/")


if __name__ == "__main__":
    main()
