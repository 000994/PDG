"""
Classic ML Baselines for Multivariate Time Series Classification.

Compares Logistic Regression, Random Forest, SVM, KNN, and Gradient Boosting
on the same three datasets used by PDGNetV2, with two preprocessing modes:
  - flat:       (N, sensors, timesteps) -> (N, sensors * timesteps)
  - period:     same period-slicing as PDGNetV2, then flatten across periods
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
import csv
import os

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = "../data"
DATASETS = [
    # original
    "condition", "FingerMovements", "output_gear",
    # UEA (newly downloaded)
    "NATOPS", "ERing", "SelfRegulationSCP2", "HandMovementDirection",
    "BasicMotions", "Epilepsy", "UWaveGestureLibrary", "Cricket",
    "Libras", "AtrialFibrillation",
]
PERIOD_LEN = 10
NUM_PERIODS = 5

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM_RBF":            SVC(kernel="rbf", random_state=42, probability=False),
    "KNN_k5":             KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}


# ── Preprocessing ───────────────────────────────────────────────────

def z_score_normalize(X):
    """Per-sample Z-score normalization (same as PDGNetV2 data_process.py)."""
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def period_slice(x, period_len, num_periods):
    """Slice each sample into periods (same as PDGNetV2 data_process.py)."""
    N, D, L = x.shape
    total_len = period_len * num_periods
    if L < total_len:
        pad = np.zeros((N, D, total_len - L))
        x = np.concatenate([x, pad], axis=-1)
    else:
        x = x[..., :total_len]
    return x.reshape(N, num_periods, D, period_len)


def flatten(X):
    """Flatten (N, sensors, timesteps) -> (N, sensors * timesteps)."""
    return X.reshape(X.shape[0], -1)


def period_flatten(X):
    """
    Apply PDGNetV2 period-slicing then flatten:
      (N, sensors, timesteps) -> (N, 5, sensors, 10) -> (N, 5 * sensors * 10)
    """
    X = z_score_normalize(X)
    X = period_slice(X, PERIOD_LEN, NUM_PERIODS)  # (N, 5, sensors, 10)
    return X.reshape(X.shape[0], -1)


def preprocess(X_train, X_test, mode):
    """
    Apply one of two preprocessing pipelines:
      'flat'   – raw flatten + StandardScaler
      'period' – period-slice + flatten + StandardScaler (like PDGNetV2)
    """
    if mode == "flat":
        X_train_2d = flatten(X_train)
        X_test_2d = flatten(X_test)
    elif mode == "period":
        X_train_2d = period_flatten(X_train)
        X_test_2d = period_flatten(X_test)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    return X_train_scaled, X_test_scaled


# ── Main ────────────────────────────────────────────────────────────

def main():
    results = []

    for dataset in DATASETS:
        # Load
        X_train = np.load(os.path.join(DATA_DIR, dataset, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, dataset, "y_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, dataset, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, dataset, "y_test.npy"))

        # Squeeze labels if needed
        if y_train.ndim > 1:
            y_train = y_train.squeeze()
        if y_test.ndim > 1:
            y_test = y_test.squeeze()

        n_train, n_test = len(y_train), len(y_test)

        for mode in ["flat", "period"]:
            try:
                X_tr, X_te = preprocess(X_train, X_test, mode)
            except Exception as e:
                for name in MODELS:
                    results.append({
                        "dataset": dataset, "mode": mode, "model": name,
                        "acc": f"ERR: {e}", "f1": "—"
                    })
                continue

            for model_name, model in MODELS.items():
                try:
                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_tr, y_train)
                    preds = model_clone.predict(X_te)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

                    acc_str = f"{acc:.4f}"
                    f1_str = f"{f1:.4f}"
                except Exception as e:
                    acc_str = f"ERR"
                    f1_str = str(e)[:40]

                results.append({
                    "dataset": dataset, "mode": mode, "model": model_name,
                    "acc": acc_str, "f1": f1_str
                })
                print(f"  {dataset:20s} | {mode:6s} | {model_name:20s} | Acc={acc_str:8s} | F1={f1_str:8s}")

    # ── Pretty table ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY: Classic ML Baselines vs PDGNetV2")
    print("=" * 110)

    # PDGNetV2 results for reference (to be filled after running PDGNetV2)
    pdgnet = {
        "condition": ("0.99", "0.99"),
        "FingerMovements": ("0.54", "0.53"),
        "output_gear": ("NaN", "—"),
        "NATOPS": ("TBD", "TBD"),
        "ERing": ("TBD", "TBD"),
        "SelfRegulationSCP2": ("TBD", "TBD"),
        "HandMovementDirection": ("TBD", "TBD"),
        "BasicMotions": ("TBD", "TBD"),
        "Epilepsy": ("TBD", "TBD"),
        "UWaveGestureLibrary": ("TBD", "TBD"),
        "Cricket": ("TBD", "TBD"),
        "Libras": ("TBD", "TBD"),
        "AtrialFibrillation": ("TBD", "TBD"),
    }

    # Group by dataset
    for dataset in DATASETS:
        print(f"\n{'─' * 80}")
        print(f"  Dataset: {dataset}")
        print(f"{'─' * 80}")
        header = f"  {'Model':22s} | {'flat Acc':>8s} | {'flat F1':>8s} | {'period Acc':>8s} | {'period F1':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for model_name in MODELS:
            flat_row = [r for r in results if r["dataset"] == dataset and r["mode"] == "flat" and r["model"] == model_name]
            period_row = [r for r in results if r["dataset"] == dataset and r["mode"] == "period" and r["model"] == model_name]
            flat_acc = flat_row[0]["acc"] if flat_row else "—"
            flat_f1 = flat_row[0]["f1"] if flat_row else "—"
            period_acc = period_row[0]["acc"] if period_row else "—"
            period_f1 = period_row[0]["f1"] if period_row else "—"
            print(f"  {model_name:22s} | {flat_acc:>8s} | {flat_f1:>8s} | {period_acc:>8s} | {period_f1:>8s}")

        # PDGNetV2 reference line
        pdg_acc, pdg_f1 = pdgnet[dataset]
        print(f"  {'─' * 70}")
        print(f"  {'PDGNetV2 (reference)':22s} | {'—':>8s} | {'—':>8s} | {pdg_acc:>8s} | {pdg_f1:>8s}")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "mode", "model", "acc", "f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
