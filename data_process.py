import numpy as np


NORM_MODES = ("sample_global", "train_channel", "sample_channel", "train_global", "none")


def z_score_normalize(X):
    """Original per-sample global Z-score normalization."""
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def normalize_data(X_train, X_test, train_split_mask, norm_mode="sample_global"):
    """Normalize train/validation/test data without leaking validation/test statistics.

    ``train_split_mask`` identifies precisely the samples used in the training loss.
    It is used exclusively to estimate statistics for train-based modes.
    """
    if norm_mode not in NORM_MODES:
        raise ValueError(f"Unknown norm_mode: {norm_mode}. Choose from {NORM_MODES}")
    if norm_mode == "none":
        return X_train, X_test
    if norm_mode == "sample_global":
        return z_score_normalize(X_train), z_score_normalize(X_test)
    if norm_mode == "sample_channel":
        train_mean = X_train.mean(axis=2, keepdims=True)
        train_std = X_train.std(axis=2, keepdims=True) + 1e-8
        test_mean = X_test.mean(axis=2, keepdims=True)
        test_std = X_test.std(axis=2, keepdims=True) + 1e-8
        return (X_train - train_mean) / train_std, (X_test - test_mean) / test_std

    fit_data = X_train[train_split_mask]
    if norm_mode == "train_channel":
        mean = fit_data.mean(axis=(0, 2), keepdims=True)
        std = fit_data.std(axis=(0, 2), keepdims=True) + 1e-8
    else:  # train_global
        mean = fit_data.mean()
        std = fit_data.std() + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def process_data(X_train, X_test):
    """Backward-compatible wrapper for the original preprocessing behavior."""
    return normalize_data(X_train, X_test, np.ones(len(X_train), dtype=bool), "sample_global")
