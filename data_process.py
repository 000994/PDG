import numpy as np


def period_slice(data, period_len):
    """
    周期感知切片：将多元时间序列截断或零填充至固定长度。
    Baseline 中采用前端对齐（取前 period_len 个时间步），
    可扩展为基于自相关/FFT 的动态周期检测与对齐。

    Args:
        data: np.ndarray, shape [N, D, L]
        period_len: int, 目标周期长度

    Returns:
        np.ndarray, shape [N, D, period_len]
    """
    N, D, L = data.shape
    if L >= period_len:
        data = data[..., :period_len]
    else:
        pad = np.zeros((N, D, period_len - L), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=-1)
    return data


def z_score_normalize(X_train, X_test):
    """
    按变量维度进行 Z-Score 标准化。
    使用训练集的均值和方差对训练集与测试集同时归一化。

    Args:
        X_train: np.ndarray, shape [N_train, D, L]
        X_test: np.ndarray, shape [N_test, D, L]

    Returns:
        (X_train_norm, X_test_norm)
    """
    # 在样本 N 和时间 L 维度上计算每个变量的统计量 => [1, D, 1]
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def process_data(X_train, X_test, period_len):
    """
    完整的时序预处理流程：周期切片 + 标准化。

    Returns:
        (X_train_processed, X_test_processed)
    """
    X_train = period_slice(X_train, period_len)
    X_test = period_slice(X_test, period_len)
    X_train, X_test = z_score_normalize(X_train, X_test)
    return X_train, X_test
