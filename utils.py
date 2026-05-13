import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MTSDataset(Dataset):
    """多元时间序列数据集。"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(name, path):
    """
    加载 .npy 格式的 UEA 数据集，并自动处理标签形状。

    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train = np.load(f"{path}/{name}/X_train.npy")
    y_train = np.load(f"{path}/{name}/y_train.npy")
    X_test = np.load(f"{path}/{name}/X_test.npy")
    y_test = np.load(f"{path}/{name}/y_test.npy")

    # 若标签是二维 [N, 1]，squeeze 成一维 [N]
    if y_train.ndim > 1:
        y_train = y_train.squeeze()
    if y_test.ndim > 1:
        y_test = y_test.squeeze()
    return X_train, y_train, X_test, y_test


def get_loader(X, y, batch_size, shuffle=True):
    """构建 PyTorch DataLoader。"""
    dataset = MTSDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true, y_pred):
    """
    计算分类评估指标。

    Returns:
        dict: {accuracy, f1, precision, recall}
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
