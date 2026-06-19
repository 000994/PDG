import torch
import numpy as np


def build_sample_graph(features, k=5):
    """
    基于余弦相似度构建样本相似性图。

    Args:
        features: torch.Tensor, shape [N, F]  样本全局特征（如 GRU 最终隐状态）
        k: int, 每个样本保留的最近邻数

    Returns:
        adj: torch.Tensor, shape [N, N]  对称归一化后的带权邻接矩阵
    """
    N = features.shape[0]
    device = features.device

    # 1) 计算余弦相似度矩阵
    # L2 归一化，使每个样本特征模长为 1
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    # 余弦相似度 = 归一化后的特征矩阵乘其转置
    sim = torch.mm(features_norm, features_norm.t())

    # 处理数值误差，确保自身相似度为 1，范围在 [-1, 1]
    sim = torch.clamp(sim, -1.0, 1.0)

    # 2) 将相似度映射到 [0, 1] 区间作为边权重（也可直接使用 sim）
    # 这里使用 (sim + 1) / 2 将 [-1,1] 映射到 [0,1]，保证非负
    # 如果特征本身就是非负的，可以直接用 sim
    sim = (sim + 1.0) / 2.0

    # 3) TopK 稀疏化：保留每个样本最相似的 k 个邻居
    _, idx = torch.topk(sim, k, dim=-1)
    mask = torch.zeros_like(sim)
    mask.scatter_(-1, idx, 1.0)
    adj = sim * mask

    # 4) 对称化（保证图无向）
    adj = torch.max(adj, adj.t())

    # 5) 添加自环
    eye = torch.eye(N, device=device, dtype=adj.dtype)
    adj = adj + eye

    # 6) 对称归一化: D^{-1/2} A D^{-1/2}，适配 GCN 消息传递
    rowsum = adj.sum(dim=-1, keepdim=True)
    d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
    adj = d_inv_sqrt * adj * d_inv_sqrt.transpose(-2, -1)

    return adj
