import torch
import torch.nn.functional as F


def sym_normalize(adj):
    """
    对称归一化: D^{-1/2} A D^{-1/2}
    适用于 GCN 的消息传递，防止数值爆炸/消失。

    Args:
        adj: torch.Tensor, shape [B, D, D]

    Returns:
        torch.Tensor, shape [B, D, D]
    """
    rowsum = adj.sum(dim=-1, keepdim=True)           # [B, D, 1]
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(-2, -1)


def build_graph(x, topk=5, graph_type="correlation"):
    """
    基于时间序列相似度构建批量图邻接矩阵。

    Args:
        x: torch.Tensor, shape [B, D, L]  (B: batch, D: 变量/节点数, L: 时间长度)
        topk: int, KNN 稀疏化保留的最近邻数
        graph_type: str, "correlation" | "knn" | "cosine"

    Returns:
        adj: torch.Tensor, shape [B, D, D], 已做对称归一化
    """
    B, D, L = x.shape

    if graph_type == "correlation":
        # 皮尔逊相关系数图
        mean = x.mean(dim=-1, keepdim=True)             # [B, D, 1]
        xm = x - mean
        cov = torch.matmul(xm, xm.transpose(-2, -1))    # [B, D, D]
        std = torch.sqrt((xm ** 2).sum(dim=-1, keepdim=True))  # [B, D, 1]
        adj = cov / (std * std.transpose(-2, -1) + 1e-8)
    elif graph_type == "knn":
        # 欧氏距离 + 高斯核
        dist = torch.cdist(x, x)                        # [B, D, D]
        adj = torch.exp(-dist ** 2)
    elif graph_type == "cosine":
        # 余弦相似度
        x_norm = F.normalize(x, p=2, dim=-1)            # [B, D, L]
        adj = torch.matmul(x_norm, x_norm.transpose(-2, -1))  # [B, D, D]
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # ReLU 抑制负值（保留正相关/相似度）
    adj = F.relu(adj)

    # TopK 稀疏化：只保留每个节点最相似的 topk 个邻居
    if 0 < topk < D:
        topk_val, topk_idx = torch.topk(adj, topk, dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_idx, 1.0)
        adj = adj * mask

    # 添加自环
    eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, D, D]
    adj = adj + eye

    # 对称归一化
    adj = sym_normalize(adj)
    return adj
