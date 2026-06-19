import torch


def euclidean_dist(x):
    """
    计算批次内节点间的欧氏距离矩阵。
    Args:
        x: torch.Tensor, shape [B, N, F]
    Returns:
        torch.Tensor, shape [B, N, N]
    """
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    dist = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    return torch.sqrt(torch.clamp(dist, min=1e-8))


def build_hypergraph(x, k=5):
    """
    构建超图关联矩阵 H 及超图卷积所需的对称归一化结构。

    超边生成规则：
    1. 基于节点特征相似度（欧氏距离），对每个节点使用 KNN 选取 k 个最近邻。
    2. 每个节点与其 k 个近邻构成一条超边；由于 KNN 的对称性，互为近邻的节点将
       同时出现在彼此的超边中，从而实现“若多个节点互为近邻，则归入同一条超边”
       的语义（以各节点为中心的超边天然包含其近邻）。

    Args:
        x: torch.Tensor, shape [B, N, F]  (B: batch, N: 节点/传感器数, F: 特征维度)
        k: int, KNN 近邻数（包含自身，因为自身距离为 0 最小）

    Returns:
        H:           torch.Tensor, shape [B, N, E]  节点-超边二值关联矩阵 (E=N)
        Dv_inv_sqrt: torch.Tensor, shape [B, N, 1]  节点度矩阵 Dv 的 -1/2 次方
        De_inv:      torch.Tensor, shape [B, E, 1]  超边度矩阵 De 的 -1 次方
    """
    B, N, F = x.shape
    # 1) 计算节点间欧氏距离
    dist = euclidean_dist(x)  # [B, N, N]

    # 2) 每个节点选取 k 个最近邻（自身距离为 0，必然包含在 topk 中）
    _, idx = torch.topk(-dist, k, dim=-1)  # [B, N, k]

    # 3) 构建超图关联矩阵 H: [B, N, E]，此处 E = N（每个节点作为一条超边的中心）
    H = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
    for b in range(B):
        for i in range(N):
            # 超边 i 包含节点 i 及其 k 个近邻
            H[b, idx[b, i], i] = 1.0

    # 4) 计算节点度矩阵 Dv 和超边度矩阵 De
    Dv = H.sum(dim=-1, keepdim=True)   # [B, N, 1]
    De = H.sum(dim=-2, keepdim=True)   # [B, 1, E]

    # 5) 对称归一化所需的度矩阵逆
    Dv_inv_sqrt = torch.pow(Dv + 1e-8, -0.5)   # [B, N, 1]
    De_inv = torch.pow(De + 1e-8, -1.0)         # [B, 1, E]
    # 调整为 [B, E, 1] 以便后续在超边维度上做广播乘法
    De_inv = De_inv.transpose(1, 2)             # [B, E, 1]

    return H, Dv_inv_sqrt, De_inv
