import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    图卷积层：基于邻接矩阵做消息聚合 + 线性变换 + ReLU + Dropout + BatchNorm。
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, use_bn=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        """
        Args:
            x:    torch.Tensor, shape [B, D, F_in]
            adj:  torch.Tensor, shape [B, D, D]

        Returns:
            torch.Tensor, shape [B, D, F_out]
        """
        # 消息聚合: adj @ x  => [B, D, F_in]
        out = torch.matmul(adj, x)
        # 特征变换
        out = self.linear(out)                # [B, D, F_out]
        out = F.relu(out)
        out = self.dropout(out)
        # BatchNorm: 需要在通道维度上做 => [B, F_out, D]
        if self.use_bn:
            out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return out


class GCN(nn.Module):
    """
    多层图卷积网络。
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hidden_dim, dropout))
        for _ in range(max(0, num_layers - 2)):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
        if num_layers >= 2:
            self.layers.append(GCNLayer(hidden_dim, out_dim, dropout))

    def forward(self, x, adj):
        """
        Args:
            x:   torch.Tensor, shape [B, D, F_in]
            adj: torch.Tensor, shape [B, D, D]

        Returns:
            torch.Tensor, shape [B, D, F_out]
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x
