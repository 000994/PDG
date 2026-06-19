import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiGCNConv(nn.Module):
    """
    半监督图卷积层：在样本相似性图上执行消息聚合 + 线性变换。
    """
    def __init__(self, in_dim, out_dim, use_relu=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_relu = use_relu

    def forward(self, x, adj):
        x = self.linear(adj @ x)
        if self.use_relu:
            x = F.relu(x)
        return x


class SemiGCN(nn.Module):
    """
    多层半监督图卷积网络。
    将样本全局特征作为节点特征，在样本相似性图上执行 GCN 卷积，
    实现特征平滑与标签扩散。最后一层不激活，直接输出分类 logits。
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SemiGCNConv(in_dim, hidden_dim, use_relu=True))
        for _ in range(num_layers - 2):
            self.layers.append(SemiGCNConv(hidden_dim, hidden_dim, use_relu=True))
        self.layers.append(SemiGCNConv(hidden_dim, out_dim, use_relu=False))

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x
