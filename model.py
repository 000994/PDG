import torch.nn as nn
import config as cfg
from layers import GCN
from build_graph import build_graph


class BaselineModel(nn.Module):
    """
    PDGNet-Baseline:
        输入 [B, D, L] -> 图构建 -> GCN -> 全局平均池化 -> 分类。
    其中 D 为多元变量数（节点数），L 为周期对齐后的时间长度（节点初始特征维度）。
    """
    def __init__(self, input_len, hidden_dim, num_classes, num_nodes,
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.gcn = GCN(
            in_dim=input_len,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape [B, D, L]

        Returns:
            torch.Tensor, shape [B, num_classes] (logits)
        """
        # 1) 图建模：基于时序相似度构建邻接矩阵
        adj = build_graph(x, cfg.TOPK, cfg.GRAPH_TYPE)   # [B, D, D]

        # 2) 图卷积：在变量（节点）维度上做消息传递
        feat = self.gcn(x, adj)                          # [B, D, hidden_dim]

        # 3) 读出层：全局平均池化聚合所有节点特征
        feat = feat.mean(dim=1)                          # [B, hidden_dim]
        feat = self.dropout(feat)

        # 4) 分类头
        out = self.fc(feat)                              # [B, num_classes]
        return out
