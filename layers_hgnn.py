import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNConv(nn.Module):
    """
    超图卷积层：显式执行 节点→超边聚合、超边→节点分发。

    为了缓解严格对称归一化（D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}）
    在多层堆叠时带来的梯度消失问题，此处采用单侧节点度归一化
    并引入残差连接，保证深层网络的训练稳定性。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = None

    def forward(self, x, H, Dv_inv_sqrt, De_inv):
        """
        Args:
            x:           torch.Tensor, shape [B, N, F_in]  节点初始特征
            H:           torch.Tensor, shape [B, N, E]      超图关联矩阵
            Dv_inv_sqrt: torch.Tensor, shape [B, N, 1]      节点度 Dv 的 -1/2 次方
            De_inv:      torch.Tensor, shape [B, E, 1]      超边度 De 的 -1 次方

        Returns:
            torch.Tensor, shape [B, N, F_out]  超图增强后的节点特征
        """
        x_in = x

        # 阶段1：节点 → 超边信息聚合，并进行超边度归一化
        x = torch.bmm(H.transpose(1, 2), x)   # [B, E, F_in]
        x = x * De_inv                         # D_e^{-1} H^T X

        # 阶段2：超边 → 节点信息分发，并进行节点度归一化（单侧）
        x = torch.bmm(H, x)                    # H D_e^{-1} H^T X
        x = x * Dv_inv_sqrt                    # D_v^{-1/2} H D_e^{-1} H^T X

        # 阶段3：特征变换 + 批归一化
        x = self.linear(x)
        x = self.bn(x)

        # 阶段4：残差连接 + 激活
        if self.residual is not None:
            x = x + self.residual(x_in)
        else:
            x = x + x_in
        return F.relu(x)


class HGNN(nn.Module):
    """
    多层超图卷积网络。
    """
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HGNNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(HGNNConv(hidden_dim, hidden_dim))
        self.layers.append(HGNNConv(hidden_dim, hidden_dim))

    def forward(self, x, H, Dv_inv_sqrt, De_inv):
        for layer in self.layers:
            x = layer(x, H, Dv_inv_sqrt, De_inv)
        return x
