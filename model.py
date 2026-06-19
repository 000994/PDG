import torch
import torch.nn as nn
import torch.nn.functional as F
from build_hypergraph import build_hypergraph
from layers_hgnn import HGNN
from layers_semi import SemiGCN
import config as cfg


class PDGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D CNN 全局特征: (B, N, L) → (B, HGNN_DIM)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(cfg.NUM_NODES, cfg.HGNN_DIM, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cfg.HGNN_DIM),
            nn.Conv1d(cfg.HGNN_DIM, cfg.HGNN_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 逐节点特征提取（Depthwise Conv，每个传感器独立压缩）
        # (B, N, L) → (B, N*HGNN_DIM, 1) → (B, N, HGNN_DIM)
        self.node_feat = nn.Sequential(
            nn.Conv1d(cfg.NUM_NODES, cfg.NUM_NODES * cfg.HGNN_DIM,
                      kernel_size=5, padding=2, groups=cfg.NUM_NODES),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 超图卷积网络（建模传感器间关系）
        self.hgnn = HGNN(cfg.HGNN_DIM, cfg.HGNN_DIM, cfg.HGNN_LAYERS)

        # GRU 处理时间序列: (B, L, N) → (B, GRU_DIM)
        self.gru = nn.GRU(
            input_size=cfg.NUM_NODES,
            hidden_size=cfg.GRU_DIM,
            num_layers=cfg.GRU_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.GRU_LAYERS > 1 else 0
        )

        # 三通道融合: CNN + HGNN + GRU
        fusion_dim = cfg.HGNN_DIM + cfg.HGNN_DIM + cfg.GRU_DIM
        self.feat_fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.GRU_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        # SemiGCN 处理样本间关系
        self.semi = SemiGCN(cfg.GRU_DIM, cfg.SEMI_DIM, cfg.NUM_CLASSES, cfg.SEMI_LAYERS)
        self.drop = nn.Dropout(cfg.DROPOUT)

    def forward(self, x, sample_adj=None):
        B, N, L = x.shape

        # === 长序列自适应下采样 ===
        if L > cfg.MAX_TIMESTEPS:
            x = F.adaptive_avg_pool1d(x, cfg.MAX_TIMESTEPS)
            L = cfg.MAX_TIMESTEPS

        # 1) CNN 全局特征: (B, N, L) → (B, HGNN_DIM)
        x_cnn = self.conv_layers(x).squeeze(-1)

        # 2) HGNN 超图特征（节点数≥5时启用，太少则超图无意义）
        if N >= 5:
            x_node = self.node_feat(x).squeeze(-1)     # (B, N*HGNN_DIM)
            x_node = x_node.reshape(B, N, -1)           # (B, N, HGNN_DIM)
            k_hyper = min(cfg.K_HYPER, N - 1)
            H, Dv_inv_sqrt, De_inv = build_hypergraph(x_node, k_hyper)
            x_hgnn = self.hgnn(x_node, H, Dv_inv_sqrt, De_inv)
            x_hgnn_pooled = x_hgnn.mean(dim=1)          # (B, HGNN_DIM)
        else:
            # 节点太少，HGNN分支输出零向量
            x_hgnn_pooled = torch.zeros(B, cfg.HGNN_DIM, device=x.device)

        # 3) GRU 时序特征: (B, N, L) → (B, L, N) → (B, GRU_DIM)
        x_time = x.transpose(1, 2)
        _, h_gru = self.gru(x_time)
        x_gru = h_gru[-1]

        # 4) 三通道融合
        xf = torch.cat([x_cnn, x_hgnn_pooled, x_gru], dim=1)
        xf = self.feat_fusion(xf)

        if sample_adj is None:
            return xf

        logits = self.semi(self.drop(xf), sample_adj)
        return logits
