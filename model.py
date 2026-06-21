import torch
import torch.nn as nn
import torch.nn.functional as F

from build_hypergraph import build_hypergraph
from layers_hgnn import HGNN
from layers_semi import SemiGCN
import config as cfg


class LowNChannelRelation(nn.Module):
    """Channel-relation encoder for small-N multivariate time series."""

    def __init__(self, use_self_attention=True):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.node_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, cfg.HGNN_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.HGNN_DIM),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        if use_self_attention:
            self.channel_attention = nn.MultiheadAttention(
                embed_dim=cfg.HGNN_DIM, num_heads=4, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(cfg.HGNN_DIM)
        self.pool_score = nn.Linear(cfg.HGNN_DIM, 1)

    def forward(self, x):
        batch_size, num_nodes, seq_len = x.shape
        node_input = x.reshape(batch_size * num_nodes, 1, seq_len)
        node_features = self.node_encoder(node_input).squeeze(-1)
        node_features = node_features.reshape(batch_size, num_nodes, cfg.HGNN_DIM)

        if self.use_self_attention:
            attended, _ = self.channel_attention(node_features, node_features, node_features, need_weights=False)
            node_features = self.attention_norm(node_features + attended)

        weights = torch.softmax(self.pool_score(node_features), dim=1)
        return (weights * node_features).sum(dim=1)


class PDGNet(nn.Module):
    """PDGNet with switches for first-round branch ablation experiments."""

    ABLATIONS = {
        "full_current": {"cnn": True, "hgnn": True, "gru": True, "sample_gcn": True},
        "no_sample_gcn": {"cnn": True, "hgnn": True, "gru": True, "sample_gcn": False},
        "cnn_gru_only": {"cnn": True, "hgnn": False, "gru": True, "sample_gcn": False},
        "cnn_only": {"cnn": True, "hgnn": False, "gru": False, "sample_gcn": False},
        "gru_only": {"cnn": False, "hgnn": False, "gru": True, "sample_gcn": False},
        "no_hgnn_but_gcn": {"cnn": True, "hgnn": False, "gru": True, "sample_gcn": True},
    }

    ENCODER_MODES = {
        "current",
        "ms_cnn",
        "gru_pool",
        "ms_cnn_gru_pool",
        "ms_cnn_gru_pool_delta",
    }
    RELATION_MODES = {
        "hgnn_zero",
        "lowN_channel_attn",
        "lowN_node_pool",
        "channel_attn_all",
    }

    def __init__(
        self,
        ablation="full_current",
        graph_mode="current",
        use_residual_gcn=True,
        encoder_mode="current",
        relation_mode="hgnn_zero",
    ):
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}. Choose from {sorted(self.ABLATIONS)}")
        if encoder_mode not in self.ENCODER_MODES:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}. Choose from {sorted(self.ENCODER_MODES)}")
        if relation_mode not in self.RELATION_MODES:
            raise ValueError(f"Unknown relation_mode: {relation_mode}. Choose from {sorted(self.RELATION_MODES)}")

        self.ablation = ablation
        self.encoder_mode = encoder_mode
        self.relation_mode = relation_mode
        switches = self.ABLATIONS[ablation]
        self.use_cnn = switches["cnn"]
        self.use_hgnn = switches["hgnn"]
        self.use_gru = switches["gru"]
        self.use_sample_gcn = switches["sample_gcn"]
        self.graph_mode = graph_mode
        self.use_residual_gcn = use_residual_gcn
        self.graph_enabled = self.use_sample_gcn and graph_mode != "no_graph"
        fusion_dims = []

        self.use_ms_cnn = encoder_mode in {"ms_cnn", "ms_cnn_gru_pool", "ms_cnn_gru_pool_delta"}
        self.use_gru_pool = encoder_mode in {"gru_pool", "ms_cnn_gru_pool", "ms_cnn_gru_pool_delta"}
        self.use_delta_input = encoder_mode == "ms_cnn_gru_pool_delta"
        encoder_channels = cfg.NUM_NODES * 2 if self.use_delta_input else cfg.NUM_NODES

        if self.use_cnn:
            if self.use_ms_cnn:
                self.ms_cnn_branches = nn.ModuleList(
                    [
                        nn.Conv1d(encoder_channels, 32, kernel_size=3, padding=1),
                        nn.Conv1d(encoder_channels, 32, kernel_size=5, padding=2),
                        nn.Conv1d(encoder_channels, 32, kernel_size=9, padding=4),
                        nn.Conv1d(encoder_channels, 32, kernel_size=17, padding=8),
                    ]
                )
                self.ms_cnn_head = nn.Sequential(
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, cfg.HGNN_DIM, kernel_size=1),
                    nn.BatchNorm1d(cfg.HGNN_DIM),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
            else:
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(encoder_channels, cfg.HGNN_DIM, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.BatchNorm1d(cfg.HGNN_DIM),
                    nn.Conv1d(cfg.HGNN_DIM, cfg.HGNN_DIM, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
            fusion_dims.append(cfg.HGNN_DIM)

        if self.use_hgnn:
            if relation_mode == "channel_attn_all":
                self.channel_relation = LowNChannelRelation(use_self_attention=True)
            else:
                self.node_feat = nn.Sequential(
                    nn.Conv1d(
                        cfg.NUM_NODES,
                        cfg.NUM_NODES * cfg.HGNN_DIM,
                        kernel_size=5,
                        padding=2,
                        groups=cfg.NUM_NODES,
                    ),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.hgnn = HGNN(cfg.HGNN_DIM, cfg.HGNN_DIM, cfg.HGNN_LAYERS)
                # Instantiate low-N replacements only when they can be used. This
                # keeps the N>=5 parameter initialization identical to hgnn_zero.
                if relation_mode == "lowN_channel_attn" and cfg.NUM_NODES < 5:
                    self.low_n_relation = LowNChannelRelation(use_self_attention=True)
                elif relation_mode == "lowN_node_pool" and cfg.NUM_NODES < 5:
                    self.low_n_relation = LowNChannelRelation(use_self_attention=False)
            fusion_dims.append(cfg.HGNN_DIM)

        if self.use_gru:
            self.gru = nn.GRU(
                input_size=encoder_channels,
                hidden_size=cfg.GRU_DIM,
                num_layers=cfg.GRU_LAYERS,
                batch_first=True,
                dropout=cfg.DROPOUT if cfg.GRU_LAYERS > 1 else 0,
            )
            if self.use_gru_pool:
                self.gru_pool_head = nn.Sequential(
                    nn.Linear(cfg.GRU_DIM * 2, cfg.GRU_DIM),
                    nn.ReLU(),
                    nn.Dropout(cfg.DROPOUT),
                )
            fusion_dims.append(cfg.GRU_DIM)

        # Disabled branches are omitted from the concatenation and from this input
        # dimension. The output representation remains cfg.GRU_DIM for every run.
        self.feat_fusion = nn.Sequential(
            nn.Linear(sum(fusion_dims), cfg.GRU_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
        )

        # The base classifier is always present. It is the complete classifier in
        # no-graph experiments and anchors residual graph experiments.
        self.base_classifier = nn.Linear(cfg.GRU_DIM, cfg.NUM_CLASSES)
        if self.graph_enabled:
            self.semi = SemiGCN(cfg.GRU_DIM, cfg.SEMI_DIM, cfg.NUM_CLASSES, cfg.SEMI_LAYERS)
            self.gcn_alpha_logit = nn.Parameter(torch.tensor(-3.0))
        else:
            self.register_parameter("gcn_alpha_logit", None)
        self.drop = nn.Dropout(cfg.DROPOUT)

    def gcn_alpha(self):
        if not self.graph_enabled:
            return 0.0
        return torch.sigmoid(self.gcn_alpha_logit)

    def forward(self, x, sample_adj=None):
        """Return embeddings without sample_adj, otherwise return class logits."""
        batch_size, num_nodes, seq_len = x.shape
        # The HGNN receives the original channels. Delta augmentation is prepared
        # separately for the CNN and GRU paths below.
        encoder_x = x
        if self.use_delta_input:
            delta_x = torch.zeros_like(x)
            delta_x[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            encoder_x = torch.cat([x, delta_x], dim=1)
        if seq_len > cfg.MAX_TIMESTEPS:
            x = F.adaptive_avg_pool1d(x, cfg.MAX_TIMESTEPS)
            encoder_x = F.adaptive_avg_pool1d(encoder_x, cfg.MAX_TIMESTEPS)

        features = []

        if self.use_cnn:
            if self.use_ms_cnn:
                multi_scale = torch.cat([branch(encoder_x) for branch in self.ms_cnn_branches], dim=1)
                features.append(self.ms_cnn_head(multi_scale).squeeze(-1))
            else:
                features.append(self.conv_layers(encoder_x).squeeze(-1))

        if self.use_hgnn:
            if self.relation_mode == "channel_attn_all":
                features.append(self.channel_relation(x))
            elif num_nodes >= 5:
                x_node = self.node_feat(x).squeeze(-1).reshape(batch_size, num_nodes, -1)
                k_hyper = min(cfg.K_HYPER, num_nodes - 1)
                incidence, dv_inv_sqrt, de_inv = build_hypergraph(x_node, k_hyper)
                x_hgnn = self.hgnn(x_node, incidence, dv_inv_sqrt, de_inv)
                features.append(x_hgnn.mean(dim=1))
            elif self.relation_mode in {"lowN_channel_attn", "lowN_node_pool"}:
                features.append(self.low_n_relation(x))
            else:
                # Retain the existing full-model behavior for datasets with <5 channels.
                features.append(torch.zeros(batch_size, cfg.HGNN_DIM, device=x.device))

        if self.use_gru:
            gru_out, hidden = self.gru(encoder_x.transpose(1, 2))
            if self.use_gru_pool:
                mean_pool = gru_out.mean(dim=1)
                max_pool = gru_out.max(dim=1).values
                features.append(self.gru_pool_head(torch.cat([mean_pool, max_pool], dim=1)))
            else:
                features.append(hidden[-1])

        embedding = self.feat_fusion(torch.cat(features, dim=1))
        if sample_adj is None:
            return embedding

        embedding = self.drop(embedding)
        logits_base = self.base_classifier(embedding)
        if not self.graph_enabled:
            return logits_base

        logits_gcn = self.semi(embedding, sample_adj)
        if self.use_residual_gcn:
            return logits_base + self.gcn_alpha() * logits_gcn
        return logits_gcn
