import torch

# ==================== 设备 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据集 ====================
DATASET = "FingerMovements"
DATA_DIR = "./data"

# ==================== 周期感知预处理 ====================
PERIOD_LENGTH = 10          # 周期切片后固定长度（可视为周期对齐后的时间步）

# ==================== 图构建 ====================
TOPK = 5                    # KNN 稀疏化 TopK
GRAPH_TYPE = "correlation"  # 可选: "correlation" | "knn" | "cosine"

# ==================== 模型维度 ====================
NUM_NODES = 28              # 多元变量数 D（节点数）
INPUT_LEN = PERIOD_LENGTH   # 每个节点的输入特征长度 = 时间步长
HIDDEN_DIM = 128            # 图卷积隐层维度
NUM_CLASSES = 2             # 分类类别数
NUM_LAYERS = 2              # GCN 层数
DROPOUT = 0.3               # Dropout 比率

# ==================== 训练 ====================
EPOCHS = 100
LR = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 15         # 早停耐心值
VAL_RATIO = 0.2             # 从训练集划分验证集比例
