import torch

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset (will be overridden per run)
DATASET = 'condition'
DATA_DIR = './data'
NUM_NODES = 17
NUM_CLASSES = 3

# long sequence downsampling
MAX_TIMESTEPS = 200

# hypergraph
K_HYPER = 5
HGNN_DIM = 64
HGNN_LAYERS = 2

# GRU
GRU_DIM = 64
GRU_LAYERS = 1

# sample graph (DTW)
K_SAMPLE = 5

# semi-supervised GCN
SEMI_DIM = 64
SEMI_LAYERS = 2

# training
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
VAL_RATIO = 0.2
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4
DROPOUT = 0.2
