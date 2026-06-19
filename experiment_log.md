# PDGNetV2 实验记录

> **日期**: 2026-06-19
> **环境**: Windows 11, Python 3.12, PyTorch 2.5.1 (CPU), scikit-learn 1.x
> **数据**: 3 个原始数据集 + 10 个 UEA 新下载数据集

---

## 1. PDGNetV2 模型配置

### 默认超参数 (config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| PERIOD_LEN | 10 | 每个周期的长度 |
| NUM_PERIODS | 5 | 每个样本切分的周期数 |
| K_HYPER | 5 | 超图 KNN 的 k 值 |
| HGNN_DIM | 64 | 超图卷积维度 |
| HGNN_LAYERS | 2 | 超图层数 |
| GRU_DIM | 64 | GRU 隐藏维度 |
| GRU_LAYERS | 1 | GRU 层数 |
| K_SAMPLE | 5 | 样本图 KNN 的 k 值 |
| SEMI_DIM | 64 | 半监督 GCN 维度 |
| SEMI_LAYERS | 2 | 半监督 GCN 层数 |
| EPOCHS | 100 | 最大训练轮数 |
| BATCH_SIZE | 32 | 批次大小（未使用） |
| LR | 1e-3 | 初始学习率 |
| VAL_RATIO | 0.2 | 验证集比例 |
| PATIENCE | 10 | 早停耐心值 |
| DROPOUT | 0.2 | Dropout 比例 |

### 各数据集适配的配置

| 数据集 | NUM_NODES | NUM_CLASSES | K_HYPER |
|--------|-----------|-------------|---------|
| condition | 17 | 3 | 5 |
| FingerMovements | 28 | 2 | 5 |
| output_gear | 3 | 5 | 2 |
| NATOPS | 24 | 6 | 5 |
| Epilepsy | 3 | 4 | 2 |
| SelfRegulationSCP2 | 7 | 2 | 5 |
| HandMovementDirection | 10 | 4 | 5 |
| BasicMotions | 6 | 4 | 5 |

---

## 2. 数据集特征

| 数据集 | 传感器(N) | 时间步(L) | 训练数 | 测试数 | 类别数 | 有效时间步(截断后) | 来源 |
|--------|----------|----------|-------|-------|-------|-------------------|------|
| condition | 17 | 60 | 1764 | 441 | 3 | 50 | 原始 |
| FingerMovements | 28 | 50 | 316 | 100 | 2 | 50 | 原始 |
| output_gear | 3 | 1024 | 8 | 2 | 5 | 50 (含NaN) | 原始 |
| NATOPS | 24 | 51 | 180 | 180 | 6 | 50 | UEA |
| ERing | 4 | 65 | 30 | 270 | 6 | 50 | UEA |
| SelfRegulationSCP2 | 7 | 1152 | 200 | 180 | 2 | 50 | UEA |
| HandMovementDirection | 10 | 400 | 160 | 74 | 4 | 50 | UEA |
| BasicMotions | 6 | 100 | 40 | 40 | 4 | 50 | UEA |
| Epilepsy | 3 | 206 | 137 | 138 | 4 | 50 | UEA |
| UWaveGestureLibrary | 3 | 315 | 120 | 320 | 8 | 50 | UEA |
| Cricket | 6 | 1197 | 108 | 72 | 12 | 50 | UEA |
| Libras | 2 | 45 | 180 | 180 | 15 | 50 (补零) | UEA |
| AtrialFibrillation | 2 | 640 | 15 | 15 | 3 | 50 | UEA |

> **注意**: 所有数据集的原始时间步均被周期切片截断/补零至 50 (5 周期 × 10 步)。对于长序列数据集 (如 SelfRegulationSCP2 的 1152 步)，超过 95% 的数据被丢弃。

---

## 3. PDGNetV2 训练详情

### 训练流程

- **阶段一 (预训练)**: 30 epochs, 使用单位矩阵替代样本图, Adam LR=1e-3
- **阶段二 (端到端)**: 最多 100 epochs, 动态样本图, LR 降至 3e-4, ReduceLROnPlateau, 早停 patience=10

### condition (17N, 60L→50, 3类, 1764/441)

```
Phase 1: Ep 10 Loss 1.03 Val 0.62 | Ep 20 Loss 0.79 Val 0.71 | Ep 30 Loss 0.51 Val 0.95
Phase 2: Ep  5 Loss 0.50 Val 0.96 | Ep 10 Loss 0.45 Val 0.97 | Ep 20 Loss 0.38 Val 0.99
          Ep 35 Loss 0.29 Val 0.99 | Early stop at Ep 38 (best val: 0.9943)
Test: Acc=0.99  F1=0.99
```

### FingerMovements (28N, 50L→50, 2类, 316/100)

```
Phase 1: Ep 10 Loss 0.64 Val 0.67 | Ep 20 Loss 0.54 Val 0.54 | Ep 30 Loss 0.36 Val 0.57
Phase 2: Ep  5 Loss 0.65 Val 0.67 | Ep 10 Loss 0.63 Val 0.59 | Early stop at Ep 13 (best val: 0.6667)
Test: Acc=0.54  F1=0.53
```

### output_gear (3N, 1024L→50, 5类, 8/2)

```
Phase 1: Loss=nan, Val=0.00 (数据含 NaN)
Result: FAILED - 原始数据包含 NaN 值
```

### NATOPS (24N, 51L→50, 6类, 180/180)

```
Phase 1: Ep 10 Loss 1.41 Val 0.64 | Ep 20 Loss 0.95 Val 0.78 | Ep 30 Loss 0.58 Val 0.72
Phase 2: Ep  5 Loss 0.89 Val 0.72 | Ep 10 Loss 0.82 Val 0.72 | Early stop at Ep 13 (best val: 0.7500)
Test: Acc=0.72  F1=0.72
```

### SelfRegulationSCP2 (7N, 1152L→50, 2类, 200/180)

```
Phase 1: Ep 10 Loss 0.64 Val 0.60 | Ep 20 Loss 0.56 Val 0.50 | Ep 30 Loss 0.46 Val 0.57
Phase 2: Ep  5 Loss 0.66 Val 0.50 | Ep 10 Loss 0.65 Val 0.53 | Early stop at Ep 11 (best val: 0.5500)
Test: Acc=0.53  F1=0.51
```

### HandMovementDirection (10N, 400L→50, 4类, 160/74)

```
Phase 1: Ep 10 Loss 1.27 Val 0.22 | Ep 20 Loss 1.04 Val 0.28 | Ep 30 Loss 0.72 Val 0.31
Phase 2: Ep  5 Loss 1.20 Val 0.28 | Ep 10 Loss 1.17 Val 0.28 | Early stop at Ep 11 (best val: 0.3125)
Test: Acc=0.30  F1=0.24
```

### BasicMotions (6N, 100L→50, 4类, 40/40)

```
Phase 1: Ep 10 Loss 1.01 Val 0.38 | Ep 20 Loss 0.60 Val 0.50 | Ep 30 Loss 0.25 Val 0.50
Phase 2: Ep  5 Loss 1.29 Val 0.25 | Ep 10 Loss 1.24 Val 0.25 | Early stop at Ep 11 (best val: 0.3750)
Test: Acc=0.35  F1=0.29
```

### Epilepsy (3N, 206L→50, 4类, 137/138)

```
Phase 1: Ep 10 Loss 1.18 Val 0.37 | Ep 20 Loss 0.92 Val 0.56 | Ep 30 Loss 0.59 Val 0.67
Phase 2: Ep  5 Loss 0.59 Val 0.70 | Ep 10 Loss 0.53 Val 0.74 | Ep 20 Loss 0.48 Val 0.81
          Ep 25 Loss 0.45 Val 0.81 | Early stop at Ep 29 (best val: 0.8148)
Test: Acc=0.67  F1=0.64
```

---

## 4. 经典 ML 基线 (完整结果)

五种算法：LogisticRegression (max_iter=5000), RandomForest (n=200), SVM_RBF, KNN (k=5), GradientBoosting (n=200)

两种预处理：**flat** (直接展平) / **period** (PDGNetV2 同款周期切片后展平)

### condition

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 1.0000 | 1.0000 | 0.9977 | 0.9977 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM_RBF | 0.9977 | 0.9977 | 1.0000 | 1.0000 |
| KNN_k5 | 1.0000 | 1.0000 | 0.9977 | 0.9977 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### FingerMovements

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| **LogisticRegression** | **0.6600** | **0.6578** | 0.6000 | 0.5974 |
| RandomForest | 0.5500 | 0.5478 | 0.5100 | 0.5096 |
| SVM_RBF | 0.5600 | 0.5572 | 0.4700 | 0.4699 |
| KNN_k5 | 0.5900 | 0.5900 | 0.5300 | 0.5277 |
| GradientBoosting | 0.5800 | 0.5785 | 0.4900 | 0.4887 |

### output_gear

| Model | flat | period |
|-------|------|--------|
| All models | ERR (NaN in data) | ERR (NaN in data) |

### NATOPS

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| **LogisticRegression** | 0.8944 | 0.8965 | **0.9056** | **0.9072** |
| RandomForest | 0.8500 | 0.8493 | 0.8556 | 0.8560 |
| SVM_RBF | 0.8056 | 0.8101 | 0.8000 | 0.8053 |
| KNN_k5 | 0.8000 | 0.8053 | 0.7833 | 0.7880 |
| GradientBoosting | 0.7333 | 0.7453 | 0.7778 | 0.7841 |

### ERing

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| **LogisticRegression** | **0.9556** | **0.9557** | 0.9222 | 0.9220 |
| RandomForest | 0.9407 | 0.9404 | 0.9185 | 0.9187 |
| SVM_RBF | 0.9296 | 0.9294 | 0.9185 | 0.9189 |
| KNN_k5 | 0.9296 | 0.9292 | 0.8704 | 0.8692 |
| GradientBoosting | 0.7370 | 0.7324 | 0.6963 | 0.6925 |

### SelfRegulationSCP2

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.4611 | 0.4610 | 0.5278 | 0.5278 |
| RandomForest | 0.4944 | 0.4941 | 0.5389 | 0.5372 |
| SVM_RBF | 0.5278 | 0.5274 | 0.5389 | 0.5389 |
| **KNN_k5** | 0.4889 | 0.4848 | **0.5556** | **0.5542** |
| GradientBoosting | 0.5111 | 0.5096 | 0.5167 | 0.5133 |

### HandMovementDirection

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| **LogisticRegression** | **0.5811** | **0.5719** | 0.3378 | 0.3453 |
| RandomForest | 0.4730 | 0.4548 | 0.2973 | 0.2796 |
| SVM_RBF | 0.5405 | 0.5267 | 0.2973 | 0.3018 |
| KNN_k5 | 0.3514 | 0.3501 | 0.3108 | 0.2709 |
| GradientBoosting | 0.3108 | 0.3081 | 0.2703 | 0.2634 |

### BasicMotions

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.7250 | 0.7000 | 0.8000 | 0.8083 |
| **RandomForest** | **0.9250** | **0.9246** | 0.9000 | 0.8990 |
| SVM_RBF | 0.8500 | 0.8465 | 0.9000 | 0.9015 |
| KNN_k5 | 0.3250 | 0.2311 | 0.6500 | 0.6123 |
| GradientBoosting | 0.7500 | 0.7370 | 0.6000 | 0.5948 |

### Epilepsy

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.6522 | 0.6253 | 0.2754 | 0.2547 |
| RandomForest | 0.7899 | 0.7795 | 0.7391 | 0.7429 |
| **SVM_RBF** | **0.8406** | **0.8265** | 0.5942 | 0.5939 |
| KNN_k5 | 0.5217 | 0.4256 | 0.5580 | 0.5357 |
| GradientBoosting | 0.7464 | 0.7408 | 0.6377 | 0.6331 |

### UWaveGestureLibrary

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.7812 | 0.7816 | 0.5219 | 0.5163 |
| RandomForest | 0.8688 | 0.8687 | 0.5750 | 0.5774 |
| SVM_RBF | 0.8719 | 0.8714 | 0.5188 | 0.5107 |
| **KNN_k5** | **0.8750** | **0.8737** | 0.5062 | 0.4948 |
| GradientBoosting | 0.6594 | 0.6587 | 0.5000 | 0.5018 |

### Cricket

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.9306 | 0.9260 | 0.8472 | 0.8397 |
| RandomForest | 0.9028 | 0.9005 | 0.8194 | 0.8061 |
| **SVM_RBF** | **0.9444** | **0.9430** | 0.8472 | 0.8474 |
| KNN_k5 | 0.9167 | 0.9121 | 0.7917 | 0.7772 |
| GradientBoosting | 0.5278 | 0.4971 | 0.6528 | 0.6356 |

### Libras

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.7778 | 0.7707 | 0.7111 | 0.6992 |
| **RandomForest** | 0.7333 | 0.7346 | **0.8000** | **0.7957** |
| SVM_RBF | 0.7333 | 0.7159 | 0.7278 | 0.7226 |
| KNN_k5 | 0.6278 | 0.6249 | 0.6278 | 0.6145 |
| GradientBoosting | 0.6556 | 0.6497 | 0.5833 | 0.5829 |

### AtrialFibrillation

| Model | flat Acc | flat F1 | period Acc | period F1 |
|-------|----------|---------|------------|-----------|
| LogisticRegression | 0.4000 | 0.3444 | 0.2667 | 0.2095 |
| RandomForest | 0.2000 | 0.1818 | 0.2667 | 0.2095 |
| **SVM_RBF** | **0.5333** | **0.4850** | 0.2667 | 0.2645 |
| KNN_k5 | 0.4000 | 0.3205 | 0.3333 | 0.2762 |
| GradientBoosting | 0.3333 | 0.2667 | 0.2667 | 0.2323 |

---

## 5. 最终对比汇总

| 数据集 | N | 训练数 | 最佳经典 ML | PDGNetV2 | 胜者 |
|--------|---|-------|------------|----------|------|
| condition | 17 | 1764 | **1.000** (RF/LR/KNN-GB flat) | 0.99 | Classic ML |
| FingerMovements | 28 | 316 | **0.660** (LR flat) | 0.54 | Classic ML |
| output_gear | 3 | 8 | NaN (数据含NaN) | NaN | — |
| NATOPS | 24 | 180 | **0.906** (LR period) | 0.72 | Classic ML |
| ERing | 4 | 30 | **0.956** (LR flat) | — | — |
| SelfRegulationSCP2 | 7 | 200 | **0.556** (KNN period) | 0.53 | Classic ML (微弱) |
| HandMovementDirection | 10 | 160 | **0.581** (LR flat) | 0.30 | Classic ML |
| BasicMotions | 6 | 40 | **0.925** (RF flat) | 0.35 | Classic ML |
| Epilepsy | 3 | 137 | **0.841** (SVM flat) | 0.67 | Classic ML |
| UWaveGestureLibrary | 3 | 120 | **0.875** (KNN flat) | — | — |
| Cricket | 6 | 108 | **0.944** (SVM flat) | — | — |
| Libras | 2 | 180 | **0.800** (RF period) | — | — |
| AtrialFibrillation | 2 | 15 | **0.533** (SVM flat) | — | — |

> PDGNetV2 跑通的 7 个数据集中，0 个优于经典 ML。

---

## 6. 分析结论

1. **PDGNetV2 在所有非平凡数据集上均劣于至少一种经典 ML 算法**
2. 周期切片将大部分数据集截断至 50 时间步，对长序列造成严重信息丢失
3. SemiGCN 样本图在小训练集上表现不佳（BasicMotions 40 样本, HandMovementDirection 160 样本）
4. 模型超参数似乎专为 condition 数据集调优，泛化能力差
5. 简单逻辑回归在 7/13 数据集上取得最优或接近最优结果

### 文件清单

| 文件 | 说明 |
|------|------|
| `config.py` | PDGNetV2 超参数配置 |
| `model.py` | PDGNetV2 模型定义 (HGNN+GRU+SemiGCN) |
| `trainer.py` | 训练脚本 (两阶段) |
| `data_process.py` | 数据预处理 (Z-score + 周期切片) |
| `build_hypergraph.py` | 传感器超图构建 |
| `layers_hgnn.py` | HGNN 卷积层 |
| `build_sample_graph.py` | 样本图构建 |
| `layers_semi.py` | 半监督 GCN 层 |
| `baselines/classic_ml.py` | 经典 ML 基线脚本 |
| `baselines/results.csv` | 经典 ML 基线结果 |
| `download_uea.py` | UEA 数据集下载脚本 |
| `data/` | 13 个数据集 (.npy 格式) |
