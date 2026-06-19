# PDGNetV2 实验结果存储库

这个文件夹用来存储PDGNetV2模型的各次改进和对比实验的结果。

## 文件结构

每个实验会生成两个文件：
- `YYYYMMDD_HHMMSS_comparison.json` - 结构化的JSON格式数据，便于程序读取和分析
- `YYYYMMDD_HHMMSS_comparison.md` - 可读的Markdown格式报告

## 关键指标

### 每个数据集的结果包含

| 字段 | 说明 |
|------|------|
| nodes | 传感器/特征数 |
| classes | 分类类别数 |
| timesteps | 原始序列长度 |
| train_samples | 训练集大小 |
| test_samples | 测试集大小 |
| pdgnet_acc | PDGNetV2模型的准确率 |
| classical_acc | 经典ML最佳准确率 |
| improvement | 相对于经典ML的改进百分比 |

### 性能对比说明

- **improvement > 0**: 模型超过经典ML基线 ✓
- **improvement < 0**: 模型劣于经典ML基线 ✗

## 现有实验

| 日期 | 实验名称 | 关键变化 | 结果文件 |
|------|---------|---------|---------|
| 2026-06-19 | No Period Slicing | 移除周期切片，保留完整时间序列 | 等待完成... |

## 使用方法

### 快速查看最新结果
```bash
ls -ltr results/ | tail -2  # 查看最新的两个文件
cat results/LATEST_comparison.md  # 读取最新的Markdown报告
```

### 对比多个实验
```python
import json

# 加载两个实验的结果
exp1 = json.load(open('results/20260619_090000_comparison.json'))
exp2 = json.load(open('results/20260619_100000_comparison.json'))

# 对比同一数据集的结果
for dataset in exp1['datasets']:
    acc1 = exp1['datasets'][dataset]['comparison']['pdgnet_acc']
    acc2 = exp2['datasets'][dataset]['comparison']['pdgnet_acc']
    print(f"{dataset}: {acc1:.4f} -> {acc2:.4f} ({(acc2-acc1)*100:+.2f}%)")
```

## 实验记录

### 原始基准（使用周期切片）
见 `experiment_log.md`
- PDGNetV2平均性能：在7个数据集上全部劣于经典ML
- 问题：周期切片导致长序列数据丢失95%以上

### 当前改进（移除周期切片）
见本文件夹中的最新comparison文件
- 目标：验证移除周期切片是否改善性能
- BasicMotions预期：0.35 → 0.80（+45%）
- condition预期：可能下降（原本就优化过）

## 下一步实验计划

1. **自适应周期切片** - 根据数据集长度动态决定是否切片
2. **Transformer模型** - 用自注意力机制替代GRU
3. **混合策略** - 针对不同类型数据集选择不同架构
4. **超参数优化** - 为小样本数据集增加正则化

## 参数配置参考

### 数据处理 (config.py)
```python
# 原始配置（使用周期切片）
PERIOD_LEN = 10
NUM_PERIODS = 5

# 当前配置（移除周期切片）
# 注意：period_slice()已从data_process.py中删除
```

### 模型配置 (config.py)
```python
HGNN_DIM = 64      # 特征维度
GRU_DIM = 64       # GRU隐藏维度
SEMI_DIM = 64      # SemiGCN维度
DROPOUT = 0.2      # Dropout比例
```

### 训练配置 (config.py)
```python
EPOCHS = 100       # 最大训练轮数
LR = 1e-3         # 初始学习率
PATIENCE = 10      # 早停耐心值
VAL_RATIO = 0.2   # 验证集比例
```

## 快速索引

- 最新实验：等待中...
- 原始基准：见 `../experiment_log.md`
- 模型代码：`../model.py`
- 训练脚本：`../trainer.py`
- 基线模型：`../baselines/classic_ml.py`
