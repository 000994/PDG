# PDGNet：多变量时间序列分类实验

本项目实现了一个面向多变量时间序列分类（MTS）的 PyTorch 模型，并包含从基础模型、消融实验到训练协议与调度诊断的一组可复现实验脚本。

当前推荐配置：

```text
--norm-mode train_global
--graph-mode no_graph
--encoder-mode current
--relation-mode lowN_channel_attn
--train-protocol train_all_fixed_100
```

该配置不使用样本图 GCN；使用完整训练集做全局标准化并训练 100 个 epoch。对于变量数少于 5 的数据集，模型用通道注意力关系分支替代原先的零向量 HGNN 输出。

## 环境

项目使用 Python 3.12 和 CUDA 12.4 版 PyTorch。建议在项目根目录创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` 固定为 `torch==2.6.0+cu124`。机器需要有兼容 CUDA 12.4 的 NVIDIA 驱动。

## 数据格式

每个数据集放在 `data/<dataset>/` 目录，包含：

```text
data/<dataset>/
├── X_train.npy  # [num_train, num_channels, seq_len]
├── y_train.npy  # [num_train]
├── X_test.npy   # [num_test, num_channels, seq_len]
└── y_test.npy   # [num_test]
```

已有数据集包括 `condition`、`FingerMovements`、`NATOPS`、`Epilepsy`、`BasicMotions`、`HandMovementDirection`、`SelfRegulationSCP2`、`ERing`、`UWaveGestureLibrary`、`Cricket`、`Libras` 和 `AtrialFibrillation`。

如需下载部分 UEA 数据集并转换为 `.npy`：

```powershell
.\.venv\Scripts\python.exe download_uea.py
```

## 模型结构

输入为 `[B, N, L]`：

- CNN 分支：提取全局局部时序模式，输出 64 维。
- HGNN / relation 分支：建模通道关系，输出 64 维。
- GRU 分支：提取时序依赖，输出 64 维。
- 融合层：拼接三路 64 维向量，得到 192 维后映射为 64 维。
- 分类器：`base_classifier(64, num_classes)`。

若启用样本图，SemiGCN 作为残差分支：

```text
logits = logits_base + sigmoid(gcn_alpha_logit) * logits_gcn
```

默认推荐配置使用 `no_graph`，因此最终 logits 只来自 `base_classifier`。

## 单次训练

使用推荐配置训练一个数据集：

```powershell
.\.venv\Scripts\python.exe trainer.py `
  --data data\Epilepsy `
  --norm-mode train_global `
  --graph-mode no_graph `
  --encoder-mode current `
  --relation-mode lowN_channel_attn `
  --train-protocol train_all_fixed_100 `
  --seed 0
```

`trainer.py` 会根据数据自动设置通道数和类别数，无需手动改写 `config.py`。

## 主要参数

| 参数 | 可选值 | 说明 |
|---|---|---|
| `--ablation` | `full_current`、`no_sample_gcn`、`cnn_gru_only`、`cnn_only`、`gru_only`、`no_hgnn_but_gcn` | 第一轮分支消融 |
| `--norm-mode` | `sample_global`、`train_channel`、`sample_channel`、`train_global`、`none` | 标准化策略 |
| `--graph-mode` | `current`、`mutual_knn`、`mutual_knn_threshold`、`no_graph` | 样本图构造 |
| `--encoder-mode` | `current`、`ms_cnn`、`gru_pool`、`ms_cnn_gru_pool`、`ms_cnn_gru_pool_delta` | CNN / GRU 编码方式 |
| `--relation-mode` | `hgnn_zero`、`lowN_channel_attn`、`lowN_node_pool`、`channel_attn_all` | 通道关系分支 |
| `--train-protocol` | `current_split`、`stratified_split`、`retrain_trainval`、`train_all_fixed_50`、`train_all_fixed_100` | 训练/验证协议 |
| `--seed` | 整数 | 随机种子 |

`train_channel` 与 `train_global` 只使用最终用于训练损失的训练样本估计 mean/std；不使用验证或测试样本统计量。

### 低变量数关系分支

`lowN_channel_attn` 的行为：

- `N >= 5`：保留原 HGNN。
- `N < 5`：每个通道经过共享一维卷积编码，再经 4-head 自注意力、残差 LayerNorm 与 attention pooling，输出 `[B, 64]`。

因此融合层输入始终是 `CNN[64] + relation[64] + GRU[64] = 192` 维。

## 实验脚本

| 脚本 | 内容 | 主要输出 |
|---|---|---|
| `run_ablation_experiments.py` | 分支消融 | `ablation_results.csv`、`summary_results.csv` |
| `run_graph_v2_experiments.py` | 残差 GCN、互惠 kNN、DropEdge | `graph_v2_results.csv`、`graph_v2_summary.csv` |
| `run_norm_experiments.py` | 五种标准化策略 | `norm_results.csv`、`norm_summary.csv` |
| `run_encoder_experiments.py` | CNN / GRU 时序编码器比较 | `encoder_results.csv`、`encoder_summary.csv` |
| `run_relation_experiments.py` | 低变量数 relation branch | `relation_results.csv`、`relation_summary.csv`、`relation_diagnostic.csv` |
| `run_protocol_experiments.py` | 训练/验证协议比较 | `protocol_results.csv`、`protocol_summary.csv`、`protocol_diagnostic.csv` |
| `run_final12_evaluation.py` | 推荐配置的 12 数据集、5 seed 重评估 | `final12_*.csv` |
| `run_schedule_experiments.py` | 学习率、训练轮数、checkpoint 调度诊断 | `schedule_*.csv` |

运行任一批量实验：

```powershell
.\.venv\Scripts\python.exe run_relation_experiments.py
```

### 完整 12 数据集评估

```powershell
.\.venv\Scripts\python.exe run_final12_evaluation.py
```

输出：

- `final12_results.csv`：每个 dataset/seed 的结果。
- `final12_summary.csv`：5 seed 的均值与标准差。
- `final12_compare.csv`：相对原始 PDGNet 和传统最佳基线的差异。
- `final12_training_curve.csv`：epoch 1、10、25、50、75、100 的训练 loss/accuracy。

### 训练调度诊断

```powershell
.\.venv\Scripts\python.exe run_schedule_experiments.py
```

该脚本默认完整运行 `condition` 的 S0--S12，并为其他目标数据集运行 S0、S3、S4、S5、S7、S8。若要所有数据集运行完整网格：

```powershell
.\.venv\Scripts\python.exe run_schedule_experiments.py --full-grid
```

调度参数示例：

```powershell
.\.venv\Scripts\python.exe trainer.py `
  --data data\condition `
  --norm-mode train_global `
  --graph-mode no_graph `
  --encoder-mode current `
  --relation-mode lowN_channel_attn `
  --max-epochs 150 `
  --lr-mult 0.5 `
  --checkpoint-mode best_train_loss `
  --seed 0 `
  --results-file schedule_results.csv `
  --schedule-curve-file schedule_training_curve.csv `
  --collapse-diagnostic-file condition_collapse_diagnostic.csv `
  --results-schema schedule
```

`best_train_loss` 仅根据完整训练集的训练 loss 保存 checkpoint，不读取测试集；`condition_collapse_diagnostic.csv` 记录选中 checkpoint 的训练/测试预测类别计数，用于诊断类别坍缩。

## 已有最终评估结果

推荐配置在 12 数据集、5 seed 下的汇总见 `final12_summary.csv`。其中 Epilepsy、BasicMotions 的平均准确率超过项目记录的传统最佳基线；ERing、NATOPS 等数据集已接近传统基线。详细对比见 `final12_compare.csv`。

## 注意事项

- 所有批量脚本默认要求 CUDA 可用；若 `torch.cuda.is_available()` 为 `False` 会直接报错。
- 实验脚本会覆盖同名结果 CSV，请先备份需要保留的结果。
- `best.pth` 与 `best_pretrain.pth` 是中间检查点，已被 `.gitignore` 忽略。
- 传统机器学习基线位于 `baselines/classic_ml.py`；从 `baselines` 目录运行时使用相对数据路径。
