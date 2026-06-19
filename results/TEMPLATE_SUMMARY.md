# 实验结果总结模板

## 实验信息

**实验名称**: [输入实验名称]  
**日期**: [自动填充]  
**模型版本**: PDGNetV2  
**主要改动**: [描述修改]  

## 快速对比

### 性能概览
| 指标 | 值 |
|------|-----|
| 测试数据集数 | 12 |
| 超过经典ML的数据集 | X |
| 劣于经典ML的数据集 | Y |
| 平均改进 | Z% |

### 数据集胜负统计
- ✓ 胜利: [数据集1], [数据集2], ...
- ✗ 失败: [数据集3], [数据集4], ...
- ➖ 平衡: [数据集5], ...

## 详细结果

见 `YYYYMMDD_HHMMSS_comparison.md` 文件。

## 关键发现

### 优势
1. ...
2. ...

### 劣势
1. ...
2. ...

## 后续改进方向

1. **短期** (下个实验)
   - [ ] ...
   
2. **中期** (1-2个月)
   - [ ] ...
   
3. **长期** (半年+)
   - [ ] ...

## 对比前版本

如果有之前的实验版本，可以这样对比：

```bash
python compare_experiments.py \
  results/PREVIOUS_VERSION_comparison.json \
  results/CURRENT_VERSION_comparison.json
```

## 文件清单

- `YYYYMMDD_HHMMSS_comparison.json` - 结构化数据
- `YYYYMMDD_HHMMSS_comparison.md` - 详细报告
- 本文件 - 快速总结

---

**生成时间**: [自动]  
**相关代码**: model.py, data_process.py, trainer.py  
**配置**: 见 config.py  
