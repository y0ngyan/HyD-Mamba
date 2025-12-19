# HyD-Mamba 训练指南 (V2.3)

## 1. 核心思想与架构优势
本架构 (G-HyD-Mamba) 为**混合嫁接架构**，旨在利用大模型 (VMamba) 的预训练特征，在小规模数据集上实现快速迁移。

*   **V2.3 核心改进**: 补齐了全网 Normalization，消除了 NaN 崩溃风险。
*   **深层监督**: 推理端隐藏 AuxHead，训练端通过 1/16 尺度的 AuxLoss 提速。

---

## 2. 训练配置 (Recommended Config)

基于 `configs/nyuv2_hyd.json`，建议生产环境参数如下：

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| **Input Size** | 512x512 | 最佳平衡点 (Jetson 适配) |
| **LR Strategy** | 5e-4 (Cosine) | 配合 AdamW 优化器 |
| **freeze_backbone_epochs** | 5-10 | **关键**: 首先微调 FPN/DuSA 等随机初始化层 |
| **Main Loss** | OhemCE | 挖掘难例像素 |
| **Aux Loss Weight** | 0.4 | 平衡中间特征的学习强度 |
| **Detail Loss** | DetailAggregate | 强化深度梯度边界 |

---

## 3. 分阶段训练流程 (Staged Process)

### 第一阶段：Grafted Fine-tuning (NYUv2)
直接加载我们准备好的 `load_grafted_weights`，在 NYUv2 上进行初步训练。
1.  **Freeze 阶段**: 训练脚本会自动锁定 RGB Backbone。
2.  **Unfreeze 阶段**: 5 轮后自动解锁全网，提升语义分割 mIoU。

### 第二阶段：Gazebo Sim-to-Real (目标场景)
1.  在仿真环境中布置无人机关键障碍物（如 0.5cm 粗细的电线）。
2.  录制同步 RGB+Depth 数据。
3.  以 NYUv2 模型作为预训练权重，在此数据集上进行 Domain Adaptation。

---

## 4. 常见问题排查 (Troubleshooting)

*   **Q: 训练初期 Loss 为 0?**
    *   A: V2.3 已大幅优化，如果仍出现，请检查输入深度图是否全黑或越界。
*   **Q: 显存不足?**
    *   A: 适当调小 Batch Size (建议为 8 或 16)。由于采用了 Mamba 和稀疏 DuSA，显存压力已远低于标准 Transformer。
*   **Q: 推理速度慢?**
    *   A: 确保在推理脚本中使用 `model.eval()`。AuxHead 会被自动短路，不再消耗计算量。

---
## 5. 命令参考
```bash
# 1. 挂载 NYUv2 训练 (自动处理加载嫁接权重)
python3 train.py --config configs/nyuv2_hyd.json

# 2. 快速验证架构正确性 (包含收敛测试)
python3 tools/quick_verify.py
```
