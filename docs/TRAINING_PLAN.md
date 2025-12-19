# HyD-Mamba 训练指南 (V2.5 PGI)

## 1. 核心思想与架构优势

本架构 (G-HyD-Mamba) 为**混合嫁接架构**，旨在利用大模型 (VMamba) 的预训练特征，配合 **PGI (Progressive Geometric Injection)** 实现模态深度协作。

* **V2.5 核心改进**: 引入可学习注入因子 $\alpha$，解决“同色异谱”障碍物识别难题。
* **PGI 保护**: $\alpha$ 初始化为 0，防止训练初期深度噪声冲击主干预训练特征。

---

## 2. 训练配置 (Recommended Config)

基于 `configs/nyuv2_hyd.json`，建议生产环境参数如下：

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| **Input Size** | 512x512 | 最佳平衡点 (Jetson 适配) |
| **LR Strategy** | 5e-4 (Cosine) | 配合 AdamW 优化器 |
| **freeze_backbone_epochs** | 5-10 | **关键**: 首先微调 PGI/FPN/DuSA 等随机初始化模块 |
| **Main Loss** | OhemCE | 挖掘难例像素 |
| **Aux Loss Weight** | 0.4 | 平衡中间特征的学习强度 |
| **Detail Loss** | DetailAggregate | 强化深度梯度边界 |

---

## 3. PGI 监控建议 (PGI Monitoring)

在训练过程中，建议通过 TensorBoard 或日志监控 `encoder.inject_scale1` 到 `inject_scale4` 的数值变化：
1. **初期 (Epoch 0-5)**: 数值应保持在 0 附近（处于冻结或极低学习率阶段）。
2. **中期 (Epoch 10-50)**: 数值应缓慢上升。如果 $\alpha$ 持续为 0，说明深度信息对当前任务无贡献；如果 $\alpha$ 剧烈波动，说明 NRGM 过滤可能失效。

---

## 4. 分阶段训练流程 (Staged Process)

### 第一阶段：Grafted Fine-tuning (NYUv2)
1. **Freeze 阶段**: 训练脚本会自动锁定 RGB Backbone，仅更新融合层与解码器。
2. **Unfreeze 阶段**: 随着 $\alpha$ 解锁，深度信息开始渗透进主干网络，提升全局语义的一致性。

### 第二阶段：仿真到现实 (Sim-to-Real)
1. 在 Gazebo 仿真环境中使用真实障碍物模型（如线缆、碎石）生成训练集。
2. 借助于 PGI 的鲁棒性，模型能更快适应真实传感器（如 RealSense D435）的噪声分布。

---

## 5. 常见问题排查 (Troubleshooting)

* **Q: 注入因子 $\alpha$ 为负数?**
  * A: 正常的优化过程可能产生负权重（表示抑制），但如果大幅偏移，请检查深度图极性。
* **Q: 显存不足?**
  * A: 适当调小 Batch Size (建议为 8 或 16)。
* **Q: 模型不收敛?**
  * A: 降低学习率至 1e-4，并确保 `load_grafted_weights` 路径正确。

---

## 6. 命令参考
```bash
# 1. 挂载 NYUv2 训练 (自动处理加载嫁接权重)
python3 train.py --config configs/nyuv2_hyd.json

# 2. 快速验证架构正确性 (包含 PGI 交互测试)
python3 tools/quick_verify.py
```
