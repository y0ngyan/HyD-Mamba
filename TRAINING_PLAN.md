# HyD-Mamba 训练指南与进展安排

## 1. 论文阅读与核心思想
本网络 (HyD-Mamba) 参考了多篇前沿论文的核心思想，你不需要通读全篇，只需重点关注以下部分：

*   **TUNI (RGB-D Fusion)**: 重点看它的 decoder 部分如何处理 RGB 和 Thermal (类比 Depth) 的融合。你的架构中 GGFM 模块即源于此。
*   **VMamba (Visual State Space Model)**: 重点理解 `VSSBlock` 和 `SS2D` 机制。理解为什么它能以线性复杂度 $O(N)$ 替代 Transformer 的 $O(N^2)$。
*   **Small-Object Sensitive Segmentation**: 理解为什么要在 Loss 中加入对小目标的权重，以及为什么在 Decoder 末端加 Attention 能找回细节。

## 2. 数据集选择 (Datasets)

### 2.1 预训练/基准数据集 (Pre-training / Benchmarking)
建议首先在公开的标准 RGB-D 数据集上跑通过程，验证模型收敛性。

*   **Cityscapes (RGB + Disparity)**
    *   **用途**: 验证模型在复杂城市场景下的语义分割能力。
    *   **处理**: Cityscapes 提供的 Disparity（视差图）可以直接转换为 Depth 图（`Depth = Baseline * Focal / Disparity`）。
    *   **优点**: 数据量大 (5000张精细标注)，是各大论文对比 mIoU 的标准战场。

*   **NYUv2 (室内 RGB-D)**
    *   **用途**: 如果你的无人机应用场景包含室内（如搜救），这是必测的数据集。
    *   **优点**: 原生 RGB-D，无需转换。

### 2.2 目标数据集 (Target Dataset) - 核心竞争力
这是你论文 "Sim-to-Real" 故事的核心。

*   **Gazebo 仿真数据集 (Simulated)**
    *   **来源**: 利用 Gazebo 搭建无人机飞行环境（城市峡谷、废墟）。
    *   **录制**: 控制无人机飞行，同步录制 RGB 图像和深度图（Depth Camera 插件）。
    *   **标注**: 在仿真器中可以通过 Shader 直接获取 Ground Truth Segmentation Mask（这是完全免费且精确的“上帝视角”标注）。
    *   **优势**: 可以专门生成“细小障碍物”（布置大量的电线、栏杆），证明你的网络在处理小目标上的优势。

---

## 3. 训练策略 (Training Strategy)

### 3.1配置建议
基于 `HyD-Mamba/configs/base_cfg.py` 进行修改：

| 参数 | 建议值 | 说明 |
| :--- | :--- | :--- |
| **Input** | RGB-D | 4 通道输入 (3 RGB + 1 Depth) |
| **Optimizer** | AdamW | Mamba 对优化器敏感，推荐 AdamW |
| **Learning Rate** | 5e-4 | 配合 Cosine Annealing 或 Poly 策略 |
| **Weight Decay** | 0.01 | 防止过拟合 |
| **Loss** | OHEM + Detail | 使用我们实现的组合 Loss |
| **Image Size** | 512x512 | 平衡精度与速度 (Jetson 建议尺寸) |

### 3.2 训练流程
1.  **Backbone Pre-training (可选)**: 如果有条件，先在 ImageNet 上预训练 HyDEncoder 的 RGB 分支（去掉 Depth 分支），这能显著提升收敛速度。如果没有，直接从头训练 (Scratch) 需要更多 Epoch (例如 500+)。
2.  **Stage 1: Cityscapes 训练**: 使用公开数据集训练网络，直到 mIoU 达到基准线（例如 70%+）。
3.  **Stage 2: Gazebo 迁移**: 加载 Stage 1 的权重，在你的 Gazebo 数据集上进行 Fine-tuning。这一步能让模型适应无人机视角和小目标。

---

## 4. 详细进展安排 (Detailed Progress Schedule)

建议按以下 4 个阶段推进，总耗时约 3-4 周（视算力而定）。

### 第一阶段：环境与数据准备 (Week 1)
*   [ ] **Day 1-2**: 解决 Jetson/本地 PC 环境配置。尝试编译 `mamba_ssm`。如果失败，确保 `mock` 版本能跑通训练流程。
*   [ ] **Day 3-4**: 准备数据集。下载 Cityscapes，写一个脚本将 Disparity 转为 Depth，制作为 PyTorch Dataloader。
*   [ ] **Day 5**: 跑通 `train.py`。使用少量数据（Dummy Data）测试 Loss 下降，确保无梯度报错。

### 第二阶段：基准训练与 Ablation (Week 2)
*   [ ] **Day 1-3**: 在 Cityscapes 上进行完整训练。记录 mIoU、参数量、FLOPs。
*   [ ] **Day 4-5**: **消融实验 (Ablation Study)**。这是写论文的关键：
    *   `HyD-Net` (完整版) vs `No-Depth` (去掉深度流) -> 证明双流有效。
    *   `HyD-Net` vs `No-DuSA` (去掉解码器注意力) -> 证明 DuSA 有效。
    *   `HyD-Net` vs `CNN-Only` (把 Mamba 换回 CNN) -> 证明 Mamba 有效。

### 第三阶段：仿真与实机验证 (Week 3)
*   [ ] **Day 1-3**: **Sim-to-Real 实验**。在 Gazebo 中录制一段无人机穿越电线的视频。使用训练好的模型进行推理，生成可视化视频。
*   [ ] **Day 4-5**: **Jetson 部署**。使用 `torch.onnx.export` 导出模型。测试在 Jetson 上的 FPS（帧率）。如果慢，尝试 TensorRT 加速。

### 第四阶段：论文撰写与整理 (Week 4)
*   [ ] **Day 1-2**: 整理实验数据，绘制 `mIoU vs FPS` 的对比图（对比 SegFormer, PIDNet）。
*   [ ] **Day 3-5**: 撰写论文。将 README 中的架构图画精美，截取 Gazebo 中的小目标分割效果图（这往往是论文的 Highlight）。

---

## 5. 关键命令速查
```bash
# 1. 验证模型结构
python3 hyd_mamba.py

# 2. 启动训练 (假设你写好了 train.py)
python3 train.py --config configs/cityscapes_hyd.json

# 3. 导出 ONNX
python3 tools/export_onnx.py --model hyd_mamba.pth
```
