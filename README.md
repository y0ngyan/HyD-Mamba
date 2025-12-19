# HyD-Mamba: 面向资源受限无人机的轻量级 RGB-D 语义分割网络 (V2.3)

## 1. 项目简介 (Project Overview)
**HyD-Mamba (Hybrid Depth-aware Mamba Network)** 是一个专为嵌入式无人机（如 NVIDIA Jetson 平台）设计的轻量级、高精度 RGB-D 语义分割网络。

### 核心痛点解决
*   **计算受限**: 采用线性复杂度的 **Mamba (State Space Model)** 替代高计算量的 Transformer，在保持全局感受野的同时显著降低算力需求。
*   **双模态冗余**: 设计 **非对称双流架构 (Asymmetric Dual-Stream)**，深度分支通道数仅为 RGB 的 1/4，专注于几何结构提取。
*   **边缘模糊**: 引入 **Top-Down FPN (特征金字塔)** 结构与 **DuSA (稀疏注意力)** 机制，利用深度边缘引导，精准重建电线、障碍物边缘。
*   **训练稳定性**: 采用 **权重嫁接 (Weight Grafting)** 技术与 **深层辅助监控 (AuxLoss)**，确保模型在混合架构下快速平稳收敛。

---

## 2. 目录结构 (Directory Structure)
```text
HyD-Mamba/
├── backbone_model/
│   ├── vmamba_layer.py    # [核心] SS2Dv2 和 VSSBlock (含 Mock 实现)
│   ├── hyd_encoder.py     # [核心] 嫁接式编码器 (MobileNetV3 + VMamba)
│   └── side.py            # [预处理] SIDE (尺度不变) + NRGM (抗噪)
├── decoder/
│   ├── hyd_head.py        # [核心] Top-Down FPN 解码头 + 多级融合
│   └── dusa_attention.py  # [创新] DuSA 稀疏注意力 (深度边缘引导)
├── tools/
│   └── losses.py          # [工具] 鲁棒 OHEM + DetailAggregateLoss
├── configs/               # [配置] 训练与模型参数配置
├── hyd_mamba.py           # [入口] HyDNet 主模型定义
└── README.md              # 本文档
```

---

## 3. 核心组件升级 (Key Upgrades)

### 3.1 嫁接式非对称编码器 (Grafted Encoder)
*   **RGB 分支**: 巧妙复用 MobileNetV3 (Stage 1-2) 的高效纹理提取与 VMamba (Stage 3-4) 的全局建模能力。
*   **精简深度流**: Stage 1-3 采用大核卷积 (LK-Conv)，Stage 4 简化为轻量级 Conv-BN-ReLU 结构，平衡精度与功耗。

### 3.2 异步 Top-Down 解码器 (FPN Decoder)
*   **原理**: 改变传统的 Aggregation 拼接模式，采用逐级上采样融合 (P4->P3->P2->P1)。
*   **DuSA 迁移**: 将稀疏注意力模块迁移至 1/4 尺度融合点，利用深度梯度图引导注意力流向物体边界。

### 3.3 深层辅助监控 (Deep Supervision)
*   **AuxHead**: 在 Stage 3 (1/16) 尺度引入多层预测头 (Conv-BN-ReLU-Dropout-Conv)，权重设为 0.4，加速语义信息的反向传播。

---

## 4. 模型性能 (Model Statistics)
*   **输入**: RGB (512x512), Depth (512x512)
*   **参数量 (Params)**: **~5.3 M**（推理端不计入辅助头）。
*   **稳定性**: 通过标准化 Normalization 与全局初始化方案，彻底解决 `loss=nan` 风险。

---

## 5. 快速开始 (Quick Start)

### 实例化模型
```python
import torch
from hyd_mamba import HyDNet

# 初始化 (支持 Stage 1-2/3-4 嫁接权重加载)
model = HyDNet(num_classes=19).cuda()

# 前向推理 (推理模式自动跳过 AuxHead)
model.eval()
rgb = torch.randn(1, 3, 512, 512).cuda()
depth = torch.randn(1, 1, 512, 512).cuda()
# 推理仅返回 Main Output
output = model(rgb, depth) # (1, 19, 512, 512)
```

### 多损失训练
```python
# 训练模式返回 (Main_Logits, Aux_Logits)
model.train()
main_out, aux_out = model(rgb, depth)
# 总 Loss = 主 Loss + 0.4 * 辅助 Loss
loss = main_criterion(main_out, target) + 0.4 * aux_criterion(aux_out, target)
```

---

## 6. 后续计划 (Next Steps)
1.  **数据适配**: 已完成 NYUv2 数据适配器的开发。
2.  **仿真验证**: 在仿真器生成的复杂障碍物环境下验证边际对齐精度。
3.  **TensorRT**: 优化 SS2D 算子，实现 30+ FPS 推理。
