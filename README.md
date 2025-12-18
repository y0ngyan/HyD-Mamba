# HyD-Mamba: 面向资源受限无人机的轻量级 RGB-D 语义分割网络

## 1. 项目简介 (Project Overview)
**HyD-Mamba (Hybrid Depth-aware Mamba Network)** 是一个专为嵌入式无人机（如 NVIDIA Jetson 平台）设计的轻量级、高精度 RGB-D 语义分割网络。

### 核心痛点解决
*   **计算受限**: 采用线性复杂度的 **Mamba (State Space Model)** 替代高计算量的 Transformer，在保持全局感受野的同时显著降低算力需求。
*   **双模态冗余**: 设计 **非对称双流架构 (Asymmetric Dual-Stream)**，RGB 流负责语义，深度流仅负责几何边缘（通道数仅为 1/4），避免计算量翻倍。
*   **小目标丢失**: 引入 **DuSA (Dual-Stage Sparse Attention)** 机制，利用深度边缘引导，仅对画面中 10% 的关键区域（如线缆、障碍物边缘）进行精细化计算。

---

## 2. 目录结构 (Directory Structure)
```text
HyD-Mamba/
├── backbone_model/
│   ├── vmamba_layer.py    # [核心] VMamba 基础模块 (VSSBlock, SS2Dv2) - 含无 Cuda 环境的 Mock 实现
│   └── hyd_encoder.py     # [核心] 非对称双流编码器 (RGB CNN-Mamba + Depth CNN)
├── decoder/
│   ├── dusa_attention.py  # [创新] DuSA 稀疏注意力模块 (Top-k Selection)
│   ├── hyd_head.py        # [核心] 解码头 (多尺度融合 + DuSA 修正)
│   └── __init__.py
├── tools/
│   └── losses.py          # [工具] 自定义损失函数 (OHEM + DetailAggregateLoss)
├── configs/               # [配置] 从 TUNI 继承的配置文件
├── hyd_mamba.py           # [入口] HyDNet 主模型定义
└── README.md              # 本文档
```

---

## 3. 核心组件详解 (Key Components)

### 3.1 非对称混合编码器 (Asymmetric Hybrid Encoder)
*   **文件**: `backbone_model/hyd_encoder.py`
*   **RGB 分支**:
    *   Stage 1-2: 使用轻量级卷积 (ConvBNAct) 提取局部纹理。
    *   Stage 3-4: 使用 **VSSBlock (VMamba)** 提取全局上下文信息。
*   **深度分支 (Depth Stream)**:
    *   4 阶段纯卷积网络，通道数极低（RGB 的 1/4），专注于提取梯度和边缘。
*   **融合模块 (GGFM)**:
    *   **Geometry-Gated Fusion Module**: $F_{rgb} = F_{rgb} + F_{rgb} \times \sigma(F_{depth})$
    *   利用深度特征作为门控，在物体边缘处增强 RGB 特征。

### 3.2 细节增强解码器 (Detail-Refining Decoder)
*   **文件**: `decoder/dusa_attention.py`, `decoder/hyd_head.py`
*   **原理**:
    1.  融合多尺度特征 (1/4, 1/8, 1/16, 1/32)。
    2.  计算“重要性图” (Importance Map)，通常结合了深度边缘信息。
    3.  **Top-k 选择**: 仅选取重要性最高的 top-k (如 10%) 像素。
    4.  **稀疏注意力**: 仅对选中的像素进行 Self-Attention 计算，大幅节省算力。

### 3.3 损失函数 (Loss Functions)
*   **文件**: `tools/losses.py`
*   **OHEM Cross Entropy**: 在线挖掘难例，专注于难以分类的像素。
*   **Detail Aggregate Loss**: 利用深度图的梯度（边缘）对 Loss 进行加权，强制网络关注几何边界。

---

## 4. 环境与依赖 (Environment)
本项目在 NVIDIA Jetson 等嵌入式平台上运行时，可能会遇到 `mamba_ssm` 的编译问题。为此我们做了如下设计：

*   **智能回退 (Graceful Fallback)**: `vmamba_layer.py` 内部检测环境。
    *   如果安装了 `selective_scan_cuda`，则使用高性能 CUDA 核。
    *   如果未安装（开发/调试模式），自动回退到 Python 实现的 **Mock 算子**，确保模型能跑通（仅用于验证结构，不可用于训练收敛）。

### 推荐依赖
```bash
pip install torch torchvision
# 可选，用于最大化性能
# pip install mamba_ssm causal_conv1d
```

---

## 5. 模型参数与性能 (Model Statistics)
我们对模型进行了初步验证（`python -m HyD-Mamba.hyd_mamba`）：

*   **输入**: RGB (512x512), Depth (512x512)
*   **输出**: Class Mask (19类, 512x512)
*   **参数量 (Params)**: **2.74 M**
    *   这是一个极具竞争力的轻量化指标（相比 SegFormer-B0 的 ~3.7M 更轻）。

---

## 6. 快速开始 (Quick Start)

### 实例化模型
```python
import torch
from hyd_mamba import HyDNet

# 初始化 (默认 19 类)
model = HyDNet(num_classes=19).cuda()

# 准备数据
rgb = torch.randn(1, 3, 512, 512).cuda()
depth = torch.randn(1, 1, 512, 512).cuda()

# 前向推理
output = model(rgb, depth) # (1, 19, 512, 512)
print(output.shape)
```

### 计算 Loss
```python
from tools.losses import DetailAggregateLoss

criterion = DetailAggregateLoss()
loss = criterion(output, target_mask, depth)
loss.backward()
```

---

## 7. 后续计划 (Next Steps)
1.  **数据适配**: 编写 Dataset 类加载实际采集的 RGB-D 数据。
2.  **训练**: 编写 `train.py` 挂载损失函数进行训练。
3.  **部署**: 尝试导出 ONNX 并使用 TensorRT 加速。
