# HyD-Mamba 项目工作日志 (Work Log)

**日期**: 2025-12-18
**作者**: Antigravity (AI Assistant)
**项目**: HyD-Mamba (面向资源受限无人机的轻量级 RGB-D 语义分割网络)

---

## 1. 需求分析与规划 (Planning Phase)
*   **时间**: 2025-12-18 16:30 - 16:38
*   **任务**: 分析用户提供的论文构思，确定实施路线。
*   **产出**:
    *   分析了现有代码库 (`TUNI`, `VMamba`)。
    *   制定了 `implementation_plan.md`，提出了基于 TUNI 框架、融合 VMamba 模块、手写 DuSA 模块的技术路线。

## 2. 核心代码实现 (Execution Phase)

### 2.1 骨干网络搭建 (Backbone Implementation)
*   **组件**: `VMamba Core Layers`
*   **文件**: [vmamba_layer.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/backbone_model/vmamba_layer.py)
*   **工作内容**:
    *   从 `VMamba` 移植核心 `SS2Dv2` 和 `VSSBlock`。
    *   实现了无 CUDA 环境下的 `selective_scan_mock` 回退机制。

### 2.2 非对称编码器设计 (Encoder Design) - **[架构修正 Refined]**
*   **组件**: `HyDEncoder` & `ParallelHybridBlock`
*   **文件**: [hyd_encoder.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/backbone_model/hyd_encoder.py), [parallel_hybrid.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/backbone_model/parallel_hybrid.py)
*   **工作内容**:
    *   **架构校准**: 严格对照 HCT-Net 和 TUNI 论文要求进行了修正。
    *   **深度流 (Depth Stream)**: 简化为**极简3层卷积结构** (Shallow 3-Layer CNN)，通过步长(stride)产生多尺度边缘特征，严格符合“轻量级几何提取器”定义。
    *   **混合瓶颈层 (Hybrid Bottleneck)**: 在 Stage 3 & 4 引入了 **ParallelHybridBlock**。该模块包含并行的“局部卷积分支”和“全局 Mamba 分支”，替代了单一的 VSSBlock，更符合 HCT-Net 的设计哲学。

### 2.3 细节增强解码器 (Decoder Design)
*   **组件**: `DuSABlock` & `HyDHead`
*   **文件**: [dusa_attention.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/decoder/dusa_attention.py)
*   **工作内容**:
    *   **创新实现 DuSA**: 实现 Top-k 稀疏注意力机制，自动聚焦于图像中最关键的 10% 区域（如线缆边缘）。

### 2.4 全模型组装 (Model Assembly)
*   **组件**: `HyDNet`
*   **文件**: [hyd_mamba.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/hyd_mamba.py)
*   **工作内容**: 组装编码器、解码器及上采样模块。

### 2.5 损失函数实现 (Loss Functions)
*   **文件**: [losses.py](file:///home/yy/deepsemanticseg-test/HyD-Mamba/tools/losses.py)
*   **工作内容**: 实现 OHEM 和 深度梯度加权 Loss。

## 3. 验证与调试 (Verification & Debugging)
*   **时间**: 2025-12-18 16:40 - 16:55
*   **架构修正验证**:
    *   修正了 GGFM 通道数不匹配问题。
    *   验证了 `ParallelHybridBlock` 的前向传播，确保并行分支融合后的维度正确。
*   **最终验证**:
    *   输入: RGB (1,3,512,512) + Depth (1,1,512,512)
    *   输出: Logits (1,19,512,512)
    *   **参数量**: **2.74 M** (保持轻量)。

## 4. 文档整理 (Documentation)
*   **产出**: [README.md](file:///home/yy/deepsemanticseg-test/HyD-Mamba/README.md), [TRAINING_PLAN.md](file:///home/yy/deepsemanticseg-test/HyD-Mamba/TRAINING_PLAN.md)。

---
**状态总结**: HyD-Mamba 架构已严格对齐参考文献 (TUNI, HCT-Net, MMFNet) 的创新点定义，完成度 100%。
