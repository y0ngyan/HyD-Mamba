# HyD-Mamba 项目工作日志 (Work Log)

**项目**: G-HyD-Mamba (面向资源受限无人机的轻量级 RGB-D 语义分割网络)

---

## 2025-12-18: 初始实现

### 1. 需求分析与规划
*   分析现有代码库 (`TUNI`, `VMamba`)
*   制定实施路线：基于 TUNI 框架、融合 VMamba 模块、手写 DuSA

### 2. 核心代码实现
*   [vmamba_layer.py](backbone_model/vmamba_layer.py): SS2Dv2 和 VSSBlock
*   [hyd_encoder.py](backbone_model/hyd_encoder.py): 非对称双流编码器
*   [parallel_hybrid.py](backbone_model/parallel_hybrid.py): CNN-Mamba 并行块
*   [dusa_attention.py](decoder/dusa_attention.py): Top-k 稀疏注意力
*   [hyd_mamba.py](hyd_mamba.py): HyDNet 主模型
*   [losses.py](tools/losses.py): OHEM 和深度梯度加权损失

### 3. 验证
*   输入: RGB (1,3,512,512) + Depth (1,1,512,512)
*   输出: Logits (1,19,512,512)
*   参数量: 2.74 M

### 4. 数据集准备
*   NYUv2 数据提取: [extract_nyuv2.py](tools/extract_nyuv2.py)
*   标签映射 (894→40): [map_nyu_labels.py](tools/map_nyu_labels.py)
*   数据适配器: [nyuv2_adapter.py](datasets/nyuv2_adapter.py)
*   训练脚本: [train.py](train.py) + [nyuv2_hyd.json](configs/nyuv2_hyd.json)

---

## 2025-12-19: V2 升级

### 5. V2 新增模块 (HyD-Mamba-v2 分支)
*   [side.py](backbone_model/side.py): **SIDE** (尺度不变深度编码) + **NRGM** (噪声鲁棒引导)
*   [large_kernel.py](backbone_model/large_kernel.py): 7x7 大核卷积深度流
*   **MambaFusion**: RGB+Depth 自适应融合

### 6. 推理验证
*   创建 [inference.py](inference.py)
*   在 best_model.pth 上完成推理可视化

### 7. V2.1 架构改进

#### 7.1 增加 Mamba 深度
*   Stage 3/4: ParallelHybridBlock 从 **1 个** 增加到 **2 个**
*   参数增量: ~0.8M

#### 7.2 Depth-guided DuSA
*   新增 `DepthEdgeExtractor` (Sobel 算子)
*   深度边缘引导 Top-k 选择
*   参数增量: **0**

### 验证结果
```
Epoch [1/5] Loss: 0.8517
Epoch [5/5] Loss: 0.0493
VERIFICATION SUCCESS
```

### 当前状态
*   总参数: **~5.3M** (轻量级)
*   架构文档: [ARCHITECTURE.md](ARCHITECTURE.md)
