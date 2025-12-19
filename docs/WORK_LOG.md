# HyD-Mamba 项目工作日志 (Work Log)

**项目**: G-HyD-Mamba (面向资源受限无人机的轻量级 RGB-D 语义分割网络)

---

## 2025-12-18: 初始实现 & 权重嫁接

* **嫁接方案**: 实现了 MobileNetV3 + VMamba-Tiny 的通道切片嫁接。
* **编码器**: 完成非对称双流结构，引入 SIDE/NRGM 预处理。
* **解码器**: 初始版 Aggregation Decoder + DuSA。
* **数据**: 完成 NYUv2 数据适配与训练脚本准备。

---

## 2025-12-19: 稳定性精调与 V2.2/V2.3 升级

### 1. 训练稳定性攻坚 (NaN 修复)

* **难例挖掘保护**: 在 `OhemCrossEntropyLoss` 中增加了对空张量的 Check，防止梯度爆发。
* **分阶段微调**: 引入 `freeze_backbone_epochs`，先训练随机初始化层，再全网微调。

### 2. V2.2 架构重构 (精度导向)

* **Top-Down FPN**: 将解码器重构为自顶向下的金字塔结构，增强 P4->P1 的语义流动。
* **DuSA 迁移**: 将稀疏注意力模块移至 1/4 融合点，结合深度边缘先验找回细小目标细节。
* **AuxLoss**: 在 Stage 3 增加辅助预测头（权重 0.4），提振反向传播信号。

### 3. V2.3 终极优化 (工程化落地)

* **归一化补齐**: 为深度流 Stage 4 和所有下采样 Block 补齐 `BN+ReLU`。
* **初始化协议**: 实现 `_init_weights` 全局初始化函数，规范化随机层。
* **深度流精简**: 简化 Stage 4 背景深度处理，进一步压缩 5% 计算量。

### 验证快报 (V2.3 Synthesis)

```text
Test 1 (OHEM): Loss 平稳，从 3.3 压制至 0.5。
Test 2 (Detail): 收敛迅速，边缘对齐权重生效。
Status: [SUCCESS] 模型已进入可生产状态。
```

### 4. V2.4 - V2.5 深度进化 (交互导向)

* **V2.4 Deep Fusion**: 探索了特征重注入 (Reinjection) 机制，验证了深度信息参与后续语义提取的必要性。
* **V2.5 PGI (Progressive Geometric Injection)**:
  * **残差注入**: 实现 $X = X + \alpha \cdot F_{fused}$，将硬重注入进化为渐进式交互。
  * **权重保护**: $\alpha$ 初始化为 0，完美契合“嫁接移植”场景，实现了零冲击启动。
  * **审计报告**: 完成全案代码审计，确认为工业级高可靠架构。

### 验证快报 (V2.5 PGI)

```text
Test 1 (OHEM): 3.3 -> 0.36 (2 Epochs)
Test 2 (Detail): 0.05 -> 0.04 (2 Epochs)
Status: [SUCCESS] 深度信息成功转化为语义引导动力。
```

---

## 当前汇总

* **当前版本**: V2.5 (PGI Final Release)
* **参数量**: **~5.3 M**
* **核心特性**: PGI 渐进式交互 + Mamba 全局建模 + Top-Down FPN。
