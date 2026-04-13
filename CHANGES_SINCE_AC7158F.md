# 代码修改文档 (从 commit ac7158f 开始)

## 概述

本文档记录了从 commit `ac7158f`（包含该提交）到当前版本的所有重要修改。

**统计信息：**
- 提交数量：10个（包含 ac7158f）
- 修改文件：54个（去重后）
- 新增代码：11,425行
- 删除代码：154行
- 时间跨度：2026年3月27日 - 2026年4月9日

---

## 提交历史详解

### 0. 基本功能已经打通，MTP和SP均可使用
**Commit:** `ac7158f` (基准提交)  
**作者:** NENGXU003  
**时间:** 2026年3月27日  
**修改文件:** 17个文件，新增815行，删除55行

**修改内容：**
这是整个项目的基准提交，建立了完整的训练框架和基础设施。

#### 新增文件：
1. **训练配置和脚本**
   - `sft_qwen3_5_35B_config.py` (196行) - 完整的训练配置文件
   - `sft_qwen3_5_35B.sh` (107行) - 训练启动脚本
   - `hyf_test/config_gpu.py` (81行) - GPU 配置
   - `hyf_test/test_simple_qwen35_35b.sh` (88行) - 测试脚本

2. **基础设施**
   - `.gitignore` - Git 忽略配置
   - `kernel_meta/buildPidInfo.json` (30行) - 内核构建信息
   - `ls/node_0.txt` (77行) - 节点信息
   - `requirements.txt` - 依赖更新
   - `requirements/runtime.txt` - 运行时依赖

#### 核心修改：
1. **gated_deltanet.py** (257行修改)
   - 大幅重构 Gated DeltaNet 实现
   - 支持 MTP (Multi-Token Prediction) 和 SP (Sequence Parallelism)
   - 优化注意力机制实现

2. **算子优化**
   - `rms_norm/__init__.py` (11行修改) - RMS 归一化优化
   - `rotary_emb.py` (8行修改) - 旋转位置编码优化

3. **其他修改**
   - `xtuner/_testing/testcase.py` - 测试用例更新
   - `xtuner/v1/data_proto/utils.py` - 数据工具更新
   - `xtuner/v1/loss/ce_loss.py` - Loss 函数调整
   - `xtuner/v1/model/moe/moe.py` - MoE 模型调整

**技术亮点：**
- 完整的 Qwen3.5-35B-A3B 训练框架
- 支持 MTP 和 SP 两种并行策略
- 集成 Ascend NPU 硬件支持
- 建立了完善的测试和配置体系

**影响范围：**
- 建立了整个项目的基础架构
- 为后续优化提供了稳定的基础
- 支持大规模分布式训练

---

### 1. 修改causal_conv1d和固定vision确定性
**Commit:** `73e4a976`  
**作者:** NengXu001  
**时间:** 2026年3月31日  
**修改文件:** 3个文件，新增71行，删除13行

**修改内容：**
- 修改了 `gated_deltanet.py` 中的 causal_conv1d 实现
- 固定了 vision 模块的确定性
- 在 `rms_norm/__init__.py` 中添加了相关功能

**影响范围：**
- `xtuner/v1/model/compose/qwen3_vl/modeling_qwen3_vl.py`
- `xtuner/v1/module/attention/gated_deltanet.py`
- `xtuner/v1/ops/rms_norm/__init__.py`

---

### 2. 使用mojo的triton算子，64k，sp2，swap，mtp，不开随机性
**Commit:** `da0c8820`  
**作者:** NENGXU003  
**时间:** 2026年4月2日  
**修改文件:** 4个文件，新增174行，删除19行

**修改内容：**
- 集成了 mojo 的 triton 算子
- 支持配置：64k、sp2、swap、mtp
- 关闭了随机性以保证确定性
- 新增 `causal_conv1d.py` 实现（104行）

**影响范围：**
- `hyf_test/config_gpu.py` - 配置更新
- `hyf_test/test_simple_qwen35_35b.sh` - 测试脚本更新
- `xtuner/v1/module/attention/causal_conv1d.py` - 新增文件
- `xtuner/v1/module/attention/gated_deltanet.py` - 功能增强

---

### 3. base triton
**Commit:** `d7a6f175`  
**作者:** NengXu001  
**时间:** 2026年4月3日  
**修改文件:** 19个文件，新增7,906行

**修改内容：**
这是最大的一个提交，新增了大量 triton 算子实现：

#### 新增文件（triton目录）：
1. **chunk_gated_delta_rule_npu/triton/**
   - `chunk_gated_delta_rule.py` (375行)
   - `flash_gated_delta_rule.py` (504行)
   - `chunk_delta_h.py` (552行)
   - `chunk_o.py` (563行)
   - `chunk_scaled_dot_kkt.py` (337行)
   - `cumsum.py` (130行)
   - `solve_tril.py` (302行)
   - `utils.py` (346行)
   - `wy_fast.py` (351行)

2. **triton_core/**
   - `chunk_delta_h.py` (810行)
   - `chunk_o.py` (623行)
   - `chunk_o.py.bak` (备份)
   - `chunk_o.py.origin` (原始版本)
   - `chunk_scaled_dot_kkt.py` (348行)
   - `cumsum.py` (144行)
   - `l2norm.py` (314行)
   - `solve_tril.py` (277行)
   - `utils.py` (347行)
   - `wy_fast.py` (340行)

**技术亮点：**
- 实现了完整的 chunk-based gated delta rule 算法
- 提供了 triton 和 triton_core 两套实现
- 包含优化的 cumsum、wy_fast 等核心算子

---

### 4. using ascend c
**Commit:** `67beaa2c`  
**作者:** NengXu001  
**时间:** 2026年4月3日  
**修改文件:** 1个文件，新增6行，删除1行

**修改内容：**
- 在 `gated_deltanet.py` 中集成 ascend c 实现
- 小规模调整以适配 ascend 硬件

---

### 5. bindcore faster 25s to 22s
**Commit:** `2773acfd`  
**作者:** NENGXU003  
**时间:** 2026年4月3日  
**修改文件:** 4个文件，新增698行

**修改内容：**
性能优化：将训练时间从25秒优化到22秒

#### 新增文件：
1. **cpu_binder.json** (26行)
   - CPU 核心绑定配置

2. **cpu_binder.py** (577行)
   - CPU 亲和性绑定工具
   - 支持进程级别的 CPU 绑定
   - 优化多进程调度

3. **hyf_test/bind_irq.sh** (94行)
   - IRQ 中断绑定脚本
   - 减少 CPU 中断干扰

**性能提升：**
- 训练时间：25s → 22s（提升约12%）

---

### 6. fastest 21.7s
**Commit:** `d0b95a03`  
**作者:** NENGXU003  
**时间:** 2026年4月3日  
**修改文件:** 9个文件，新增152行，删除65行

**修改内容：**
进一步性能优化：达到最快21.7秒

**优化点：**
- 更新 `.gitignore` 配置
- 优化 `config_gpu.py` 配置参数
- 改进 `flash_gated_delta_rule.py` 实现
- 重构 `gated_deltanet.py`（91行修改）
- 优化 `rms_norm` 实现
- 在 `trainer.py` 中添加性能优化代码

**新增文件：**
- `kernel_meta/buildPidInfo.json` - 内核构建信息

**性能提升：**
- 训练时间：22s → 21.7s（进一步提升约1.4%）

---

### 7. [Feature] Layer-wise MoE balance loss computation
**Commit:** `8e7b9842`  
**作者:** wentiange  
**时间:** 2026年4月1日  
**修改文件:** 4个文件，新增531行，删除37行

**修改内容：**
新增 MoE（Mixture of Experts）层级平衡损失计算功能

#### 新增文件：
1. **xtuner/v1/loss/layer_moe_loss.py** (318行)
   - 实现层级 MoE 平衡损失
   - 支持异步 offload
   - 包含 all_reduce 梯度传播

2. **xtuner/v1/utils/router_offload.py** (64行)
   - Router 状态异步 offload 工具
   - 支持 CPU-GPU 异步数据传输

**修改文件：**
- `xtuner/v1/loss/__init__.py` - 导出新 loss
- `xtuner/v1/model/moe/moe.py` - 集成 MoE loss 计算（176行修改）

**功能特性：**
- 支持层级负载均衡
- 异步数据传输优化
- 分布式训练支持

---

### 8. ascendc的gdn精度正常，添加了chunkLoss的triton，但是没调通
**Commit:** `3d44bdad`  
**作者:** NENGXU003  
**时间:** 2026年4月7日  
**修改文件:** 5个文件，新增453行，删除5行

**修改内容：**
- AscendC 的 GDN（Gated DeltaNet）精度验证通过
- 添加 ChunkLoss 的 triton 实现（初步版本）

#### 新增文件：
- **xtuner/v1/loss/loss_fn_triton.py** (391行)
  - ChunkLoss 的 triton 实现
  - 支持大词汇表分块处理
  - 解决 UB 溢出问题

**修改文件：**
- `hyf_test/config_gpu.py` - 配置更新
- `hyf_test/test_simple_qwen35_35b.sh` - 测试脚本
- `xtuner/v1/loss/ce_loss.py` - 添加 chunk loss 支持
- `xtuner/v1/module/attention/flash_gated_delta_rule.py` - 功能调整

---

### 9. 消除causal_conv1d前后的transpose和GDNRMSNormGated的cast，chunkloss的triton版本可用，使用solve_tril_fast版本
**Commit:** `e3a8ff99` (HEAD)  
**作者:** NENGXU003  
**时间:** 2026年4月9日  
**修改文件:** 8个文件，新增719行，删除59行

**修改内容：**
最新提交，包含多项重要优化：

#### 核心优化：
1. **消除不必要的 transpose 操作**
   - 在 causal_conv1d 前后移除 transpose
   - 减少内存拷贝和计算开销

2. **消除 GDNRMSNormGated 的 cast 操作**
   - 优化数据类型转换
   - 提升计算效率

3. **ChunkLoss triton 版本可用**
   - 修复了之前版本的问题
   - 实现了完整的 triton 版本

4. **使用 solve_tril_fast 版本**
   - 新增优化的三角求解算法

#### 新增文件：
- **xtuner/v1/module/attention/triton_core/solve_tril_fast.py** (532行)
  - 快速三角矩阵求解
  - 优化的 triton 实现

**修改文件：**
- `hyf_test/config_gpu.py` - 配置优化
- `hyf_test/test_simple_qwen35_35b.sh` - 测试更新
- `xtuner/v1/loss/ce_loss.py` - loss 优化
- `xtuner/v1/loss/loss_fn_triton.py` - 大幅增强（202行修改）
- `xtuner/v1/module/attention/causal_conv1d.py` - 优化实现
- `xtuner/v1/module/attention/flash_gated_delta_rule.py` - 功能调整
- `xtuner/v1/module/attention/gated_deltanet.py` - 核心优化

---

## 主要功能模块

### 1. Triton 算子库

**目录结构：**
```
xtuner/v1/module/attention/
├── chunk_gated_delta_rule_npu/
│   └── triton/           # NPU 优化的 triton 实现
│       ├── chunk_gated_delta_rule.py
│       ├── flash_gated_delta_rule.py
│       ├── chunk_delta_h.py
│       ├── chunk_o.py
│       ├── chunk_scaled_dot_kkt.py
│       ├── cumsum.py
│       ├── solve_tril.py
│       ├── utils.py
│       └── wy_fast.py
└── triton_core/          # 核心算子实现
    ├── chunk_delta_h.py
    ├── chunk_o.py
    ├── chunk_scaled_dot_kkt.py
    ├── cumsum.py
    ├── l2norm.py
    ├── solve_tril.py
    ├── solve_tril_fast.py  # 最新优化版本
    ├── utils.py
    └── wy_fast.py
```

**核心算子说明：**
- `chunk_delta_h.py`: 分块计算 delta_h
- `chunk_o.py`: 分块输出计算
- `chunk_scaled_dot_kkt.py`: 分块缩放点积
- `cumsum.py`: 累积求和
- `l2norm.py`: L2 归一化
- `solve_tril.py`: 三角矩阵求解
- `solve_tril_fast.py`: 快速三角求解（优化版）
- `wy_fast.py`: WY 表示快速算法

### 2. Loss 函数优化

**新增文件：**
- `xtuner/v1/loss/layer_moe_loss.py` - MoE 平衡损失
- `xtuner/v1/loss/loss_fn_triton.py` - Triton 优化的 loss

**特性：**
- 支持大词汇表分块处理（解决 UB 溢出）
- 层级 MoE 负载均衡
- 数值稳定性优化
- Mask 处理支持

### 3. 性能优化工具

**CPU 绑定工具：**
- `cpu_binder.py` - CPU 亲和性管理
- `cpu_binder.json` - 绑定配置
- `hyf_test/bind_irq.sh` - IRQ 中断优化

**优化效果：**
- 减少 CPU 上下文切换
- 降低中断干扰
- 提升缓存命中率

### 4. Attention 模块增强

**修改文件：**
- `causal_conv1d.py` - 新增因果卷积实现
- `gated_deltanet.py` - 核心优化
- `flash_gated_delta_rule.py` - Flash 注意力变体

**优化点：**
- 消除不必要的 transpose
- 优化 cast 操作
- 集成 AscendC 实现

---

## 性能优化成果

### 训练速度提升

| 版本 | 训练时间 | 提升幅度 |
|------|---------|---------|
| 基线版本 | 25.0s | - |
| CPU 绑定优化 | 22.0s | 12.0% |
| 最终优化版本 | 21.7s | 13.2% |

### 优化技术

1. **算子级优化**
   - Triton 算子实现
   - 消除冗余操作（transpose、cast）
   - 内存访问优化

2. **系统级优化**
   - CPU 核心绑定
   - IRQ 中断优化
   - 进程调度优化

3. **算法级优化**
   - Chunk-based 处理
   - Fast 三角求解
   - Flash 注意力机制

---

## 文件变更统计

### 新增文件（54个文件）

**基准提交（ac7158f）新增：**
- `sft_qwen3_5_35B_config.py` (196行) - 训练配置
- `sft_qwen3_5_35B.sh` (107行) - 训练脚本
- `hyf_test/config_gpu.py` (81行) - GPU 配置
- `hyf_test/test_simple_qwen35_35b.sh` (88行) - 测试脚本
- `kernel_meta/buildPidInfo.json` (30行) - 内核构建信息
- `ls/node_0.txt` (77行) - 节点信息

**核心功能：**
- `xtuner/v1/loss/layer_moe_loss.py` (318行)
- `xtuner/v1/loss/loss_fn_triton.py` (511行)
- `xtuner/v1/utils/router_offload.py` (64行)
- `xtuner/v1/module/attention/causal_conv1d.py` (104行)

**Triton 算子（19个文件）：**
- chunk_gated_delta_rule_npu/triton/ 目录下 9 个文件
- triton_core/ 目录下 10 个文件

**性能优化：**
- `cpu_binder.py` (577行)
- `cpu_binder.json` (26行)
- `hyf_test/bind_irq.sh` (94行)

### 修改文件

**主要修改：**
- `xtuner/v1/module/attention/gated_deltanet.py` - 大幅重构
- `xtuner/v1/model/moe/moe.py` - MoE 功能增强
- `hyf_test/config_gpu.py` - 配置优化
- `hyf_test/test_simple_qwen35_35b.sh` - 测试脚本更新

---

## 技术亮点

### 1. Triton 算子生态

建立了完整的 Triton 算子库，包括：
- 两套实现（triton 和 triton_core）
- 完整的 chunk-based 算法
- 优化的核心算子（cumsum、wy_fast、solve_tril_fast）

### 2. MoE 训练优化

- 层级负载均衡损失
- 异步 offload 机制
- 分布式训练支持

### 3. 硬件适配

- Ascend NPU 支持
- AscendC 集成
- 多硬件平台优化

### 4. 性能工程

- 系统级优化（CPU 绑定、IRQ 优化）
- 算子级优化（消除冗余操作）
- 算法级优化（fast 算法）

---

## 后续计划

根据提交信息，以下功能仍在开发中：
1. ChunkLoss triton 版本的进一步优化
2. 更多算子的 triton 实现
3. 性能持续优化

---

## 贡献者

- **NENGXU003** - 主要贡献者（7个提交，包括基准提交）
- **NengXu001** - Triton 算子开发（2个提交）
- **wentiange** - MoE 功能开发（1个提交）

---

## 总结

从 commit `ac7158f`（包含该提交）到现在，项目经历了重大升级：

1. **代码量增长**：新增超过11,400行高质量代码
2. **性能提升**：训练速度提升13.2%
3. **功能增强**：新增 MoE 支持、Triton 算子库
4. **工程优化**：系统级性能优化工具
5. **基础设施**：完整的训练框架和配置体系

**项目发展历程：**
- **阶段一（ac7158f）**：建立基础框架，支持 MTP 和 SP
- **阶段二（后续提交）**：持续优化性能，引入 Triton 算子，增强 MoE 功能

这些修改为项目带来了显著的性能提升和功能增强，特别是在 Triton 算子实现和 MoE 训练优化方面取得了重要进展。项目从基础框架逐步演进为高性能、功能完善的训练系统。
