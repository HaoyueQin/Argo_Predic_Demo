# DenseTNT Optimization 后处理验证报告

> 注：本报告的验证脚本 `eval_optimization.py` 已适配并加入公开仓库
> （`scripts/eval/eval_optimization.py`），指标数据固化于
> `outputs/eval_output/optimization_comparison.json`；复现方法：训练 →
> `eval_all_models.py` 评估 → `python scripts/eval/eval_optimization.py`。
> 原始全量运行日志（`baseline_full.log` / `optimization_full.log`）未随仓库发布，
> 指标结果可直接由脚本重新生成。

## 1. 概述

本报告验证 DenseTNT（ICCV 2021）中的 **Optimization 后处理** 相对默认的 **NMS top-k** 方案的指标提升幅度。在 Argoverse 1 验证集（39,472 场景）上完成了全量对比。

### 核心结论

| Metric | Baseline (NMS top-k) | Optimization | Δ | Δ% |
|--------|:---------------------:|:------------:|:---:|:---:|
| **minADE** | 1.2199 | **0.9446** | -0.2752 | **-22.56%** |
| **minFDE** | 2.0991 | **1.3502** | -0.7490 | **-35.68%** |
| **MR** | 30.07% | **8.38%** | -21.69pp | **-72.14%** |

### 对照论文基线

注意：本验证是**离线后处理**——直接对 Dense Goal Set heatmap 做优化，**没有**训练 goal set predictor 来蒸馏。对应论文（DenseTNT, ICCV 2021）中描述的 **离线（offline）模式**，而非在线推理的 goal set predictor。

**论文 Table 2 — 离线模式指标**：

| Sampling | minFDE | MR |
|:---------|:------:|:--:|
| 2m 网格 | ~1.42 | ~9.8% |
| 1m 网格 | ~1.27 | ~7.0% |

**我们的结果**：

| Metric | 我们的值 | 与论文对比 |
|:-------|:--------:|:----------:|
| minADE | 0.94 | 不在 Table 2 离线指标中，参考用 |
| minFDE | **1.35** | 落在 2m（1.42）~ 1m（1.27）之间 |
| MR | **8.4%** | 落在 2m（9.8%）~ 1m（7.0%）之间 |

结果落在论文离线模式的正常区间。我们的表现略优，是因为 `other_params` 中启用了 `subdivide` 参数，使目标采样比论文默认更密，给优化提供了更大搜索空间。

---

## 2. 方法原理

### 2.0 在线 vs 离线模式

论文描述了两条路线：

```
Dense Goal Set Heatmap
  │
  ├─ 在线 (Online): 训练一个 goal set predictor 蒸馏优化结果
  │    ├─ 推理时无额外延迟
  │    └─ 指标: minADE~0.89, minFDE~1.47, MR~12.7%
  │
  └─ 离线 (Offline): 直接对 heatmap 做模拟退火优化（本验证）
       ├─ 每个场景 ~1.4s 后处理延迟
       └─ 指标: minFDE~1.42(2m)/~1.27(1m), MR~9.8%(2m)/~7.0%(1m)
```

本报告验证的是 **离线模式**。

### 2.1 整体数据流

```
输入: 场景轨迹数据
  │
  ▼
VectorNet 编码器 ──→ 特征向量
  │
  ▼
Decoder + Dense Goal Set Prediction ──→ (goals_2D, scores)
  │                                                │
  ├─ NMS top-k (Baseline)                          │
  │     ├─ 非极大值抑制，选出 k 个候选点              │
  │     └─ 用这些点作为最终 6 条轨迹的终点             │
  │                                                │
  └─ Optimization (Cython 后处理)                   │
        ├─ 从 Dense Goal Set 采样 6 个初始候选点        │
        ├─ 模拟退火优化 (Simulated Annealing)          │
        │    ├─ 在每个候选点周围随机扰动                  │
        │    ├─ 计算目标函数值 (MR / minFDE / MRminFDE) │
        │    ├─ 以一定概率接受更差的解（跳出局部最优）      │
        │    └─ 学习率随迭代指数衰减                      │
        ├─ 8 次独立优化（run_times=8），选最优            │
        └─ 输出: 6 个优化后的终点 + 概率                 │
```

### 2.2 Optimization 目标函数

Cython 函数 `get_value`（`utils_cython.pyx:138`）计算给定 6 个候选终点时的"期望损失"：

```
对每个 Dense Goal Point:
  1. 在该点附近 cnt_sample(36) 个亚格点采样（6x6 网格）
  2. 对每个亚格点，计算到 6 个候选终点的最小距离
  3. 如果最小距离 <= 2.0m，视为"命中"（miss_error=0）
  4. 否则 miss_error=10.0m
  5. 加权: minFDE * (1-MRratio) + miss_error * MRratio

总期望值 = Σ score_i * 亚格点平均损失
```

目标函数由 `objective` 参数控制，可选：
- `'MR'`: 纯 Miss Rate 优化（MRratio 控制 miss 惩罚权重）
- `'minFDE'`: 纯终点误差优化
- `'MRminFDE'`: 两者混合（通过 `MRratio` 调节）

### 2.3 模拟退火流程

`_get_optimal_targets`（`utils_cython.pyx:191`）：

1. **初始化**: 从 Dense Goal Set 中随机选 6 个点作为起始候选
2. **迭代优化** (num_step 次):
   - 计算当前学习率 `lr = exp(-(step/num_step) * 2)`（指数衰减）
   - 对每个候选点以 30% 概率在 [-lr, +lr] 范围内随机扰动
   - 计算扰动后的期望损失
   - 如果新损失更小，接受；否则以 1% 概率接受（跳出局部最优）
   - 记录全局最优解
3. **后处理**:
   - 用每个候选点作为"锚点"重新计算概率
   - 按概率降序排列，输出 6 个终点 + 概率

### 2.4 与 Baseline 的差异

| 环节 | Baseline (NMS top-k) | Optimization |
|:----|:--------------------:|:------------:|
| 后处理算法 | 非极大值抑制 | 模拟退火优化 |
| 候选点来源 | Dense Goal Set 聚类+NMS | Dense Goal Set 直接优化 |
| 轨迹终点 | 从候选点直接选取 | 在目标函数引导下迭代优化 |
| 稳定性 | 依赖 NMS 阈值 | 8 次独立运行取最优 |
| 计算开销 | 几乎为 0 | ~0.18s/次，8 次/场景 |

---

## 3. 代码文件清单

### 3.1 核心文件

| 文件 | 路径 | 说明 |
|:----|:----|:----|
| 验证脚本 | `scripts/eval/eval_optimization.py` | 独立验证脚本，包含完整的 Baseline/Optimization 双模式对比流程 |
| Optimization 调度 | `src/utils.py` `run_process` / `select_goals_by_optimization` | `run_process` 将任务派发给 Cython 子进程; `select_goals_by_optimization` 管理多进程池和 8 次运行 |
| Cython 优化核心 | `src/utils_cython.pyx` `get_value` / `_get_optimal_targets` | `get_value`（目标函数计算）+ `_get_optimal_targets`（模拟退火） |
| 编译产物 | `src/utils_cython.*.so` / `.pyd` | 编译后的 Cython 扩展（随环境生成，不入库；编译方法见 README） |
| 解码器 args 同步 | `src/modeling/decoder.py` `Decoder.__init__` | `global args; args = args_` 模块级 args |
| VectorNet args 同步 | `src/modeling/vectornet.py` `VectorNet.__init__` | 同上，`global args` |
| 结果文件 | `outputs/eval_output/optimization_comparison.json` | 包含 Baseline + Optimization 完整对比数据 |

### 3.2 运行日志

原始全量运行日志（`baseline_full.log` / `optimization_full.log`，开发期产物）未随仓库
发布；指标结果可由 `scripts/eval/eval_optimization.py` 直接重新生成。

---

## 4. 关键 Bug 修复

### Bug 1: 模块级 `global args` 未同步

**位置**: `decoder.py` `Decoder.__init__`、`vectornet.py` `VectorNet.__init__`

```
def __init__(self, args_: utils.Args):
    global args
    args = args_        # ← 建立模块级全局 args
```

`decoder.py` 和 `vectornet.py` 各自维护独立的模块级 `global args` 变量，通过 `import modeling.decoder as _decoder_mod; _decoder_mod.args` 访问。仅修改 `utils.args` 不会影响 Decoder/VectorNet 运行时行为。

**修复**（`scripts/eval/eval_optimization.py` 的 baseline/optimization 两个 pass 中）:

```python
_decoder_mod.args = args
_vectornet_mod.args = args
```

### Bug 2: Cython `cnt_sample != square` 断言

**位置**: `utils_cython.pyx` `get_value`（`cnt_len == 0` 时）

`get_value` 需要在 6x6 亚格点网格上采样，要求 `cnt_sample` 为完全平方数。默认值 `cnt_sample=2` 不是完全平方数，导致断言失败（当前 pyx 已改为抛出带说明的 `ValueError`）。

**修复**（`scripts/eval/eval_optimization.py` `make_args`）:

```python
a.other_params['cnt_sample'] = 36  # 6² = 36
```

### Bug 3: `MRratio` UnboundLocalError

**位置**: `src/utils.py` `run_process`

```python
if 'MRminFDE' in args.other_params:
    assert 'cnt_sample' in args.other_params
    MRratio = float(args.other_params['MRminFDE']) if ... else 1.0

if 'cnt_sample' in args.other_params:
    ...
    kwargs.update(dict(
        ...
        MRratio=MRratio,    # ← MRratio 只在上一个 if 中定义
    ))
```

当 `cnt_sample` 加入 `other_params` 但 `MRminFDE` 未同时加入时，`MRratio` 变量未定义。

**修复**：已合入公开仓库 `src/utils.py`——`run_process` 中无条件初始化
`MRratio = 1.0`（纯 MR 目标），`'MRminFDE'` 存在时按参数覆盖。任何调用方
（含 `--other_params optimization cnt_sample=36`）不再依赖外部设置 `MRminFDE`。
`scripts/eval/eval_optimization.py` 仍显式设置 `MRminFDE=1.0` 以与历史评估
配置保持一致。

---

## 5. 运行环境

| 项目 | 配置 |
|:-----|:-----|
| 系统 | WSL2 (Ubuntu 22.04) |
| Python | 3.12.3 |
| 模型权重 | `model_save_full_chunked/model_save/model.16.bin` |
| GPU | CUDA 12.1 (硬件来源 WSL) |
| Cython 扩展 | `utils_cython.cpython-312-x86_64-linux-gnu.so` |
| 依赖 | numpy 1.26.4, torch 2.5.1+cu121, pandas, tqdm, matplotlib, scipy, argoverse-api |
| 核心数 | 4 (`core_num=4`) |
| Batch size | 16 |

### 依赖安装

```bash
pip install -r requirements_densetnt.txt          # 见 README
pip install -e argoverse-api/ --no-deps           # vendored Argoverse API
cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

---

## 6. 运行说明

### 6.1 全量对比（两者都跑）

```bash
python scripts/eval/eval_optimization.py
```

### 6.2 只跑 Optimization（复用 Baseline 缓存）

```bash
python scripts/eval/eval_optimization.py --optim-only --reuse-cache
```

### 6.3 限制场景数（快速验证）

```bash
python scripts/eval/eval_optimization.py --max-scenes 2000
```

### 6.4 参数说明

| 参数 | 默认值 | 说明 |
|:-----|:------:|:-----|
| `--max-scenes` | None（全量） | 限制推理场景数 |
| `--baseline-only` | False | 只跑 NMS top-k |
| `--optim-only` | False | 只跑 Optimization |
| `--reuse-cache` | False | 复用已缓存的 temp_file（跳过预处理） |
| `--model` | `model_save_full_chunked/model_save/model.16.bin` | 模型检查点路径 |
| `--data-dir` | `val/data` | 验证集数据目录 |

---

## 7. 运行时间统计

| 阶段 | 场景数 | 耗时 | 速度 |
|:----|:------:|:----:|:----:|
| Baseline 推理 | 39,472 | 841s | 46.9 scenes/s |
| Optimization 推理 | 39,472 | 7,618s (~2.1h) | 5.2 scenes/s |
| 500 场景优化测试 | 500 | 101s | 5.0 scenes/s |
| 缓存构建（首次） | 39,472 | ~276s | - |

Optimization 慢的原因：
- 每个场景需运行 8 次独立优化（`run_times=8`）
- 每次优化执行 1,000 步迭代（`num_step=1000`，`opti_time` 默认 10000s 不触发 Cython 覆盖）
- 每步需在 36 个亚格点 × 6 个候选点上计算距离

---

## 8. 结果验证

### 8.1 小规模验证（500 场景）

| Metric | Baseline | Optimization | Δ% |
|--------|:--------:|:------------:|:---:|
| minADE | 1.2770 | 0.9458 | -25.94% |
| minFDE | 2.1708 | 1.2754 | -41.25% |
| MR | 28.80% | 5.80% | -79.86% |

### 8.2 全量验证（39,472 场景）

| Metric | Baseline | Optimization | Δ% |
|--------|:--------:|:------------:|:---:|
| minADE | 1.2199 | 0.9446 | -22.56% |
| minFDE | 2.0991 | 1.3502 | -35.68% |
| MR | 30.07% | 8.38% | -72.14% |

### 8.3 与论文离线模式对比

| Metric | 论文离线（2m） | 论文离线（1m） | 我们的值 | 对比结论 |
|:-------|:-------------:|:-------------:|:--------:|:--------:|
| minFDE | ~1.42 | ~1.27 | 1.35 | 落在区间内 |
| MR | ~9.8% | ~7.0% | 8.4% | 落在区间内 |

结果在论文离线模式正常范围内。`subdivide` 参数带来的更密采样使优化略优于 2m 网格默认设置。

---

## 9. 技术细节

### 9.1 Optimization 激活条件

在 `utils.py` 的 `run_process` 中，需要在 `args.other_params` 中包含：
- `'optimization'` → 触发调用 `select_goals_by_optimization`
- `'cnt_sample'` → 传入 Cython 的采样网格大小（必须为完全平方数）
- `'MRminFDE'` (可选) → 定义 `MRratio`（缺省时默认为 1.0，即纯 MR 目标；
  该默认值已合入 `src/utils.py`，见第 4 节 Bug 3）
- `'opti_time'` (可选) → 每个优化调用的 CPU 时间上限（默认 10000s，不触发 Cython 的 `num_step` 覆盖）

### 9.2 Cython 编译注意事项

Cython 文件编译需要 NumPy 1.x（不兼容 NumPy 2.x），编译命令：

```bash
cd src/
cython -a utils_cython.pyx
python setup.py build_ext --inplace
```

编译产物为 `utils_cython.cpython-312-x86_64-linux-gnu.so`，只能在 Linux x86_64 + Python 3.12 环境下使用。

### 9.3 多进程池模型

`select_goals_by_optimization` 维护一个全局进程池（单例模式）：
- `args.core_num` 个 Worker 进程
- 主进程通过 `multiprocessing.Queue` 分发任务
- 8 次独立优化通过 8 轮 `queue.put` 实现
- Workers 通过 `queue_res` 返回结果
- 每轮结束时主进程合并最优结果
