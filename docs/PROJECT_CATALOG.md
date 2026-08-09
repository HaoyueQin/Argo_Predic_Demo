# DenseTNT 轨迹预测项目文件目录

> 模型架构: VectorNet (DenseTNT) | 数据集: Argoverse 1 | 预测: 3秒(30帧)

---

## 1. DenseTNT 核心代码（仓库根 `src/`）

| 文件 | 说明 |
|------|------|
| `train_v4.py` | **当前训练脚本** — Python epoch 循环、LR 衰减、inline 验证、`--resume` 续训 |
| `dataset_argoverse_chunked.py` | **当前数据集** — 分块磁盘缓存加载，避免 OOM |
| `dataset_argoverse.py` | 原始数据集（整块加载，含 Pool 防死锁改造） |
| `do_eval.py` | 上游评估入口 |
| `run.py` | 上游训练入口 + 改造（`--resume`、取消 batch_size 断言） |
| `utils.py` | 工具函数（参数解析、batch 处理、loss 计算等） |
| `utils_cython.pyx` | Cython 加速核心 |
| `structs.py` | 数据结构定义 |
| `setup.py` | Cython 编译配置 |

### 模型定义 (`src/modeling/`)

| 文件 | 说明 |
|------|------|
| `vectornet.py` | VectorNet/DenseTNT — 向量化编码 + 目标预测 + 轨迹生成 |
| `decoder.py` | 解码器 — 目标评分、轨迹补全（含 top-K 容错改造） |
| `lib.py` | GNN 子图构建、LayerNorm 等基础层 |

## 2. 训练/评估工具（仓库根）

| 文件 | 说明 |
|------|------|
| `eval_all_models.py` | 批量评估 `model_save_*/model_save/` 下全部 epoch 模型 |
| `eval_single.py` | 单个模型快速评估 |
| `dashboard.py` | 训练监控 Web 面板（:8080） |
| `watchdog.sh` | 训练看门狗（崩溃自动重启） |
| `mk50k.sh` | 创建 N 条样本的训练子集（symlink） |

## 3. 模型输出（训练生成，不入库）

| 路径 | 说明 |
|------|------|
| `model_save_full_chunked/model_save/model.{1..N}.bin` | 每 epoch 模型权重 |
| `model_save_full_chunked/model_save/checkpoint.pt` | 含 optimizer state（`--resume` 用） |
| `model_save_full_chunked/temp_file/` | 训练数据缓存 |
| `model_save_full_chunked/temp_file_val/` | 验证数据缓存（独立，不污染训练缓存） |
| `model_save_full_chunked/training_history.json` | 训练历史 |
| `model_save_full_chunked/val_results_all.txt` | 验证结果汇总 |

## 4. 数据集（自行下载，不入库）

| 目录 | 样本数 | 用途 |
|------|--------|------|
| `data/raw/train/data/` | 205,942 csv | 训练集（全量） |
| `data/raw/val/data/` | 39,472 csv | 验证集（评估用） |
| `train/data_50k/` | 50,000 symlink | 训练子集（快速实验，由 mk50k.sh 生成） |

> 数据获取方式见根目录 README。预处理脚本 `scripts/preprocess/argoverse_preprocess_v2.py` 输出到 `data/cleaned/`（DenseTNT 用）与 `data/processed/`（LSTM/Kalman 用）。

## 5. 其他目录

| 目录 | 说明 |
|------|------|
| `models/` | 自写基线模型（rule_based / learning_based / loss_common / metrics_common） |
| `scripts/` | 预处理、评估、可视化、演示脚本 |
| `notebooks/` | Jupyter 演示 notebook |
| `docs/` | 技术文档 |
| `outputs/` | 评估图表与报告 |
| `argoverse-api/` | Argoverse API 上游代码（git subtree 合入，含完整历史） |

## 快速导航

```bash
# 训练（完整参数见 docs/DenseTNT_WSL搭建指南.md）
python src/train_v4.py --do_train --data_dir train/data --data_dir_for_val val/data ...

# 批量评估
python eval_all_models.py

# 监控
python dashboard.py
```
