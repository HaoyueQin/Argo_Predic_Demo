# Argo_Predic_Demo

基于 **Argoverse 1** 运动预测数据集的车辆轨迹预测项目——复现 **DenseTNT**
（Gu et al., ICCV 2021）并附带训练/评估工具链，同时提供规则基线与学习基线
（卡尔曼滤波、匀速模型、LSTM）用于方法对比。

## 方法对比

| 方法 | 类型 | minADE (m) | minFDE (m) | MR @2m |
|------|------|:----------:|:----------:|:------:|
| CV（匀速模型） | 规则基线 | ~5.0 | ~10.0 | ~80% |
| 卡尔曼滤波 | 规则基线 | 2.26 | 5.15 | ~50% |
| LSTM | 学习基线 | 1.99 | 4.60 | ~35% |
| **DenseTNT（本项目）** | VectorNet + 地图（6 模态） | **1.20** | **2.09** | 30.0% |
| DenseTNT + 目标优化 | VectorNet + 地图 | **0.94** | **1.35** | **8.4%** |

指标在 Argoverse 1.1 验证集（39,472 场景）上计算。DenseTNT 模型在 60k 子集上
训练 16 个 epoch；详见 `outputs/charts/report_analysis.md`。

## 仓库结构

```
├── src/                        # DenseTNT 训练代码（上游 + 本项目改造）
│   ├── train_v4.py             # 训练主入口（epoch 循环、LR 衰减、inline 验证、--resume）
│   ├── dataset_argoverse_chunked.py  # 磁盘缓存数据集（205k 文件不全量进内存）
│   ├── run.py                  # 上游入口 + 改造（--resume、取消 batch_size 断言）
│   ├── dataset_argoverse.py    # 上游数据集 + Pool 防死锁改造
│   ├── do_eval.py / utils.py / structs.py / setup.py / utils_cython.pyx
│   └── modeling/               # vectornet.py / decoder.py / lib.py
├── eval_all_models.py          # 批量评估所有 epoch 检查点
├── eval_single.py              # 评估单个检查点
├── dashboard.py                # 训练监控 Web 界面（端口 8080）
├── watchdog.sh                 # 训练崩溃自动重启看门狗
├── mk50k.sh                    # 创建 N 条样本的训练子集（symlink）
├── models/                     # 自写基线：rule_based/、learning_based/、loss_common/、metrics_common/
├── scripts/                    # 预处理、评估、可视化、演示脚本
├── notebooks/                  # Jupyter 演示（CV vs Kalman vs LSTM）
├── docs/                       # 技术文档
├── outputs/                    # 评估图表与报告
└── argoverse-api/              # Argoverse API（上游仓库，含完整 git 历史）
```

## 数据获取

下载 **Argoverse 1** 运动预测数据集：

- 官方地址：<https://www.argoverse.org/av1.html>（训练集约 20.5 万场景、验证集约 3.9 万场景）
- 高精地图**不随本仓库提供**（上游 argoverse-api 的 `.gitignore` 排除了 `map_files/`），需从 Argoverse 1 官网单独下载并放到
  `argoverse-api/map_files/` —— `ArgoverseMap` 加载器按仓库相对路径解析该目录（无需环境变量，见
  `argoverse-api/argoverse/map_representation/map_api.py`）

**原始数据不包含在本仓库中**。预期目录布局：

```
data/raw/
├── train/data/*.csv            # 训练场景
└── val/data/*.csv              # 验证场景
```

高精地图放在 `argoverse-api/map_files/`（见上文）。

## 环境安装

```bash
pip install -r requirements_densetnt.txt   # 训练环境（torch、numpy、cython 等）
pip install -e argoverse-api/              # Argoverse API（仓库自带上游代码，含完整 git 历史）
cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

训练需要 CUDA GPU（在 6GB 显存笔记本 GPU + WSL2 上验证通过）。
基线模型演示可运行 `notebooks/Trajectory_Prediction_Demo.ipynb`，或
`python scripts/enhanced_demo.py --data-dir data/raw`。

## DenseTNT 训练与评估

```bash
# 1) 准备数据（也可先用 mk50k.sh 创建子集快速实验）
python src/train_v4.py --do_train \
  --data_dir train/data --data_dir_for_val val/data \
  --output_dir model_save_full_chunked \
  --train_batch_size 64 --num_train_epochs 16 --patience 5 \
  --hidden_size 128 --core_num 4 --num_workers 0 \
  --distributed_training 1 --use_map --use_centerline --argoverse \
  --other_params semantic_lane direction l1_loss goals_2D \
    enhance_global_graph subdivide goal_scoring laneGCN \
    point_sub_graph lane_scoring complete_traj complete_traj-3

# 中断后续训
python src/train_v4.py ... --resume

# 2) 批量评估所有检查点
python eval_all_models.py

# 3) 训练监控（可选）
python dashboard.py
```

验证缓存会在训练开始前由主进程预构建（见 `src/train_v4.py` 的
`build_validation_cache`）——这避免了 `mp.spawn` 子进程内 `multiprocessing.Pool`
处理验证集全部失败的已知问题。

## 测试

单元测试覆盖基线模型、损失/指标函数与 Argoverse CSV 预处理（无需 GPU 和真实数据）：

```bash
pip install pytest
python -m pytest tests/ -v
```

## 致谢

- **DenseTNT 代码**：[Tsinghua-MARS-Lab/DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT)，
  MIT License（Copyright (c) 2024 Tsinghua MARS Lab）。本仓库保留上游完整
  commit 历史，并在此基础上进行上述改造。
- **Argoverse API**：[argoai/argoverse-api](https://github.com/argoai/argoverse-api)，
  MIT License，合入 `argoverse-api/` 目录并保留完整提交历史（作者署名保留）。
- **论文**：J. Gu, C. Sun, H. Zhao, "DenseTNT: End-to-end Trajectory Prediction
  from Dense Goal Sets", ICCV 2021。 <https://arxiv.org/abs/2108.09640>

## License

MIT —— 见 [LICENSE](LICENSE)。第三方组件保留其自身许可证（详见 LICENSE 文件）。
