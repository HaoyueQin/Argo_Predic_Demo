# Argo_Predic_Demo

Trajectory prediction on the **Argoverse 1** motion forecasting dataset — a
reproduction of **DenseTNT** (Gu et al., ICCV 2021) with training/validation
tooling, plus rule-based and learning-based baselines (Kalman filter, constant
velocity, LSTM) for comparison.

## Methods

| Method | Type | minADE (m) | minFDE (m) | MR @2m |
|--------|------|:----------:|:----------:|:------:|
| CV (constant velocity) | Rule baseline | ~5.0¹ | ~10.0¹ | ~80%¹ |
| Kalman filter | Rule baseline | 2.26² | 5.15² | ~50%² |
| LSTM | Learning baseline | 1.99² | 4.60² | ~35%² |
| **DenseTNT (ours)** | VectorNet + map (6 modes) | **1.20³** | **2.09³** | 30.0%³ |
| DenseTNT + goal optimization | VectorNet + map | **0.94³** | **1.35³** | **8.4%³** |

¹ CV 为量级参考的近似值（无全量实测）。
² Kalman / LSTM 在 300 样本子集上评估（规则/学习基线对比用）。
³ DenseTNT 两行在 Argoverse 1.1 全量验证集（39,472 场景）上评估。
**不同口径的行不可直接对比**；详见 `docs/多方法对比分析报告.md` 的评估口径说明。

The DenseTNT model was trained on a 60k subset (16 epochs); see
`outputs/charts/report_analysis.md` for details. Chinese version: [README-zh.md](README-zh.md).

## Repository layout

```
├── src/                        # DenseTNT training code (upstream + modifications)
│   ├── train_v4.py             # Main training entry (epoch loop, LR decay,
│   │                           #   inline validation, --resume)
│   ├── dataset_argoverse_chunked.py  # Disk-cached dataset (avoids OOM on 205k files)
│   ├── run.py                  # Upstream entry + resume/batch-size patches
│   ├── dataset_argoverse.py    # Upstream dataset + Pool deadlock fix
│   ├── do_eval.py / utils.py / structs.py / setup.py / utils_cython.pyx
│   └── modeling/               # vectornet.py / decoder.py / lib.py
├── eval_all_models.py          # Batch-evaluate all epoch checkpoints
├── eval_single.py              # Evaluate a single checkpoint
├── dashboard.py                # Training monitoring web UI (port 8080)
├── watchdog.sh                 # Crash-restart watchdog for training
├── mk50k.sh                    # Create an N-sample training subset (symlinks)
├── scripts/eval/eval_optimization.py  # Reproduce the goal-optimization results
│                                   #   (baseline vs optimization, see docs/
│                                   #   OPTIMIZATION_VERIFICATION_REPORT.md)
├── models/                     # Own baselines: rule_based/, learning_based/,
│                               #   loss_common/, metrics_common/
├── scripts/                    # Preprocessing, evaluation, visualization, demo
├── notebooks/                  # Jupyter demo (CV vs Kalman vs LSTM)
├── docs/                       # Technical documentation (Chinese)
├── outputs/                    # Evaluation charts and reports
└── argoverse-api/              # Argoverse API (upstream repo, full git history)
```

## Data

Download the **Argoverse 1** motion forecasting dataset:

- Official: <https://www.argoverse.org/av1.html> (train ~205k scenes, val ~39k scenes);
  for the course reproduction we downloaded the dataset via the Baidu PaddlePaddle
  AI Studio mirror (multi-volume archives; verification and extraction steps are
  documented in `docs/data_cleaning_report.md`)
- HD maps are **not** shipped with this repo (the upstream Argoverse API
  `.gitignore` excludes `map_files/`). Download them from Argoverse 1
  (<https://www.argoverse.org/av1.html>) and place them under
  `argoverse-api/map_files/` — the `ArgoverseMap` loader resolves this
  directory relative to the repo (no environment variable needed, see
  `argoverse-api/argoverse/map_representation/map_api.py`)

The raw data is **not** stored in this repository. Expected layout:

```
data/raw/
├── train/data/*.csv            # training scenes
└── val/data/*.csv              # validation scenes
```

HD maps go to `argoverse-api/map_files/` (see above).

## Environment

```bash
pip install -r requirements_densetnt.txt   # training env (torch, numpy, cython, ...)
pip install -e argoverse-api/ --no-deps    # Argoverse API (vendored upstream, full git history);
                                           # runtime deps are already covered above, so skip the
                                           # upstream's pinned legacy pins (numpy==1.19, ...)
cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

Training requires a CUDA GPU (tested on 6 GB VRAM laptop GPU, WSL2 recommended).
Note: the upstream `argoverse-api/setup.py` refuses to run on native Windows
("Argoverse currently does not support Windows"), so use WSL2 / Linux / macOS.
`pip install -e argoverse-api/ --no-deps` skips the upstream's pinned legacy
dependencies (`numpy==1.19`, `hydra-core==1.1.0`, ...) — the runtime deps are
already covered by `requirements_densetnt.txt`.
Run the demo notebook (`notebooks/Trajectory_Prediction_Demo.ipynb`) or
`python scripts/enhanced_demo.py --data-dir data/raw` for the baselines.

## Train & evaluate DenseTNT

```bash
# 1) Prepare data (or use mk50k.sh to make a subset first)
python src/train_v4.py --do_train \
  --data_dir train/data --data_dir_for_val val/data \
  --output_dir model_save_full_chunked \
  --train_batch_size 64 --num_train_epochs 16 --patience 5 \
  --hidden_size 128 --core_num 4 --num_workers 0 \
  --distributed_training 1 --use_map --use_centerline --argoverse \
  --other_params semantic_lane direction l1_loss goals_2D \
    enhance_global_graph subdivide goal_scoring laneGCN \
    point_sub_graph lane_scoring complete_traj complete_traj-3

# resume after interruption
python src/train_v4.py ... --resume

# 2) Evaluate all saved checkpoints
python eval_all_models.py

# 3) Monitor training (optional)
python dashboard.py
```

The validation cache is pre-built in the main process before training starts
(see `build_validation_cache` in `src/train_v4.py`) — this avoids a known
`multiprocessing.Pool`-inside-`mp.spawn` failure on the validation set.

## Tests

Unit tests cover the baseline models, loss/metrics and the Argoverse CSV
preprocessor (no GPU or real data required):

```bash
pip install pytest
python -m pytest tests/ -v
```

## Acknowledgements

- **DenseTNT** code: [Tsinghua-MARS-Lab/DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT),
  MIT License (Copyright (c) 2024 Tsinghua MARS Lab). This repo keeps the
  upstream commit history and applies the modifications described above.
- **Argoverse API**: [argoai/argoverse-api](https://github.com/argoai/argoverse-api),
  MIT License, merged into `argoverse-api/` with its full commit history
  preserved (authorship intact).
- **Paper**: J. Gu, C. Sun, H. Zhao, "DenseTNT: End-to-end Trajectory Prediction
  from Dense Goal Sets", ICCV 2021. <https://arxiv.org/abs/2108.09640>

## License

MIT — see [LICENSE](LICENSE). Third-party components retain their own licenses
(see the LICENSE file for details).
