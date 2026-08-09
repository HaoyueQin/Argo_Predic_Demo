# Argo_Predic_Demo

Trajectory prediction on the **Argoverse 1** motion forecasting dataset — a
reproduction of **DenseTNT** (Gu et al., ICCV 2021) with training/validation
tooling, plus rule-based and learning-based baselines (Kalman filter, constant
velocity, LSTM) for comparison.

## Methods

| Method | Type | minADE (m) | minFDE (m) | MR @2m |
|--------|------|:----------:|:----------:|:------:|
| CV (constant velocity) | Rule baseline | ~5.0 | ~10.0 | ~80% |
| Kalman filter | Rule baseline | 2.26 | 5.15 | ~50% |
| LSTM | Learning baseline | 1.99 | 4.60 | ~35% |
| **DenseTNT (ours)** | VectorNet + map (6 modes) | **1.20** | **2.09** | 30.0% |
| DenseTNT + goal optimization | VectorNet + map | **0.94** | **1.35** | **8.4%** |

Metrics are computed on the Argoverse 1.1 validation split (39,472 scenes).
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

- Official: <https://www.argoverse.org/av1.html> (train ~205k scenes, val ~39k scenes)
- HD maps are **not** shipped with this repo (the upstream Argoverse API
  `.gitignore` excludes `map_files/`). Download them from Argoverse 1
  (<https://www.argoverse.org/av1.html>) and set `ARGOVERSE_MAP_DIR`
  (`argoverse-api/argoverse/utils/` loader expects it)

The raw data is **not** stored in this repository. Expected layout:

```
data/raw/
├── train/data/*.csv            # training scenes
├── val/data/*.csv              # validation scenes
└── argoverse_data/map_files/   # HD maps
```

## Environment

```bash
pip install -r requirements_densetnt.txt   # training env (torch, numpy, cython, ...)
cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

Training requires a CUDA GPU (tested on 6 GB VRAM laptop GPU, WSL2 recommended).
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
