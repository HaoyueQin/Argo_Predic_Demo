# DenseTNT Optimization Post-processing Verification Report

> Chinese version: [optimization-verification-report-zh.md](optimization-verification-report-zh.md)
>
> Note: the verification script `eval_optimization.py` has been adapted and
> included in this repository (`scripts/eval/eval_optimization.py`); the metric
> data is fixed in `outputs/eval_output/optimization_comparison.json`.
> Reproduction: train → `scripts/eval/eval_all_models.py` evaluation → run
> `python scripts/eval/eval_optimization.py`. The original full-run logs
> (`baseline_full.log` / `optimization_full.log`) are not published with the
> repo; the numbers can be regenerated directly by the script.

## 1. Overview

This report verifies the improvement of the **Optimization** post-processing in
DenseTNT (ICCV 2021) over the default **NMS top-k** scheme. The full comparison
was run on the Argoverse 1 validation set (39,472 scenes).

### Core conclusion

| Metric | Baseline (NMS top-k) | Optimization | Δ | Δ% |
|--------|:---------------------:|:------------:|:---:|:---:|
| **minADE** | 1.2199 | **0.9446** | -0.2752 | **-22.56%** |
| **minFDE** | 2.0991 | **1.3502** | -0.7490 | **-35.68%** |
| **MR** | 30.07% | **8.38%** | -21.69pp | **-72.14%** |

### Comparison with the paper baseline

Note: this verification is **offline post-processing** — the optimization runs
directly on the Dense Goal Set heatmap, and no goal set predictor was trained to
distill it. This corresponds to the **offline mode** of the paper (DenseTNT,
ICCV 2021), not the online goal set predictor.

**Paper Table 2 — offline mode metrics**:

| Sampling | minFDE | MR |
|:---------|:------:|:--:|
| 2m grid | ~1.42 | ~9.8% |
| 1m grid | ~1.27 | ~7.0% |

**Our results**:

| Metric | Our value | vs paper |
|:-------|:--------:|:----------:|
| minADE | 0.94 | not in Table 2 offline metrics; for reference |
| minFDE | **1.35** | between 2m (1.42) and 1m (1.27) |
| MR | **8.4%** | between 2m (9.8%) and 1m (7.0%) |

The results fall in the normal range of the paper's offline mode. Ours are
slightly better because the `subdivide` parameter in `other_params` produces a
denser goal sampling than the paper default, giving the optimizer a larger
search space.

---

## 2. Method

### 2.0 Online vs offline mode

The paper describes two routes:

```
Dense Goal Set Heatmap
  │
  ├─ Online: train a goal set predictor to distill the optimization results
  │    ├─ no extra latency at inference
  │    └─ metrics: minADE~0.89, minFDE~1.47, MR~12.7%
  │
  └─ Offline: run simulated annealing directly on the heatmap (this verification)
       ├─ ~1.4s post-processing per scene
       └─ metrics: minFDE~1.42 (2m) / ~1.27 (1m), MR~9.8% (2m) / ~7.0% (1m)
```

This report verifies the **offline mode**.

### 2.1 Overall data flow

```
Input: scene trajectory data
  │
  ▼
VectorNet encoder ──→ feature vectors
  │
  ▼
Decoder + Dense Goal Set Prediction ──→ (goals_2D, scores)
  │                                                │
  ├─ NMS top-k (Baseline)                          │
  │     ├─ non-maximum suppression, pick k candidates │
  │     └─ use them as the endpoints of the final 6 trajectories │
  │                                                │
  └─ Optimization (Cython post-processing)         │
        ├─ sample 6 initial candidates from the Dense Goal Set │
        ├─ simulated annealing                      │
        │    ├─ perturb each candidate randomly         │
        │    ├─ evaluate the objective (MR / minFDE / MRminFDE) │
        │    ├─ accept worse solutions with some probability (escape local optima) │
        │    └─ learning rate decays exponentially over iterations │
        ├─ 8 independent runs (run_times=8), keep the best │
        └─ output: 6 optimized endpoints + probabilities │
```

### 2.2 Optimization objective

The Cython function `get_value` (`utils_cython.pyx:138`) computes the "expected
loss" for a given set of 6 candidate endpoints:

```
For each Dense Goal Point:
  1. sample cnt_sample (36) sub-grid points around it (6x6 grid)
  2. for each sub-grid point, compute the min distance to the 6 candidates
  3. if min distance <= 2.0m, count as "hit" (miss_error=0)
  4. otherwise miss_error=10.0m
  5. combine: minFDE * (1-MRratio) + miss_error * MRratio

Total expectation = Σ score_i * average sub-grid loss
```

The objective is controlled by the `objective` parameter:
- `'MR'`: pure Miss Rate optimization (MRratio weights the miss penalty)
- `'minFDE'`: pure endpoint error optimization
- `'MRminFDE'`: a mix of both (tuned via `MRratio`)

### 2.3 Simulated annealing flow

`_get_optimal_targets` (`utils_cython.pyx:191`):

1. **Initialization**: randomly pick 6 points from the Dense Goal Set as start
   candidates
2. **Iterative optimization** (num_step iterations):
   - current learning rate `lr = exp(-(step/num_step) * 2)` (exponential decay)
   - perturb each candidate randomly in `[-lr, +lr]` with 30% probability
   - evaluate the expected loss after perturbation
   - accept if smaller; otherwise accept with 1% probability (escape local
     optima)
   - keep track of the global best
3. **Post-processing**:
   - re-compute probabilities using each candidate as an "anchor"
   - sort by probability descending, output 6 endpoints + probabilities

### 2.4 Differences from the baseline

| Stage | Baseline (NMS top-k) | Optimization |
|:------|:--------------------:|:------------:|
| post-processing | non-maximum suppression | simulated annealing |
| candidate source | Dense Goal Set clustering + NMS | direct optimization over the Dense Goal Set |
| trajectory endpoints | picked directly from candidates | iteratively optimized under the objective |
| stability | depends on the NMS threshold | best of 8 independent runs |
| compute cost | nearly zero | ~0.18s/run, 8 runs/scene |

---

## 3. Code Files

### 3.1 Core files

| File | Path | Description |
|:-----|:-----|:------------|
| verification script | `scripts/eval/eval_optimization.py` | standalone script with the full Baseline/Optimization dual-mode comparison |
| Optimization scheduler | `src/utils.py` `run_process` / `select_goals_by_optimization` | `run_process` dispatches tasks to Cython workers; `select_goals_by_optimization` manages the process pool and the 8 runs |
| Cython core | `src/utils_cython.pyx` `get_value` / `_get_optimal_targets` | objective computation + simulated annealing |
| build artifact | `src/utils_cython.*.so` / `.pyd` | compiled extension (built per environment, not committed; see README) |
| decoder args sync | `src/modeling/decoder.py` `Decoder.__init__` | module-level `global args; args = args_` |
| VectorNet args sync | `src/modeling/vectornet.py` `VectorNet.__init__` | same, `global args` |
| result file | `outputs/eval_output/optimization_comparison.json` | full Baseline + Optimization comparison data |

### 3.2 Run logs

The original full-run logs (`baseline_full.log` / `optimization_full.log`,
development artifacts) are not published; the metrics can be regenerated
directly by `scripts/eval/eval_optimization.py`.

---

## 4. Key Bug Fixes

### Bug 1: module-level `global args` not synced

**Location**: `decoder.py` `Decoder.__init__`, `vectornet.py` `VectorNet.__init__`

```
def __init__(self, args_: utils.Args):
    global args
    args = args_        # ← establishes the module-level global args
```

`decoder.py` and `vectornet.py` each keep their own module-level `global args`,
accessed via `import modeling.decoder as _decoder_mod; _decoder_mod.args`.
Modifying only `utils.args` does not affect Decoder/VectorNet runtime behavior.

**Fix** (in both passes of `scripts/eval/eval_optimization.py`):

```python
_decoder_mod.args = args
_vectornet_mod.args = args
```

### Bug 2: Cython `cnt_sample != square` assertion

**Location**: `utils_cython.pyx` `get_value` (when `cnt_len == 0`)

`get_value` samples a 6x6 sub-grid and requires `cnt_sample` to be a perfect
square. The default `cnt_sample=2` is not a perfect square, which fails the
assertion (the current pyx raises an explanatory `ValueError` instead).

**Fix** (`make_args` in `scripts/eval/eval_optimization.py`):

```python
a.other_params['cnt_sample'] = 36  # 6² = 36
```

### Bug 3: `MRratio` UnboundLocalError

**Location**: `src/utils.py` `run_process`

```python
if 'MRminFDE' in args.other_params:
    assert 'cnt_sample' in args.other_params
    MRratio = float(args.other_params['MRminFDE']) if ... else 1.0

if 'cnt_sample' in args.other_params:
    ...
    kwargs.update(dict(
        ...
        MRratio=MRratio,    # ← MRratio only defined in the previous if
    ))
```

When `cnt_sample` is present in `other_params` but `MRminFDE` is not, `MRratio`
is undefined.

**Fix**: merged into the public repo `src/utils.py` — `run_process`
unconditionally initializes `MRratio = 1.0` (pure MR objective), overridden by
the parameter when `'MRminFDE'` exists. No caller (including
`--other_params optimization cnt_sample=36`) depends on externally setting
`MRminFDE` anymore. `scripts/eval/eval_optimization.py` still explicitly sets
`MRminFDE=1.0` to stay consistent with the historical evaluation config.

---

## 5. Runtime Environment

| Item | Configuration |
|:-----|:--------------|
| OS | WSL2 (Ubuntu 22.04) |
| Python | 3.12.3 |
| Model weights | `model_save_full_chunked/model_save/model.16.bin` |
| GPU | CUDA 12.1 (hardware via WSL) |
| Cython extension | `utils_cython.cpython-312-x86_64-linux-gnu.so` |
| Dependencies | numpy 1.26.4, torch 2.5.1+cu121, pandas, tqdm, matplotlib, scipy, argoverse-api |
| Cores | 4 (`core_num=4`) |
| Batch size | 16 |

### Dependency installation

```bash
pip install -r requirements_densetnt.txt          # see README
pip install -e argoverse-api/ --no-deps           # vendored Argoverse API
cd src && cython -a utils_cython.pyx && python setup.py build_ext --inplace
```

---

## 6. How to Run

### 6.1 Full comparison (both passes)

```bash
python scripts/eval/eval_optimization.py
```

### 6.2 Optimization only (reuse the baseline cache)

```bash
python scripts/eval/eval_optimization.py --optim-only --reuse-cache
```

### 6.3 Limit the number of scenes (quick verification)

```bash
python scripts/eval/eval_optimization.py --max-scenes 2000
```

### 6.4 Parameters

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `--max-scenes` | None (full) | limit the number of inference scenes |
| `--baseline-only` | False | run only NMS top-k |
| `--optim-only` | False | run only Optimization |
| `--reuse-cache` | False | reuse cached temp_file (skip preprocessing) |
| `--model` | `model_save_full_chunked/model_save/model.16.bin` | model checkpoint path |
| `--data-dir` | `val/data` | validation data directory |

---

## 7. Runtime Statistics

| Stage | Scenes | Time | Speed |
|:------|:------:|:----:|:-----:|
| Baseline inference | 39,472 | 841s | 46.9 scenes/s |
| Optimization inference | 39,472 | 7,618s (~2.1h) | 5.2 scenes/s |
| 500-scene optimization test | 500 | 101s | 5.0 scenes/s |
| Cache build (first run) | 39,472 | ~276s | - |

Why Optimization is slow:
- 8 independent optimizations per scene (`run_times=8`)
- 1,000 iterations per optimization (`num_step=1000`; default `opti_time=10000s`
  does not trigger the Cython override)
- each step computes distances over 36 sub-grid points × 6 candidates

---

## 8. Result Verification

### 8.1 Small-scale verification (500 scenes)

| Metric | Baseline | Optimization | Δ% |
|--------|:--------:|:------------:|:---:|
| minADE | 1.2770 | 0.9458 | -25.94% |
| minFDE | 2.1708 | 1.2754 | -41.25% |
| MR | 28.80% | 5.80% | -79.86% |

### 8.2 Full verification (39,472 scenes)

| Metric | Baseline | Optimization | Δ% |
|--------|:--------:|:------------:|:---:|
| minADE | 1.2199 | 0.9446 | -22.56% |
| minFDE | 2.0991 | 1.3502 | -35.68% |
| MR | 30.07% | 8.38% | -72.14% |

### 8.3 Comparison with the paper's offline mode

| Metric | Paper offline (2m) | Paper offline (1m) | Our value | Conclusion |
|:-------|:------------------:|:------------------:|:---------:|:-----------|
| minFDE | ~1.42 | ~1.27 | 1.35 | inside the range |
| MR | ~9.8% | ~7.0% | 8.4% | inside the range |

The results are within the paper's offline-mode range. The denser sampling from
the `subdivide` parameter makes the optimization slightly better than the 2m
grid default.

---

## 9. Technical Details

### 9.1 Activation conditions of Optimization

In `run_process` of `utils.py`, `args.other_params` must contain:
- `'optimization'` → triggers `select_goals_by_optimization`
- `'cnt_sample'` → sampling grid size passed to Cython (must be a perfect square)
- `'MRminFDE'` (optional) → defines `MRratio` (defaults to 1.0, i.e. the pure MR
  objective; the default is merged into `src/utils.py`, see Bug 3 in §4)
- `'opti_time'` (optional) → CPU-time budget per optimization call (default
  10000s, does not trigger the Cython `num_step` override)

### 9.2 Cython compilation notes

Compiling the Cython file requires NumPy 1.x (not compatible with NumPy 2.x):

```bash
cd src/
cython -a utils_cython.pyx
python setup.py build_ext --inplace
```

The artifact is `utils_cython.cpython-312-x86_64-linux-gnu.so`, usable only on
Linux x86_64 with Python 3.12.

### 9.3 Multiprocess pool model

`select_goals_by_optimization` maintains a global process pool (singleton):
- `args.core_num` worker processes
- the main process dispatches tasks via `multiprocessing.Queue`
- the 8 independent optimizations are dispatched as 8 rounds of `queue.put`
- workers return results through `queue_res`
- the main process merges the best result at the end of every round
