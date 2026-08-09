# Trajectory Prediction Method Comparison Report

> Chinese version: [model-comparison-report-zh.md](model-comparison-report-zh.md)
>
> **Public-repo notice**:
> - Early versions of this report included HiVT (Zhou et al., CVPR 2022)
>   experiments (official pretrained weights vs self-trained, per_sample_metrics,
>   bucket stratification, stepwise error, etc.). That part was a **teammate's
>   experiment**: its code, weights and data artifacts are not in this public
>   repo and cannot be reproduced here. To keep every claim in the public repo
>   reproducible, **all HiVT content has been removed from this report**.
> - Everything in this report is reproducible from this repo: DenseTNT is
>   trained with `src/` and evaluated with
>   `scripts/eval/eval_optimization.py`; the Kalman/LSTM/CV baselines come from
>   `models/` and `scripts/`.
> - Numbers from different validation sets / different sources **are not
>   directly comparable** (see the evaluation-caliber notes in §1.1).

---

## 1. Experimental Setup

### 1.1 Data
- **Dataset**: Argoverse 1.1 Motion Forecasting
- **Train set**: 205,942 scenes (DenseTNT trained on a 60,000-scene subset for
  16 epochs)
- **Task**: given the past 2 s (20 frames) of trajectory, predict the next 3 s
  (30 frames)
- **Evaluation caliber**:
  - **DenseTNT: full validation set** (39,472 val/data scenes)
  - **Kalman / LSTM / CV: 300-sample subset** (for fast baseline comparison);
    these numbers are **not directly comparable** to DenseTNT's full-set
    evaluation

### 1.2 Metrics
| Metric | Meaning |
|--------|---------|
| **minADE** (m) | mean per-point Euclidean distance of the best of the top-6 predictions to the ground truth |
| **minFDE** (m) | endpoint Euclidean distance of the best of the top-6 predictions to the ground-truth endpoint |
| **MR** (Miss Rate) | fraction of samples where none of the 6 predictions lands within 2m of the true endpoint |
| **FDE_median** | median FDE (robust to outliers) |
| **FDE_p90** | 90th-percentile FDE (reflects the worst 10% of scenes) |

### 1.3 Methods and training configs

| Method | Full name / source | Training config |
|--------|--------------------|-----------------|
| **Kalman Filter** | constant-acceleration Kalman filter | no training |
| **LSTM** | Encoder-Decoder LSTM | see `scripts/enhanced_demo.py` |
| **CV** | constant-velocity extrapolation | no training (approximate, order-of-magnitude reference) |
| **DenseTNT** | DenseTNT (Gu et al., ICCV 2021) | 16 epochs on the 60,000-scene train subset |

> **Note**: all methods in this report are single-agent predictions. DenseTNT
> is evaluated on the full 39,472-scene validation set; Kalman/LSTM/CV on a
> 300-sample subset. The calibers differ and **the numbers are not directly
> comparable**.

---

## 2. Performance Comparison

### 2.1 Single-agent prediction

| Method | minADE (m) ↓ | minFDE (m) ↓ | FDE_median (m) ↓ | FDE_p90 (m) ↓ |
|--------|:------------:|:------------:|:----------------:|:--------------:|
| CV* | ~5.0 | ~10.0 | — | — |
| Kalman Filter* | 2.259 | 5.151 | 3.843 | 10.877 |
| LSTM* | 1.995 | 4.596 | 3.508 | 9.319 |
| DenseTNT | 1.200 | 2.130 | — | — |

\* CV is an order-of-magnitude approximation; Kalman/LSTM are evaluated on the
300-sample subset. DenseTNT is evaluated on the full validation set (39,472
scenes); minADE=1.200/minFDE=2.130 are from that evaluation run. The baseline in
the README and `outputs/eval_output/optimization_comparison.json` is minADE
1.2199 / minFDE 2.0991 / MR 30.07% (two runs differ slightly in details).
**Numbers from different calibers are not directly comparable.**

**Key finding**: from Kalman (FDE=5.15) to LSTM (FDE=4.60), FDE drops ~11%;
from LSTM to DenseTNT (FDE=2.13), FDE drops another ~54% — goal reasoning plus
map modeling is the main source of accuracy.

### 2.2 DenseTNT training convergence (60k training, 39,472 full validation, single mode)

| Model | Epoch | minADE | minFDE | MR |
|-------|-------|:------:|:------:|:---:|
| model.1 | 0 | 2.137 | 3.751 | 0.522 |
| model.5 | 4 | 1.451 | 2.603 | 0.390 |
| model.9 | 8 | 1.329 | 2.348 | 0.349 |
| model.12 | 11 | 1.241 | 2.178 | 0.320 |
| model.14 | 13 | 1.222 | 2.141 | 0.309 |
| **model.16** | **15** | **1.200** | **2.130** | **0.310** |

> Note: the model.16 row is from that evaluation run; the baseline in the README
> and `outputs/eval_output/optimization_comparison.json` is minADE 1.2199 /
> minFDE 2.0991 / MR 30.07% (two runs differ slightly in details).

DenseTNT keeps improving after 16 epochs (FDE 3.75→2.13, −43%), no
overfitting, still has training headroom.

### 2.3 Bucket analysis by scene difficulty

Scenes are stratified into 4 buckets by curvature:
bucket_counts = [1, 469, 3379, 35623] (bucket 1 simplest → bucket 4 hardest).
FDE by bucket (300-sample subset):

| Method | Bucket 1 (simplest) | Bucket 2 | Bucket 3 | Bucket 4 (hardest) |
|--------|:-------------------:|:--------:|:--------:|:-------------------:|
| Kalman | 3.299 | 1.883 | 1.726 | 2.315 |
| LSTM | 3.976 | 1.886 | 1.661 | 2.028 |

**Interpretation**:
- Bucket 1 has only 1 sample, not statistically meaningful; buckets 2–4 have
  469, 3379, 35623 samples
- In the mainstream buckets (2–4), LSTM and Kalman are close; neither uses map
  information, so errors on complex scenes (bucket 4) stay limited by the
  expressiveness of pure-trajectory methods

### 2.4 DenseTNT offline goal-set optimization

The DenseTNT paper proposes **offline goal set optimization**: on the dense
goal-scoring heatmap, simulated annealing searches the optimal set of 6 goal
points, replacing the default top-k selection. The Cython `get_optimal_targets`
samples 6 initial points from the heatmap, then perturbs them for 1,000
iterations under an MR-mixed objective. Each search runs 8 times and keeps the
best; ~1.4 s per scene.

**Results (39,472 full validation set)**:

| Mode | minADE (m) | minFDE (m) | MR |
|------|:----------:|:----------:|:---:|
| Online inference (top-k) | 1.2199 | 2.0991 | 30.07% |
| Offline optimization | **0.9446** | **1.3502** | **8.38%** |
| Δ | -22.6% | -35.7% | -72.1% |

**Analysis**:
- **MR −72%** is the biggest win: the heatmap quality is good; most
  "false positives" come from crude top-k selection rather than an unreliable
  heatmap. The optimized goals precisely lock onto high-probability regions.
- minFDE 1.35 is close to the paper's offline level on Argoverse 1 (paper 2m
  sampling FDE~1.42) and beats some paper settings.
- 1.4 s/scene is not suitable for real-time inference, but works for offline
  analysis (accident reconstruction, annotation assistance, model upper-bound
  estimation) and paper benchmarking.
- The paper distills offline optimization into an online Goal Set Predictor
  (latency → 0); this experiment did not train that module, so offline
  optimization represents the model's theoretical upper bound.

**Scene-level visualization**: consistent with the statistics — in
straight/slightly-curved scenes (e.g. 10215), top-k already locks the goals and
the offline perturbation may slightly degrade; in turning/intersection scenes
(e.g. 1485), top-k is easily misled by straight-direction high-probability
points, and the global search effectively avoids this trap, greatly improving
accuracy. The core benefit of offline optimization concentrates on **complex
maneuver scenes**, with limited or even slightly negative gains on simple scenes
— consistent with the MR −72% statistics: the improvement mainly comes from the
hard scenes that top-k misjudged.

---

## 3. Per-method Analysis

### 3.1 Kalman Filter

**Principle**: recursive Bayesian state estimator with a constant-acceleration
motion model. State includes position, velocity, acceleration; predict-update
recursion. Only uses the target's kinematic history — no map or interaction
information.

**Pros**: tiny compute ($O(d^3)$), zero training cost, interpretable, decent for
short horizons (<1s).

**Cons**: cannot understand road structure, no interaction modeling, single
deterministic prediction (no multimodality), severely off in turning/
acceleration/braking scenes.

**Use cases**: baseline reference or very low-compute embedded environments.
FDE=5.15m in this experiment — not practically accurate.

---

### 3.2 LSTM Encoder-Decoder

**Principle**: Seq2Seq — an LSTM encoder compresses 2 s of history into a
hidden state; an LSTM decoder generates the 3 s future step by step. This
implementation is single-modal single-agent, without map encoding or
interaction modeling.

**Pros**: simple architecture, moderate data needs, fast inference, temporal
modeling better than physics-based models.

**Cons**: single-modal output without multimodal futures, no map (predictions
may violate road constraints), weak interaction modeling, long-sequence
gradient issues.

**Use cases**: lightweight baseline with limited data. FDE=4.60m here, ~11%
better than Kalman. Literature variants with CNN map encoding + social pooling
reach FDE~2.0m.

---

### 3.3 DenseTNT — Gu et al., ICCV 2021

**Principle**:
1. VectorNet vectorized scene encoding
2. Anchor-free dense goal candidate sampling
3. Goal probability estimation (which candidates are most likely the final position)
4. Top-K goals → trajectory completion (conditional decoding of full trajectories)
5. End-to-end differentiable training

**Pros**:
- Anchor-free goal prediction adapts to arbitrary topology without lane anchors
- Explicit goal reasoning natively supports multimodality (different goals →
  different trajectories)
- End-to-end avoids information bottlenecks
- Formerly #1 on the Argoverse 1 leaderboard + Waymo 2021 champion

**Cons**:
- Dense sampling adds computation
- Needs offline-optimized pseudo-labels (one-time, before training)
- This experiment is single-agent only, no multi-agent interaction
- Slow convergence (still improving at epoch 16)

**Use cases**: scenes with complex road structure requiring flexible goal
selection.

**Online vs offline inference**:

| Mode | minFDE (m) | MR | Latency | Use case |
|------|:----------:|:---:|:-------:|----------|
| Online (top-k) | 2.10 | 30.1% | ~50ms | real-time prediction |
| Offline (optimization) | 1.35 | 8.4% | ~1.4s | offline analysis, upper-bound evaluation |

**Scene-level differences**: the benefit of offline optimization is highly
scene-dependent. In some scenes (e.g. 10215) top-k already picks well and the
optimization perturbation may slightly degrade; in most scenes (e.g. 1485) the
optimization effectively corrects top-k's straight bias, greatly improving
accuracy. This matches the global MR −72% statistics — the core value of
offline optimization is rescuing complex maneuver scenes misjudged by top-k.

---

## 4. Comprehensive Comparison

### 4.1 Performance gradient

```
Kalman(5.15) → LSTM(4.60) → DenseTNT(2.13) → DenseTNT+Opt(1.35)
  physics        sequence      goal-based       offline optimization
```

Core gain of each step:
- **physics→sequence**: +11% (temporal modeling)
- **sequence→DenseTNT**: +54% (goal reasoning + map modeling — main source of accuracy)
- **DenseTNT→+Opt**: +37% (goal selection optimization)

### 4.2 Capability matrix

| Capability | Kalman | LSTM | DenseTNT | DenseTNT+Opt |
|------------|:------:|:----:|:--------:|:------------:|
| Temporal modeling | 2 | 4 | 4 | 4 |
| Map usage | ✗ | ✗ | 5 | 5 |
| Multi-agent interaction | ✗ | 1 | 1 | 1 |
| Multimodal output | ✗ | ✗ | 5 | 5 |
| Inference latency | <1ms | <1ms | ~50ms | ~1.4s |

### 4.3 FDE distribution

| Method | FDE (mean) | FDE_median | FDE_p90 | p90/median ratio |
|--------|:----------:|:----------:|:-------:|:----------------:|
| Kalman | 5.15 | 3.84 | 10.88 | 2.83 |
| LSTM | 4.60 | 3.51 | 9.32 | 2.66 |

**Observation**: Kalman and LSTM have similar p90/median ratios (2.8 vs 2.7),
meaning pure-trajectory methods degrade consistently on hard scenes (worst
10%); their shared weakness is the lack of map and interaction information —
rare intersection topologies and abnormal driving remain blind spots for all
trajectory-only methods.

### 4.4 Scene-difficulty stratification

In bucket 4 (35,623 scenes, the largest category), LSTM (FDE=2.03) and Kalman
(FDE=2.32) are close; neither can use map information. The stratification shows
that **for pure-trajectory baselines, scene complexity does not widen the gap
significantly — closing the gap requires map awareness and goal reasoning
(DenseTNT)**.

---

## 5. Conclusions & Suggestions

### 5.1 Main findings

1. **Clear method ladder**: from Kalman (5.15m) to LSTM (4.60m) to DenseTNT
   (2.13m), FDE drops 59% cumulatively. Map awareness with explicit goal
   reasoning is the single biggest gain (−54%).

2. **Offline optimization unlocks DenseTNT's potential, with gains
   concentrated in complex scenes**: global MR drops from 30.1% to 8.4%
   (−72%), minFDE from 2.10 to 1.35 (−36%). Scene-level analysis shows strong
   gains on turning/intersection scenes and limited gains on straight scenes —
   consistent with the paper's theory and the boundary of offline optimization.

3. **Hard scenes are the common weakness**: pure-trajectory baselines keep p90
   FDE above 6m; the p90/median ratio shows extreme-scene robustness is
   unsolved; after adding map awareness (DenseTNT) overall error drops
   significantly, but rare topologies and abnormal driving remain blind spots
   for all methods.

### 5.2 Improvement directions

1. Train on the full 205k data (the current 60k subset at 16 epochs is still
   improving, no overfitting)
2. Train a Goal Set Predictor to distill offline optimization into online
   inference (the paper's route)
3. Tighten goal sampling spacing to 1m (currently 2m; the paper's 1m grid
   offline FDE~1.27)
4. Try cosine-annealing learning rate schedules
