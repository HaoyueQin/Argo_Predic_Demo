# DenseTNT Method Overview & Experiment Record

> Chinese version: [densetnt-overview-zh.md](densetnt-overview-zh.md)

## 1. Method Overview

### 1.1 What is DenseTNT

DenseTNT (Dense Target-driven Trajectory Prediction) is one of the SOTA methods
in the Argoverse trajectory forecasting competition, proposed by the Tsinghua
MARS Lab (Gu et al., ICCV 2021, "DenseTNT: End-to-end Trajectory Prediction from
Dense Goal Sets"). Its core ideas are:

1. **Dense goal sampling**: densely sample candidate goal points on the map
   (instead of sparse sampling)
2. **Lane graph encoding (laneGCN)**: encode lane features with a graph neural
   network over the HD-map lane topology
3. **Lane scoring (lane_scoring)**: score each lane and keep the lanes relevant
   to prediction
4. **Goal scoring (goal_scoring)**: score every candidate goal point and pick
   the most likely goals
5. **Trajectory completion (complete_traj)**: generate the full trajectory
   backwards from the selected goal

### 1.2 Model architecture

```
Input (2s history trajectory + lane map)
    ↓
[Point Sub-Graph] trajectory encoding per agent
    ↓
[Global Graph] interaction modeling between agents
    ↓
[LaneGCN] lane graph neural network encoding
    ↓
[Lane Scoring] keep relevant lanes
    ↓
[Goal Scoring] score dense candidate goals
    ↓
[NMS] non-maximum suppression dedup
    ↓
[Trajectory Completion] generate 6 candidate trajectories
    ↓
Output (6 future 3s trajectories + probabilities)
```

### 1.3 Key technical characteristics

- **Multimodal output**: 6 candidate trajectories covering different possible
  futures (straight, left turn, right turn, etc.)
- **Map awareness**: lane information from the HD map keeps predictions aligned
  with the road layout
- **Probability scoring**: every trajectory carries a probability used for
  ranking and evaluation

---

## 2. Concrete Work

### 2.1 Environment setup

| Item | Configuration |
|------|---------------|
| OS | Windows 11 + WSL 2 (Ubuntu 24.04) |
| GPU | NVIDIA RTX 4050 Laptop (6 GB) |
| CUDA | 12.2 (note: versions drift over time; follow `requirements*.txt` and the README) |
| Python | 3.12 (note: versions drift over time; follow `requirements*.txt` and the README) |
| PyTorch | 2.12 + CUDA |
| Dataset | Argoverse 1 (8k train + 2k val) (note: this is the early experiment scale; the final training used a 60k split, see README) |

### 2.2 Data preparation

- Argoverse 1 dataset
- HD map files
- Preprocessing: extract trajectory coordinates, lane line information, build
  scene instances

### 2.3 Model training

| Parameter | Value |
|-----------|-------|
| Epochs | 16 |
| Batch size | 16 (the paper uses 64; reduced due to GPU VRAM limits) |
| Learning rate | 1e-3 with warmup + cosine annealing |
| Training time | ~1.5 h |
| Final training loss | 5.46 |

**Training loss curve**:
- Epoch 1: 12.28 → Epoch 16: 5.46
- Loss drops clearly at the three LR decay steps (epoch 5/10/15)

### 2.4 Model validation

Evaluated on 2,000 validation scenes:

> Note: this section reports the early experiment on the 8k training split. The
> final 60k-trained model metrics (minADE 1.20 / minFDE 2.09 / MR 30.0%, and
> 0.94 / 1.35 / 8.4% with goal optimization) are in the README and
> `outputs/eval_output/optimization_comparison.json`.

| Metric | Value | Meaning |
|--------|-------|---------|
| **minADE** | 1.036 m | average distance between the best candidate trajectory and the ground truth |
| **minFDE** | 1.502 m | distance between the best candidate endpoint and the ground-truth endpoint |
| **MR (2m)** | 10.9% | fraction of scenes whose endpoint error exceeds 2 m |

**Comparison with the paper**:
- Paper (8 GPUs + 200k training data): minADE 0.73–0.82, minFDE 1.05–1.37, MR 7–9.8%
- This work (1 GPU + 8k training data): minADE 1.036, minFDE 1.502, MR 10.9%
- **Conclusion**: with 1/25 of the data we reached 70–80% of the paper's
  performance, which shows the implementation is correct and effective.

### 2.5 Visualization

Complete visualization scripts were developed that can:
- Overlay predicted trajectories and the lane map in BEV (bird's-eye view)
- Color-code the 6 candidate trajectories and annotate their probabilities
- Support single-scene, batch-scene, and full-scene visualization
- Generate 2000+ high-quality images (600 DPI)

---

## 3. Result Analysis

### 3.1 Quantitative analysis

**Method comparison table** (final 60k-trained model metrics, consistent with
the README; the early 8k-split DenseTNT numbers are in §2.4):

| Method | Type | minADE (m) | minFDE (m) | MR @2m |
|--------|------|------------|------------|--------|
| CV (constant velocity) | rule baseline | ~5.0 | ~10.0 | ~80% |
| Kalman filter | rule baseline | 2.26 | 5.15 | ~50% |
| LSTM | deep learning | 1.99 | 4.60 | ~35% |
| **DenseTNT** | **deep learning + map** | **1.20** | **2.09** | **30.0%** |

**Analysis**:
- DenseTNT vs LSTM: minADE −39.7% (1.99→1.20), minFDE −54.6% (4.60→2.09)
- DenseTNT vs Kalman: minADE −46.9% (2.26→1.20), minFDE −59.4% (5.15→2.09)
- Introducing map information is the key to the improvement

### 3.2 Qualitative analysis (visualization)

**Straight scenes**: predicted trajectories closely match the ground truth; the
6 candidates nearly overlap.

**Turning scenes**: the model recognizes the turning intent; predictions bend
along the lanes and match the ground-truth direction.

**Intersection scenes**: multiple candidates cover different turning options
(straight, left, right), demonstrating the advantage of multimodal prediction.

### 3.3 Limitations

1. **Data scale**: only 8k training scenes (the paper uses 200k); confidence on
   some complex scenes is limited
2. **GPU VRAM**: batch size reduced from 64 to 16, slightly lower training
   efficiency
3. **Some turning scenes**: the highest-probability prediction may still go
   straight, though the correct direction is present among the candidates

---

## 4. Key Code Files

### Inference & visualization

| File | Description |
|------|-------------|
| visualize_map.py | main visualization script: loads the model, runs inference, produces images |
| metrics_comparison.py | generates the metric comparison bar chart |
| enhanced_demo.py | enhanced demo script (multi-method comparison) |

### Training core code (training/ directory)

| File | Description |
|------|-------------|
| run.py | main training script with resume (--resume) and checkpoint saving |
| utils.py | utilities: argument definitions, data preprocessing, evaluation metrics |
| modeling/vectornet.py | model architecture (VectorNet) |
| modeling/decoder.py | DenseTNT decoder: goal scoring, trajectory completion |
| modeling/lib.py | basic NN components (MLP, Attention, etc.) |

## 5. Model Files

| File | Description |
|------|-------------|
| `model.16.bin` | trained weights (epoch 16) |
| `checkpoint.pt` | full checkpoint (includes optimizer state, usable for resume) |

---

## 6. Visualization Gallery

9 representative scene images were selected, in three categories:

1. **Straight scenes**: 10002, 10005, 10020
2. **Turning/curved scenes**: 10215, 10047, 10036
3. **Intersection/complex scenes**: 10014, 10022, 10084

Legend of the visualization images:
- Gray lines: lane map
- Black dashed line: history trajectory (past 2 s)
- Green solid line: ground-truth future trajectory (next 3 s)
- Red thick line: highest-probability prediction
- Colored thin lines: other candidate predictions
- Square marker: current vehicle position

---

## 7. PPT Presentation Suggestions (for reference)

1. Problem definition: input (2 s history + lanes) → output (6 future 3 s
   trajectories + probabilities)
2. Method comparison table: use the table above
3. Visualization examples: pick one straight, one turning, one intersection
4. Metric explanation: minADE = average error, minFDE = endpoint error, MR = miss rate
5. Conclusion: DenseTNT leads across the board; map information is the key
