# Argoverse v1.1 Data Preprocessing & Cleaning Report

> Chinese version: [data-cleaning-report-zh.md](data-cleaning-report-zh.md)

## 1. Overview

This report documents the full preprocessing pipeline of the Argoverse v1.1
Motion Forecasting dataset: data validation, anomaly detection and repair,
scene classification, coordinate transformation, and final processing
statistics.

**Data source**: full Argoverse v1.1 dataset downloaded from the Baidu PaddlePaddle (Baidu AI Studio) platform
**Processing date**: 2026-06-02
**Environment**: Windows 11, Python 3.13, NumPy 2.4.6, 4 parallel processes
(note: environment versions drift over time; follow `requirements*.txt` and the README)

> **Version note (important)**: this document is a snapshot of the results of an
> **early version** of `scripts/preprocess/argoverse_preprocess_v2.py` run on
> 2026-06-02. The current script in this repo has been updated: scenes whose
> AGENT trajectory is shorter than 50 frames (history ≥20 frames but the future
> is truncated) are **no longer padded with the last frame**; they are marked as
> `skip_short` and skipped directly — padding artificially creates a "stationary
> future" (a short trajectory treated as stopping), and the DenseTNT training
> pipeline requires exactly 50 AGENT rows, so the padded CSV would be discarded
> anyway. Therefore **re-running with the current script produces some
> filtering** (`skip_short` / `skip_no_av` / `skip_no_agent`), and the filtered
> counts differ from the early results (0 filtered) recorded here. The numbers
> in this document represent only that 2026-06-02 run, not the current script's
> behavior.

---

## 2. Dataset Overview

| Item | Train | Val | Total |
|------|-------|-----|-------|
| Raw scenes | 205,942 | 39,472 | 245,414 |
| Valid scenes | 205,942 | 39,472 | **245,414 (100%)** |
| Filtered scenes | 0 | 0 | 0 |

> Every scene contains complete AV and AGENT trajectories, so no filtering was
> needed (2026-06-02 early-version result; current script behavior is described
> in the version note at the top).

---

## 3. Data Format

### Raw data format

Each scene is one CSV file with 50 frames (10 Hz, 5 seconds total):
- **Frames 0-19**: history observations (2 s)
- **Frames 20-49**: future to predict (3 s)

CSV columns:

```
TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME
```

`OBJECT_TYPE` has three values:
- **AV**: the autonomous (ego) vehicle
- **AGENT**: the target vehicle to predict
- **OTHERS**: surrounding traffic participants

---

## 4. Cleaning Pipeline

### Step 1: Basic validation

- Check CSV format integrity (at least 5 columns)
- Confirm the scene contains both AV and AGENT object types
- Missing either → skip the scene

**Result**: all passed; 0 scenes filtered.

### Step 2: Trajectory completeness check

- Check that the AGENT history has ≥ 20 frames
- Fewer than 20 → skip (`skip_short`)
- Fewer than 50 frames (history ≥20 but truncated future) → **the early
  version padded with the last-frame coordinates; the current version skips**
  (`skip_short`, see the version note at the top)

**Result** (2026-06-02 early version): all passed; 0 scenes filtered.

### Step 3: Anomaly detection & repair

For each AGENT trajectory, detect anomalies between consecutive frames:

| Anomaly type | Threshold | Condition | Repair |
|--------------|-----------|-----------|--------|
| Speed anomaly | 50 m/s (180 km/h) | consecutive-frame speed > threshold | flagged, coordinates unchanged |
| Frame jump | 5 m/frame | consecutive-frame displacement > threshold | fill with the previous frame's coordinates |

> Note: only consecutive frames with a time gap in 0.05–0.2 s are checked;
> frame pairs with unusual time gaps are skipped.

**Repair logic**:

```python
# when a jump is detected, overwrite the current frame with the previous one
if displacement > 5.0m:
    xy[current_frame] = xy[previous_frame]
```

### Step 4: Scene classification

Scenes are classified by the AGENT trajectory's heading change:

Computation:
1. direction vector of the first 10 frames `start_dir = pts[10] - pts[0]`
2. direction vector of the last 10 frames `end_dir = pts[-1] - pts[-10]`
3. angle between the two vectors `heading_change = atan2(cross, dot)`

| Scene type | Condition | Meaning |
|------------|-----------|---------|
| straight | \|heading_change\| < 15° | going straight |
| left_turn | heading_change > 15° | left turn (counter-clockwise) |
| right_turn | heading_change < -15° | right turn (clockwise) |
| complex | 5° ≤ \|heading_change\| < 15° | complex/gentle curve |

### Step 5: Coordinate transformation

**Choice**: after a 2,000-sample comparison experiment, **option B
(Agent-centered rotation)** was chosen.

| Option | X range (mean) | Y range (mean) | endpoint distance (mean) | Conclusion |
|--------|----------------|----------------|--------------------------|------------|
| A: AV-centered rotation | 13.39m | 20.32m | 40.63m | too large (AV is far away) |
| **B: Agent-centered rotation** | **14.24m** | **19.00m** | **29.41m** | **✓ best** |
| C: translation only | 16.12m | 20.27m | 29.41m | range too large |

**Option B steps**:
1. **Translate**: origin at the AGENT position in frame 19 (last history frame)
2. **Rotate**: align the AGENT motion direction from frame 18→19 with the +x axis
3. Apply the same translation+rotation to all coordinates (AGENT, AV, OTHERS)

**Transformation formulas**:

```
angle = atan2(y[19] - y[18], x[19] - x[18])
x' = (x - x[19]) * cos(-angle) - (y - y[19]) * sin(-angle)
y' = (x - x[19]) * sin(-angle) + (y - y[19]) * cos(-angle)
```

---

## 5. Cleaning Results

### 5.1 Train set (205,942 scenes)

| Metric | Count | Ratio |
|--------|-------|-------|
| Valid scenes | 205,942 | 100% |
| Scenes with speed anomalies | 3,365 | 1.6% |
| Total speed anomaly points | 2,708 | — |
| Total frame-jump points | 3,482 | — |
| Stationary vehicle scenes | 0 | 0% |

**Scene type distribution:**

| Type | Count | Ratio |
|------|-------|-------|
| straight | 137,463 | 66.8% |
| left_turn | 28,525 | 13.9% |
| complex | 20,421 | 9.9% |
| right_turn | 19,533 | 9.3% |

### 5.2 Validation set (39,472 scenes)

| Metric | Count | Ratio |
|--------|-------|-------|
| Valid scenes | 39,472 | 100% |
| Scenes with speed anomalies | 1,247 | 3.2% |
| Total speed anomaly points | 1,121 | — |
| Total frame-jump points | 1,275 | — |
| Stationary vehicle scenes | 0 | 0% |

**Scene type distribution:**

| Type | Count | Ratio |
|------|-------|-------|
| straight | 27,981 | 70.9% |
| left_turn | 4,227 | 10.7% |
| complex | 4,217 | 10.7% |
| right_turn | 3,047 | 7.7% |

---

## 6. Analysis & Conclusions

### 6.1 Data quality

- **Very high completeness**: all 245,414 scenes contain complete AV and AGENT
  trajectories, none filtered (2026-06-02 early-version result; the current
  script filters short/no-AV/no-AGENT scenes, see the version note)
- **Very low anomaly ratio**: speed anomalies only 1.6%–3.2%, few frame jumps —
  the Argoverse dataset is high quality
- **No stationary vehicles**: all AGENTs move noticeably, suitable for
  trajectory forecasting

### 6.2 Scene distribution characteristics

- **Straight dominates**: ~67%–71% straight scenes, consistent with real urban
  driving
- **Turns ~24%**: left turns (11–14%) slightly outnumber right turns (8–9%),
  possibly related to right-hand traffic rules
- **~10% complex scenes**: gentle curves, lane changes, etc. — the hardest part
  to predict

### 6.3 Coordinate option choice

Why option B (Agent-centered rotation):
- most reasonable endpoint distance (29.4m ≈ 3 s × ~10 m/s average speed)
- balanced X/Y ranges, easier for the model to learn
- centered on the predicted target, semantically clearest

---

## 7. Output Files

### 7.1 DenseTNT format: `data_cleaned/{train,val}/*.csv`

- **Format**: headerless CSV, each row `TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME`
- **Coordinate system**: original world coordinates (DenseTNT does its own
  transformation internally)
- **Cleaning**: AGENT jumps repaired; other objects kept as-is
- **Used by**: DenseTNT training and evaluation

### 7.2 LSTM/Kalman format: `data_processed/{train,val}/*.npz`

- **Format**: NumPy .npz archives
- **Coordinate system**: Agent-centered rotated coordinates (option B)
- **Fields**:

| Field | Shape | Description |
|-------|-------|-------------|
| `scene_id` | int | scene ID |
| `hist` | (20, 2) | history trajectory (rotated) |
| `gt` | (30, 2) | future trajectory (rotated) |
| `hist_global` | (20, 2) | history trajectory (world) |
| `gt_global` | (30, 2) | future trajectory (world) |
| `cent_x` | float | origin x |
| `cent_y` | float | origin y |
| `angle` | float | rotation angle (radians) |
| `city` | str | city name (MIA/PIT) |
| `scene_type` | str | scene type |
| `is_stationary` | bool | whether stationary |
| `speed_anomalies` | int | speed anomaly point count |
| `jump_anomalies` | int | jump anomaly point count |

- **Used by**: models needing normalized input, e.g. LSTM, Kalman filter

### 7.3 Data difference note

Both formats are cleaned by the **exact same rules** (same code path); the only
difference is the coordinate system:
- `data_cleaned/`: keeps original world coordinates → DenseTNT transforms
  internally
- `data_processed/`: additionally applies Agent-centered rotation → used
  directly by LSTM/Kalman

---

## 8. Appendix: Coordinate transformation comparison

Three options were compared on 2,000 validation scenes:

| Option | X range (mean±std) | Y range (mean±std) | endpoint distance (mean±std) |
|--------|--------------------|--------------------|------------------------------|
| A: AV-centered rotation | 13.39±10.91 | 20.32±16.28 | 40.63±32.84 |
| **B: Agent-centered rotation** | **14.24±11.39** | **19.00±16.81** | **29.41±13.31** |
| C: translation only | 16.12±13.44 | 20.27±11.16 | 29.41±13.31 |

Option B has the smallest std of the endpoint distance (13.31), meaning the
prediction target is more stable after the transform, which helps learning.
