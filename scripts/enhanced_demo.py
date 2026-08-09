"""
Trajectory Prediction — Three-Method Comparison Pipeline
========================================================
Methods:  Constant Velocity (CV) | Kalman Filter | LSTM

Usage:
    python enhanced_demo.py

Output:
    - Console: ADE/FDE comparison table
    - Plots: trajectory comparison figures
    - Saved figures to ./scripts/results/ directory
"""

import argparse
import os, sys, glob, random, warnings
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Project imports ──────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.learning_based.lstm_predictor import LSTMTrajectoryModel
from models.rule_based.kalman_predictor import KalmanTrajectoryPredictor
from models.rule_based.cv_predictor import ConstantVelocityPredictor
from models.loss_common import ade_loss, fde_loss, WeightedSmoothL1Loss
from models.metrics_common import compute_ade, compute_fde
from scripts.preprocess.argo1_dataset import ArgoverseV1Dataset

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Trajectory prediction demo: CV vs Kalman vs LSTM")
parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "raw"),
                    help="Argoverse 1 raw CSV directory (default: <repo>/data/raw)")
parser.add_argument("--processed-dir", default=os.path.join(REPO_ROOT, "data", "processed"),
                    help="output directory for preprocessed .pt samples")
parser.add_argument("--max-files", type=int, default=4000,
                    help="max number of samples to use")
parser.add_argument("--device", default=None,
                    help="torch device (default: cuda if available)")
args_cli = parser.parse_args()

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = args_cli.data_dir
PROCESSED_DIR = args_cli.processed_dir
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device(args_cli.device if args_cli.device
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[INFO] Device: {DEVICE}")

# Data settings
MAX_FILES = args_cli.max_files
TRAIN_RATIO = 0.8
SEED = 42

# LSTM hyperparameters
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.5
LSTM_LR = 1e-3
LSTM_EPOCHS = 30          # increased from demo's 10
PATIENCE = 5              # early stopping

print(f"[INFO] Root: {ROOT}")
print(f"[INFO] Data dir: {DATA_DIR}")
print(f"[INFO] Processed dir: {PROCESSED_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 1: Data Loading")
print("=" * 60)

if not os.path.isdir(PROCESSED_DIR) or len(glob.glob(os.path.join(PROCESSED_DIR, "*.pt"))) == 0:
    raw_csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not raw_csvs:
        print(f"[ERROR] No .csv files found directly under {DATA_DIR}.")
        print("        Argoverse 1 official layout is data/raw/{train,val}/data/*.csv —")
        print("        pass --data-dir pointing at the directory that directly contains")
        print("        the scene CSVs (e.g. --data-dir data/raw/val/data).")
        sys.exit(1)
    print("[INFO] Processed data not found. Running dataset.process() ...")
    ds = ArgoverseV1Dataset(REPO_ROOT, raw_dir=DATA_DIR, processed_dir=PROCESSED_DIR)
    ds.process()
    print("[DONE] Data processing complete.")

files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.pt")))
print(f"[INFO] Found {len(files)} processed samples")

# Split: prefer official split subdirectories (processed_dir/train, /val) when
# present; otherwise fall back to a random 80/20 file split and warn — a random
# split of one pool is NOT the official split, so if the pool mixes official
# train and val scenes, validation metrics would be optimistic (leak).
train_dir = os.path.join(PROCESSED_DIR, "train")
val_dir = os.path.join(PROCESSED_DIR, "val")
train_files = sorted(glob.glob(os.path.join(train_dir, "*.pt")))
val_files = sorted(glob.glob(os.path.join(val_dir, "*.pt")))
if train_files and val_files:
    train_files = train_files[:MAX_FILES]
    print(f"[INFO] Using official split dirs: train={len(train_files)}, val={len(val_files)}")
else:
    random.seed(SEED)
    random.shuffle(files)
    files = files[:MAX_FILES]
    print(f"[INFO] Using {len(files)} samples (max={MAX_FILES})")
    split = int(len(files) * TRAIN_RATIO)
    train_files = files[:split]
    val_files = files[split:]
    print(f"[WARN] No {PROCESSED_DIR}/{{train,val}} subdirs found — using a random 80/20 file split.")
    print(f"       If this pool mixes official train AND val scenes, validation is leaked;")
    print(f"       prefer processing official splits into {PROCESSED_DIR}/train and /val.")
print(f"[INFO] Train: {len(train_files)} | Val: {len(val_files)}")

def load_sample(path):
    d = torch.load(path, weights_only=False)
    idx = int(d["agent_index"]) if "agent_index" in d else 0
    return d["x"][idx], d["y"][idx], str(d["seq_id"])

# ═══════════════════════════════════════════════════════════════════════════
# 2. LSTM Training
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2: LSTM Training")
print("=" * 60)

lstm = LSTMTrajectoryModel(
    input_size=2, hidden_size=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
).to(DEVICE)
optimizer = torch.optim.Adam(lstm.parameters(), lr=LSTM_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3,
)
criterion = WeightedSmoothL1Loss(beta=1.0, reduction="mean")

best_val_loss = float("inf")
patience_counter = 0
history_loss = []

for epoch in range(LSTM_EPOCHS):
    lstm.train()
    train_loss = 0.0
    for path in train_files:
        hist, fut, _ = load_sample(path)
        hist, fut = hist.to(DEVICE), fut.to(DEVICE)
        pred = lstm(hist.unsqueeze(0), steps=fut.shape[0]).squeeze(0)
        loss = criterion(pred.unsqueeze(0), fut.unsqueeze(0)) \
               + 0.5 * fde_loss(pred.unsqueeze(0), fut.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / max(1, len(train_files))

    # Validation (subset for speed)
    lstm.eval()
    val_loss = 0.0
    val_subset = val_files[:min(500, len(val_files))]
    with torch.no_grad():
        for path in val_subset:
            hist, fut, _ = load_sample(path)
            hist, fut = hist.to(DEVICE), fut.to(DEVICE)
            pred = lstm(hist.unsqueeze(0), steps=fut.shape[0]).squeeze(0)
            # same loss composition as training (smooth-L1 + 0.5 * FDE)
            val_loss += (criterion(pred.unsqueeze(0), fut.unsqueeze(0))
                         + 0.5 * fde_loss(pred.unsqueeze(0), fut.unsqueeze(0))).item()
    avg_val = val_loss / max(1, len(val_subset))

    scheduler.step(avg_val)
    history_loss.append((avg_train, avg_val))

    # Early stopping
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(lstm.state_dict(), os.path.join(RESULTS_DIR, "lstm_best.pt"))
    else:
        patience_counter += 1

    print(f"epoch {epoch:3d} | train_loss {avg_train:.4f} | val_loss {avg_val:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

    if patience_counter >= PATIENCE:
        print(f"[INFO] Early stopping at epoch {epoch}")
        break

# Load best checkpoint
lstm.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "lstm_best.pt"), weights_only=True))
lstm.eval()
print(f"[DONE] LSTM training complete. Best val_loss: {best_val_loss:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Batch Evaluation — Three Methods
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: Three-Method Evaluation")
print("=" * 60)

results = {
    "CV":     {"ade": [], "fde": []},
    "Kalman": {"ade": [], "fde": []},
    "LSTM":   {"ade": [], "fde": []},
}

# Also store per-sample data for later visualization
sample_data = {}

for i, path in enumerate(val_files):
    hist, fut, seq_id = load_sample(path)

    # ---- CV Prediction ----
    cv = ConstantVelocityPredictor(dt=1.0).fit(hist.numpy())
    cv_pred = torch.tensor(cv.forecast(fut.shape[0]), dtype=torch.float32)

    # ---- Kalman Prediction ----
    kal = KalmanTrajectoryPredictor(dt=1.0).fit(hist.numpy())
    kal_pred = torch.tensor(kal.forecast(fut.shape[0]), dtype=torch.float32)

    # ---- LSTM Prediction ----
    with torch.no_grad():
        lstm_pred = lstm(
            hist.to(DEVICE).unsqueeze(0), steps=fut.shape[0]
        ).squeeze(0).cpu()

    # ---- Metrics ----
    for name, pred in [("CV", cv_pred), ("Kalman", kal_pred), ("LSTM", lstm_pred)]:
        ade = compute_ade(pred.unsqueeze(0), fut.unsqueeze(0))
        fde = compute_fde(pred.unsqueeze(0), fut.unsqueeze(0))
        results[name]["ade"].append(ade)
        results[name]["fde"].append(fde)

    # Store sample-level data (limit to 500 for memory)
    if i < 500:
        sample_data[path] = {
            "seq_id": seq_id,
            "hist": hist.numpy(),
            "fut": fut.numpy(),
            "cv_pred": cv_pred.numpy(),
            "kal_pred": kal_pred.numpy(),
            "lstm_pred": lstm_pred.numpy(),
        }

    if (i + 1) % 200 == 0:
        print(f"  Evaluated {i + 1}/{len(val_files)} samples")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Results Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4: Results")
print("=" * 60)

print(f"\n{'Method':<12} {'ADE':>10} {'FDE':>10}")
print("-" * 35)
for name in ["CV", "Kalman", "LSTM"]:
    ade_mean = np.mean(results[name]["ade"])
    fde_mean = np.mean(results[name]["fde"])
    print(f"{name:<12} {ade_mean:>10.4f} {fde_mean:>10.4f}")

# Save results to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write(f"Trajectory Prediction — Three Method Comparison\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Val samples: {len(val_files)}\n\n")
    f.write(f"{'Method':<12} {'ADE':>10} {'FDE':>10}\n")
    f.write("-" * 35 + "\n")
    for name in ["CV", "Kalman", "LSTM"]:
        ade_mean = np.mean(results[name]["ade"])
        fde_mean = np.mean(results[name]["fde"])
        ade_std  = np.std(results[name]["ade"])
        fde_std  = np.std(results[name]["fde"])
        f.write(f"{name:<12} {ade_mean:>10.4f} ±{ade_std:.4f}    {fde_mean:>10.4f} ±{fde_std:.4f}\n")
print(f"\n[INFO] Results saved to: {results_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: Visualization")
print("=" * 60)


def plot_trajectory(hist, fut, predictions, title, save_path):
    """Plot history, ground truth, and all predictions."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {"CV": "orange", "Kalman": "red", "LSTM": "blue"}
    markers = {"CV": "s", "Kalman": "x", "LSTM": "x"}

    # History & Ground Truth
    ax.plot(hist[1:, 0], hist[1:, 1], "k--", marker=".", label="History")
    ax.plot(fut[1:, 0], fut[1:, 1], "g-", marker=".", label="Ground Truth")

    for name, pred in predictions.items():
        ade = compute_ade(
            torch.tensor(pred).unsqueeze(0), torch.tensor(fut).unsqueeze(0)
        )
        ax.plot(
            pred[:, 0], pred[:, 1],
            color=colors.get(name, "gray"),
            marker=markers.get(name, "."),
            label=f"{name} (ADE={ade:.2f})",
        )

    # Agent rectangle at last history position
    if len(hist) >= 2:
        current_pos = hist[-1]
        prev_pos = hist[-2]
        heading = np.arctan2(current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0])
        length, width = 4.0, 2.0
        rect = plt.Rectangle(
            (-length / 2, -width / 2), length, width,
            color="orangered", alpha=0.8, label="Agent",
        )
        t = matplotlib.transforms.Affine2D().rotate(heading).translate(
            current_pos[0], current_pos[1]
        ) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        arrow_len = 3.0
        ax.arrow(
            current_pos[0], current_pos[1],
            arrow_len * np.cos(heading), arrow_len * np.sin(heading),
            head_width=0.5, head_length=0.5, fc="orangered", ec="orangered",
        )

    ax.axis("equal")
    ax.legend(loc="best")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 5a. Random samples ──
print("\n[5a] Visualizing random validation samples ...")
n_viz = 5
indices = random.sample(range(min(len(val_files), 500)), min(n_viz, min(len(val_files), 500)))
for j, idx in enumerate(indices):
    path = val_files[idx]
    d = sample_data.get(path)
    if d is None:
        hist, fut, seq_id = load_sample(path)
        hist_np = hist.numpy()
        fut_np = fut.numpy()
        cv_pred = (ConstantVelocityPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0]))
        kal_pred = KalmanTrajectoryPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0])
        with torch.no_grad():
            lstm_pred_np = lstm(hist.to(DEVICE).unsqueeze(0), steps=fut.shape[0]).squeeze(0).cpu().numpy()
    else:
        seq_id = d["seq_id"]
        hist_np = d["hist"]
        fut_np = d["fut"]
        cv_pred = d["cv_pred"]
        kal_pred = d["kal_pred"]
        lstm_pred_np = d["lstm_pred"]

    plot_trajectory(
        hist_np, fut_np,
        {"CV": cv_pred, "Kalman": kal_pred, "LSTM": lstm_pred_np},
        title=f"Trajectory Prediction — Seq {seq_id} (Val #{idx})",
        save_path=os.path.join(RESULTS_DIR, f"sample_{seq_id}.png"),
    )

# ── 5b. Turning scenario: 10215.csv ──
print("\n[5b] Visualizing 10215.csv (turning scenario) ...")

# Find 10215 in validation set
turning_path = None
turning_idx = None
for i, p in enumerate(val_files):
    _, _, sid = load_sample(p)
    if sid == "10215":
        turning_path = p
        turning_idx = i
        break

if turning_path is not None:
    print(f"  Found seq_id=10215 at val index {turning_idx}")
    hist, fut, seq_id = load_sample(turning_path)
    hist_np = hist.numpy()
    fut_np = fut.numpy()

    cv_pred = ConstantVelocityPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0])
    kal_pred = KalmanTrajectoryPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0])
    with torch.no_grad():
        lstm_pred_np = lstm(hist.to(DEVICE).unsqueeze(0), steps=fut.shape[0]).squeeze(0).cpu().numpy()

    plot_trajectory(
        hist_np, fut_np,
        {"CV": cv_pred, "Kalman": kal_pred, "LSTM": lstm_pred_np},
        title="Turning Scenario — Seq 10215 (Three-Method Comparison)",
        save_path=os.path.join(RESULTS_DIR, "turning_10215.png"),
    )

    # Also save a zoomed version
    plot_trajectory(
        hist_np, fut_np,
        {"CV": cv_pred, "Kalman": kal_pred, "LSTM": lstm_pred_np},
        title="Turning Scenario — Seq 10215 (Zoomed)",
        save_path=os.path.join(RESULTS_DIR, "turning_10215_zoomed.png"),
    )
else:
    print("  [WARN] seq_id=10215 not found in validation set. "
          "It may be in the training set or not loaded.")
    # Try to find it in all files
    for p in files:
        _, _, sid = load_sample(p)
        if sid == "10215":
            print(f"  Found 10215 at index {files.index(p)} (train set). "
                  f"Generating prediction anyway...")
            hist, fut, seq_id = load_sample(p)
            hist_np = hist.numpy()
            fut_np = fut.numpy()
            cv_pred = ConstantVelocityPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0])
            kal_pred = KalmanTrajectoryPredictor(dt=1.0).fit(hist_np).forecast(fut.shape[0])
            with torch.no_grad():
                lstm_pred_np = lstm(hist.to(DEVICE).unsqueeze(0), steps=fut.shape[0]).squeeze(0).cpu().numpy()
            plot_trajectory(
                hist_np, fut_np,
                {"CV": cv_pred, "Kalman": kal_pred, "LSTM": lstm_pred_np},
                title="Turning Scenario — Seq 10215 (Three-Method Comparison)",
                save_path=os.path.join(RESULTS_DIR, "turning_10215.png"),
            )
            break

# ── 5c. Loss curve ──
print("\n[5c] Plotting training loss curve ...")
fig, ax = plt.subplots(figsize=(8, 5))
train_l, val_l = zip(*history_loss)
ax.plot(train_l, label="Train Loss")
ax.plot(val_l, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("LSTM Training Curve")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=150)
plt.close(fig)

# ── 5d. ADE/FDE histogram comparison ──
print("\n[5d] Plotting ADE/FDE distribution ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for j, metric in enumerate(["ade", "fde"]):
    ax = axes[j]
    for name, color in [("CV", "orange"), ("Kalman", "red"), ("LSTM", "blue")]:
        ax.hist(
            results[name][metric], bins=50, alpha=0.5, label=name,
            color=color, density=True,
        )
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Density")
    ax.set_title(f"{metric.upper()} Distribution")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "ade_fde_hist.png"), dpi=150)
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"\nResults saved to: {RESULTS_DIR}")
print("\nFiles:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  - {f}")

print(f"\nFinal Comparison:")
print(f"{'Method':<12} {'ADE':>10} {'FDE':>10}")
print("-" * 35)
for name in ["CV", "Kalman", "LSTM"]:
    print(f"{name:<12} {np.mean(results[name]['ade']):>10.4f} {np.mean(results[name]['fde']):>10.4f}")
