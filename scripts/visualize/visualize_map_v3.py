#!/usr/bin/env python3
"""
DenseTNT 轨迹预测可视化 (v3 — 增强版)
========================================
- 轨迹占画布 ~80%
- 不同轨迹点用圆点/三角/叉号标注
- 更大画布 + 美观车道线
- 正确的图层顺序
- 模型路径可配置

用法:
    python visualize_map_v3.py 10215
    python visualize_map_v3.py 10215 --model model_save_full_chunked/model_save/model.16.bin
    python visualize_map_v3.py --scenes 10215,10002,10005
    python visualize_map_v3.py --model model_save_no_ct3/model_save/model.16.bin --scenes 10215
"""

import os, sys, argparse, logging, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import torch

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
# 本脚本位于 <repo>/scripts/visualize/，DenseTNT 代码在仓库根 src/ 下
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "model_save_full_chunked", "model_save", "model.16.bin")
VAL_DATA = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "visualizations")

# DenseTNT 代码按仓库根布局 import（utils/modeling 位于 <repo>/src/）
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_DIR)
# ArgoverseMap 按仓库相对路径 <repo>/argoverse-api/map_files/ 加载地图（见 map_api.py），
# 此处环境变量仅作兼容备用
os.environ['ARGOVERSE_MAP_DIR'] = os.path.join(REPO_ROOT, 'argoverse-api', 'map_files')

from argoverse.map_representation.map_api import ArgoverseMap
import utils
from modeling.vectornet import VectorNet

# speed_scale_factor 已由 utils.py 提供（utils_cython 上游缺口已在 utils 侧修复，
# 见 utils.speed_scale_factor），此处无需 monkey-patch。


# ═══════════════════════════════════════════════════════════════════════════
# Color & Style Configuration
# ═══════════════════════════════════════════════════════════════════════════

# 6 prediction trajectories (sorted by model probability)
PRED_COLORS = ["#E53935", "#FB8C00", "#8E24AA", "#6D4C41", "#00ACC1", "#43A047"]
PRED_ALPHAS = [1.0, 0.55, 0.45, 0.35, 0.30, 0.25]

# History (past 2 seconds)
HIST_COLOR = "#1565C0"        # deep blue
HIST_LINEWIDTH = 2.0
HIST_MARKER_SIZE = 110

# Ground truth (future 3 seconds)
GT_COLOR = "#2E7D32"          # forest green
GT_LINEWIDTH = 2.5
GT_MARKER_SIZE = 120

# Agent current position
AGENT_COLOR = "#C62828"       # dark red
AGENT_SIZE = 240

# Lane styles
LANE_COLOR = "#78909C"        # blue-grey
LANE_LINEWIDTH = 1.2
LANE_ALPHA = 0.5

# Figure
FIGURE_DPI = 300               # output DPI
TRAJECTORY_FRAC = 0.65        # trajectory occupies ~65% of canvas


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def load_model(args, model_path):
    """Load DenseTNT VectorNet model from checkpoint."""
    model = VectorNet(args)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def inverse_transform(traj_local, cent_x, cent_y, angle):
    """Convert local (agent-centric) coordinates back to global."""
    from utils import rotate
    traj_global = np.zeros_like(traj_local)
    for i in range(traj_local.shape[0]):
        tx, ty = rotate(traj_local[i, 0], traj_local[i, 1], -angle)
        traj_global[i, 0] = tx + cent_x
        traj_global[i, 1] = ty + cent_y
    return traj_global


def get_trajectory_endpoints(traj):
    """Get start and end points of a trajectory segment for marker placement."""
    return traj[0], traj[-1]


# ═══════════════════════════════════════════════════════════════════════════
# Main visualization function
# ═══════════════════════════════════════════════════════════════════════════

def visualize(scene_id, model, dense_args, am, device, output_dir, data_dir):
    """Generate a single scene visualization."""

    csv_path = os.path.join(data_dir, f"{scene_id}.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    # ── Load & prepare data ──────────────────────────────────────────
    with open(csv_path) as f:
        all_lines = f.readlines()
    lines = all_lines[1:] if all_lines[0].strip().upper().startswith("TIMESTAMP") else all_lines

    import dataset_argoverse
    instance = dataset_argoverse.argoverse_get_instance(lines, f"{scene_id}.csv", dense_args)
    if instance is None:
        print(f"  SKIP: scene {scene_id} could not be parsed")
        return

    mapping = instance
    mapping["matrix"] = torch.tensor(mapping["matrix"], dtype=torch.float)

    # Add speed / labels_is_valid if missing
    if "speed" not in mapping:
        agent_traj = mapping["agents"][0]
        mapping["speed"] = float(np.linalg.norm(agent_traj[-1] - agent_traj[-2]) / 0.1) if len(agent_traj) >= 2 else 0.0
    if "labels_is_valid" not in mapping:
        mapping["labels_is_valid"] = np.ones(30)

    # ── Model forward ────────────────────────────────────────────────
    with torch.no_grad():
        pred_trajs_batch, pred_probs_batch, _ = model([mapping], device)

    # Model output may be torch tensor or numpy array
    pred_trajs = pred_trajs_batch[0]
    pred_probs_arr = pred_probs_batch[0]
    preds_global = pred_trajs.cpu().numpy() if hasattr(pred_trajs, 'cpu') else pred_trajs
    pred_probs = pred_probs_arr.cpu().numpy() if hasattr(pred_probs_arr, 'cpu') else pred_probs_arr

    # ── Trajectory data extraction ───────────────────────────────────
    cent_x, cent_y = mapping["cent_x"], mapping["cent_y"]
    angle = mapping["angle"]
    city = mapping["city_name"]

    origin_labels = mapping.get("origin_labels", None)
    gt_local = mapping["labels"]
    gt_global = origin_labels if origin_labels is not None else inverse_transform(gt_local, cent_x, cent_y, angle)
    hist_local = mapping["agents"][0]
    hist_global = inverse_transform(hist_local, cent_x, cent_y, angle)

    # ── Sort predictions by probability ───────────────────────────────
    probs = pred_probs / pred_probs.sum()
    order = np.argsort(probs)[::-1]
    top_idx = order[0]

    # ── Compute view bounds (trajectory area ~80% of canvas) ─────────
    all_x, all_y = [], []
    for traj in [hist_global, gt_global]:
        all_x.extend(traj[:, 0]); all_y.extend(traj[:, 1])
    for i in range(min(6, len(preds_global))):
        all_x.extend(preds_global[i, :, 0])
        all_y.extend(preds_global[i, :, 1])

    all_x, all_y = np.array(all_x), np.array(all_y)
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()

    # Pad so trajectory occupies ~65% of canvas
    margin_frac = (1 - TRAJECTORY_FRAC) / 2  # margin on each side
    margin_x = max(x_range * margin_frac / TRAJECTORY_FRAC, 3.0)
    margin_y = max(y_range * margin_frac / TRAJECTORY_FRAC, 3.0)

    x_center = (all_x.min() + all_x.max()) / 2
    y_center = (all_y.min() + all_y.max()) / 2
    x_min, x_max = x_center - (x_range/2 + margin_x), x_center + (x_range/2 + margin_x)
    y_min, y_max = y_center - (y_range/2 + margin_y), y_center + (y_range/2 + margin_y)

    # Adaptive figure size: maintain data aspect ratio, target ~13 inches on longer side
    # Prefer wider-than-tall rectangle for trajectory scenes
    data_aspect = (y_max - y_min) / max(x_max - x_min, 0.01)
    target_long = 13.0
    if data_aspect > 1.2:
        fig_w, fig_h = target_long / data_aspect, target_long
    elif data_aspect < 0.6:
        fig_w, fig_h = target_long, target_long * data_aspect
    else:
        fig_w, fig_h = target_long, target_long * 0.78  # slightly wider than tall
    fig_w = max(fig_w, 10); fig_h = max(fig_h, 8)

    # ── Load lane map ────────────────────────────────────────────────
    # Extend search radius to cover the full view
    search_radius = max((x_max - x_min) * 0.55, (y_max - y_min) * 0.55, 50)
    lane_ids = am.get_lane_ids_in_xy_bbox(cent_x, cent_y, city, search_radius)
    lanes = []
    for lid in lane_ids:
        try:
            cl = am.get_lane_segment_centerline(lid, city)
            lanes.append(cl[:, :2])
        except Exception:
            pass

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # ---- Layer 1: Lanes (lowest) ----
    for lane in lanes:
        ax.plot(lane[:, 0], lane[:, 1],
                color=LANE_COLOR, linewidth=LANE_LINEWIDTH, alpha=LANE_ALPHA,
                solid_capstyle="round", zorder=1)

    # ---- Layer 2: Non-top predictions (background) ----
    for rank, k in enumerate(order[1:], 1):
        traj = preds_global[k]
        color = PRED_COLORS[rank]
        alpha = PRED_ALPHAS[rank]

        # Line
        ax.plot(traj[:, 0], traj[:, 1],
                color=color, linewidth=1.2, alpha=alpha,
                linestyle="--", dashes=(4, 3), zorder=2)
        # Endpoint marker: triangle ▼ at destination
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   marker="v", s=95, color=color, alpha=alpha,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # ---- Layer 3: History track (past 2s) ----
    ax.plot(hist_global[:, 0], hist_global[:, 1],
            color=HIST_COLOR, linewidth=HIST_LINEWIDTH, linestyle="-",
            alpha=0.9, zorder=4, solid_capstyle="round")

    # History start point: circle ○
    ax.scatter(hist_global[0, 0], hist_global[0, 1],
               marker="o", s=HIST_MARKER_SIZE, color=HIST_COLOR,
               edgecolors="white", linewidths=1.5, zorder=5)

    # History waypoints: small circles ● (every 2nd to show clearer trajectory)
    for j in range(1, len(hist_global) - 1, 2):
        ax.scatter(hist_global[j, 0], hist_global[j, 1],
                   marker="o", s=20, color=HIST_COLOR, alpha=0.55, zorder=5)

    # ---- Layer 4: Ground truth (future 3s) ----
    ax.plot(gt_global[:, 0], gt_global[:, 1],
            color=GT_COLOR, linewidth=GT_LINEWIDTH, linestyle="-",
            alpha=0.85, zorder=6, solid_capstyle="round")

    # GT waypoints: triangles ▲ (every 2nd)
    for j in range(1, len(gt_global) - 1, 2):
        ax.scatter(gt_global[j, 0], gt_global[j, 1],
                   marker="^", s=40, color=GT_COLOR, alpha=0.55,
                   edgecolors="white", linewidths=0.5, zorder=7)

    # GT end point: star marker
    ax.scatter(gt_global[-1, 0], gt_global[-1, 1],
               marker="*", s=GT_MARKER_SIZE * 1.5, color=GT_COLOR,
               edgecolors="white", linewidths=1.2, zorder=8)

    # ---- Layer 5: Top prediction ----
    top_traj = preds_global[top_idx]
    top_color = PRED_COLORS[0]

    ax.plot(top_traj[:, 0], top_traj[:, 1],
            color=top_color, linewidth=2.2, linestyle="-",
            alpha=0.95, zorder=9, solid_capstyle="round")

    # Top prediction waypoints: diamonds ◆ (every 2nd)
    for j in range(1, len(top_traj) - 1, 2):
        ax.scatter(top_traj[j, 0], top_traj[j, 1],
                   marker="D", s=45, color=top_color, alpha=0.65,
                   edgecolors="white", linewidths=0.7, zorder=10)

    # Top prediction endpoint: cross ✕
    ax.scatter(top_traj[-1, 0], top_traj[-1, 1],
               marker="X", s=160, color=top_color,
               edgecolors="white", linewidths=1.5, zorder=11)

    # ---- Layer 6: Other prediction endpoints (× marker) ----
    for rank, k in enumerate(order[1:3], 1):
        traj = preds_global[k]
        color = PRED_COLORS[rank]
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   marker="x", s=105, color=color, alpha=0.8,
                   linewidths=1.5, zorder=10)

    # ---- Layer 7: Agent current position (top) ----
    # Use a filled arrow/direction indicator
    agent_x, agent_y = hist_global[-1, 0], hist_global[-1, 1]
    if len(hist_global) >= 2:
        heading = np.arctan2(
            hist_global[-1, 1] - hist_global[-2, 1],
            hist_global[-1, 0] - hist_global[-2, 0]
        )
    else:
        heading = 0

    car_length = max((x_max - x_min) * 0.032, 1.2)
    car_width = car_length * 0.5

    car = mpatches.FancyBboxPatch(
        (-car_width / 2, -car_length / 2), car_width, car_length,
        boxstyle="round,pad=0.1",
        facecolor=AGENT_COLOR, edgecolor="#4A0000", linewidth=1.5, alpha=0.95,
        zorder=12
    )
    t = matplotlib.transforms.Affine2D().rotate(-heading + np.pi / 2).translate(agent_x, agent_y) + ax.transData
    car.set_transform(t)
    ax.add_patch(car)

    # Heading arrow
    arrow_len = car_length * 0.6
    ax.annotate("", xy=(agent_x + arrow_len * np.cos(heading),
                        agent_y + arrow_len * np.sin(heading)),
                xytext=(agent_x, agent_y),
                arrowprops=dict(arrowstyle="-|>", color="#4A0000",
                                lw=2.5, alpha=0.9),
                zorder=13)

    # ---- Layer 8: Probability annotations ───────────────────────────
    # Only annotate top 3 predictions to reduce clutter
    for rank, k in enumerate(order[:3]):
        traj = preds_global[k]
        color = PRED_COLORS[rank]
        is_top = (k == top_idx)
        pct = probs[k]

        # Offset direction: away from the trajectory midpoint center
        mid_x, mid_y = traj[15, 0], traj[15, 1]
        dx_end = traj[-1, 0] - mid_x
        dy_end = traj[-1, 1] - mid_y
        norm = max(np.sqrt(dx_end**2 + dy_end**2), 0.1)
        offset_x = dx_end / norm * 6
        offset_y = dy_end / norm * 6

        label_text = f"↑ {pct:.0%}" if is_top else f"{pct:.0%}"
        fontsize = 10 if is_top else 8
        fontweight = "bold" if is_top else "normal"

        ax.annotate(label_text,
                    xy=(traj[-1, 0], traj[-1, 1]),
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=fontsize, fontweight=fontweight, color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec=color, alpha=0.92, linewidth=1.2),
                    zorder=14)

    # ── Axes settings ────────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.40, linestyle="--", linewidth=0.3, zorder=0)

    # Clean axis — no tick labels for a cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    # ── Title ────────────────────────────────────────────────────────
    ade = float(np.mean(np.linalg.norm(top_traj - gt_global, axis=1)))
    fde = float(np.linalg.norm(top_traj[-1] - gt_global[-1]))
    ax.set_title(
        f"DenseTNT Trajectory Prediction — Scene {scene_id}  ({city})",
        fontsize=15, fontweight="bold", pad=12
    )

    # ── Legend ───────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color=HIST_COLOR, linewidth=HIST_LINEWIDTH,
               markersize=7, label=f"History (past 2s)"),
        Line2D([0], [0], marker="^", color=GT_COLOR, linewidth=GT_LINEWIDTH,
               markersize=7, label=f"Ground Truth (future 3s)"),
        Line2D([0], [0], color=top_color, linewidth=2.2,
               marker="X", markersize=8, label=f"Best Pred (ADE={ade:.2f}m, FDE={fde:.2f}m, {probs[top_idx]:.0%})"),
    ]
    # Add remaining predictions
    for rank, k in enumerate(order[1:4], 1):
        pade = np.mean(np.linalg.norm(preds_global[k] - gt_global, axis=1))
        pfde = np.linalg.norm(preds_global[k][-1] - gt_global[-1])
        legend_elements.append(
            Line2D([0], [0], color=PRED_COLORS[rank], linewidth=1.2,
                   linestyle="--", alpha=PRED_ALPHAS[rank],
                   label=f"Alt Pred {rank+1} (ADE={pade:.2f}m, {probs[k]:.0%})")
        )

    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8.5, framealpha=0.92, edgecolor="#CCCCCC",
              ncol=1, borderpad=0.8)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{scene_id}_denseTNT_v3.png")
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DenseTNT Trajectory Prediction Visualization v3")
    parser.add_argument("scene_id", type=str, nargs="?", default=None,
                        help="Scene ID (e.g. 10215)")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene IDs (e.g. 10215,10002)")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all scenes in val/data")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model checkpoint path (default: {DEFAULT_MODEL})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--data_dir", type=str, default=VAL_DATA,
                        help=f"Validation data directory (default: {VAL_DATA})")
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI,
                        help=f"Output DPI (default: {FIGURE_DPI})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    args_cli = parser.parse_args()

    # ── Scene selection ──────────────────────────────────────────────
    if args_cli.all:
        scene_ids = sorted([
            f.replace(".csv", "") for f in os.listdir(args_cli.data_dir)
            if f.endswith(".csv")
        ])
        print(f"Batch mode: {len(scene_ids)} scene(s)")
    elif args_cli.scenes:
        scene_ids = [s.strip() for s in args_cli.scenes.split(",")]
    elif args_cli.scene_id:
        scene_ids = [args_cli.scene_id]
    else:
        parser.print_help()
        sys.exit(1)

    # ── Model path validation ────────────────────────────────────────
    if not os.path.exists(args_cli.model):
        print(f"ERROR: Model not found: {args_cli.model}")
        sys.exit(1)

    print(f"DenseTNT Visualization v3")
    print(f"  Model:   {args_cli.model}")
    print(f"  Data:    {args_cli.data_dir}")
    print(f"  Output:  {args_cli.output_dir}")
    print(f"  Scenes:  {len(scene_ids)}")

    # ── Setup DenseTNT args ──────────────────────────────────────────
    import argparse as ap
    arg_parser = ap.ArgumentParser()
    utils.add_argument(arg_parser)
    dense_args = arg_parser.parse_args([])
    dense_args.hidden_size = 128
    dense_args.future_frame_num = 30
    dense_args.use_map = True
    dense_args.use_centerline = True
    dense_args.core_num = 4
    dense_args.other_params = [
        "semantic_lane", "direction", "l1_loss",
        "goals_2D", "enhance_global_graph", "subdivide",
        "goal_scoring", "laneGCN", "point_sub_graph",
        "lane_scoring", "complete_traj", "complete_traj-3",
    ]
    dense_args.output_dir = os.path.join(args_cli.output_dir, ".tmp")
    dense_args.data_dir = args_cli.data_dir
    dense_args.data_dir_for_val = args_cli.data_dir
    dense_args.do_eval = True
    dense_args.argoverse = True
    dense_args.nms_threshold = 0.5
    os.makedirs(dense_args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(dense_args.output_dir, "model_save"), exist_ok=True)
    os.makedirs(os.path.join(dense_args.output_dir, "temp_file"), exist_ok=True)

    viz_logger = logging.getLogger("viz")
    logging.basicConfig(level=logging.INFO)
    utils.init(dense_args, viz_logger)

    # ── Load model ───────────────────────────────────────────────────
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    print(f"  Device:  {device}")
    print(f"Loading model: {args_cli.model}")
    model = load_model(dense_args, args_cli.model).to(device)

    # ── Load ArgoverseMap ────────────────────────────────────────────
    import dataset_argoverse
    dataset_argoverse.am = ArgoverseMap()
    am = ArgoverseMap()

    # ── Run ──────────────────────────────────────────────────────────
    print(f"\nProcessing {len(scene_ids)} scene(s)...")
    saved = []
    for i, sid in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] Scene {sid}")
        result = visualize(sid, model, dense_args, am, device,
                          args_cli.output_dir, args_cli.data_dir)
        if result:
            saved.append(result)

    print(f"\n{'='*60}")
    print(f"Done! {len(saved)}/{len(scene_ids)} scenes visualized.")
    print(f"Output: {args_cli.output_dir}")


if __name__ == "__main__":
    main()
