"""Generate PPT metrics chart (English labels for font compatibility)

数据来源：
  - CV / Kalman / LSTM 为 README 记录的基线实验值（CV 为近似值 `~`）
  - DenseTNT 两行来自 outputs/eval_output/optimization_comparison.json
    （Argoverse v1.1 验证集 39472 场景的真实评估结果）
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(OUT, '..', '..', 'outputs', 'eval_output', 'optimization_comparison.json')
with open(JSON_PATH, encoding='utf-8') as f:
    _data = json.load(f)

methods = ["CV", "Kalman", "LSTM", "DenseTNT\n(ours)", "DenseTNT\n+optimization"]
# README 基线值：CV ~5.0 / ~10.0 / ~80%（近似，无实测，仅量级参考），
# Kalman / LSTM 为 300 样本子集实验值（与 DenseTNT 的全量验证集口径不同，见 README 脚注）
minADE = [5.0, 2.26, 1.99, _data['baseline']['minADE'], _data['optimization']['minADE']]
minFDE = [10.0, 5.15, 4.60, _data['baseline']['minFDE'], _data['optimization']['minFDE']]
MR     = [80.0, 50.0, 35.0, _data['baseline']['MR'] * 100, _data['optimization']['MR'] * 100]
# CV 行是近似值：数值标签加 ~ 前缀，避免与实测值混淆
APPROX = [True, False, False, False, False]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
colors = ["#aec7e8", "#aec7e8", "#aec7e8", "#d62728", "#1f77b4"]

for ax, vals, title in zip(
    axes,
    [minADE, minFDE, MR],
    ["minADE (m)", "minFDE (m)", "Miss Rate (%)"]
):
    bars = ax.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, v, approx in zip(bars, vals, APPROX):
        label = f"~{v:.1f}" if approx and v >= 10 else (f"~{v:.2f}" if approx else (f"{v:.1f}" if v >= 10 else f"{v:.2f}"))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                label,
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Trajectory Prediction Methods Comparison\n(CV approximate; Kalman/LSTM on 300-scene subset; DenseTNT on full 39,472 val)",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: metrics_comparison.png")
