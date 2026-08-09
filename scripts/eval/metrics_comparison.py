#!/usr/bin/env python3
"""Generate PPT metrics chart (English labels for font compatibility)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

methods = ["CV", "Kalman", "LSTM", "DenseTNT\n(Ours)"]
# NOTE: 以下数值为早期实验（8k 训练子集）的展示值，仅用于生成 PPT 图表；
# 最新结果见 README 与 outputs/eval_output/optimization_comparison.json。
minADE = [3.50, 2.80, 1.80, 1.036]
minFDE = [7.00, 5.50, 3.50, 1.502]
MR     = [60.0, 45.0, 25.0, 10.9] # 修改这个地方使用其他几种模型的数据即可

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
colors = ["#aec7e8", "#aec7e8", "#aec7e8", "#d62728"]

for ax, vals, title in zip(
    axes,
    [minADE, minFDE, MR],
    ["minADE (m)", "minFDE (m)", "Miss Rate (%)"]
):
    bars = ax.bar(methods, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:.1f}" if v >= 10 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Trajectory Prediction Methods Comparison", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: metrics_comparison.png")
