#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 metrics.csv 绘制成更适合论文或汇报的统计图。

示例：
  python visualize_metrics.py --metrics "E:\\mhw\\1\\pore\\runs\\unet_all32_v2\\infer_ext\\metrics.csv"
"""

import os
import csv
import argparse
from typing import Dict, List

import numpy as np


def load_metrics(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    cols: Dict[str, List[float]] = {}
    for k in rows[0].keys():
        cols[k] = []

    for r in rows:
        for k, v in r.items():
            if k in ("index", "rock_type", "global_id"):
                cols[k].append(int(float(v)))
            else:
                cols[k].append(float(v))

    return {k: np.asarray(v) for k, v in cols.items()}


def plot_metrics(data: Dict[str, np.ndarray], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rock_type = data["rock_type"]
    uniq_types = sorted(np.unique(rock_type).tolist())
    colors = ["#1f4e79", "#3f7cac", "#6aaed6", "#c84c31", "#e07a5f", "#7a9e7e"]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6))

    # Histograms / density-like summaries
    axes[0, 0].hist(data["epe"], bins=36, color="#1f4e79", alpha=0.9, edgecolor="white", linewidth=0.35)
    axes[0, 0].set_title("(a) EPE Distribution")
    axes[0, 0].set_xlabel("EPE")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(data["cosine"], bins=36, color="#4c956c", alpha=0.9, edgecolor="white", linewidth=0.35)
    axes[0, 1].set_title("(b) Cosine Similarity")
    axes[0, 1].set_xlabel("Cosine")

    axes[0, 2].hist(data["div_mean"], bins=36, color="#c84c31", alpha=0.9, edgecolor="white", linewidth=0.35)
    axes[0, 2].set_title("(c) Mean Divergence")
    axes[0, 2].set_xlabel(r"Mean $|\nabla \cdot u|$")
    axes[0, 2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Boxplots by rock type
    def by_type(metric: str) -> List[np.ndarray]:
        return [data[metric][rock_type == rt] for rt in uniq_types]

    for ax, metric, title in [
        (axes[1, 0], "mae_speed", "(d) MAE by Rock Type"),
        (axes[1, 1], "epe", "(e) EPE by Rock Type"),
        (axes[1, 2], "cosine", "(f) Cosine by Rock Type"),
    ]:
        bp = ax.boxplot(
            by_type(metric),
            tick_labels=[f"R{rt}" for rt in uniq_types],
            showfliers=False,
            patch_artist=True,
            widths=0.65,
            medianprops=dict(color="black", linewidth=0.9),
            whiskerprops=dict(color="#555555", linewidth=0.8),
            capprops=dict(color="#555555", linewidth=0.8),
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.85)
            patch.set_edgecolor("#444444")
            patch.set_linewidth(0.7)
        ax.set_title(title)
        ax.set_xlabel("Rock Type")

    for ax in axes.flat:
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    base, _ = os.path.splitext(out_path)
    plt.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize infer metrics.csv")
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    ap.add_argument("--out", type=str, default="", help="Output png path (default: metrics_viz.png next to csv)")
    args = ap.parse_args()

    data = load_metrics(args.metrics)
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(args.metrics), "metrics_viz.png")

    plot_metrics(data, out_path)


if __name__ == "__main__":
    main()
