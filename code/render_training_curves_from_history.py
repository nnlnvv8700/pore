#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 training_history.csv 和 training_history_by_rock_type.csv 重绘训练曲线。

适用场景：
- 训练已经完成，不想重跑
- 需要统一训练曲线风格
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def maybe_array(rows: List[Dict[str, str]], key: str) -> np.ndarray | None:
    if not rows or key not in rows[0]:
        return None
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def configure_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_dual(fig, out_png: str) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    base, _ = os.path.splitext(out_png)
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main_curves(rows: List[Dict[str, str]], out_dir: str, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    epochs = np.asarray([int(r["epoch"]) for r in rows], dtype=np.int32)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8))
    axes = axes.flatten()

    train_loss = maybe_array(rows, "train_loss")
    val_loss = maybe_array(rows, "val_loss")
    train_rmse = maybe_array(rows, "train_RMSE_pore_mean")
    val_rmse = maybe_array(rows, "val_RMSE_pore_mean")
    train_mae = maybe_array(rows, "train_MAE_pore_mean")
    val_mae = maybe_array(rows, "val_MAE_pore_mean")
    train_r2 = maybe_array(rows, "train_q_R2")
    val_r2 = maybe_array(rows, "val_q_R2")
    val_rel_mean = maybe_array(rows, "val_RelFluxErr_mean")
    val_rel_median = maybe_array(rows, "val_RelFluxErr_median")
    val_rel_p90 = maybe_array(rows, "val_RelFluxErr_p90")
    val_rel_p95 = maybe_array(rows, "val_RelFluxErr_p95")
    lr = maybe_array(rows, "lr")

    # (a) loss
    if train_loss is not None and val_loss is not None:
        ax = axes[0]
        ax.plot(epochs, train_loss, color="#1f4e79", label="Train")
        ax.plot(epochs, val_loss, color="#c84c31", label="Validation")
        ax.set_title("(a) Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        style_axes(ax)

    # (b) rel flux err
    ax = axes[1]
    plotted = False
    for arr, label, color in [
        (val_rel_mean, "Mean", "#1f4e79"),
        (val_rel_median, "Median", "#4c956c"),
        (val_rel_p90, "P90", "#c84c31"),
        (val_rel_p95, "P95", "#8e6c8a"),
    ]:
        if arr is not None:
            ax.plot(epochs, arr, label=label, color=color)
            plotted = True
    ax.set_title("(b) Validation Relative Flux Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RelFluxErr")
    if plotted:
        ax.legend(frameon=False, ncol=2)
        if np.all(np.isfinite(np.concatenate([a for a in [val_rel_mean, val_rel_median, val_rel_p90, val_rel_p95] if a is not None]))) and np.nanmax(np.concatenate([a for a in [val_rel_mean, val_rel_median, val_rel_p90, val_rel_p95] if a is not None])) > 0:
            ax.set_yscale("log")
    style_axes(ax)

    # (c) R2
    ax = axes[2]
    if train_r2 is not None and val_r2 is not None:
        ax.plot(epochs, train_r2, color="#1f4e79", label="Train")
        ax.plot(epochs, val_r2, color="#c84c31", label="Validation")
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(frameon=False)
    ax.set_title("(c) Flux $R^2$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$R^2$")
    style_axes(ax)

    # (d) RMSE
    ax = axes[3]
    if train_rmse is not None and val_rmse is not None:
        ax.plot(epochs, train_rmse, color="#1f4e79", label="Train")
        ax.plot(epochs, val_rmse, color="#c84c31", label="Validation")
        ax.legend(frameon=False)
    ax.set_title("(d) RMSE in Pore Region")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    style_axes(ax)

    # (e) MAE
    ax = axes[4]
    if train_mae is not None and val_mae is not None:
        ax.plot(epochs, train_mae, color="#1f4e79", label="Train")
        ax.plot(epochs, val_mae, color="#c84c31", label="Validation")
        ax.legend(frameon=False)
    ax.set_title("(e) MAE in Pore Region")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    style_axes(ax)

    # (f) learning rate
    ax = axes[5]
    if lr is not None:
        ax.plot(epochs, lr, color="#e09f3e")
        ax.set_yscale("log")
    ax.set_title("(f) Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    style_axes(ax)

    plt.suptitle(f"{title_prefix} Training Curves", y=0.995, fontsize=11)
    plt.tight_layout()
    save_dual(fig, str(Path(out_dir) / "training_curves_paper.png"))


def plot_by_type(rows: List[Dict[str, str]], out_dir: str, title_prefix: str) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt

    rock_types = sorted({int(r["rock_type"]) for r in rows})
    if not rock_types:
        return

    colors = ["#1f4e79", "#4c956c", "#3f7cac", "#c84c31", "#8e6c8a", "#bfa23a", "#5c677d", "#2a9d8f"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.8))
    axes = axes.flatten()

    for i, rt in enumerate(rock_types):
        rt_rows = [r for r in rows if int(r["rock_type"]) == rt]
        epochs = np.asarray([int(r["epoch"]) for r in rt_rows], dtype=np.int32)
        col = colors[i % len(colors)]
        axes[0].plot(epochs, [float(r["RelFluxErr_p90"]) for r in rt_rows], color=col, label=f"R{rt}")
        axes[1].plot(epochs, [float(r["RelFluxErr_median"]) for r in rt_rows], color=col, label=f"R{rt}")
        axes[2].plot(epochs, [float(r["RelFluxErr_mean"]) for r in rt_rows], color=col, label=f"R{rt}")
        axes[3].plot(epochs, [float(r["q_R2"]) for r in rt_rows], color=col, label=f"R{rt}")

    axes[0].set_title("(a) RelFluxErr P90")
    axes[1].set_title("(b) RelFluxErr Median")
    axes[2].set_title("(c) RelFluxErr Mean")
    axes[3].set_title("(d) Flux $R^2$")
    for idx in [0, 1, 2]:
        axes[idx].set_yscale("log")
    axes[3].axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[3].set_ylim(-0.1, 1.05)

    final_epoch = max(int(r["epoch"]) for r in rows)
    final_rows = sorted([r for r in rows if int(r["epoch"]) == final_epoch], key=lambda x: int(x["rock_type"]))
    x = np.arange(len(final_rows))
    p90_vals = [float(r["RelFluxErr_p90"]) for r in final_rows]
    r2_vals = [float(r["q_R2"]) for r in final_rows]
    bar_colors = [colors[int(r["rock_type"]) % len(colors)] for r in final_rows]
    axes[4].bar(x, p90_vals, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    axes[4].set_xticks(x)
    axes[4].set_xticklabels([f"R{int(r['rock_type'])}" for r in final_rows])
    axes[4].set_title(f"(e) Final P90 at Epoch {final_epoch}")
    axes[4].set_ylabel("RelFluxErr P90")
    axes[5].bar(x, r2_vals, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=0.5)
    axes[5].set_xticks(x)
    axes[5].set_xticklabels([f"R{int(r['rock_type'])}" for r in final_rows])
    axes[5].set_title(f"(f) Final $R^2$ at Epoch {final_epoch}")
    axes[5].set_ylabel("$R^2$")
    axes[5].set_ylim(0, 1.05)

    for ax in axes[:4]:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        style_axes(ax)

    plt.suptitle(f"{title_prefix} Per-Rock Training Curves", y=0.995, fontsize=11)
    plt.tight_layout()
    save_dual(fig, str(Path(out_dir) / "training_curves_by_rock_type_paper.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="根据已有 history csv 重绘训练曲线。")
    parser.add_argument("--history", type=str, required=True, help="training_history.csv")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--by-type", type=str, default="", help="training_history_by_rock_type.csv")
    parser.add_argument("--title-prefix", type=str, default="Model", help="图标题前缀")
    args = parser.parse_args()

    configure_style()
    rows = load_csv_rows(args.history)
    plot_main_curves(rows, args.out_dir, args.title_prefix)
    if args.by_type:
        by_type_rows = load_csv_rows(args.by_type)
        plot_by_type(by_type_rows, args.out_dir, args.title_prefix)

    print(f"saved training curve renders to: {args.out_dir}")


if __name__ == "__main__":
    main()
