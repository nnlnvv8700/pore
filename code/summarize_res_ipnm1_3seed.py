#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize 3-seed Res-IPNM1-FlowNet experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    "best_epoch",
    "best_val_loss",
    "pixel_r2_all",
    "pixel_r2_val",
    "pixel_r2_test",
    "q_mean_re_all",
    "q_mean_re_val",
    "q_mean_re_test",
    "q_r2_all",
    "q_r2_val",
    "q_r2_test",
    "table_g_mean_re_all",
    "table_g_r2_all",
    "true_g_mean_re_all",
    "true_g_r2_all",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_seed(seed_dir: Path) -> dict:
    run_dir = seed_dir / "res_ed"
    summary = read_json(run_dir / "summary.json")
    conductance = read_json(run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json")
    row = {
        "seed": int(seed_dir.name.split("_")[-1]),
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "pixel_r2_all": summary["all"]["pixel_r2"],
        "pixel_r2_val": summary["val"]["pixel_r2"],
        "pixel_r2_test": summary["test"]["pixel_r2"],
        "q_mean_re_all": summary["all"]["q_mean_re"],
        "q_mean_re_val": summary["val"]["q_mean_re"],
        "q_mean_re_test": summary["test"]["q_mean_re"],
        "q_r2_all": summary["all"]["q_r2"],
        "q_r2_val": summary["val"]["q_r2"],
        "q_r2_test": summary["test"]["q_r2"],
        "table_g_mean_re_all": conductance["g_metrics_vs_table"]["mean_rel_error"],
        "table_g_median_re_all": conductance["g_metrics_vs_table"]["median_rel_error"],
        "table_g_r2_all": conductance["g_metrics_vs_table"]["r2"],
        "true_g_mean_re_all": conductance["g_metrics_vs_true_velocity"]["mean_rel_error"],
        "true_g_r2_all": conductance["g_metrics_vs_true_velocity"]["r2"],
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(root: Path, rows: list[dict], stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in rows:
        hist_path = root / f"seed_{row['seed']}" / "res_ed" / "training_history.csv"
        hist = list(csv.DictReader(hist_path.open("r", encoding="utf-8")))
        ep = [int(x["epoch"]) for x in hist]
        axes[0].plot(ep, [float(x["val_loss"]) for x in hist], label=f"seed {row['seed']}")
        axes[1].plot(ep, [float(x["val_flux"]) for x in hist], label=f"seed {row['seed']}")
    axes[0].set_title("Res-IPNM1-FlowNet validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation integrated-flux loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative q loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "res_ipnm1_3seed_loss_curves.png", dpi=180)
    plt.close(fig)

    labels = ["Velocity R2", "q R2", "Table g R2", "q mean RE", "Table g mean RE"]
    means = [
        stats["pixel_r2_all"]["mean"],
        stats["q_r2_all"]["mean"],
        stats["table_g_r2_all"]["mean"],
        100.0 * stats["q_mean_re_all"]["mean"],
        100.0 * stats["table_g_mean_re_all"]["mean"],
    ]
    stds = [
        stats["pixel_r2_all"]["std"],
        stats["q_r2_all"]["std"],
        stats["table_g_r2_all"]["std"],
        100.0 * stats["q_mean_re_all"]["std"],
        100.0 * stats["table_g_mean_re_all"]["std"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=["#4c78a8", "#4c78a8", "#4c78a8", "#f58518", "#f58518"])
    ax.set_title("Res-IPNM1-FlowNet 3-seed summary")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "res_ipnm1_3seed_metric_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"E:\mhw\1\cup\runs\res_ipnm1_flownet_100e_3seed_20260527")
    args = parser.parse_args()
    root = Path(args.root)
    rows = [collect_seed(p) for p in sorted(root.glob("seed_*")) if (p / "res_ed" / "summary.json").exists()]
    rows.sort(key=lambda x: x["seed"])
    write_csv(root / "res_ipnm1_3seed_metrics.csv", rows)

    stats = {}
    for metric in METRICS:
        arr = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        stats[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    (root / "res_ipnm1_3seed_summary.json").write_text(json.dumps({"rows": rows, "stats": stats}, indent=2), encoding="utf-8")
    plot_summary(root, rows, stats)

    lines = [
        "# Res-IPNM1-FlowNet 100 Epoch 3-Seed Summary",
        "",
        "| Metric | Mean +/- std |",
        "| --- | ---: |",
        f"| Best epoch | {stats['best_epoch']['mean']:.1f} +/- {stats['best_epoch']['std']:.1f} |",
        f"| Velocity pixel R2, all | {stats['pixel_r2_all']['mean']:.4f} +/- {stats['pixel_r2_all']['std']:.4f} |",
        f"| Velocity pixel R2, val | {stats['pixel_r2_val']['mean']:.4f} +/- {stats['pixel_r2_val']['std']:.4f} |",
        f"| Velocity pixel R2, test | {stats['pixel_r2_test']['mean']:.4f} +/- {stats['pixel_r2_test']['std']:.4f} |",
        f"| q mean RE, all | {100.0 * stats['q_mean_re_all']['mean']:.2f}% +/- {100.0 * stats['q_mean_re_all']['std']:.2f}% |",
        f"| q R2, all | {stats['q_r2_all']['mean']:.4f} +/- {stats['q_r2_all']['std']:.4f} |",
        f"| Table g mean RE, all | {100.0 * stats['table_g_mean_re_all']['mean']:.2f}% +/- {100.0 * stats['table_g_mean_re_all']['std']:.2f}% |",
        f"| Table g R2, all | {stats['table_g_r2_all']['mean']:.4f} +/- {stats['table_g_r2_all']['std']:.4f} |",
        f"| True-g mean RE, all | {100.0 * stats['true_g_mean_re_all']['mean']:.2f}% +/- {100.0 * stats['true_g_mean_re_all']['std']:.2f}% |",
        f"| True-g R2, all | {stats['true_g_r2_all']['mean']:.4f} +/- {stats['true_g_r2_all']['std']:.4f} |",
        "",
        "Figures:",
        "",
        "- `res_ipnm1_3seed_loss_curves.png`",
        "- `res_ipnm1_3seed_metric_summary.png`",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "README.md")


if __name__ == "__main__":
    main()
