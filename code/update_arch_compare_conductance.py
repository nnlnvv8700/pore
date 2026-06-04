#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Refresh architecture comparison tables/figures with postprocessed conductance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(compare_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(x["model"]) for x in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for item in rows:
        hist = list(csv.DictReader((compare_dir / str(item["model"]) / "training_history.csv").open("r", encoding="utf-8")))
        ep = [int(r["epoch"]) for r in hist]
        axes[0].plot(ep, [float(r["val_loss"]) for r in hist], label=str(item["model"]))
        axes[1].plot(ep, [float(r["val_flux"]) for r in hist], label=str(item["model"]))
    axes[0].set_title("Validation total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation flux loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative q loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "loss_curves_comparison.png", dpi=180)
    plt.close(fig)

    metrics = ["pixel_r2_all", "q_r2_all", "table_g_r2_all"]
    titles = ["Velocity pixel R2", "q R2", "Table g R2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [float(x[metric]) for x in rows]
        ax.bar(labels, vals, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylim(max(-0.05, min(vals) - 0.08), 1.01)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "r2_metrics_comparison.png", dpi=180)
    plt.close(fig)

    metrics = ["q_mean_re_all", "table_g_mean_re_all"]
    titles = ["q mean relative error", "Table g mean relative error"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [100.0 * float(x[metric]) for x in rows]
        ax.bar(labels, vals, color="#f58518")
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "relative_error_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", type=str, default=r"E:\mhw\1\cup\runs\arch_compare_20260527")
    args = parser.parse_args()
    compare_dir = Path(args.compare_dir)
    rows = []
    for model_dir in sorted([p for p in compare_dir.iterdir() if p.is_dir()]):
        summary_path = model_dir / "summary.json"
        conductance_path = model_dir / "conductance_ipnm1_rho1" / "conductance_summary.json"
        if not summary_path.exists() or not conductance_path.exists():
            continue
        summary = read_json(summary_path)
        conductance = read_json(conductance_path)
        table_g = conductance["g_metrics_vs_table"]
        true_g = conductance["g_metrics_vs_true_velocity"]
        summary["all"]["table_g_mean_re"] = table_g["mean_rel_error"]
        summary["all"]["table_g_median_re"] = table_g["median_rel_error"]
        summary["all"]["table_g_r2"] = table_g["r2"]
        summary["all"]["true_g_mean_re"] = true_g["mean_rel_error"]
        summary["all"]["true_g_r2"] = true_g["r2"]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        row = {
            "model": summary["model"],
            "best_epoch": summary["best_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        for split_name, metrics in [("all", summary["all"]), ("val", summary["val"]), ("test", summary["test"])]:
            for key, value in metrics.items():
                if not key.startswith("table_g_"):
                    row[f"{key}_{split_name}"] = value
        row["table_g_mean_re_all"] = table_g["mean_rel_error"]
        row["table_g_median_re_all"] = table_g["median_rel_error"]
        row["table_g_r2_all"] = table_g["r2"]
        row["true_g_mean_re_all"] = true_g["mean_rel_error"]
        row["true_g_r2_all"] = true_g["r2"]
        rows.append(row)

    rows.sort(key=lambda x: float(x["pixel_r2_all"]), reverse=True)
    write_csv(compare_dir / "architecture_metrics_summary.csv", rows)
    (compare_dir / "architecture_metrics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_outputs(compare_dir, rows)

    lines = [
        "# Architecture Comparison",
        "",
        "Unified split and loss: `field + 0.10 * integrated_flux`.",
        "",
        "| Model | Pixel R2 all | Pixel R2 val | q mean RE all | q R2 all | Table g mean RE | Table g R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {float(r['pixel_r2_all']):.4f} | {float(r['pixel_r2_val']):.4f} | "
            f"{100.0 * float(r['q_mean_re_all']):.2f}% | {float(r['q_r2_all']):.4f} | "
            f"{100.0 * float(r['table_g_mean_re_all']):.2f}% | {float(r['table_g_r2_all']):.4f} |"
        )
    lines += [
        "",
        "Figures:",
        "",
        "- `loss_curves_comparison.png`",
        "- `r2_metrics_comparison.png`",
        "- `relative_error_comparison.png`",
    ]
    (compare_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {compare_dir}")


if __name__ == "__main__":
    main()
