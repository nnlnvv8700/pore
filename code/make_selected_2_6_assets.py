#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create selected rank 2-6 tables and figures for manuscript plotting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEEP_MODELS = ["res_ed", "ipnm1_flownet", "multitask_cnn", "convnext_ed", "unet"]
LABEL = {
    "res_ed": "Res-IPNM1-FlowNet",
    "ipnm1_flownet": "IPNM1-FlowNet",
    "multitask_cnn": "Multi-task CNN",
    "convnext_ed": "ConvNeXt-ED",
    "unet": "U-Net",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_selected(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: KEEP_MODELS.index(r["model"]))
    labels = [LABEL[r["model"]] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [
        ("pixel_r2_all", "Velocity pixel R2"),
        ("q_r2_all", "q R2"),
        ("table_g_r2_all", "Table g R2"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        vals = [float(r[f"{metric}_mean"]) for r in rows]
        errs = [float(r[f"{metric}_std"]) for r in rows]
        ax.bar(labels, vals, yerr=errs, capsize=4, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylim(max(0.9, min(vals) - 0.03), 1.001)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_rank2_6_r2_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    specs = [
        ("q_mean_re_all", "q mean relative error"),
        ("table_g_mean_re_all", "Table g mean relative error"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        vals = [100.0 * float(r[f"{metric}_mean"]) for r in rows]
        errs = [100.0 * float(r[f"{metric}_std"]) for r in rows]
        ax.bar(labels, vals, yerr=errs, capsize=4, color="#f58518")
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_rank2_6_error_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"E:\mhw\1\cup\runs\final_100e_3seed_comparison_20260527")
    args = parser.parse_args()
    root = Path(args.root)
    summary_rows = read_csv(root / "all_models_100e_3seed_summary.csv")
    per_seed_rows = read_csv(root / "all_models_100e_3seed_per_seed.csv")
    selected_summary = [r for r in summary_rows if r["model"] in KEEP_MODELS]
    selected_per_seed = [r for r in per_seed_rows if r["model"] in KEEP_MODELS]
    selected_summary.sort(key=lambda r: KEEP_MODELS.index(r["model"]))
    selected_per_seed.sort(key=lambda r: (KEEP_MODELS.index(r["model"]), int(r["seed"])))

    write_csv(root / "selected_rank2_6_summary.csv", selected_summary)
    write_csv(root / "selected_rank2_6_per_seed.csv", selected_per_seed)
    (root / "selected_rank2_6_summary.json").write_text(
        json.dumps({"models": KEEP_MODELS, "summary": selected_summary, "per_seed": selected_per_seed}, indent=2),
        encoding="utf-8",
    )
    plot_selected(root, selected_summary)

    lines = [
        "# Selected Rank 2-6 Models",
        "",
        "| Model | Velocity R2 | q mean RE | q R2 | Table g mean RE | Table g R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in selected_summary:
        lines.append(
            f"| {LABEL[r['model']]} | "
            f"{float(r['pixel_r2_all_mean']):.4f} +/- {float(r['pixel_r2_all_std']):.4f} | "
            f"{100*float(r['q_mean_re_all_mean']):.2f}% +/- {100*float(r['q_mean_re_all_std']):.2f}% | "
            f"{float(r['q_r2_all_mean']):.4f} +/- {float(r['q_r2_all_std']):.4f} | "
            f"{100*float(r['table_g_mean_re_all_mean']):.2f}% +/- {100*float(r['table_g_mean_re_all_std']):.2f}% | "
            f"{float(r['table_g_r2_all_mean']):.4f} +/- {float(r['table_g_r2_all_std']):.4f} |"
        )
    lines += [
        "",
        "Figures:",
        "",
        "- `selected_rank2_6_r2_summary.png`",
        "- `selected_rank2_6_error_summary.png`",
    ]
    (root / "SELECTED_RANK2_6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "SELECTED_RANK2_6.md")


if __name__ == "__main__":
    main()
