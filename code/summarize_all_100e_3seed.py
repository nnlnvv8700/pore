#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize all 100-epoch 3-seed architecture runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODEL_LABEL = {
    "res_ed": "Res-IPNM1-FlowNet",
    "convnext_ed": "ConvNeXt-ED",
    "fno": "FNO",
    "multitask_cnn": "Multi-task CNN",
    "ipnm1_flownet": "IPNM1-FlowNet",
    "unet": "U-Net",
    "deeponet_lite": "DeepONet-lite",
}

METRICS = [
    "best_epoch",
    "pixel_r2_all",
    "pixel_r2_val",
    "pixel_r2_test",
    "q_mean_re_all",
    "q_r2_all",
    "table_g_mean_re_all",
    "table_g_r2_all",
    "true_g_mean_re_all",
    "true_g_r2_all",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run(seed: int, model: str, run_dir: Path) -> dict:
    summary = read_json(run_dir / "summary.json")
    conductance = read_json(run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json")
    return {
        "seed": seed,
        "model": model,
        "label": MODEL_LABEL.get(model, model),
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
        "pixel_r2_all": summary["all"]["pixel_r2"],
        "pixel_r2_val": summary["val"]["pixel_r2"],
        "pixel_r2_test": summary["test"]["pixel_r2"],
        "q_mean_re_all": summary["all"]["q_mean_re"],
        "q_r2_all": summary["all"]["q_r2"],
        "table_g_mean_re_all": conductance["g_metrics_vs_table"]["mean_rel_error"],
        "table_g_r2_all": conductance["g_metrics_vs_table"]["r2"],
        "true_g_mean_re_all": conductance["g_metrics_vs_true_velocity"]["mean_rel_error"],
        "true_g_r2_all": conductance["g_metrics_vs_true_velocity"]["r2"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for model in sorted(set(r["model"] for r in rows)):
        sub = [r for r in rows if r["model"] == model]
        row = {"model": model, "label": MODEL_LABEL.get(model, model), "n_seeds": len(sub)}
        for metric in METRICS:
            arr = np.asarray([float(r[metric]) for r in sub], dtype=np.float64)
            row[f"{metric}_mean"] = float(arr.mean())
            row[f"{metric}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(row)
    # Overall ranking: prefer velocity and q/g R2, penalize q/g errors.
    for row in out:
        row["score"] = (
            row["pixel_r2_all_mean"]
            + row["q_r2_all_mean"]
            + row["table_g_r2_all_mean"]
            - row["q_mean_re_all_mean"]
            - row["table_g_mean_re_all_mean"]
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def plot(root: Path, summary: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("pixel_r2_all", "Velocity pixel R2"),
        ("q_r2_all", "q R2"),
        ("table_g_r2_all", "Table g R2"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        vals = [r[f"{metric}_mean"] for r in summary]
        errs = [r[f"{metric}_std"] for r in summary]
        ax.bar(labels, vals, yerr=errs, capsize=4, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylim(max(-0.05, min(vals) - 0.08), 1.01)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(root / "all_models_100e_3seed_r2_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    specs = [
        ("q_mean_re_all", "q mean relative error"),
        ("table_g_mean_re_all", "Table g mean relative error"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        vals = [100.0 * r[f"{metric}_mean"] for r in summary]
        errs = [100.0 * r[f"{metric}_std"] for r in summary]
        ax.bar(labels, vals, yerr=errs, capsize=4, color="#f58518")
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(root / "all_models_100e_3seed_error_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=str, default=r"E:\mhw\1\cup\runs\res_ipnm1_flownet_100e_3seed_20260527")
    parser.add_argument("--compare-root", type=str, default=r"E:\mhw\1\cup\runs\arch_compare_100e_3seed_20260527")
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\final_100e_3seed_comparison_20260527")
    args = parser.parse_args()

    main_root = Path(args.main_root)
    compare_root = Path(args.compare_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed_dir in sorted(main_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        rows.append(collect_run(seed, "res_ed", seed_dir / "res_ed"))
    for seed_dir in sorted(compare_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        for run_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir()]):
            if (run_dir / "summary.json").exists() and (run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json").exists():
                rows.append(collect_run(seed, run_dir.name, run_dir))

    write_csv(out_dir / "all_models_100e_3seed_per_seed.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "all_models_100e_3seed_summary.csv", summary)
    (out_dir / "all_models_100e_3seed_summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    plot(out_dir, summary)

    lines = [
        "# Final 100-Epoch 3-Seed Architecture Comparison",
        "",
        "| Rank | Model | Velocity R2 | q mean RE | q R2 | Table g mean RE | Table g R2 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(summary, start=1):
        lines.append(
            f"| {i} | {r['label']} | "
            f"{r['pixel_r2_all_mean']:.4f} +/- {r['pixel_r2_all_std']:.4f} | "
            f"{100*r['q_mean_re_all_mean']:.2f}% +/- {100*r['q_mean_re_all_std']:.2f}% | "
            f"{r['q_r2_all_mean']:.4f} +/- {r['q_r2_all_std']:.4f} | "
            f"{100*r['table_g_mean_re_all_mean']:.2f}% +/- {100*r['table_g_mean_re_all_std']:.2f}% | "
            f"{r['table_g_r2_all_mean']:.4f} +/- {r['table_g_r2_all_std']:.4f} |"
        )
    lines += [
        "",
        "Figures:",
        "",
        "- `all_models_100e_3seed_r2_summary.png`",
        "- `all_models_100e_3seed_error_summary.png`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
