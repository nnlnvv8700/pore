#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect IPNM1 manuscript-supporting experiments into one result folder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(r"E:\mhw\1\cup")
OUT = ROOT / "runs" / "ipnm1_manuscript_experiments_20260527"
FLUX_ROOT = ROOT / "runs" / "res_flux_ablation_100e_seed42_20260527"
SCALAR_ROOT = ROOT / "runs" / "scalar_q_baseline_100e_3seed_20260527"
TEACHER_ROOT = ROOT / "runs" / "teacher_ipnm1_audit_20260527"
MAIN_SUMMARY = ROOT / "runs" / "res_ipnm1_flownet_100e_3seed_20260527" / "res_ipnm1_3seed_summary.json"


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_flux_ablation() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    mapping = [("0", 0.0), ("0p01", 0.01), ("0p10", 0.10), ("1p0", 1.0)]
    for tag, lam in mapping:
        run_dir = FLUX_ROOT / f"flux_{tag}" / "res_ed"
        train = read_json(run_dir / "summary.json")
        cond = read_json(run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json")
        rows.append(
            {
                "lambda_flux": lam,
                "best_epoch": train["best_epoch"],
                "pixel_r2_all": train["all"]["pixel_r2"],
                "pixel_r2_val": train["val"]["pixel_r2"],
                "pixel_r2_test": train["test"]["pixel_r2"],
                "q_mean_re_all": cond["q_metrics_vs_true_velocity"]["mean_rel_error"],
                "q_r2_all_post": cond["q_metrics_vs_true_velocity"]["r2"],
                "table_g_mean_re_all": cond["g_metrics_vs_table"]["mean_rel_error"],
                "table_g_r2_all": cond["g_metrics_vs_table"]["r2"],
                "true_g_mean_re_all": cond["g_metrics_vs_true_velocity"]["mean_rel_error"],
                "true_g_r2_all": cond["g_metrics_vs_true_velocity"]["r2"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_all(out_dir: Path, flux_rows: List[Dict[str, object]], scalar_rows: List[Dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r["lambda_flux"]) for r in flux_rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in [
        (axes[0], "pixel_r2_all", "Velocity pixel R2"),
        (axes[1], "q_mean_re_all", "q relative error"),
        (axes[2], "table_g_mean_re_all", "Table g relative error"),
    ]:
        vals = [float(r[key]) for r in flux_rows]
        if "relative error" in title:
            vals = [100.0 * v for v in vals]
            ax.set_ylabel("%")
        ax.plot(labels, vals, marker="o", color="#4c78a8")
        ax.set_title(title)
        ax.set_xlabel("lambda_flux")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "flux_ablation_summary.png", dpi=180)
    plt.close(fig)

    targets = [r["target"] for r in scalar_rows]
    all_re = [100.0 * float(r["all_mean_re_mean"]) for r in scalar_rows]
    test_re = [100.0 * float(r["test_mean_re_mean"]) for r in scalar_rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(targets))
    w = 0.34
    ax.bar(x - w / 2, all_re, width=w, label="all")
    ax.bar(x + w / 2, test_re, width=w, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("Direct scalar fitting baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scalar_baseline_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flux_rows = collect_flux_ablation()
    write_csv(OUT / "flux_ablation_summary.csv", flux_rows)

    scalar_summary = read_csv(SCALAR_ROOT / "scalar_baseline_summary.csv")
    write_csv(OUT / "scalar_baseline_summary.csv", scalar_summary)
    scalar_per_seed = read_csv(SCALAR_ROOT / "scalar_baseline_per_seed.csv")
    write_csv(OUT / "scalar_baseline_per_seed.csv", scalar_per_seed)
    teacher_summary = read_csv(TEACHER_ROOT / "teacher_ipnm1_audit_summary.csv")
    write_csv(OUT / "teacher_ipnm1_audit_summary.csv", teacher_summary)

    main = read_json(MAIN_SUMMARY)
    (OUT / "main_model_3seed_summary.json").write_text(json.dumps(main, indent=2), encoding="utf-8")
    combined = {
        "main_model_3seed": main,
        "flux_ablation_seed42": flux_rows,
        "scalar_baseline_3seed": scalar_summary,
        "teacher_ipnm1_audit": teacher_summary,
    }
    (OUT / "ipnm1_manuscript_experiments_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    plot_all(OUT, flux_rows, scalar_summary)

    best_flux = min(flux_rows, key=lambda r: float(r["table_g_mean_re_all"]))
    main_stats = main["stats"]
    lines = [
        "# IPNM1 Manuscript Experiments",
        "",
        "Scope: IPNM1 only. FNO is not used in the manuscript-supporting experiment set.",
        "",
        "## Main Model",
        "",
        "Final model: Res-IPNM1-FlowNet.",
        "",
        "| Metric | Mean +/- std |",
        "| --- | ---: |",
        f"| Velocity pixel R2 | {main_stats['pixel_r2_all']['mean']:.4f} +/- {main_stats['pixel_r2_all']['std']:.4f} |",
        f"| q mean relative error | {100*main_stats['q_mean_re_all']['mean']:.2f}% +/- {100*main_stats['q_mean_re_all']['std']:.2f}% |",
        f"| q R2 | {main_stats['q_r2_all']['mean']:.4f} +/- {main_stats['q_r2_all']['std']:.4f} |",
        f"| Table g mean relative error | {100*main_stats['table_g_mean_re_all']['mean']:.2f}% +/- {100*main_stats['table_g_mean_re_all']['std']:.2f}% |",
        f"| Table g R2 | {main_stats['table_g_r2_all']['mean']:.4f} +/- {main_stats['table_g_r2_all']['std']:.4f} |",
        "",
        "## Flux-Loss Ablation",
        "",
        "| lambda_flux | Velocity R2 | q RE | q R2 post | Table g RE | Table g R2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in flux_rows:
        lines.append(
            f"| {float(r['lambda_flux']):.2f} | {float(r['pixel_r2_all']):.4f} | "
            f"{100*float(r['q_mean_re_all']):.2f}% | {float(r['q_r2_all_post']):.4f} | "
            f"{100*float(r['table_g_mean_re_all']):.2f}% | {float(r['table_g_r2_all']):.4f} |"
        )
    lines += [
        "",
        f"Best flux setting by table-g error: lambda_flux={float(best_flux['lambda_flux']):.2f}.",
        "",
        "## Direct Scalar Baseline",
        "",
        "| Target | All mean RE | All R2 | Test mean RE | Test R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in scalar_summary:
        lines.append(
            f"| {r['target']} | {100*float(r['all_mean_re_mean']):.2f}% +/- {100*float(r['all_mean_re_std']):.2f}% | "
            f"{float(r['all_r2_mean']):.4f} +/- {float(r['all_r2_std']):.4f} | "
            f"{100*float(r['test_mean_re_mean']):.2f}% +/- {100*float(r['test_mean_re_std']):.2f}% | "
            f"{float(r['test_r2_mean']):.4f} +/- {float(r['test_r2_std']):.4f} |"
        )
    lines += [
        "",
        "Interpretation: direct q_true fitting can estimate the integrated scalar, but it does not provide a velocity field. Direct q_table/g fitting is much weaker and less stable, supporting the velocity-field-first IPNM1 route.",
        "",
        "## Teacher 300x200x200 IPNM1 Data",
        "",
        "| Rock | Valid sections | Conductance rows | Valid sections matched | Median pore pixels | Median g |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in teacher_summary:
        lines.append(
            f"| {r['rock']} | {r['n_valid_cross_sections']} | {r['n_conductivity_rows']} | "
            f"{r['n_matched_by_local_id']} | {float(r['pore_pixels_median']):.1f} | "
            f"{float(r['conductivity_g_median']):.4g} |"
        )
    lines += [
        "",
        "Generated files:",
        "",
        "- `flux_ablation_summary.csv`",
        "- `scalar_baseline_summary.csv`",
        "- `scalar_baseline_per_seed.csv`",
        "- `teacher_ipnm1_audit_summary.csv`",
        "- `flux_ablation_summary.png`",
        "- `scalar_baseline_summary.png`",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
