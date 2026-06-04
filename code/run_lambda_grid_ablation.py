#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run local lambda grid ablation for IPNM1 conductance-first selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\mhw\1\cup")
PYTHON = Path(r"D:\anaconda\envs\nnlnvv\python.exe")
TRAIN = ROOT / "code" / "train_architecture_comparison.py"
POST = ROOT / "code" / "postprocess_conductance.py"
H5 = ROOT / "dataset_all_32.h5"
DATA = ROOT / "data"


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_command(cmd: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:])
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def postprocess(run_dir: Path) -> None:
    cmd = [
        str(PYTHON),
        str(POST),
        "--pred-file",
        str(run_dir / "predictions.npz"),
        "--pred-key",
        "pred_ux",
        "--true-file",
        str(H5),
        "--true-key",
        "Y",
        "--mask-file",
        str(H5),
        "--mask-key",
        "X",
        "--scale-file",
        str(H5),
        "--scale-key",
        "scale_s",
        "--index-file",
        str(run_dir / "predictions.npz"),
        "--index-key",
        "index",
        "--meta-file",
        str(H5),
        "--spatial-axis-order",
        "yz",
        "--flow-axis",
        "x",
        "--flow-mode",
        "pressure",
        "--conductance-mode",
        "ipnm2",
        "--delta-p",
        "1.0",
        "--rho",
        "1.0",
        "--ax",
        "1.0e-4",
        "--segment-length",
        "4.0",
        "--mu-lbm",
        "0.5",
        "--target-mu",
        "1.0",
        "--undo-velocity-normalization",
        "--undo-area-normalization",
        "--permeability-root",
        str(DATA),
        "--output-dir",
        str(run_dir / "conductance_ipnm1_rho1"),
    ]
    run_command(cmd, run_dir / "postprocess.log")


def train_one(out_dir: Path, lam_flux: float, lam_dist: float, seed: int, epochs: int, overwrite: bool) -> Path:
    run_name = f"flux_{tag(lam_flux)}__dist_{tag(lam_dist)}__seed_{seed}"
    case_dir = out_dir / run_name
    run_dir = case_dir / "res_ed"
    if (run_dir / "summary.json").exists() and not overwrite:
        return run_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(TRAIN),
        "--out-dir",
        str(case_dir),
        "--models",
        "res_ed",
        "--epochs",
        str(epochs),
        "--batch-size",
        "24",
        "--lambda-flux",
        str(lam_flux),
        "--lambda-dist",
        str(lam_dist),
        "--lambda-poisson",
        "0",
        "--lambda-head",
        "0.05",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--patience",
        str(epochs),
        "--overwrite",
    ]
    run_command(cmd, case_dir / "train.log")
    return run_dir


def collect_row(run_dir: Path, lam_flux: float, lam_dist: float, seed: int) -> dict:
    summary = read_json(run_dir / "summary.json")
    cond = read_json(run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json")
    return {
        "lambda_flux": lam_flux,
        "lambda_dist": lam_dist,
        "seed": seed,
        "best_epoch": summary["best_epoch"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "pixel_r2_all": summary["all"]["pixel_r2"],
        "pixel_r2_val": summary["val"]["pixel_r2"],
        "pixel_r2_test": summary["test"]["pixel_r2"],
        "q_mean_re_all_raw": summary["all"]["q_mean_re"],
        "q_mean_re_val_raw": summary["val"]["q_mean_re"],
        "q_mean_re_test_raw": summary["test"]["q_mean_re"],
        "q_r2_all_raw": summary["all"]["q_r2"],
        "q_r2_val_raw": summary["val"]["q_r2"],
        "q_r2_test_raw": summary["test"]["q_r2"],
        "q_mean_re_all_post": cond["q_metrics_vs_true_velocity"]["mean_rel_error"],
        "q_r2_all_post": cond["q_metrics_vs_true_velocity"]["r2"],
        "table_g_mean_re_all": cond["g_metrics_vs_table"]["mean_rel_error"],
        "table_g_median_re_all": cond["g_metrics_vs_table"]["median_rel_error"],
        "table_g_r2_all": cond["g_metrics_vs_table"]["r2"],
        "true_g_mean_re_all": cond["g_metrics_vs_true_velocity"]["mean_rel_error"],
        "true_g_r2_all": cond["g_metrics_vs_true_velocity"]["r2"],
        "run_dir": str(run_dir),
    }


def plot_heatmap(out_dir: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fluxes = sorted({float(r["lambda_flux"]) for r in rows})
    dists = sorted({float(r["lambda_dist"]) for r in rows})
    z = np.full((len(dists), len(fluxes)), np.nan, dtype=float)
    for row in rows:
        yi = dists.index(float(row["lambda_dist"]))
        xi = fluxes.index(float(row["lambda_flux"]))
        z[yi, xi] = 100.0 * float(row["table_g_mean_re_all"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(z, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(fluxes)), [f"{x:g}" for x in fluxes])
    ax.set_yticks(range(len(dists)), [f"{y:g}" for y in dists])
    ax.set_xlabel("lambda_flux")
    ax.set_ylabel("lambda_dist")
    ax.set_title("Table-g mean relative error (%)")
    for y in range(len(dists)):
        for x in range(len(fluxes)):
            ax.text(x, y, f"{z[y, x]:.2f}", ha="center", va="center", color="white" if z[y, x] > np.nanmean(z) else "black")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_grid_table_g_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "lambda_grid_seed42_20260529"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flux_grid = [0.05, 0.10, 0.20]
    dist_grid = [0.0, 0.005, 0.01, 0.02]
    rows: list[dict] = []
    for lam_flux in flux_grid:
        for lam_dist in dist_grid:
            print(f"Running lambda_flux={lam_flux:g}, lambda_dist={lam_dist:g}, seed={args.seed}", flush=True)
            run_dir = train_one(out_dir, lam_flux, lam_dist, args.seed, args.epochs, args.overwrite)
            if not (run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json").exists() or args.overwrite:
                postprocess(run_dir)
            rows.append(collect_row(run_dir, lam_flux, lam_dist, args.seed))
            rows_sorted = sorted(rows, key=lambda r: float(r["table_g_mean_re_all"]))
            write_csv(out_dir / "lambda_grid_summary_partial.csv", rows_sorted)
    rows = sorted(rows, key=lambda r: float(r["table_g_mean_re_all"]))
    write_csv(out_dir / "lambda_grid_summary.csv", rows)
    (out_dir / "lambda_grid_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_heatmap(out_dir, rows)
    lines = [
        "# Lambda Grid Ablation",
        "",
        "| Rank | lambda_flux | lambda_dist | table-g RE | table-g R2 | q RE | velocity R2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['lambda_flux']:.3g} | {row['lambda_dist']:.3g} | "
            f"{100.0 * row['table_g_mean_re_all']:.2f}% | {row['table_g_r2_all']:.4f} | "
            f"{100.0 * row['q_mean_re_all_post']:.2f}% | {row['pixel_r2_all']:.4f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
