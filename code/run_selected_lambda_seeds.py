#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run selected lambda candidates on multiple seeds and summarize."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_lambda_grid_ablation import ROOT, collect_row, plot_heatmap, postprocess, train_one, write_csv


def summarize_group(rows: list[dict]) -> list[dict]:
    import statistics

    groups: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((float(row["lambda_flux"]), float(row["lambda_dist"])), []).append(row)
    out = []
    metric_keys = [
        "table_g_mean_re_all",
        "table_g_r2_all",
        "q_mean_re_all_post",
        "q_r2_all_post",
        "pixel_r2_all",
        "q_mean_re_test_raw",
        "pixel_r2_test",
    ]
    for (lam_flux, lam_dist), items in groups.items():
        row = {"lambda_flux": lam_flux, "lambda_dist": lam_dist, "n_seeds": len(items)}
        for key in metric_keys:
            vals = [float(x[key]) for x in items]
            row[f"{key}_mean"] = statistics.mean(vals)
            row[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        row["seeds"] = ",".join(str(x["seed"]) for x in sorted(items, key=lambda x: int(x["seed"])))
        out.append(row)
    return sorted(out, key=lambda r: float(r["table_g_mean_re_all_mean"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "lambda_selected_3seed_20260529"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(0.20, 0.005), (0.05, 0.005), (0.10, 0.005), (0.10, 0.01)]
    seeds = [42, 43, 44]
    rows = []
    for lam_flux, lam_dist in candidates:
        for seed in seeds:
            print(f"Selected run lambda_flux={lam_flux:g}, lambda_dist={lam_dist:g}, seed={seed}", flush=True)
            run_dir = train_one(out_dir, lam_flux, lam_dist, seed, args.epochs, args.overwrite)
            if not (run_dir / "conductance_ipnm1_rho1" / "conductance_summary.json").exists() or args.overwrite:
                postprocess(run_dir)
            rows.append(collect_row(run_dir, lam_flux, lam_dist, seed))
            write_csv(out_dir / "selected_lambda_seed_rows_partial.csv", rows)
            write_csv(out_dir / "selected_lambda_group_summary_partial.csv", summarize_group(rows))
    write_csv(out_dir / "selected_lambda_seed_rows.csv", rows)
    grouped = summarize_group(rows)
    write_csv(out_dir / "selected_lambda_group_summary.csv", grouped)
    (out_dir / "selected_lambda_group_summary.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
    try:
        plot_heatmap(out_dir, rows)
    except Exception:
        pass
    lines = [
        "# Selected Lambda Three-seed Summary",
        "",
        "| Rank | lambda_flux | lambda_dist | table-g RE | table-g R2 | q RE | velocity R2 | seeds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, row in enumerate(grouped, 1):
        lines.append(
            f"| {i} | {row['lambda_flux']:.3g} | {row['lambda_dist']:.3g} | "
            f"{100.0 * row['table_g_mean_re_all_mean']:.2f}% +/- {100.0 * row['table_g_mean_re_all_std']:.2f}% | "
            f"{row['table_g_r2_all_mean']:.4f} | "
            f"{100.0 * row['q_mean_re_all_post_mean']:.2f}% +/- {100.0 * row['q_mean_re_all_post_std']:.2f}% | "
            f"{row['pixel_r2_all_mean']:.4f} | {row['seeds']} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
