#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Render publication-style figures for local throat conductance surrogate results.

Main outputs:
1. Parity scatter: g_pred vs reference conductance.
2. Relative error histogram.
3. Relative error boxplot by rock type.
4. Ranked relative error curve.

The script prefers conductance labels from permeability_<id>.dat / conductivity.dat
when g_table is available; otherwise it falls back to velocity-derived g_true_velocity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize local conductance surrogate results in publication style."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to conductance_results.csv")
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Optional conductance_summary.json. Used for figure annotations when available.",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default="",
        help="Optional split_info.json for filtering a specific split by (rock_type, global_id).",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Subset scope when split-json is provided.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="auto",
        choices=("auto", "table", "velocity"),
        help="Reference conductance source used in the plots.",
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--figure-name",
        type=str,
        default="conductance_analysis_paper",
        help="Basename without extension for the main figure.",
    )
    parser.add_argument(
        "--by-rock-name",
        type=str,
        default="conductance_by_rock_type.csv",
        help="CSV summary by rock type.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output resolution for PNG export.",
    )
    return parser.parse_args()


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
            "lines.linewidth": 1.4,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def maybe_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def maybe_int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return -1
    try:
        return int(float(value))
    except ValueError:
        return -1


def load_split_keys(path: str, scope: str) -> Optional[set[tuple[int, int]]]:
    if not path or scope == "all":
        return None
    with open(path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    out: set[tuple[int, int]] = set()
    for rock_type_str, item in split_info["by_rock_type"].items():
        rock_type = int(rock_type_str)
        key = f"{scope}_global_ids"
        for gid in item.get(key, []):
            out.add((rock_type, int(gid)))
    return out


def select_reference(rows: List[Dict[str, str]], requested: str) -> tuple[str, str, str]:
    has_table = any(np.isfinite(maybe_float(r, "g_table")) for r in rows)
    has_velocity = any(np.isfinite(maybe_float(r, "g_true_velocity")) for r in rows)

    if requested == "table":
        if not has_table:
            raise ValueError("reference=table was requested but g_table is missing.")
        return "g_table", "Table Reference", "g_rel_error_table"
    if requested == "velocity":
        if not has_velocity:
            raise ValueError("reference=velocity was requested but g_true_velocity is missing.")
        return "g_true_velocity", "Velocity-Derived Reference", "g_rel_error_velocity"
    if has_table:
        return "g_table", "Table Reference", "g_rel_error_table"
    if has_velocity:
        return "g_true_velocity", "Velocity-Derived Reference", "g_rel_error_velocity"
    raise ValueError("No valid conductance reference columns were found in the CSV.")


def compute_stats(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-12) -> Dict[str, float]:
    diff = y_pred - y_true
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(y_true), eps)
    true_mean = float(np.mean(y_true))
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((y_true - true_mean) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, eps)
    return {
        "n": int(y_true.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mre": float(np.mean(rel_err)),
        "median_re": float(np.median(rel_err)),
        "p90_re": float(np.percentile(rel_err, 90.0)),
        "r2": float(r2),
    }


def merge_summary_metrics(stats: Dict[str, float], metrics: Dict[str, object]) -> Dict[str, float]:
    merged = dict(stats)
    mapping = {
        "n": "n_samples",
        "mae": "mean_abs_error",
        "rmse": "rmse",
        "mre": "mean_rel_error",
        "median_re": "median_rel_error",
        "r2": "r2",
    }
    for target_key, source_key in mapping.items():
        if source_key in metrics:
            merged[target_key] = float(metrics[source_key])
    return merged


def write_by_rock_summary(out_path: Path, rock_types: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    unique_types = sorted(np.unique(rock_types).tolist())
    fieldnames = ["rock_type", "n_samples", "mae", "rmse", "mean_rel_error", "median_rel_error", "p90_rel_error", "r2"]
    rows_out: List[Dict[str, object]] = []
    for rock_type in unique_types:
        mask = rock_types == rock_type
        stats = compute_stats(y_true[mask], y_pred[mask])
        rows_out.append(
            {
                "rock_type": int(rock_type),
                "n_samples": int(np.sum(mask)),
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "mean_rel_error": stats["mre"],
                "median_rel_error": stats["median_re"],
                "p90_rel_error": stats["p90_re"],
                "r2": stats["r2"],
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def plot_figure(
    out_base: Path,
    rock_types: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_label: str,
    stats: Dict[str, float],
    title_suffix: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#1f4e79", "#3f7cac", "#6aaed6", "#c84c31", "#e07a5f", "#7a9e7e"]
    unique_types = sorted(np.unique(rock_types).tolist())
    color_map = {rt: colors[i % len(colors)] for i, rt in enumerate(unique_types)}

    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.maximum(np.abs(y_true), 1.0e-12)
    rel_err_pct = 100.0 * rel_err

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    ax_scatter, ax_hist, ax_box, ax_rank = axes.flatten()

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    lo = max(lo, 1.0e-12)
    hi = max(hi, lo * 10.0)

    for rock_type in unique_types:
        mask = rock_types == rock_type
        ax_scatter.scatter(
            y_true[mask],
            y_pred[mask],
            s=13,
            alpha=0.72,
            color=color_map[rock_type],
            edgecolors="none",
            label=f"Rock {int(rock_type)}",
        )
    ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color="#4a4a4a", linewidth=0.9)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_xlabel(f"{reference_label} Conductance")
    ax_scatter.set_ylabel("Predicted Conductance")
    ax_scatter.set_title("(a) Conductance Parity")
    ax_scatter.legend(frameon=False, ncol=2, fontsize=7)
    annotation = (
        f"$R^2$ = {stats['r2']:.4f}\n"
        f"MAE = {stats['mae']:.2f}\n"
        f"MedRE = {100.0 * stats['median_re']:.2f}%\n"
        f"P90 = {100.0 * stats['p90_re']:.2f}%"
    )
    ax_scatter.text(
        0.04,
        0.96,
        annotation,
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#777777", linewidth=0.6, alpha=0.92),
    )

    bins = np.linspace(0.0, max(5.0, float(np.percentile(rel_err_pct, 99.0)) * 1.05), 36)
    ax_hist.hist(rel_err_pct, bins=bins, color="#c84c31", alpha=0.88, edgecolor="white", linewidth=0.4)
    ax_hist.axvline(100.0 * stats["median_re"], color="#1f4e79", linestyle="--", linewidth=1.0, label="Median")
    ax_hist.axvline(100.0 * stats["p90_re"], color="#4c956c", linestyle="--", linewidth=1.0, label="P90")
    ax_hist.set_xlabel("Relative Error [%]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("(b) Relative Error Distribution")
    ax_hist.legend(frameon=False)

    box_data = [rel_err_pct[rock_types == rt] for rt in unique_types]
    bp = ax_box.boxplot(
        box_data,
        tick_labels=[f"R{int(rt)}" for rt in unique_types],
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
    ax_box.set_xlabel("Rock Type")
    ax_box.set_ylabel("Relative Error [%]")
    ax_box.set_title("(c) Relative Error by Rock Type")

    order = np.argsort(rel_err_pct)
    ranked = rel_err_pct[order]
    x = np.arange(1, ranked.size + 1, dtype=np.int32)
    ax_rank.plot(x, ranked, color="#1f4e79")
    ax_rank.axhline(100.0 * stats["median_re"], color="#c84c31", linestyle="--", linewidth=0.9, label="Median")
    ax_rank.axhline(100.0 * stats["p90_re"], color="#4c956c", linestyle="--", linewidth=0.9, label="P90")
    ax_rank.set_xlabel("Sample Rank")
    ax_rank.set_ylabel("Relative Error [%]")
    ax_rank.set_title("(d) Ranked Error Curve")
    ax_rank.legend(frameon=False)

    for ax in axes.flatten():
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Local Conductance Surrogate Performance ({title_suffix})", y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv)
    split_keys = load_split_keys(args.split_json, args.scope)
    if split_keys is not None:
        rows = [
            row
            for row in rows
            if (maybe_int(row, "rock_type"), maybe_int(row, "global_id")) in split_keys
        ]
        if not rows:
            raise ValueError(f"No rows remained after applying scope={args.scope}.")

    ref_key, ref_label, rel_key = select_reference(rows, args.reference)

    rock_types: List[int] = []
    y_true: List[float] = []
    y_pred: List[float] = []
    kept_rows = 0
    for row in rows:
        ref_val = maybe_float(row, ref_key)
        pred_val = maybe_float(row, "g_pred")
        if not (np.isfinite(ref_val) and np.isfinite(pred_val) and ref_val > 0.0 and pred_val > 0.0):
            continue
        rock_types.append(maybe_int(row, "rock_type"))
        y_true.append(ref_val)
        y_pred.append(pred_val)
        kept_rows += 1

    if kept_rows <= 0:
        raise ValueError("No finite conductance pairs were found for visualization.")

    rock_arr = np.asarray(rock_types, dtype=np.int16)
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    stats = compute_stats(true_arr, pred_arr)

    if args.summary and os.path.exists(args.summary):
        with open(args.summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary_key = "g_metrics_vs_table" if ref_key == "g_table" else "g_metrics_vs_true_velocity"
        if summary.get(summary_key):
            stats = merge_summary_metrics(stats, summary[summary_key])

    title_suffix = args.scope.upper() if args.scope != "all" else f"n = {kept_rows}"
    out_base = out_dir / args.figure_name
    plot_figure(
        out_base=out_base,
        rock_types=rock_arr,
        y_true=true_arr,
        y_pred=pred_arr,
        reference_label=ref_label,
        stats=stats,
        title_suffix=title_suffix,
        dpi=int(args.dpi),
    )
    write_by_rock_summary(out_dir / args.by_rock_name, rock_arr, true_arr, pred_arr)
    print(f"Saved figure: {out_base}.png/.pdf")
    print(f"Saved by-rock summary: {out_dir / args.by_rock_name}")


if __name__ == "__main__":
    main()
