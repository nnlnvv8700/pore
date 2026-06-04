#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare two predictions.npz files and quantify model advantage.

Outputs:
- comparison_summary.json
- comparison_table.csv
- advantage_table.csv
- advantage_summary.png / advantage_summary.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np


METRIC_DIRECTIONS = {
    "RelFluxErr_mean": "lower",
    "RelFluxErr_median": "lower",
    "RelFluxErr_p90": "lower",
    "AbsFluxErr_mean": "lower",
    "RMSE_pore_mean": "lower",
    "MAE_pore_mean": "lower",
    "Flux_R2": "higher",
}

METRIC_DISPLAY_NAMES = {
    "RelFluxErr_mean": "Mean RelFluxErr",
    "RelFluxErr_median": "Median RelFluxErr",
    "RelFluxErr_p90": "P90 RelFluxErr",
    "AbsFluxErr_mean": "Mean AbsFluxErr",
    "RMSE_pore_mean": "Mean RMSE",
    "MAE_pore_mean": "Mean MAE",
    "Flux_R2": "Flux $R^2$",
}


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def load_split_indices(split_json_path: str, rock_type: np.ndarray, global_id: np.ndarray) -> Dict[str, np.ndarray]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    index_by_key = {}
    for idx, (rt, gid) in enumerate(zip(rock_type.tolist(), global_id.tolist())):
        index_by_key[(int(rt), int(gid))] = idx

    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for rt_str, rt_info in split_info["by_rock_type"].items():
        rt = int(rt_str)
        for split_name in out.keys():
            gids = rt_info[f"{split_name}_global_ids"]
            for gid in gids:
                out[split_name].append(index_by_key[(rt, int(gid))])

    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def subset_by_dataset_index(pred: Dict[str, np.ndarray], keep_indices: Iterable[int]) -> Dict[str, np.ndarray]:
    keep = np.asarray(sorted(set(int(x) for x in keep_indices)), dtype=np.int64)
    mask = np.isin(pred["index"].astype(np.int64), keep)
    return {k: v[mask] for k, v in pred.items() if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]}


def summarize(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    q_true = pred["q_true"]
    q_pred = pred["q_pred"]
    rel_err = pred["rel_err"]
    abs_err = pred["abs_err"]
    rmse = pred["rmse_pore"]
    mae = pred["mae_pore"]
    return {
        "N": int(q_true.shape[0]),
        "RelFluxErr_mean": float(np.mean(rel_err)),
        "RelFluxErr_median": float(np.median(rel_err)),
        "RelFluxErr_p90": float(np.percentile(rel_err, 90)),
        "AbsFluxErr_mean": float(np.mean(abs_err)),
        "RMSE_pore_mean": float(np.mean(rmse)),
        "MAE_pore_mean": float(np.mean(mae)),
        "Flux_R2": compute_r2(q_pred, q_true),
    }


def compute_advantage_rows(scope_name: str, ours_sum: Dict[str, float], baseline_sum: Dict[str, float]) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        ours_val = float(ours_sum[metric])
        baseline_val = float(baseline_sum[metric])
        if direction == "lower":
            advantage_abs = baseline_val - ours_val
        else:
            advantage_abs = ours_val - baseline_val

        denom = abs(baseline_val) if abs(baseline_val) > 1.0e-12 else np.nan
        improvement_pct = float(100.0 * advantage_abs / denom) if np.isfinite(denom) else float("nan")

        if advantage_abs > 1.0e-12:
            winner = "ours"
        elif advantage_abs < -1.0e-12:
            winner = "baseline"
        else:
            winner = "tie"

        rows.append(
            {
                "scope": scope_name,
                "metric": metric,
                "metric_display": METRIC_DISPLAY_NAMES.get(metric, metric),
                "direction": direction,
                "ours": ours_val,
                "baseline": baseline_val,
                "ours_advantage_abs": advantage_abs,
                "ours_advantage_pct": improvement_pct,
                "winner": winner,
            }
        )
    return rows


def save_advantage_figure(rows: List[Dict[str, float | str]], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping advantage plot.")
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

    scope_order = ["all", "test"]
    metrics = list(METRIC_DIRECTIONS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, scope_name in zip(axes, scope_order):
        scope_rows = [r for r in rows if r["scope"] == scope_name]
        y_pos = np.arange(len(metrics))
        vals = [float(next(r["ours_advantage_pct"] for r in scope_rows if r["metric"] == m)) for m in metrics]
        colors = ["#1f4e79" if v >= 0 else "#c84c31" for v in vals]
        bars = ax.barh(y_pos, vals, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.axvline(0.0, color="#444444", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([METRIC_DISPLAY_NAMES[m] for m in metrics])
        ax.invert_yaxis()
        ax.set_title(f"{scope_name.capitalize()} Split")
        ax.set_xlabel("Ours Advantage over Baseline (%)")
        ax.grid(True, axis="x", alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, vals):
            x = bar.get_width()
            ha = "left" if x >= 0 else "right"
            pad = 1.1 if x >= 0 else -1.1
            ax.text(x + pad, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", ha=ha, fontsize=7.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    base, _ = os.path.splitext(out_path)
    plt.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved advantage plot: {out_path}")


def save_advantage_latex(rows: List[Dict[str, float | str]], out_path: str) -> None:
    lines = []
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\hline")
    lines.append("Scope & Metric & Ours & Baseline & Advantage & Improvement (\\%)\\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(
            f"{row['scope']} & {row['metric_display']} & "
            f"{float(row['ours']):.6f} & {float(row['baseline']):.6f} & "
            f"{float(row['ours_advantage_abs']):.6f} & {float(row['ours_advantage_pct']):.2f}\\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two predictions.npz files and quantify advantage.")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
    split = load_split_indices(args.split_json, rock_type, global_id)

    ours = {k: v for k, v in np.load(args.ours).items()}
    baseline = {k: v for k, v in np.load(args.baseline).items()}

    rows = []
    summary = {}
    advantage_rows: List[Dict[str, float | str]] = []
    for scope_name, keep in {"all": None, "test": split["test"]}.items():
        ours_scope = ours if keep is None else subset_by_dataset_index(ours, keep)
        baseline_scope = baseline if keep is None else subset_by_dataset_index(baseline, keep)
        ours_sum = summarize(ours_scope)
        baseline_sum = summarize(baseline_scope)
        delta = {k: float(baseline_sum[k] - ours_sum[k]) for k in ours_sum.keys() if k != "N"}
        summary[scope_name] = {
            "ours": ours_sum,
            "baseline": baseline_sum,
            "baseline_minus_ours": delta,
        }
        for model_name, metrics in [("ours", ours_sum), ("baseline", baseline_sum)]:
            row = {"scope": scope_name, "model": model_name}
            row.update(metrics)
            rows.append(row)
        advantage_rows.extend(compute_advantage_rows(scope_name, ours_sum, baseline_sum))

    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "advantage_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(advantage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(advantage_rows)

    with open(out_dir / "advantage_summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": advantage_rows}, f, indent=2, ensure_ascii=False)

    save_advantage_latex(advantage_rows, str(out_dir / "advantage_table.tex"))
    save_advantage_figure(advantage_rows, str(out_dir / "advantage_summary.png"))

    print(f"saved comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()
