#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate three-model local conductance surrogate performance on the common test set.

Outputs:
- three_model_conductance_summary.json
- three_model_conductance_table.csv
- three_model_conductance_metrics.png / .pdf
- three_model_conductance_advantage.png / .pdf
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from compare_conductance_reports import compute_r2, load_split_indices


METRIC_SPECS = {
    "Conductance_R2": {"direction": "higher", "display": r"Conductance $R^2$", "scale": 1.0},
    "RelConductanceErr_mean": {"direction": "lower", "display": "Mean RelErr (%)", "scale": 100.0},
    "RelConductanceErr_p90": {"direction": "lower", "display": "P90 RelErr (%)", "scale": 100.0},
    "AbsConductanceErr_mean": {"direction": "lower", "display": "Mean AbsErr", "scale": 1.0},
    "Acc@5%": {"direction": "higher", "display": r"Accuracy ($\delta \leq 5\%$)", "scale": 100.0},
    "Acc@8%": {"direction": "higher", "display": r"Accuracy ($\delta \leq 8\%$)", "scale": 100.0},
    "Tail@8%": {"direction": "lower", "display": r"High-Error Tail ($\delta > 8\%$)", "scale": 100.0},
}

MODEL_COLORS = {
    "ours": "#1e4f78",
    "pfn": "#b96a3a",
    "fno": "#3f7f6c",
}

MODEL_LIGHT = {
    "ours": "#8eb1cc",
    "pfn": "#e0b08d",
    "fno": "#9cc6bb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate three-model conductance comparison on the test set.")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline1", type=str, required=True)
    parser.add_argument("--baseline2", type=str, required=True)
    parser.add_argument("--ours-label", type=str, default="Ours")
    parser.add_argument("--baseline1-label", type=str, default="PoreFlow-Net")
    parser.add_argument("--baseline2-label", type=str, default="FNO")
    parser.add_argument(
        "--reference",
        type=str,
        default="table",
        choices=("table", "velocity"),
    )
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def row_key(row: Dict[str, str]) -> Tuple[int, int, int]:
    return (
        maybe_int(row, "dataset_index"),
        maybe_int(row, "rock_type"),
        maybe_int(row, "global_id"),
    )


def scope_keys(indices: np.ndarray, rock_type: np.ndarray, global_id: np.ndarray) -> set[Tuple[int, int]]:
    return {(int(rock_type[int(i)]), int(global_id[int(i)])) for i in indices.tolist()}


def align_common(
    paths: Dict[str, str],
    reference_key: str,
    allowed_scope: set[Tuple[int, int]],
) -> Dict[str, List[Dict[str, str]]]:
    model_maps: Dict[str, Dict[Tuple[int, int, int], Dict[str, str]]] = {}
    common_keys = None
    for name, path in paths.items():
        rows = read_csv_rows(path)
        row_map = {}
        for row in rows:
            split_key = (maybe_int(row, "rock_type"), maybe_int(row, "global_id"))
            if split_key not in allowed_scope:
                continue
            if not (np.isfinite(maybe_float(row, "g_pred")) and np.isfinite(maybe_float(row, reference_key))):
                continue
            row_map[row_key(row)] = row
        model_maps[name] = row_map
        keys = set(row_map.keys())
        common_keys = keys if common_keys is None else (common_keys & keys)

    assert common_keys is not None
    aligned: Dict[str, List[Dict[str, str]]] = {}
    sorted_keys = sorted(common_keys)
    for name, row_map in model_maps.items():
        aligned[name] = [row_map[key] for key in sorted_keys]
    return aligned


def summarize(rows: List[Dict[str, str]], reference_key: str) -> Dict[str, float]:
    y_true = np.asarray([maybe_float(r, reference_key) for r in rows], dtype=np.float64)
    y_pred = np.asarray([maybe_float(r, "g_pred") for r in rows], dtype=np.float64)
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.maximum(np.abs(y_true), 1.0e-12)
    return {
        "N": int(y_true.size),
        "Conductance_R2": compute_r2(y_pred, y_true),
        "RelConductanceErr_mean": float(np.mean(rel_err)),
        "RelConductanceErr_p90": float(np.percentile(rel_err, 90.0)),
        "AbsConductanceErr_mean": float(np.mean(abs_err)),
        "Acc@5%": float(np.mean(rel_err <= 0.05)),
        "Acc@8%": float(np.mean(rel_err <= 0.08)),
        "Tail@8%": float(np.mean(rel_err > 0.08)),
    }


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_publication_style() -> None:
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
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_metrics_figure(summary: Dict[str, Dict[str, float]], labels: Dict[str, str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configure_publication_style()
    metric_keys = ["Conductance_R2", "RelConductanceErr_mean", "RelConductanceErr_p90", "AbsConductanceErr_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0))
    axes = np.asarray(axes).reshape(-1)
    model_order = ["ours", "pfn", "fno"]
    x = np.asarray([0.00, 0.52, 1.04], dtype=np.float64)

    for ax, metric in zip(axes, metric_keys):
        vals = np.asarray([summary[m][metric] * METRIC_SPECS[metric]["scale"] for m in model_order], dtype=np.float64)
        bars = ax.bar(
            x,
            vals,
            width=0.22,
            color=[MODEL_LIGHT[m] for m in model_order],
            edgecolor=[MODEL_COLORS[m] for m in model_order],
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in model_order], rotation=0)
        ax.set_title(METRIC_SPECS[metric]["display"])
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if METRIC_SPECS[metric]["direction"] == "higher":
            best_idx = int(np.argmax(vals))
        else:
            best_idx = int(np.argmin(vals))
        best_bar = bars[best_idx]
        ax.scatter(
            [best_bar.get_x() + best_bar.get_width() / 2.0],
            [vals[best_idx] + max(np.max(vals), 1.0) * 0.04],
            marker="*",
            s=95,
            color="#d4a72c",
            edgecolor="#8a6b13",
            linewidth=0.6,
            zorder=4,
        )

    fig.suptitle("Three-Model Conductance Comparison on Test Set", y=0.995, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_advantage_figure(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configure_publication_style()
    metrics = ["Conductance_R2", "RelConductanceErr_mean", "RelConductanceErr_p90", "Acc@5%", "Acc@8%", "Tail@8%"]
    ours_vs_pfn = []
    ours_vs_fno = []
    for metric in metrics:
        ours = summary["ours"][metric]
        pfn = summary["pfn"][metric]
        fno = summary["fno"][metric]
        if METRIC_SPECS[metric]["direction"] == "lower":
            ours_vs_pfn.append(100.0 * (pfn - ours) / max(abs(pfn), 1.0e-12))
            ours_vs_fno.append(100.0 * (fno - ours) / max(abs(fno), 1.0e-12))
        else:
            ours_vs_pfn.append(100.0 * (ours - pfn) / max(abs(pfn), 1.0e-12))
            ours_vs_fno.append(100.0 * (ours - fno) / max(abs(fno), 1.0e-12))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharey=True)
    for ax, vals, title, color in [
        (axes[0], ours_vs_pfn, "Ours vs PoreFlow-Net", MODEL_COLORS["pfn"]),
        (axes[1], ours_vs_fno, "Ours vs FNO", MODEL_COLORS["fno"]),
    ]:
        y = np.arange(len(metrics))
        ax.axvline(0.0, color="#444444", linewidth=0.8)
        for yi, val in zip(y, vals):
            x0, x1 = (0.0, val) if val >= 0 else (val, 0.0)
            ax.hlines(yi, x0, x1, color=color, linewidth=1.35, alpha=0.95)
            ax.scatter([val], [yi], s=28, color=color, edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(val + (0.8 if val >= 0 else -0.8), yi, f"{val:.1f}", ha="left" if val >= 0 else "right", va="center", fontsize=7.0)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels([METRIC_SPECS[m]["display"] for m in metrics])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Relative Improvement (%)")

    fig.suptitle("Ours Advantage in Conductance Metrics", y=0.995, fontsize=10.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
    split = load_split_indices(args.split_json, rock_type, global_id)

    allowed = scope_keys(split["test"], rock_type, global_id)
    reference_key = "g_table" if args.reference == "table" else "g_true_velocity"
    aligned = align_common(
        {
            "ours": args.ours,
            "pfn": args.baseline1,
            "fno": args.baseline2,
        },
        reference_key=reference_key,
        allowed_scope=allowed,
    )

    summary = {name: summarize(rows, reference_key) for name, rows in aligned.items()}
    labels = {"ours": args.ours_label, "pfn": args.baseline1_label, "fno": args.baseline2_label}

    rows_out = []
    for name in ["ours", "pfn", "fno"]:
        row = {"model": name, "label": labels[name], "reference": args.reference}
        row.update(summary[name])
        rows_out.append(row)

    save_csv(rows_out, out_dir / "three_model_conductance_table.csv")
    with open(out_dir / "three_model_conductance_summary.json", "w", encoding="utf-8") as f:
        json.dump({"reference": args.reference, "metrics": summary, "labels": labels}, f, indent=2, ensure_ascii=False)

    save_metrics_figure(summary, labels, out_dir / "three_model_conductance_metrics.png")
    save_advantage_figure(summary, out_dir / "three_model_conductance_advantage.png")
    print(f"saved three-model conductance comparison to: {out_dir}")


if __name__ == "__main__":
    main()
