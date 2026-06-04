#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对三个固定模型做同一测试集上的 bootstrap 重复评估，并生成论文图表。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from compare_prediction_reports import compute_r2, load_split_indices, subset_by_dataset_index
from visual_compare_velocity_fields import configure_publication_style


METRIC_SPECS = {
    "Flux_R2": {"direction": "higher", "display": r"Flux $R^2$", "scale": 1.0},
    "RelFluxErr_mean": {"direction": "lower", "display": "Mean RelFluxErr (%)", "scale": 100.0},
    "RelFluxErr_p90": {"direction": "lower", "display": "P90 RelFluxErr (%)", "scale": 100.0},
    "MAE_pore_mean": {"direction": "lower", "display": "Mean MAE", "scale": 1.0},
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
    parser = argparse.ArgumentParser(description="聚合三个模型的 bootstrap 测试结果并生成论文图。")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline1", type=str, required=True)
    parser.add_argument("--baseline2", type=str, required=True)
    parser.add_argument("--ours-label", type=str, default="Ours")
    parser.add_argument("--baseline1-label", type=str, default="PoreFlow-Net")
    parser.add_argument("--baseline2-label", type=str, default="FNO")
    parser.add_argument("--bootstrap-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def load_prediction(path: str) -> Dict[str, np.ndarray]:
    p = Path(path)
    if p.is_dir():
        candidates = [p / "predictions.npz", p / "infer" / "predictions.npz"]
        for cand in candidates:
            if cand.is_file():
                p = cand
                break
    return {k: v for k, v in np.load(p).items()}


def summarize(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    rel_err = pred["rel_err"].astype(np.float64)
    abs_err = pred["abs_err"].astype(np.float64)
    q_true = pred["q_true"].astype(np.float64)
    q_pred = pred["q_pred"].astype(np.float64)
    mae = pred["mae_pore"].astype(np.float64)
    return {
        "N": int(rel_err.shape[0]),
        "Flux_R2": compute_r2(q_pred, q_true),
        "RelFluxErr_mean": float(np.mean(rel_err)),
        "RelFluxErr_p90": float(np.percentile(rel_err, 90)),
        "MAE_pore_mean": float(np.mean(mae)),
        "Acc@5%": float(np.mean(rel_err <= 0.05)),
        "Acc@8%": float(np.mean(rel_err <= 0.08)),
        "Tail@8%": float(np.mean(rel_err > 0.08)),
    }


def subset_by_positions(pred: Dict[str, np.ndarray], positions: np.ndarray) -> Dict[str, np.ndarray]:
    pos = np.asarray(positions, dtype=np.int64)
    out: Dict[str, np.ndarray] = {}
    for key, value in pred.items():
        if isinstance(value, np.ndarray) and value.shape[0] == pred["index"].shape[0]:
            out[key] = value[pos]
        else:
            out[key] = value
    return out


def aggregate_runs(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metrics = [k for k in rows[0].keys() if k != "N"]
    out: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    out["N"] = {"mean": float(rows[0]["N"]), "std": 0.0, "min": float(rows[0]["N"]), "max": float(rows[0]["N"])}
    return out


def align_common_test(
    split_test: np.ndarray,
    preds: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    scoped: Dict[str, Dict[str, np.ndarray]] = {}
    common = None
    for key, pred in preds.items():
        scope_pred = subset_by_dataset_index(pred, split_test)
        scoped[key] = scope_pred
        idx = scope_pred["index"].astype(np.int64)
        common = idx if common is None else np.intersect1d(common, idx)

    assert common is not None
    aligned: Dict[str, Dict[str, np.ndarray]] = {}
    for key, pred in scoped.items():
        order = {int(v): i for i, v in enumerate(pred["index"].astype(np.int64).tolist())}
        positions = np.asarray([order[int(i)] for i in common.tolist()], dtype=np.int64)
        aligned[key] = subset_by_positions(pred, positions)
    return aligned


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_figure(summary: Dict[str, Dict[str, Dict[str, float]]], labels: Dict[str, str], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configure_publication_style()
    metric_keys = ["Flux_R2", "RelFluxErr_mean", "RelFluxErr_p90", "MAE_pore_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0))
    axes = np.asarray(axes).reshape(-1)
    model_order = ["ours", "pfn", "fno"]
    x = np.asarray([0.00, 0.52, 1.04], dtype=np.float64)

    for ax, metric in zip(axes, metric_keys):
        vals = np.asarray([summary[m][metric]["mean"] * METRIC_SPECS[metric]["scale"] for m in model_order], dtype=np.float64)
        errs = np.asarray([summary[m][metric]["std"] * METRIC_SPECS[metric]["scale"] for m in model_order], dtype=np.float64)
        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            width=0.22,
            color=[MODEL_LIGHT[m] for m in model_order],
            edgecolor=[MODEL_COLORS[m] for m in model_order],
            linewidth=0.8,
            error_kw={"elinewidth": 0.75, "capsize": 2.5, "capthick": 0.75, "ecolor": "#333333"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in model_order], rotation=0)
        ax.set_title(METRIC_SPECS[metric]["display"])
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
        y_min = min(0.0, float(np.min(vals - errs)) * 0.95)
        y_max = float(np.max(vals + errs)) * 1.18 if np.max(vals + errs) > 0 else 1.0
        if metric == "Flux_R2":
            y_min = min(y_min, 0.90)
            y_max = max(y_max, 1.0)
        ax.set_ylim(y_min, y_max)
        if METRIC_SPECS[metric]["direction"] == "higher":
            best_idx = int(np.argmax(vals))
        else:
            best_idx = int(np.argmin(vals))
        best_bar = bars[best_idx]
        best_y = vals[best_idx] + errs[best_idx] + max(y_max - y_min, 1.0e-6) * 0.035
        ax.scatter(
            [best_bar.get_x() + best_bar.get_width() / 2.0],
            [best_y],
            marker="*",
            s=95,
            color="#d4a72c",
            edgecolor="#8a6b13",
            linewidth=0.6,
            zorder=4,
        )
        ax.text(
            best_bar.get_x() + best_bar.get_width() / 2.0,
            best_y + max(y_max - y_min, 1.0e-6) * 0.03,
            "best",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#6a5310",
        )

    for i in range(len(metric_keys), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Three-Model Comparison on Test Set", y=0.995, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def save_advantage_figure(summary: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configure_publication_style()
    metrics = ["Flux_R2", "RelFluxErr_mean", "RelFluxErr_p90", "Acc@5%", "Acc@8%", "Tail@8%"]
    ours_vs_pfn = []
    ours_vs_fno = []
    for metric in metrics:
        ours = summary["ours"][metric]["mean"]
        pfn = summary["pfn"][metric]["mean"]
        fno = summary["fno"][metric]["mean"]
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
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
        ax.set_xlabel("Improvement (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = {
        "ours": load_prediction(args.ours),
        "pfn": load_prediction(args.baseline1),
        "fno": load_prediction(args.baseline2),
    }
    labels = {
        "ours": args.ours_label,
        "pfn": args.baseline1_label,
        "fno": args.baseline2_label,
    }

    with h5py.File(args.h5, "r") as f:
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
    split = load_split_indices(args.split_json, rock_type, global_id)
    aligned = align_common_test(split["test"], preds)

    rng = np.random.default_rng(args.bootstrap_seed)
    n = int(aligned["ours"]["index"].shape[0])
    per_model_runs: Dict[str, List[Dict[str, float]]] = {"ours": [], "pfn": [], "fno": []}
    raw_rows: List[Dict[str, object]] = []

    for run_id in range(1, args.bootstrap_repeats + 1):
        positions = rng.integers(0, n, size=n, endpoint=False)
        for model_key in ["ours", "pfn", "fno"]:
            sampled = subset_by_positions(aligned[model_key], positions)
            metrics = summarize(sampled)
            per_model_runs[model_key].append(metrics)
            row = {"run_id": run_id, "model": model_key, "model_label": labels[model_key]}
            row.update(metrics)
            raw_rows.append(row)

    summary = {model_key: aggregate_runs(rows) for model_key, rows in per_model_runs.items()}

    summary_rows: List[Dict[str, object]] = []
    for model_key in ["ours", "pfn", "fno"]:
        for metric in METRIC_SPECS.keys():
            summary_rows.append(
                {
                    "model": model_key,
                    "model_label": labels[model_key],
                    "metric": metric,
                    "metric_display": METRIC_SPECS[metric]["display"],
                    "mean": summary[model_key][metric]["mean"],
                    "std": summary[model_key][metric]["std"],
                    "min": summary[model_key][metric]["min"],
                    "max": summary[model_key][metric]["max"],
                    "n_runs": args.bootstrap_repeats,
                }
            )

    with open(out_dir / "three_model_bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bootstrap_repeats": args.bootstrap_repeats,
                "labels": labels,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_csv(raw_rows, out_dir / "three_model_bootstrap_raw_runs.csv")
    save_csv(summary_rows, out_dir / "three_model_bootstrap_summary.csv")
    save_metrics_figure(summary, labels, out_dir / "three_model_metrics_test_paper.png")
    save_advantage_figure(summary, out_dir / "three_model_advantage_test_paper.png")

    print(f"saved three-model bootstrap outputs to: {out_dir}")


if __name__ == "__main__":
    main()
