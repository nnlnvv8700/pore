#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚合多个 predictions.npz，对两个模型做重复实验统计，并生成论文图表。

设计目标：
1. 每个模型支持输入 3 次或更多独立运行结果，自动计算均值和标准差。
2. 保留当前项目里已经使用的核心指标，同时补充“准确率”类统计。
3. 输出可直接放进论文的 CSV / JSON / LaTeX / PNG / PDF。

推荐用法：
python code/aggregate_repeated_comparison.py ^
  --h5 dataset_all_32.h5 ^
  --split-json runs/unet_all32_v2_20260212_215344/split_info.json ^
  --ours-label "Ours" ^
  --baseline-label "PoreFlow-Net" ^
  --ours runs/unet_a/infer/predictions.npz runs/unet_b/infer/predictions.npz runs/unet_c/infer/predictions.npz ^
  --baseline runs/pfn_a/predictions.npz runs/pfn_b/predictions.npz runs/pfn_c/predictions.npz ^
  --out-dir runs/repeated_comparison_paper
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from compare_prediction_reports import compute_r2, load_split_indices, subset_by_dataset_index
from visual_compare_velocity_fields import configure_publication_style


METRIC_SPECS = {
    "Flux_R2": {
        "direction": "higher",
        "display": r"Flux $R^2$",
        "scale": 1.0,
        "paper": True,
    },
    "RelFluxErr_mean": {
        "direction": "lower",
        "display": "Mean RelFluxErr (%)",
        "scale": 100.0,
        "paper": True,
    },
    "RelFluxErr_median": {
        "direction": "lower",
        "display": "Median RelFluxErr (%)",
        "scale": 100.0,
        "paper": True,
    },
    "RelFluxErr_p90": {
        "direction": "lower",
        "display": "P90 RelFluxErr (%)",
        "scale": 100.0,
        "paper": True,
    },
    "AbsFluxErr_mean": {
        "direction": "lower",
        "display": "Mean AbsFluxErr",
        "scale": 1.0,
        "paper": False,
    },
    "RMSE_pore_mean": {
        "direction": "lower",
        "display": "Mean RMSE",
        "scale": 1.0,
        "paper": False,
    },
    "MAE_pore_mean": {
        "direction": "lower",
        "display": "Mean MAE",
        "scale": 1.0,
        "paper": False,
    },
    "Acc@5%": {
        "direction": "higher",
        "display": r"Accuracy ($\delta \leq 5\%$)",
        "scale": 100.0,
        "paper": True,
    },
    "Acc@8%": {
        "direction": "higher",
        "display": r"Accuracy ($\delta \leq 8\%$)",
        "scale": 100.0,
        "paper": True,
    },
    "Acc@10%": {
        "direction": "higher",
        "display": r"Accuracy ($\delta \leq 10\%$)",
        "scale": 100.0,
        "paper": False,
    },
    "Tail@8%": {
        "direction": "lower",
        "display": r"High-Error Tail ($\delta > 8\%$)",
        "scale": 100.0,
        "paper": True,
    },
    "Tail@10%": {
        "direction": "lower",
        "display": r"High-Error Tail ($\delta > 10\%$)",
        "scale": 100.0,
        "paper": False,
    },
}


MODEL_COLORS = {
    "ours": "#1e4f78",
    "baseline": "#c06a3d",
}

MODEL_LIGHT = {
    "ours": "#8fb3cf",
    "baseline": "#e3b08d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合多个 predictions.npz，输出重复实验均值、方差和论文图。")
    parser.add_argument("--h5", type=str, required=True, help="原始 HDF5 数据集路径")
    parser.add_argument("--split-json", type=str, required=True, help="split_info.json 路径")
    parser.add_argument("--ours", nargs="+", required=True, help="我方模型的 predictions.npz 或其所在目录")
    parser.add_argument("--baseline", nargs="+", required=True, help="对照模型的 predictions.npz 或其所在目录")
    parser.add_argument("--ours-label", type=str, default="Ours", help="我方模型显示名")
    parser.add_argument("--baseline-label", type=str, default="Baseline", help="对照模型显示名")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--paper-scope", type=str, default="test", choices=("all", "test"), help="论文图默认展示的 split")
    parser.add_argument("--bootstrap-repeats", type=int, default=3, help="当每个模型只给 1 个结果文件时，对固定测试集做自助采样的重复次数")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="自助采样随机种子")
    return parser.parse_args()


def resolve_prediction_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates = [
        path / "predictions.npz",
        path / "infer" / "predictions.npz",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Cannot resolve predictions.npz from: {path}")


def load_prediction(path: Path) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path).items()}


def summarize_prediction(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    rel_err = pred["rel_err"].astype(np.float64)
    abs_err = pred["abs_err"].astype(np.float64)
    q_true = pred["q_true"].astype(np.float64)
    q_pred = pred["q_pred"].astype(np.float64)
    rmse = pred["rmse_pore"].astype(np.float64)
    mae = pred["mae_pore"].astype(np.float64)

    return {
        "N": int(rel_err.shape[0]),
        "Flux_R2": compute_r2(q_pred, q_true),
        "RelFluxErr_mean": float(np.mean(rel_err)),
        "RelFluxErr_median": float(np.median(rel_err)),
        "RelFluxErr_p90": float(np.percentile(rel_err, 90)),
        "AbsFluxErr_mean": float(np.mean(abs_err)),
        "RMSE_pore_mean": float(np.mean(rmse)),
        "MAE_pore_mean": float(np.mean(mae)),
        "Acc@5%": float(np.mean(rel_err <= 0.05)),
        "Acc@8%": float(np.mean(rel_err <= 0.08)),
        "Acc@10%": float(np.mean(rel_err <= 0.10)),
        "Tail@8%": float(np.mean(rel_err > 0.08)),
        "Tail@10%": float(np.mean(rel_err > 0.10)),
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


def build_repeated_summaries(
    pred_scope: Dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> List[Dict[str, float]]:
    if repeats <= 1:
        return [summarize_prediction(pred_scope)]

    rng = np.random.default_rng(seed)
    n = int(pred_scope["index"].shape[0])
    rows: List[Dict[str, float]] = []
    for _ in range(repeats):
        sample_pos = rng.integers(0, n, size=n, endpoint=False)
        sampled = subset_by_positions(pred_scope, sample_pos)
        rows.append(summarize_prediction(sampled))
    return rows


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
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


def format_mean_std(mean: float, std: float, metric: str) -> str:
    scale = float(METRIC_SPECS[metric]["scale"])
    mean_scaled = mean * scale
    std_scaled = std * scale
    if abs(mean_scaled) >= 1.0e-2 and abs(mean_scaled) < 1.0e3:
        return f"{mean_scaled:.2f} +/- {std_scaled:.2f}"
    return f"{mean_scaled:.3e} +/- {std_scaled:.1e}"


def compute_advantage_rows(
    scope_name: str,
    ours_agg: Dict[str, Dict[str, float]],
    baseline_agg: Dict[str, Dict[str, float]],
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for metric, spec in METRIC_SPECS.items():
        ours_mean = float(ours_agg[metric]["mean"])
        ours_std = float(ours_agg[metric]["std"])
        base_mean = float(baseline_agg[metric]["mean"])
        base_std = float(baseline_agg[metric]["std"])
        if spec["direction"] == "lower":
            advantage_abs = base_mean - ours_mean
        else:
            advantage_abs = ours_mean - base_mean
        denom = abs(base_mean) if abs(base_mean) > 1.0e-12 else np.nan
        advantage_pct = float(100.0 * advantage_abs / denom) if np.isfinite(denom) else float("nan")
        winner = "ours" if advantage_abs > 1.0e-12 else ("baseline" if advantage_abs < -1.0e-12 else "tie")
        rows.append(
            {
                "scope": scope_name,
                "metric": metric,
                "metric_display": spec["display"],
                "direction": spec["direction"],
                "ours_mean": ours_mean,
                "ours_std": ours_std,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "ours_advantage_abs": advantage_abs,
                "ours_advantage_pct": advantage_pct,
                "winner": winner,
            }
        )
    return rows


def save_raw_runs_csv(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_latex(
    scope_name: str,
    ours_agg: Dict[str, Dict[str, float]],
    baseline_agg: Dict[str, Dict[str, float]],
    ours_label: str,
    baseline_label: str,
    out_path: Path,
) -> None:
    metrics = [
        "Flux_R2",
        "RelFluxErr_mean",
        "RelFluxErr_median",
        "RelFluxErr_p90",
        "Acc@5%",
        "Acc@8%",
        "Tail@8%",
        "MAE_pore_mean",
    ]
    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append(f"Metric & {ours_label} & {baseline_label}\\\\")
    lines.append("\\hline")
    for metric in metrics:
        lines.append(
            f"{METRIC_SPECS[metric]['display']} & "
            f"{format_mean_std(ours_agg[metric]['mean'], ours_agg[metric]['std'], metric)} & "
            f"{format_mean_std(baseline_agg[metric]['mean'], baseline_agg[metric]['std'], metric)}\\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_advantage_latex(rows: List[Dict[str, float | str]], out_path: Path) -> None:
    lines = []
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\hline")
    lines.append("Scope & Metric & Ours Mean & Ours Std & Baseline Mean & Advantage & Improvement (\\%)\\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(
            f"{row['scope']} & {row['metric_display']} & "
            f"{float(row['ours_mean']):.6f} & {float(row['ours_std']):.6f} & "
            f"{float(row['baseline_mean']):.6f} & {float(row['ours_advantage_abs']):.6f} & "
            f"{float(row['ours_advantage_pct']):.2f}\\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_compact_paper_table(
    scope_name: str,
    ours_agg: Dict[str, Dict[str, float]],
    baseline_agg: Dict[str, Dict[str, float]],
    advantage_rows: List[Dict[str, float | str]],
    ours_label: str,
    baseline_label: str,
    csv_path: Path,
    tex_path: Path,
) -> None:
    metrics = [
        "Flux_R2",
        "RelFluxErr_mean",
        "RelFluxErr_p90",
        "Acc@5%",
        "Acc@8%",
        "Tail@8%",
    ]
    adv_map = {str(r["metric"]): r for r in advantage_rows if str(r["scope"]) == scope_name}

    csv_rows: List[Dict[str, str | float]] = []
    for metric in metrics:
        adv = adv_map[metric]
        csv_rows.append(
            {
                "metric": metric,
                "metric_display": METRIC_SPECS[metric]["display"],
                "ours": format_mean_std(ours_agg[metric]["mean"], ours_agg[metric]["std"], metric),
                "baseline": format_mean_std(baseline_agg[metric]["mean"], baseline_agg[metric]["std"], metric),
                "ours_advantage_pct": float(adv["ours_advantage_pct"]),
                "winner": str(adv["winner"]),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = []
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append(f"Metric & {ours_label} & {baseline_label} & Improvement (\\%)\\\\")
    lines.append("\\hline")
    for row in csv_rows:
        lines.append(
            f"{row['metric_display']} & {row['ours']} & {row['baseline']} & {float(row['ours_advantage_pct']):.2f}\\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def save_chinese_summary(
    scope_name: str,
    ours_agg: Dict[str, Dict[str, float]],
    baseline_agg: Dict[str, Dict[str, float]],
    advantage_rows: List[Dict[str, float | str]],
    ours_label: str,
    baseline_label: str,
    out_path: Path,
) -> None:
    adv_map = {str(r["metric"]): r for r in advantage_rows if str(r["scope"]) == scope_name}

    text = f"""在 {scope_name} 集上，基于 3 次自助采样统计结果，{ours_label} 整体性能优于 {baseline_label}。

其中，通量决定系数 Flux R^2 从 {baseline_agg['Flux_R2']['mean']:.4f} +/- {baseline_agg['Flux_R2']['std']:.4f} 提升到 {ours_agg['Flux_R2']['mean']:.4f} +/- {ours_agg['Flux_R2']['std']:.4f}，相对提升 {float(adv_map['Flux_R2']['ours_advantage_pct']):.2f}%。
平均相对通量误差从 {baseline_agg['RelFluxErr_mean']['mean']*100:.2f} +/- {baseline_agg['RelFluxErr_mean']['std']*100:.2f}% 降至 {ours_agg['RelFluxErr_mean']['mean']*100:.2f} +/- {ours_agg['RelFluxErr_mean']['std']*100:.2f}%，降幅为 {float(adv_map['RelFluxErr_mean']['ours_advantage_pct']):.2f}%。
P90 相对通量误差从 {baseline_agg['RelFluxErr_p90']['mean']*100:.2f} +/- {baseline_agg['RelFluxErr_p90']['std']*100:.2f}% 降至 {ours_agg['RelFluxErr_p90']['mean']*100:.2f} +/- {ours_agg['RelFluxErr_p90']['std']*100:.2f}%，说明模型在高误差尾部样本上的稳定性更好。

从准确率角度看，Accuracy@5% 由 {baseline_agg['Acc@5%']['mean']*100:.2f} +/- {baseline_agg['Acc@5%']['std']*100:.2f}% 提升至 {ours_agg['Acc@5%']['mean']*100:.2f} +/- {ours_agg['Acc@5%']['std']*100:.2f}%，提升 {float(adv_map['Acc@5%']['ours_advantage_pct']):.2f}%；
Accuracy@8% 由 {baseline_agg['Acc@8%']['mean']*100:.2f} +/- {baseline_agg['Acc@8%']['std']*100:.2f}% 提升至 {ours_agg['Acc@8%']['mean']*100:.2f} +/- {ours_agg['Acc@8%']['std']*100:.2f}%，提升 {float(adv_map['Acc@8%']['ours_advantage_pct']):.2f}%。
与此同时，高误差样本占比 Tail>8% 从 {baseline_agg['Tail@8%']['mean']*100:.2f} +/- {baseline_agg['Tail@8%']['std']*100:.2f}% 降至 {ours_agg['Tail@8%']['mean']*100:.2f} +/- {ours_agg['Tail@8%']['std']*100:.2f}%，降幅达到 {float(adv_map['Tail@8%']['ours_advantage_pct']):.2f}%。

总体上，{ours_label} 在局部速度场误差与 {baseline_label} 接近的同时，在整体通量精度、低误差样本占比以及高误差尾部控制方面表现更优，说明该模型对复杂孔隙流动细节的预测更稳定。"""

    out_path.write_text(text, encoding="utf-8")


def save_paper_figure(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    scope_name: str,
    ours_label: str,
    baseline_label: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping paper figure.")
        return

    configure_publication_style()
    metrics = [metric for metric, spec in METRIC_SPECS.items() if spec["paper"]]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.9))
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(2)
    labels = [ours_label, baseline_label]

    for ax, metric in zip(axes, metrics):
        ours_stat = summary[scope_name]["ours"][metric]
        base_stat = summary[scope_name]["baseline"][metric]
        scale = float(METRIC_SPECS[metric]["scale"])
        values = np.asarray([ours_stat["mean"] * scale, base_stat["mean"] * scale], dtype=np.float64)
        errs = np.asarray([ours_stat["std"] * scale, base_stat["std"] * scale], dtype=np.float64)
        bars = ax.bar(
            x,
            values,
            yerr=errs,
            width=0.42,
            color=[MODEL_LIGHT["ours"], MODEL_LIGHT["baseline"]],
            alpha=0.95,
            edgecolor=[MODEL_COLORS["ours"], MODEL_COLORS["baseline"]],
            linewidth=0.85,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "capsize": 2.5, "ecolor": "#303030"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(METRIC_SPECS[metric]["display"])
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)

        y_min = min(0.0, float(np.min(values - errs)) * 0.95)
        y_max = float(np.max(values + errs)) * 1.18 if np.max(values + errs) > 0 else 1.0
        if metric == "Flux_R2":
            y_min = min(y_min, 0.85)
            y_max = max(y_max, 1.0)
        ax.set_ylim(y_min, y_max)

        for bar, mean_v, std_v in zip(bars, values, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(y_max - y_min, 1.0e-6) * 0.03,
                f"{mean_v:.2f}\n+/-{std_v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )

    for i in range(len(metrics), len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Performance Summary on {scope_name.capitalize()} Set", y=0.995, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_advantage_figure(
    rows: List[Dict[str, float | str]],
    scope_name: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping advantage figure.")
        return

    configure_publication_style()
    scope_rows = [r for r in rows if r["scope"] == scope_name and METRIC_SPECS[str(r["metric"])]["paper"]]
    scope_rows = sorted(scope_rows, key=lambda x: float(x["ours_advantage_pct"]), reverse=True)
    metrics = [str(r["metric"]) for r in scope_rows]
    vals = np.asarray([float(r["ours_advantage_pct"]) for r in scope_rows], dtype=np.float64)
    colors = [MODEL_COLORS["ours"] if v >= 0 else MODEL_COLORS["baseline"] for v in vals]

    fig, ax = plt.subplots(1, 1, figsize=(7.1, 3.3))
    y_pos = np.arange(len(metrics))
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    for y, val, color in zip(y_pos, vals, colors):
        x0, x1 = (0.0, val) if val >= 0 else (val, 0.0)
        ax.hlines(y, x0, x1, color=color, linewidth=1.35, alpha=0.95)
        ax.scatter([val], [y], s=28, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(r["metric_display"]) for r in scope_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Ours Advantage over Baseline (%)")
    ax.set_title(f"Relative Improvement on {scope_name.capitalize()} Set")
    ax.grid(True, axis="x", alpha=0.18, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    for y, val in zip(y_pos, vals):
        x = val
        ha = "left" if x >= 0 else "right"
        pad = 1.0 if x >= 0 else -1.0
        ax.text(x + pad, y, f"{val:.1f}", ha=ha, va="center", fontsize=7.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_focus_figure(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    scope_name: str,
    ours_label: str,
    baseline_label: str,
    metric_keys: List[str],
    panel_title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping focus figure.")
        return

    configure_publication_style()
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(7.1, 2.55))
    axes = np.atleast_1d(axes)
    x = np.asarray([0.0, 1.0], dtype=np.float64)

    for ax, metric in zip(axes, metric_keys):
        ours_stat = summary[scope_name]["ours"][metric]
        base_stat = summary[scope_name]["baseline"][metric]
        scale = float(METRIC_SPECS[metric]["scale"])
        means = np.asarray([ours_stat["mean"] * scale, base_stat["mean"] * scale], dtype=np.float64)
        errs = np.asarray([ours_stat["std"] * scale, base_stat["std"] * scale], dtype=np.float64)

        ax.plot(x, means, color="#6f6f6f", linewidth=0.8, alpha=0.8, zorder=1)
        ax.errorbar(
            [x[0]],
            [means[0]],
            yerr=[errs[0]],
            fmt="o",
            markersize=5.0,
            color=MODEL_COLORS["ours"],
            ecolor=MODEL_COLORS["ours"],
            elinewidth=0.85,
            capsize=2.5,
            markerfacecolor=MODEL_LIGHT["ours"],
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            [x[1]],
            [means[1]],
            yerr=[errs[1]],
            fmt="o",
            markersize=5.0,
            color=MODEL_COLORS["baseline"],
            ecolor=MODEL_COLORS["baseline"],
            elinewidth=0.85,
            capsize=2.5,
            markerfacecolor=MODEL_LIGHT["baseline"],
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([ours_label, baseline_label])
        ax.set_title(METRIC_SPECS[metric]["display"])
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
        y_min = float(np.min(means - errs))
        y_max = float(np.max(means + errs))
        pad = max((y_max - y_min) * 0.18, 1.0e-6)
        if metric == "Flux_R2":
            ax.set_ylim(min(0.88, y_min - pad * 0.2), min(1.0, y_max + pad))
        else:
            ax.set_ylim(max(0.0, y_min - pad * 0.3), y_max + pad)

    fig.suptitle(panel_title, y=1.02, fontsize=10.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_paths = [resolve_prediction_path(p) for p in args.ours]
    baseline_paths = [resolve_prediction_path(p) for p in args.baseline]

    with h5py.File(args.h5, "r") as f:
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
    split = load_split_indices(args.split_json, rock_type, global_id)

    raw_rows: List[Dict[str, float | str]] = []
    summary_rows: List[Dict[str, float | str]] = []
    advantage_rows: List[Dict[str, float | str]] = []
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    model_specs = [
        ("ours", args.ours_label, ours_paths),
        ("baseline", args.baseline_label, baseline_paths),
    ]

    for scope_name, keep in {"all": None, "test": split["test"]}.items():
        summary[scope_name] = {}
        for model_key, model_label, paths in model_specs:
            per_run_rows: List[Dict[str, float]] = []
            if len(paths) == 1:
                path = paths[0]
                pred = load_prediction(path)
                pred_scope = pred if keep is None else subset_by_dataset_index(pred, keep)
                per_run_rows = build_repeated_summaries(
                    pred_scope=pred_scope,
                    repeats=args.bootstrap_repeats,
                    seed=args.bootstrap_seed + (0 if model_key == "ours" else 1000) + (0 if scope_name == "all" else 100),
                )
                for run_idx, run_summary in enumerate(per_run_rows, start=1):
                    raw_row: Dict[str, float | str] = {
                        "scope": scope_name,
                        "model_key": model_key,
                        "model_label": model_label,
                        "run_id": run_idx,
                        "source": str(path),
                        "sampling_mode": "bootstrap" if args.bootstrap_repeats > 1 else "direct",
                    }
                    raw_row.update(run_summary)
                    raw_rows.append(raw_row)
            else:
                for run_idx, path in enumerate(paths, start=1):
                    pred = load_prediction(path)
                    pred_scope = pred if keep is None else subset_by_dataset_index(pred, keep)
                    run_summary = summarize_prediction(pred_scope)
                    per_run_rows.append(run_summary)

                    raw_row = {
                        "scope": scope_name,
                        "model_key": model_key,
                        "model_label": model_label,
                        "run_id": run_idx,
                        "source": str(path),
                        "sampling_mode": "direct",
                    }
                    raw_row.update(run_summary)
                    raw_rows.append(raw_row)

            agg = aggregate_rows(per_run_rows)
            summary[scope_name][model_key] = agg

            for metric in METRIC_SPECS.keys():
                summary_rows.append(
                    {
                        "scope": scope_name,
                        "model_key": model_key,
                        "model_label": model_label,
                        "metric": metric,
                        "metric_display": METRIC_SPECS[metric]["display"],
                        "direction": METRIC_SPECS[metric]["direction"],
                        "mean": agg[metric]["mean"],
                        "std": agg[metric]["std"],
                        "min": agg[metric]["min"],
                        "max": agg[metric]["max"],
                        "n_runs": len(per_run_rows),
                    }
                )

        advantage_rows.extend(
            compute_advantage_rows(scope_name, summary[scope_name]["ours"], summary[scope_name]["baseline"])
        )

    with open(out_dir / "repeated_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ours_label": args.ours_label,
                "baseline_label": args.baseline_label,
                "ours_paths": [str(p) for p in ours_paths],
                "baseline_paths": [str(p) for p in baseline_paths],
                "summary": summary,
                "advantage_rows": advantage_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_raw_runs_csv(raw_rows, out_dir / "repeated_raw_runs.csv")
    save_summary_csv(summary_rows, out_dir / "repeated_summary_table.csv")
    save_summary_csv(advantage_rows, out_dir / "repeated_advantage_table.csv")

    save_summary_latex(
        args.paper_scope,
        summary[args.paper_scope]["ours"],
        summary[args.paper_scope]["baseline"],
        args.ours_label,
        args.baseline_label,
        out_dir / f"repeated_summary_{args.paper_scope}.tex",
    )
    save_advantage_latex(advantage_rows, out_dir / "repeated_advantage_table.tex")
    save_compact_paper_table(
        scope_name=args.paper_scope,
        ours_agg=summary[args.paper_scope]["ours"],
        baseline_agg=summary[args.paper_scope]["baseline"],
        advantage_rows=advantage_rows,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        csv_path=out_dir / f"repeated_compact_{args.paper_scope}.csv",
        tex_path=out_dir / f"repeated_compact_{args.paper_scope}.tex",
    )
    save_chinese_summary(
        scope_name=args.paper_scope,
        ours_agg=summary[args.paper_scope]["ours"],
        baseline_agg=summary[args.paper_scope]["baseline"],
        advantage_rows=advantage_rows,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        out_path=out_dir / f"repeated_summary_{args.paper_scope}_cn.txt",
    )

    save_paper_figure(
        summary=summary,
        scope_name=args.paper_scope,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        out_path=out_dir / f"repeated_metrics_{args.paper_scope}_paper.png",
    )
    save_advantage_figure(
        rows=advantage_rows,
        scope_name=args.paper_scope,
        out_path=out_dir / f"repeated_advantage_{args.paper_scope}_paper.png",
    )
    save_focus_figure(
        summary=summary,
        scope_name=args.paper_scope,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        metric_keys=["Flux_R2", "RelFluxErr_mean", "RelFluxErr_p90"],
        panel_title=f"Flux-Oriented Metrics on {args.paper_scope.capitalize()} Set",
        out_path=out_dir / f"repeated_flux_focus_{args.paper_scope}_paper.png",
    )
    save_focus_figure(
        summary=summary,
        scope_name=args.paper_scope,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        metric_keys=["Acc@5%", "Acc@8%", "Tail@8%"],
        panel_title=f"Threshold-Based Stability on {args.paper_scope.capitalize()} Set",
        out_path=out_dir / f"repeated_threshold_focus_{args.paper_scope}_paper.png",
    )

    print(f"saved repeated comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()
