#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据已有 predictions.npz 生成与 infer_unet_h5.py 类似的分析输出。

输出内容：
- report.csv
- report_by_rock_type.csv
- worst_cases.csv
- metrics.csv
- metrics_summary.json
- inference_results.png
- inference_analysis.png
- sample_predictions.png

适用场景：
- 对照模型没有走 infer_unet_h5.py
- 但 predictions.npz 的字段格式与现有工程保持一致
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from infer_unet_h5 import (
    compute_r2,
    compute_vector_metrics,
    discover_meta_fields,
    load_h5_field_for_indices,
    plot_inference_results,
    plot_sample_grid,
    stats_basic,
    stats_three,
)

from visual_compare_velocity_fields import configure_publication_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 predictions.npz 渲染为 infer 风格的输出目录。")
    parser.add_argument("--h5", type=str, required=True, help="HDF5 数据集路径")
    parser.add_argument("--predictions", type=str, required=True, help="predictions.npz 路径")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--topk", type=int, default=200, help="worst cases 数量")
    parser.add_argument("--sample-count", type=int, default=10, help="sample_predictions 随机样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--paper-style", action="store_true", help="额外输出论文风格 PNG/PDF")
    return parser.parse_args()


def render_inference_results_paper(
    out_dir: str,
    q_true: np.ndarray,
    q_pred: np.ndarray,
    rel_err: np.ndarray,
    rmse_pore: np.ndarray,
    pore_frac: np.ndarray,
    rock_type: np.ndarray,
    r2_val: float,
) -> None:
    import matplotlib.pyplot as plt

    configure_publication_style()
    colors = ["#1f4e79", "#4c956c", "#3f7cac", "#c84c31", "#8e6c8a", "#bfa23a"]
    rock_types_sorted = sorted(np.unique(rock_type).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))

    ax = axes[0, 0]
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type == rt
        ax.scatter(q_true[mask_rt], q_pred[mask_rt], s=8, alpha=0.45, color=colors[i % len(colors)], label=f"R{rt}")
    min_v = min(float(np.min(q_true)), float(np.min(q_pred)))
    max_v = max(float(np.max(q_true)), float(np.max(q_pred)))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="#333333", linewidth=1.0)
    ax.set_title("(a) Flux Prediction")
    ax.set_xlabel(r"$q_{\mathrm{true}}$")
    ax.set_ylabel(r"$q_{\mathrm{pred}}$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(
        0.03,
        0.97,
        rf"$R^2 = {r2_val:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#666666", alpha=0.9),
    )

    ax = axes[0, 1]
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type == rt
        ax.scatter(q_true[mask_rt], rel_err[mask_rt], s=8, alpha=0.45, color=colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_title("(b) Relative Flux Error")
    ax.set_xlabel(r"$q_{\mathrm{true}}$")
    ax.set_ylabel("RelFluxErr")

    ax = axes[1, 0]
    data = [rel_err[rock_type == rt] for rt in rock_types_sorted]
    bp = ax.boxplot(
        data,
        tick_labels=[f"R{rt}" for rt in rock_types_sorted],
        showfliers=False,
        patch_artist=True,
        widths=0.65,
        medianprops=dict(color="black", linewidth=0.9),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.7)
    ax.set_title("(c) RelFluxErr by Rock Type")
    ax.set_xlabel("Rock Type")
    ax.set_ylabel("RelFluxErr")
    ax.set_yscale("log")

    ax = axes[1, 1]
    x_pos = np.arange(len(rock_types_sorted))
    r2_by_type = [compute_r2(q_pred[rock_type == rt], q_true[rock_type == rt]) for rt in rock_types_sorted]
    bars = ax.bar(x_pos, r2_by_type, color=[colors[i % len(colors)] for i in range(len(rock_types_sorted))], alpha=0.9)
    ax.axhline(r2_val, color="#333333", linestyle="--", linewidth=1.0, label=f"Overall = {r2_val:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"R{rt}" for rt in rock_types_sorted])
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Flux $R^2$ by Rock Type")
    ax.set_xlabel("Rock Type")
    ax.set_ylabel("$R^2$")
    ax.legend(frameon=False, fontsize=7)
    for bar, val in zip(bars, r2_by_type):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    for ax in axes.flat:
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = Path(out_dir) / "inference_results_paper.png"
    pdf_path = Path(out_dir) / "inference_results_paper.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()


def render_sample_predictions_paper(samples: List[Dict], out_dir: str) -> None:
    if not samples:
        return
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    configure_publication_style()
    n_rows = len(samples)
    fig = plt.figure(figsize=(7.1, 1.25 * n_rows + 0.9))
    gs = gridspec.GridSpec(n_rows + 1, 3, height_ratios=[1.0] * n_rows + [0.08], hspace=0.10, wspace=0.06)

    vmax = max(max(float(np.max(s["y"])), float(np.max(s["pred"]))) for s in samples)
    emax = max(float(np.max(np.abs(s["pred"] - s["y"]))) for s in samples)

    for row, s in enumerate(samples):
        y = np.ma.masked_where(s["mask"] <= 0.5, s["y"])
        pred = np.ma.masked_where(s["mask"] <= 0.5, s["pred"])
        err = np.ma.masked_where(s["mask"] <= 0.5, s["pred"] - s["y"])
        for col, (title, img, cmap, vmin, vmax_local) in enumerate(
            [
                ("(a) Reference" if row == 0 else "", y, "cividis", 0.0, vmax),
                ("(b) Prediction" if row == 0 else "", pred, "cividis", 0.0, vmax),
                ("(c) Error" if row == 0 else "", err, "RdBu_r", -emax, emax),
            ]
        ):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax_local, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title, pad=6.0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("#4a4a4a")
            if col == 0:
                ax.text(
                    -0.22,
                    0.5,
                    f"idx {s['idx']}\nR{s['rock_type']}\nrel {s['rel_err']:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.0,
                )

    cax1 = fig.add_subplot(gs[n_rows, 0:2])
    cax2 = fig.add_subplot(gs[n_rows, 2])
    ColorbarBase(cax1, cmap=plt.get_cmap("cividis"), norm=Normalize(vmin=0.0, vmax=vmax), orientation="horizontal")
    cax1.set_xlabel(r"Velocity $u_x$")
    ColorbarBase(cax2, cmap=plt.get_cmap("RdBu_r"), norm=Normalize(vmin=-emax, vmax=emax), orientation="horizontal")
    cax2.set_xlabel(r"Error $\hat{u}_x-u_x$")

    fig.subplots_adjust(left=0.14, right=0.985, top=0.97, bottom=0.07)
    png_path = Path(out_dir) / "sample_predictions_paper.png"
    pdf_path = Path(out_dir) / "sample_predictions_paper.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()


def render_inference_analysis_paper(
    out_dir: str,
    q_true: np.ndarray,
    rel_err: np.ndarray,
    rmse_pore: np.ndarray,
    rock_type: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    configure_publication_style()
    colors = ["#1f4e79", "#4c956c", "#3f7cac", "#c84c31", "#8e6c8a", "#bfa23a"]
    rock_types_sorted = sorted(np.unique(rock_type).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))

    ax = axes[0, 0]
    x_max = float(np.percentile(rel_err, 98))
    for i, rt in enumerate(rock_types_sorted):
        data_rt = np.sort(rel_err[rock_type == rt])
        cdf = np.arange(1, len(data_rt) + 1) / len(data_rt)
        ax.plot(data_rt, cdf, color=colors[i % len(colors)], linewidth=1.5, label=f"R{rt}")
    ax.axvline(np.median(rel_err), color="#333333", linestyle="--", linewidth=1.0, label=f"Median = {np.median(rel_err):.3f}")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(a) CDF of Relative Flux Error")
    ax.set_xlabel("RelFluxErr")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[0, 1]
    counts = np.array([np.sum(rock_type == rt) for rt in rock_types_sorted], dtype=np.int32)
    bars = ax.bar(
        np.arange(len(rock_types_sorted)),
        counts,
        color=[colors[i % len(colors)] for i in range(len(rock_types_sorted))],
        alpha=0.9,
    )
    ax.set_xticks(np.arange(len(rock_types_sorted)))
    ax.set_xticklabels([f"R{rt}" for rt in rock_types_sorted])
    ax.set_title("(b) Sample Count by Rock Type")
    ax.set_xlabel("Rock Type")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02, f"{val}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 0]
    data_rmse = [rmse_pore[rock_type == rt] for rt in rock_types_sorted]
    bp = ax.boxplot(
        data_rmse,
        tick_labels=[f"R{rt}" for rt in rock_types_sorted],
        showfliers=False,
        patch_artist=True,
        widths=0.65,
        medianprops=dict(color="black", linewidth=0.9),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.7)
    ax.set_title("(c) RMSE in Pore Region")
    ax.set_xlabel("Rock Type")
    ax.set_ylabel("RMSE")

    ax = axes[1, 1]
    data_q = [q_true[rock_type == rt] for rt in rock_types_sorted]
    bp = ax.boxplot(
        data_q,
        tick_labels=[f"R{rt}" for rt in rock_types_sorted],
        showfliers=False,
        patch_artist=True,
        widths=0.65,
        medianprops=dict(color="black", linewidth=0.9),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.7)
    ax.set_title("(d) Flux Distribution")
    ax.set_xlabel("Rock Type")
    ax.set_ylabel(r"$q_{\mathrm{true}}$")

    for ax in axes.flat:
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    png_path = Path(out_dir) / "inference_analysis_paper.png"
    pdf_path = Path(out_dir) / "inference_analysis_paper.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.predictions)
    pred_ux = z["pred_ux"].astype(np.float32)
    q_true = z["q_true"].astype(np.float32)
    q_pred = z["q_pred"].astype(np.float32)
    rel_err = z["rel_err"].astype(np.float32)
    abs_err = z["abs_err"].astype(np.float32)
    rmse_pore = z["rmse_pore"].astype(np.float32)
    mae_pore = z["mae_pore"].astype(np.float32)
    pore_frac = z["pore_frac"].astype(np.float32)
    rock_type = z["rock_type"].astype(np.int16)
    global_id = z["global_id"].astype(np.int32)
    indices = z["index"].astype(np.int32)

    n = int(indices.shape[0])
    if pred_ux.shape[0] != n:
        raise ValueError("pred_ux 与 index 长度不一致")

    with h5py.File(args.h5, "r") as f:
        X = f["X"]
        Y = f["Y"]
        channel_order = f.attrs.get("channel_order", "")
        if isinstance(channel_order, bytes):
            channel_order = channel_order.decode("utf-8")
        channel_order = str(channel_order)
        mask_idx = 0
        order_list = [s.strip() for s in channel_order.split(",") if s.strip()]
        for i, name in enumerate(order_list):
            if name.lower() == "mask":
                mask_idx = i
                break

        # metrics.csv
        metrics_path = out_dir / "metrics.csv"
        metrics_fields = [
            "index", "rock_type", "global_id",
            "mae_ux", "rmse_ux", "mae_uy", "rmse_uy",
            "mae_speed", "rmse_speed", "epe", "cosine", "div_mean",
            "solid_speed_mean",
        ]
        metrics_sum = {k: 0.0 for k in metrics_fields if k not in ("index", "rock_type", "global_id")}
        metrics_count = {k: 0 for k in metrics_sum.keys()}
        with open(metrics_path, "w", newline="", encoding="utf-8") as mf:
            mf.write(",".join(metrics_fields) + "\n")
            for pos, ds_idx in enumerate(indices.tolist()):
                x = X[int(ds_idx)]
                y = Y[int(ds_idx)]
                mask = x[mask_idx].astype(np.float32)
                ux_t = y[0].astype(np.float32)
                uy_t = np.zeros_like(ux_t, dtype=np.float32)
                ux_p = pred_ux[pos, 0].astype(np.float32)
                uy_p = np.zeros_like(ux_p, dtype=np.float32)

                m = compute_vector_metrics(ux_t, uy_t, ux_p, uy_p, mask)
                mf.write(
                    f"{int(ds_idx)},{int(rock_type[pos])},{int(global_id[pos])},"
                    f"{m['mae_ux']:.6e},{m['rmse_ux']:.6e},{m['mae_uy']:.6e},{m['rmse_uy']:.6e},"
                    f"{m['mae_speed']:.6e},{m['rmse_speed']:.6e},{m['epe']:.6e},{m['cosine']:.6e},{m['div_mean']:.6e},"
                    f"{m['solid_speed_mean']:.6e}\n"
                )
                for k in metrics_sum.keys():
                    v = m.get(k, float("nan"))
                    if np.isfinite(v):
                        metrics_sum[k] += float(v)
                        metrics_count[k] += 1

    metrics_summary = {}
    for k in metrics_sum.keys():
        metrics_summary[k] = metrics_sum[k] / metrics_count[k] if metrics_count[k] > 0 else float("nan")
    metrics_summary["n_samples"] = n
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    # report.csv
    rel_stats = stats_basic(rel_err)
    abs_stats = stats_basic(abs_err)
    rmse_stats = stats_three(rmse_pore)
    mae_stats = stats_three(mae_pore)
    r2_val = compute_r2(q_pred, q_true)
    q_true_stats = {
        "min": float(np.min(q_true)) if q_true.size > 0 else float("nan"),
        "median": float(np.median(q_true)) if q_true.size > 0 else float("nan"),
        "p10": float(np.percentile(q_true, 10)) if q_true.size > 0 else float("nan"),
        "p90": float(np.percentile(q_true, 90)) if q_true.size > 0 else float("nan"),
    }
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        f.write(
            "N,RelFluxErr_mean,RelFluxErr_median,RelFluxErr_p90,RelFluxErr_p95,RelFluxErr_max,"
            "AbsFluxErr_mean,AbsFluxErr_median,AbsFluxErr_p90,AbsFluxErr_p95,AbsFluxErr_max,"
            "RMSE_pore_mean,RMSE_pore_median,RMSE_pore_p90,"
            "MAE_pore_mean,MAE_pore_median,MAE_pore_p90,"
            "Flux_R2,q_true_min,q_true_median,q_true_p10,q_true_p90\n"
        )
        f.write(
            f"{n},{rel_stats['mean']:.6f},{rel_stats['median']:.6f},{rel_stats['p90']:.6f},{rel_stats['p95']:.6f},{rel_stats['max']:.6f},"
            f"{abs_stats['mean']:.6f},{abs_stats['median']:.6f},{abs_stats['p90']:.6f},{abs_stats['p95']:.6f},{abs_stats['max']:.6f},"
            f"{rmse_stats['mean']:.6f},{rmse_stats['median']:.6f},{rmse_stats['p90']:.6f},"
            f"{mae_stats['mean']:.6f},{mae_stats['median']:.6f},{mae_stats['p90']:.6f},"
            f"{r2_val:.6f},{q_true_stats['min']:.6f},{q_true_stats['median']:.6f},{q_true_stats['p10']:.6f},{q_true_stats['p90']:.6f}\n"
        )

    # report_by_rock_type.csv
    with open(out_dir / "report_by_rock_type.csv", "w", newline="", encoding="utf-8") as f:
        f.write(
            "rock_type,N,RelFluxErr_mean,RelFluxErr_median,RelFluxErr_p90,RelFluxErr_p95,"
            "AbsFluxErr_mean,AbsFluxErr_median,AbsFluxErr_p90,AbsFluxErr_p95,"
            "RMSE_pore_mean,MAE_pore_mean,Flux_R2,q_true_median,q_true_p10,q_true_p90\n"
        )
        for rt in sorted(np.unique(rock_type).tolist()):
            mask_rt = rock_type == rt
            rel_s = stats_basic(rel_err[mask_rt])
            abs_s = stats_basic(abs_err[mask_rt])
            q_true_rt = q_true[mask_rt]
            r2_rt = compute_r2(q_pred[mask_rt], q_true_rt)
            f.write(
                f"{rt},{int(mask_rt.sum())},{rel_s['mean']:.6f},{rel_s['median']:.6f},{rel_s['p90']:.6f},{rel_s['p95']:.6f},"
                f"{abs_s['mean']:.6f},{abs_s['median']:.6f},{abs_s['p90']:.6f},{abs_s['p95']:.6f},"
                f"{float(np.mean(rmse_pore[mask_rt])):.6f},{float(np.mean(mae_pore[mask_rt])):.6f},"
                f"{r2_rt:.6f},{float(np.median(q_true_rt)):.6f},{float(np.percentile(q_true_rt, 10)):.6f},{float(np.percentile(q_true_rt, 90)):.6f}\n"
            )

    # worst_cases.csv
    topk = min(int(args.topk), n)
    order = np.argsort(rel_err)[::-1][:topk]
    meta_fields = discover_meta_fields(args.h5, n)
    meta_data = {field: load_h5_field_for_indices(args.h5, field, indices[order]) for field in meta_fields}
    with open(out_dir / "worst_cases.csv", "w", newline="", encoding="utf-8") as f:
        header = [
            "index", "rock_type", "global_id", "q_true", "q_pred", "rel_err", "abs_err",
            "rmse_pore", "mae_pore", "pore_frac",
        ] + meta_fields
        f.write(",".join(header) + "\n")
        for i, local_pos in enumerate(order.tolist()):
            row = [
                int(indices[local_pos]),
                int(rock_type[local_pos]),
                int(global_id[local_pos]),
                float(q_true[local_pos]),
                float(q_pred[local_pos]),
                float(rel_err[local_pos]),
                float(abs_err[local_pos]),
                float(rmse_pore[local_pos]),
                float(mae_pore[local_pos]),
                float(pore_frac[local_pos]),
            ] + [meta_data[field][i] for field in meta_fields]
            f.write(",".join([str(v) for v in row]) + "\n")

    # inference plots
    plot_inference_results(
        out_dir=str(out_dir),
        q_true=q_true,
        q_pred=q_pred,
        rel_err=rel_err,
        abs_err=abs_err,
        rmse_pore=rmse_pore,
        pore_frac=pore_frac,
        rock_type=rock_type,
        r2_val=r2_val,
    )

    # sample_predictions.png
    samples = []
    rng = np.random.default_rng(args.seed)
    pick_count = min(int(args.sample_count), n)
    picked_positions = rng.choice(n, size=pick_count, replace=False)
    with h5py.File(args.h5, "r") as f:
        X = f["X"]
        Y = f["Y"]
        channel_order = f.attrs.get("channel_order", "")
        if isinstance(channel_order, bytes):
            channel_order = channel_order.decode("utf-8")
        channel_order = str(channel_order)
        mask_idx = 0
        order_list = [s.strip() for s in channel_order.split(",") if s.strip()]
        for i, name in enumerate(order_list):
            if name.lower() == "mask":
                mask_idx = i
                break

        for pos in picked_positions.tolist():
            ds_idx = int(indices[pos])
            x = X[ds_idx]
            y = Y[ds_idx]
            samples.append(
                {
                    "kind": "random",
                    "rock_type": int(rock_type[pos]),
                    "global_id": int(global_id[pos]),
                    "idx": ds_idx,
                    "q_true": float(q_true[pos]),
                    "q_pred": float(q_pred[pos]),
                    "rel_err": float(rel_err[pos]),
                    "rmse_pore": float(rmse_pore[pos]),
                    "pore_frac": float(pore_frac[pos]),
                    "mask": x[mask_idx].astype(np.float32),
                    "y": y[0].astype(np.float32),
                    "pred": pred_ux[pos, 0].astype(np.float32),
                }
            )
    plot_sample_grid(str(out_dir / "sample_predictions.png"), samples)
    if args.paper_style:
        render_inference_results_paper(
            out_dir=str(out_dir),
            q_true=q_true,
            q_pred=q_pred,
            rel_err=rel_err,
            rmse_pore=rmse_pore,
            pore_frac=pore_frac,
            rock_type=rock_type,
            r2_val=r2_val,
        )
        render_inference_analysis_paper(
            out_dir=str(out_dir),
            q_true=q_true,
            rel_err=rel_err,
            rmse_pore=rmse_pore,
            rock_type=rock_type,
        )
        render_sample_predictions_paper(samples, str(out_dir))

    print(f"saved infer-like outputs to: {out_dir}")


if __name__ == "__main__":
    main()
