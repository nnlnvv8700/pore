#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate publication-style figures for the IPNM1 manuscript experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(r"E:\mhw\1\cup")
RUN_MAIN = ROOT / "runs" / "res_ipnm1_flownet_100e_3seed_20260527"
RUN_EXP = ROOT / "runs" / "ipnm1_manuscript_experiments_20260527"
H5_PATH = ROOT / "dataset_all_32.h5"
OUT_DIR = ROOT / "runs" / "ipnm1_paper_figures_20260527"


PALETTE = {
    "navy": "#334E68",
    "blue": "#4C78A8",
    "teal": "#5AA6A6",
    "orange": "#E08D3C",
    "red": "#C44E52",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
    "green": "#59A14F",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "legend.frameon": False,
            "figure.dpi": 160,
        }
    )


def save_figure(fig: plt.Figure, stem: str, width_mm: float = 180, height_mm: float | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if height_mm is not None:
        fig.set_size_inches(width_mm / 25.4, height_mm / 25.4)
    base = OUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / denom) if denom > 0 else float("nan")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_identity(ax: plt.Axes, values_a: np.ndarray, values_b: np.ndarray, log: bool = False) -> None:
    vals = np.concatenate([np.asarray(values_a), np.asarray(values_b)])
    vals = vals[np.isfinite(vals) & (vals > 0 if log else np.isfinite(vals))]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if log:
        lo *= 0.85
        hi *= 1.15
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        pad = 0.04 * (hi - lo)
        lo -= pad
        hi += pad
    ax.plot([lo, hi], [lo, hi], color=PALETTE["gray"], lw=0.8, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def draw_method_schematic() -> None:
    fig = plt.figure(figsize=(7.1, 2.1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    xs = [0.08, 0.34, 0.60, 0.84]
    labels = [
        ("Real throat\ncross-section", "geometry"),
        ("Res-IPNM1-\nFlowNet", "velocity surrogate"),
        ("Velocity-field\nintegration", "q = integral u dA"),
        ("IPNM1 local\nconductance", "g from q"),
    ]
    colors = ["#E8F1F2", "#EAF0F8", "#F7EFE5", "#E9F3EA"]
    for x, (title, subtitle), color in zip(xs, labels, colors):
        rect = Rectangle((x - 0.095, 0.35), 0.19, 0.30, facecolor=color, edgecolor="#334155", lw=0.8)
        ax.add_patch(rect)
        ax.text(x, 0.535, title, ha="center", va="center", fontsize=8, fontweight="bold", color=PALETTE["dark"])
        ax.text(x, 0.415, subtitle, ha="center", va="center", fontsize=6.4, color=PALETTE["gray"])
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.105, 0.50),
                (x1 - 0.105, 0.50),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color=PALETTE["navy"],
            )
        )
    ax.text(0.08, 0.18, "Input", ha="center", fontsize=6.5, color=PALETTE["gray"])
    ax.text(0.34, 0.18, "Learned local solver", ha="center", fontsize=6.5, color=PALETTE["gray"])
    ax.text(0.72, 0.18, "Explicit physics", ha="center", fontsize=6.5, color=PALETTE["gray"])
    ax.text(0.50, 0.82, "Local IPNM1 surrogate: geometry to velocity to conductance", ha="center", fontsize=9)
    save_figure(fig, "fig1_method_schematic", width_mm=180, height_mm=58)


def load_seed42_fields() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_path = RUN_MAIN / "seed_42" / "res_ed" / "predictions.npz"
    pred = np.load(pred_path)
    idx = pred["index"].astype(np.int64)
    pred_u = pred["pred_ux"].astype(np.float32)[:, 0]
    with h5py.File(H5_PATH, "r") as f:
        true_u = f["Y"][idx, 0].astype(np.float32)
        mask = f["X"][idx, 0].astype(np.float32)
    return idx, pred_u, true_u, mask


def select_velocity_examples(pred_u: np.ndarray, true_u: np.ndarray, mask: np.ndarray) -> List[int]:
    rel = []
    for i in range(pred_u.shape[0]):
        m = mask[i] > 0.5
        q_t = float(np.sum(true_u[i][m]))
        q_p = float(np.sum(pred_u[i][m]))
        rel.append(abs(q_p - q_t) / (abs(q_t) + 1.0e-12))
    rel_arr = np.asarray(rel)
    median_i = int(np.argsort(np.abs(rel_arr - np.median(rel_arr)))[0])
    p90_target = np.percentile(rel_arr, 90)
    difficult_i = int(np.argsort(np.abs(rel_arr - p90_target))[0])
    return [median_i, difficult_i]


def draw_velocity_examples() -> None:
    idx, pred_u, true_u, mask = load_seed42_fields()
    picks = select_velocity_examples(pred_u, true_u, mask)
    fig = plt.figure(figsize=(7.1, 3.7))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.10, hspace=0.18)
    titles = ["Pore mask", "Reference velocity", "Predicted velocity", "Absolute error"]
    axes = []
    vmax = float(np.percentile(true_u[mask > 0.5], 99.6))
    err_all = np.abs(pred_u - true_u) * (mask > 0.5)
    err_vmax = float(np.percentile(err_all[mask > 0.5], 99.2))
    for row, pidx in enumerate(picks):
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(titles[col], pad=4)
            if col == 0:
                im = ax.imshow(mask[pidx], cmap="gray", vmin=0, vmax=1)
            elif col == 1:
                im = ax.imshow(np.where(mask[pidx] > 0.5, true_u[pidx], np.nan), cmap="viridis", vmin=0, vmax=vmax)
            elif col == 2:
                im = ax.imshow(np.where(mask[pidx] > 0.5, pred_u[pidx], np.nan), cmap="viridis", vmin=0, vmax=vmax)
            else:
                im = ax.imshow(np.where(mask[pidx] > 0.5, np.abs(pred_u[pidx] - true_u[pidx]), np.nan), cmap="magma", vmin=0, vmax=err_vmax)
            if col == 0:
                row_name = "Representative" if row == 0 else "Higher-error"
                ax.text(-0.05, 0.5, row_name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7)
    cax1 = fig.add_axes([0.44, 0.06, 0.20, 0.018])
    cb1 = fig.colorbar(axes[1].images[0], cax=cax1, orientation="horizontal")
    cb1.set_label("Velocity", labelpad=1)
    cax2 = fig.add_axes([0.76, 0.06, 0.17, 0.018])
    cb2 = fig.colorbar(axes[3].images[0], cax=cax2, orientation="horizontal")
    cb2.set_label("Absolute error", labelpad=1)
    fig.text(0.02, 0.97, "a", fontsize=9, fontweight="bold", va="top")
    save_figure(fig, "fig2_velocity_field_examples", width_mm=180, height_mm=105)


def draw_conductance_validation() -> None:
    rows = read_csv(RUN_MAIN / "seed_42" / "res_ed" / "conductance_ipnm1_rho1" / "conductance_results.csv")
    q_pred = np.asarray([as_float(r, "q_pred_abs") for r in rows])
    q_true = np.asarray([as_float(r, "q_true_abs") for r in rows])
    q_table = np.asarray([as_float(r, "q_table") for r in rows])
    g_pred = np.asarray([as_float(r, "g_pred") for r in rows])
    g_true = np.asarray([as_float(r, "g_true_velocity") for r in rows])
    g_table = np.asarray([as_float(r, "g_table") for r in rows])
    g_rel = np.asarray([as_float(r, "g_rel_error_table") for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    ax = axes[0]
    ax.scatter(q_true, q_pred, s=6, alpha=0.35, color=PALETTE["blue"], edgecolors="none", rasterized=True)
    add_identity(ax, q_true, q_pred)
    ax.set_xlabel("Reference q")
    ax.set_ylabel("Predicted q")
    ax.set_title(f"Integrated flow, R2={r2(q_pred, q_true):.4f}")
    panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(g_table, g_pred, s=6, alpha=0.35, color=PALETTE["teal"], edgecolors="none", rasterized=True)
    add_identity(ax, g_table, g_pred, log=True)
    ax.set_xlabel("Table conductance")
    ax.set_ylabel("Predicted conductance")
    ax.set_title(f"Conductance, R2={r2(g_pred, g_table):.4f}")
    panel_label(ax, "b")

    ax = axes[2]
    clipped = np.clip(g_rel * 100.0, 0, np.percentile(g_rel * 100.0, 99))
    ax.hist(clipped, bins=34, color=PALETTE["orange"], alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(np.mean(g_rel) * 100.0, color=PALETTE["red"], lw=1.0, label=f"mean {np.mean(g_rel)*100:.2f}%")
    ax.axvline(np.median(g_rel) * 100.0, color=PALETTE["dark"], lw=1.0, ls="--", label=f"median {np.median(g_rel)*100:.2f}%")
    ax.set_xlabel("Conductance relative error (%)")
    ax.set_ylabel("Samples")
    ax.set_title("Error distribution")
    ax.legend(loc="upper right", handlelength=1.2)
    panel_label(ax, "c")
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "fig3_q_g_validation", width_mm=180, height_mm=65)


def draw_main_and_ablation() -> None:
    metrics = read_csv(RUN_MAIN / "res_ipnm1_3seed_metrics.csv")
    metric_specs = [
        ("pixel_r2_all", "Velocity\nR2", 1.0, "R2"),
        ("q_mean_re_all", "q\nRE", 100.0, "%"),
        ("q_r2_all", "q\nR2", 1.0, "R2"),
        ("table_g_mean_re_all", "Table g\nRE", 100.0, "%"),
        ("table_g_r2_all", "Table g\nR2", 1.0, "R2"),
    ]
    means = []
    stds = []
    for key, _, scale, _ in metric_specs:
        vals = np.asarray([as_float(r, key) for r in metrics]) * scale
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)))

    flux = read_csv(RUN_EXP / "flux_ablation_summary.csv")
    lambdas = np.asarray([as_float(r, "lambda_flux") for r in flux])
    vel_r2 = np.asarray([as_float(r, "pixel_r2_all") for r in flux])
    g_re = np.asarray([as_float(r, "table_g_mean_re_all") for r in flux]) * 100.0
    q_re = np.asarray([as_float(r, "q_mean_re_all") for r in flux]) * 100.0

    fig = plt.figure(figsize=(7.1, 3.0), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(metric_specs))
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["blue"], PALETTE["orange"], PALETTE["blue"]]
    ax.bar(x, means, yerr=stds, capsize=2.5, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in metric_specs])
    ax.set_title("Three-seed main-model performance")
    ax.set_ylabel("Metric value")
    ax.grid(True, axis="y", alpha=0.25)
    for i, (m, spec) in enumerate(zip(means, metric_specs)):
        text = f"{m:.3f}" if spec[3] == "R2" else f"{m:.2f}%"
        ax.text(i, m + (0.015 if spec[3] == "R2" else 0.12), text, ha="center", va="bottom", fontsize=6.3)
    ax.set_ylim(0, max(max(means) * 1.18, 1.08))
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()
    ax.plot(lambdas, g_re, marker="o", color=PALETTE["orange"], lw=1.4, label="Table g RE")
    ax.plot(lambdas, q_re, marker="s", color=PALETTE["red"], lw=1.2, label="q RE")
    ax2.plot(lambdas, vel_r2, marker="^", color=PALETTE["blue"], lw=1.2, label="Velocity R2")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks(lambdas)
    ax.set_xticklabels(["0", "0.01", "0.10", "1.00"])
    ax.set_xlabel("Flux-loss weight")
    ax.set_ylabel("Relative error (%)")
    ax2.set_ylabel("Velocity R2")
    ax.set_title("Integrated-flux constraint")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", handlelength=1.5)
    panel_label(ax, "b")
    save_figure(fig, "fig4_main_metrics_flux_ablation", width_mm=180, height_mm=80)


def draw_scalar_baseline() -> None:
    rows = read_csv(RUN_EXP / "scalar_baseline_summary.csv")
    labels = ["Direct table scalar" if r["target"] == "q_table" else "Direct integrated q" for r in rows]
    all_re = np.asarray([as_float(r, "all_mean_re_mean") for r in rows]) * 100.0
    all_re_sd = np.asarray([as_float(r, "all_mean_re_std") for r in rows]) * 100.0
    test_re = np.asarray([as_float(r, "test_mean_re_mean") for r in rows]) * 100.0
    test_re_sd = np.asarray([as_float(r, "test_mean_re_std") for r in rows]) * 100.0
    all_r2 = np.asarray([as_float(r, "all_r2_mean") for r in rows])
    all_r2_sd = np.asarray([as_float(r, "all_r2_std") for r in rows])
    test_r2 = np.asarray([as_float(r, "test_r2_mean") for r in rows])
    test_r2_sd = np.asarray([as_float(r, "test_r2_std") for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    x = np.arange(len(rows))
    w = 0.34
    ax = axes[0]
    ax.bar(x - w / 2, all_re, w, yerr=all_re_sd, capsize=2.5, color=PALETTE["orange"], label="All")
    ax.bar(x + w / 2, test_re, w, yerr=test_re_sd, capsize=2.5, color=PALETTE["red"], label="Test")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("Scalar fitting error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    panel_label(ax, "a")

    ax = axes[1]
    ax.bar(x - w / 2, all_r2, w, yerr=all_r2_sd, capsize=2.5, color=PALETTE["blue"], label="All")
    ax.bar(x + w / 2, test_r2, w, yerr=test_r2_sd, capsize=2.5, color=PALETTE["teal"], label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Scalar fitting R2")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig5_direct_scalar_baseline", width_mm=180, height_mm=70)


def write_figure_contract() -> None:
    text = """# IPNM1 Paper Figure Plan

## Figure 1. Method schematic

Claim: the proposed method is a local IPNM1 surrogate that learns velocity first and obtains conductance by explicit physics.

Panels: one schematic showing cross-section geometry, Res-IPNM1-FlowNet, velocity integration and IPNM1 conductance conversion.

Review risk addressed: avoids the misunderstanding that the network directly regresses permeability or completes network-scale IPNM2.

## Figure 2. Velocity-field examples

Claim: the model predicts spatial velocity distributions, not only scalar flow rates.

Panels: pore mask, reference velocity, predicted velocity and absolute error for a representative case and a higher-error case.

Source data: seed-42 prediction arrays and the HDF5 reference velocity/mask.

## Figure 3. q/g validation

Claim: predicted velocity fields remain physically useful after integration into q and conductance g.

Panels: predicted-vs-reference q, predicted-vs-table g and conductance relative-error distribution.

Source data: postprocessed conductance table from the main model.

## Figure 4. Main metrics and flux-loss ablation

Claim: Res-IPNM1-FlowNet is accurate and the integrated-flux term improves q/g consistency.

Panels: three-seed main metrics and lambda_flux ablation.

Source data: three-seed metric table and flux-ablation summary.

## Figure 5. Direct scalar baseline

Claim: direct scalar fitting is not equivalent to the velocity-field-first strategy.

Panels: scalar-baseline relative error and R2 for direct q_true and q_table fitting.

Source data: direct scalar baseline summary.

## Optional supplementary figures

- Training curves across three seeds.
- Dataset/preprocessing examples.
- Teacher-data audit by rock type.
- Inference-time comparison once timing measurements are finalized.
"""
    (OUT_DIR / "FIGURE_PLAN.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_figure_contract()
    draw_method_schematic()
    draw_velocity_examples()
    draw_conductance_validation()
    draw_main_and_ablation()
    draw_scalar_baseline()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
