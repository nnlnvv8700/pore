#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成三模型速度场对比论文图：
Reference / Ours / Baseline-1 / Baseline-2 / 各自误差图。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from visual_compare_velocity_fields import (
    build_index_lookup,
    configure_publication_style,
    load_split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成包含两个对照模型的速度场论文对比图。")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline1", type=str, required=True)
    parser.add_argument("--baseline2", type=str, required=True)
    parser.add_argument("--ours-label", type=str, default="Ours")
    parser.add_argument("--baseline1-label", type=str, default="Baseline-1")
    parser.add_argument("--baseline2-label", type=str, default="Baseline-2")
    parser.add_argument("--ours-alias", type=str, default="ours")
    parser.add_argument("--baseline1-alias", type=str, default="b1")
    parser.add_argument("--baseline2-alias", type=str, default="b2")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scope", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--layout", type=str, default="double", choices=("single", "double"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def velocity_colormap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap


def error_colormap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("seismic").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap


def choose_samples(
    args: argparse.Namespace,
    y_true_all: np.ndarray,
    rock_type: np.ndarray,
    global_id: np.ndarray,
    ours: np.lib.npyio.NpzFile,
    base1: np.lib.npyio.NpzFile,
    base2: np.lib.npyio.NpzFile,
    split: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    ours_lookup = build_index_lookup(args.ours)
    b1_lookup = build_index_lookup(args.baseline1)
    b2_lookup = build_index_lookup(args.baseline2)

    dataset_indices = np.arange(y_true_all.shape[0], dtype=np.int64) if args.scope == "all" else split[args.scope]
    rows: List[Dict[str, float]] = []
    for ds_idx in dataset_indices.tolist():
        if ds_idx not in ours_lookup or ds_idx not in b1_lookup or ds_idx not in b2_lookup:
            continue
        op = ours_lookup[ds_idx]
        p1 = b1_lookup[ds_idx]
        p2 = b2_lookup[ds_idx]
        ours_rel = float(ours["rel_err"][op])
        b1_rel = float(base1["rel_err"][p1])
        b2_rel = float(base2["rel_err"][p2])
        gap1 = b1_rel - ours_rel
        gap2 = b2_rel - ours_rel
        rows.append(
            {
                "dataset_index": int(ds_idx),
                "ours_pos": int(op),
                "base1_pos": int(p1),
                "base2_pos": int(p2),
                "ours_rel": ours_rel,
                "base1_rel": b1_rel,
                "base2_rel": b2_rel,
                "gap1": gap1,
                "gap2": gap2,
                "gap_mean": 0.5 * (gap1 + gap2),
                "rock_type": int(rock_type[ds_idx]),
                "global_id": int(global_id[ds_idx]),
            }
        )

    if not rows:
        return []

    strong = [r for r in rows if r["gap1"] > 0.0 and r["gap2"] > 0.0]
    medium = [r for r in rows if r["gap_mean"] > 0.0]
    pool = strong if len(strong) >= args.topk else medium
    if not pool:
        pool = rows

    pool.sort(key=lambda x: (x["gap_mean"], x["gap1"], x["gap2"]), reverse=True)
    chosen = pool[: max(1, args.topk)]
    chosen.sort(key=lambda x: x["dataset_index"])
    return chosen


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ours = np.load(args.ours)
    base1 = np.load(args.baseline1)
    base2 = np.load(args.baseline2)
    with h5py.File(args.h5, "r") as f:
        y_true_all = f["Y"][:].astype(np.float32)
        x_all = f["X"][:].astype(np.float32)
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)

    split = load_split_indices(args.split_json, rock_type, global_id)
    selected = choose_samples(args, y_true_all, rock_type, global_id, ours, base1, base2, split)

    fieldnames = [
        "dataset_index",
        "rock_type",
        "global_id",
        "ours_rel",
        "base1_rel",
        "base2_rel",
        "gap1",
        "gap2",
        "gap_mean",
    ]
    with open(out_dir / "selected_cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fieldnames} for row in selected])

    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize, PowerNorm

    vel_cmap = velocity_colormap()
    err_cmap = error_colormap()

    n_rows = len(selected)
    fig_width = 9.6 if args.layout == "double" else 4.7
    row_height = 1.18 if args.layout == "double" else 1.3
    fig = plt.figure(figsize=(fig_width, row_height * n_rows + 1.05))
    gs = gridspec.GridSpec(
        nrows=n_rows + 1,
        ncols=7,
        height_ratios=[1.0] * n_rows + [0.10],
        hspace=0.14,
        wspace=0.06,
    )

    panel_titles = [
        "Reference",
        args.ours_label,
        args.baseline1_label,
        args.baseline2_label,
        f"{args.ours_label} Error",
        f"{args.baseline1_label} Error",
        f"{args.baseline2_label} Error",
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    vmax_global = 0.0
    emax_global = 0.0
    prepared = []
    for item in selected:
        ds_idx = item["dataset_index"]
        op = item["ours_pos"]
        p1 = item["base1_pos"]
        p2 = item["base2_pos"]
        mask = x_all[ds_idx, 0] > 0.5
        true_u = y_true_all[ds_idx, 0].astype(np.float32)
        ours_u = ours["pred_ux"][op, 0].astype(np.float32)
        base1_u = base1["pred_ux"][p1, 0].astype(np.float32)
        base2_u = base2["pred_ux"][p2, 0].astype(np.float32)
        ours_e = ours_u - true_u
        base1_e = base1_u - true_u
        base2_e = base2_u - true_u
        if np.any(mask):
            vmax_global = max(
                vmax_global,
                float(np.max(true_u[mask])),
                float(np.max(ours_u[mask])),
                float(np.max(base1_u[mask])),
                float(np.max(base2_u[mask])),
            )
            emax_global = max(
                emax_global,
                float(np.max(np.abs(ours_e[mask]))),
                float(np.max(np.abs(base1_e[mask]))),
                float(np.max(np.abs(base2_e[mask]))),
            )
        prepared.append((item, mask, true_u, ours_u, base1_u, base2_u, ours_e, base1_e, base2_e))

    vmax_global = max(vmax_global, 1.0e-8)
    emax_global = max(emax_global, 1.0e-8)

    for row, (item, mask, true_u, ours_u, base1_u, base2_u, ours_e, base1_e, base2_e) in enumerate(prepared):
        panels = [
            np.ma.masked_where(~mask, true_u),
            np.ma.masked_where(~mask, ours_u),
            np.ma.masked_where(~mask, base1_u),
            np.ma.masked_where(~mask, base2_u),
            np.ma.masked_where(~mask, ours_e),
            np.ma.masked_where(~mask, base1_e),
            np.ma.masked_where(~mask, base2_e),
        ]

        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            is_error = col >= 4
            cmap = err_cmap if is_error else vel_cmap
            if is_error:
                ax.imshow(panel, cmap=cmap, vmin=-emax_global, vmax=emax_global, interpolation="nearest")
            else:
                ax.imshow(panel, cmap=cmap, norm=PowerNorm(gamma=0.72, vmin=0.0, vmax=vmax_global), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("#4a4a4a")
            if row == 0:
                ax.set_title(f"{panel_labels[col]} {panel_titles[col]}", pad=7.0)
            if col == 0:
                label = (
                    f"#{row + 1}\n"
                    f"idx {item['dataset_index']}\n"
                    f"rock {item['rock_type']}\n"
                    f"{args.ours_alias} {item['ours_rel']:.3f}\n"
                    f"{args.baseline1_alias} {item['base1_rel']:.3f}\n"
                    f"{args.baseline2_alias} {item['base2_rel']:.3f}"
                )
                ax.text(
                    -0.31,
                    0.50,
                    label,
                    transform=ax.transAxes,
                    fontsize=7.0,
                    ha="right",
                    va="center",
                    color="#333333",
                )

    cax_vel = fig.add_subplot(gs[n_rows, 0:4])
    cax_err = fig.add_subplot(gs[n_rows, 4:7])
    ColorbarBase(cax_vel, cmap=vel_cmap, norm=PowerNorm(gamma=0.72, vmin=0.0, vmax=vmax_global), orientation="horizontal")
    cax_vel.set_xlabel(r"Velocity $u_x$")
    ColorbarBase(cax_err, cmap=err_cmap, norm=Normalize(vmin=-emax_global, vmax=emax_global), orientation="horizontal")
    cax_err.set_xlabel(r"Prediction Error $\hat{u}_x-u_x$")
    cax_vel.xaxis.set_label_position("bottom")
    cax_err.xaxis.set_label_position("bottom")

    png_path = out_dir / "velocity_field_comparison_paper.png"
    pdf_path = out_dir / "velocity_field_comparison_paper.pdf"
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.12, right=0.995)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    summary = {
        "scope": args.scope,
        "topk": int(args.topk),
        "selection_rule": "ours_better_than_baselines",
        "seed": int(args.seed),
        "layout": args.layout,
        "ours_label": args.ours_label,
        "baseline1_label": args.baseline1_label,
        "baseline2_label": args.baseline2_label,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    with open(out_dir / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved publication figure: {png_path}")
    print(f"saved publication figure: {pdf_path}")


if __name__ == "__main__":
    main()
