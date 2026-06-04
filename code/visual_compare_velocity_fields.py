#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
速度场论文级对比绘图。

输出内容：
1. True / Ours / Baseline / Ours Error / Baseline Error 对比图
2. 选中样本清单 CSV
3. PNG + PDF 双格式导出

设计目标：
- 统一色标，便于横向比较
- 误差图使用对称色标
- 尽量减少重复元素，适合论文正文直接插图
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


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


def build_index_lookup(npz_path: str) -> Dict[int, int]:
    z = np.load(npz_path)
    index = z["index"].astype(np.int64)
    return {int(ds_idx): int(pos) for pos, ds_idx in enumerate(index.tolist())}


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
            "lines.linewidth": 1.2,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def masked_velocity_colormap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap


def masked_error_colormap():
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap


def choose_samples(
    args: argparse.Namespace,
    Y: np.ndarray,
    rock_type: np.ndarray,
    global_id: np.ndarray,
    ours: np.lib.npyio.NpzFile,
    baseline: np.lib.npyio.NpzFile,
    split: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    ours_lookup = build_index_lookup(args.ours)
    base_lookup = build_index_lookup(args.baseline)
    if args.scope == "all":
        dataset_indices = np.arange(Y.shape[0], dtype=np.int64)
    else:
        dataset_indices = split[args.scope]

    candidates = []
    for ds_idx in dataset_indices.tolist():
        if ds_idx not in ours_lookup or ds_idx not in base_lookup:
            continue
        op = ours_lookup[ds_idx]
        bp = base_lookup[ds_idx]
        ours_rel = float(ours["rel_err"][op])
        base_rel = float(baseline["rel_err"][bp])
        candidates.append(
            {
                "dataset_index": ds_idx,
                "ours_pos": op,
                "base_pos": bp,
                "gap": base_rel - ours_rel,
                "ours_rel": ours_rel,
                "base_rel": base_rel,
                "rock_type": int(rock_type[ds_idx]),
                "global_id": int(global_id[ds_idx]),
            }
        )

    if args.selection == "gap":
        candidates.sort(key=lambda x: x["gap"], reverse=True)
    elif args.selection == "baseline":
        candidates.sort(key=lambda x: x["base_rel"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["ours_rel"], reverse=True)

    return candidates[: max(1, args.topk)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成适合论文使用的速度场对比图。")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--ours-label", type=str, default="Ours")
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--ours-alias", type=str, default="ours")
    parser.add_argument("--baseline-alias", type=str, default="base")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=6, help="展示样本数")
    parser.add_argument("--scope", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--selection", type=str, default="gap", choices=("gap", "baseline", "ours"))
    parser.add_argument("--layout", type=str, default="double", choices=("single", "double"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ours = np.load(args.ours)
    baseline = np.load(args.baseline)
    with h5py.File(args.h5, "r") as f:
        Y = f["Y"][:].astype(np.float32)
        X = f["X"][:].astype(np.float32)
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)

    split = load_split_indices(args.split_json, rock_type, global_id)
    selected = choose_samples(args, Y, rock_type, global_id, ours, baseline, split)

    fieldnames = ["dataset_index", "rock_type", "global_id", "ours_rel", "base_rel", "gap"]
    with open(out_dir / "selected_cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fieldnames} for row in selected])

    configure_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    vel_cmap = masked_velocity_colormap()
    err_cmap = masked_error_colormap()

    n_rows = len(selected)
    fig_width = 7.1 if args.layout == "double" else 3.45
    row_height = 1.25 if args.layout == "double" else 1.4
    fig = plt.figure(figsize=(fig_width, row_height * n_rows + 1.05))
    gs = gridspec.GridSpec(
        nrows=n_rows + 1,
        ncols=5,
        height_ratios=[1.0] * n_rows + [0.11],
        hspace=0.14,
        wspace=0.06,
    )

    panel_titles = [
        "Reference",
        args.ours_label,
        args.baseline_label,
        f"{args.ours_label} Error",
        f"{args.baseline_label} Error",
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    vmax_global = 0.0
    emax_global = 0.0
    prepared = []
    for item in selected:
        ds_idx = item["dataset_index"]
        op = item["ours_pos"]
        bp = item["base_pos"]
        mask = X[ds_idx, 0] > 0.5
        true_u = Y[ds_idx, 0].astype(np.float32)
        ours_u = ours["pred_ux"][op, 0].astype(np.float32)
        base_u = baseline["pred_ux"][bp, 0].astype(np.float32)
        ours_e = ours_u - true_u
        base_e = base_u - true_u
        vmax_global = max(vmax_global, float(np.max(true_u[mask])) if np.any(mask) else 0.0, float(np.max(ours_u[mask])) if np.any(mask) else 0.0, float(np.max(base_u[mask])) if np.any(mask) else 0.0)
        emax_global = max(
            emax_global,
            float(np.max(np.abs(ours_e[mask]))) if np.any(mask) else 0.0,
            float(np.max(np.abs(base_e[mask]))) if np.any(mask) else 0.0,
        )
        prepared.append((item, mask, true_u, ours_u, base_u, ours_e, base_e))

    vmax_global = max(vmax_global, 1.0e-8)
    emax_global = max(emax_global, 1.0e-8)

    for row, (item, mask, true_u, ours_u, base_u, ours_e, base_e) in enumerate(prepared):
        panels = [
            np.ma.masked_where(~mask, true_u),
            np.ma.masked_where(~mask, ours_u),
            np.ma.masked_where(~mask, base_u),
            np.ma.masked_where(~mask, ours_e),
            np.ma.masked_where(~mask, base_e),
        ]

        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            is_error = col >= 3
            cmap = err_cmap if is_error else vel_cmap
            if is_error:
                im = ax.imshow(panel, cmap=cmap, vmin=-emax_global, vmax=emax_global, interpolation="nearest")
            else:
                im = ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=vmax_global, interpolation="nearest")
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
                    f"{args.baseline_alias} {item['base_rel']:.3f}"
                )
                ax.text(
                    -0.26,
                    0.50,
                    label,
                    transform=ax.transAxes,
                    fontsize=7.2,
                    ha="right",
                    va="center",
                )

    cax_vel = fig.add_subplot(gs[n_rows, 0:3])
    cax_err = fig.add_subplot(gs[n_rows, 3:5])
    ColorbarBase(cax_vel, cmap=vel_cmap, norm=Normalize(vmin=0.0, vmax=vmax_global), orientation="horizontal")
    cax_vel.set_xlabel(r"Velocity $u_x$")
    ColorbarBase(cax_err, cmap=err_cmap, norm=Normalize(vmin=-emax_global, vmax=emax_global), orientation="horizontal")
    cax_err.set_xlabel(r"Prediction Error $\hat{u}_x-u_x$")
    cax_vel.xaxis.set_label_position("bottom")
    cax_err.xaxis.set_label_position("bottom")

    png_path = out_dir / "velocity_field_comparison_paper.png"
    pdf_path = out_dir / "velocity_field_comparison_paper.pdf"
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.99)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)

    summary = {
        "scope": args.scope,
        "topk": int(args.topk),
        "selection_rule": args.selection,
        "layout": args.layout,
        "ours_label": args.ours_label,
        "baseline_label": args.baseline_label,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    with open(out_dir / "visual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved publication figure: {png_path}")
    print(f"saved publication figure: {pdf_path}")


if __name__ == "__main__":
    main()
