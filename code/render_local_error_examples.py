#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render velocity-field examples for local error diagnosis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_top_indices(path: Path, n: int) -> list[int]:
    out = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(int(row["dataset_index"]))
            if len(out) >= n:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, default=r"E:\mhw\1\cup\dataset_all_32.h5")
    parser.add_argument(
        "--pred-npz",
        type=str,
        default=r"E:\mhw\1\cup\runs\lambda_selected_3seed_20260529\flux_0p2__dist_0p005__seed_42\res_ed\predictions.npz",
    )
    parser.add_argument(
        "--top-csv",
        type=str,
        default=r"E:\mhw\1\cup\runs\local_error_diagnosis_seed42_20260530\top_g_error_samples.csv",
    )
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\local_error_diagnosis_seed42_20260530")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = read_top_indices(Path(args.top_csv), args.n)
    pred = np.load(args.pred_npz)["pred_ux"].astype(np.float32)
    with h5py.File(args.h5, "r") as f:
        true = f["Y"][:].astype(np.float32)
        mask = f["X"][:, 0].astype(np.float32)

    fig, axes = plt.subplots(len(idxs), 4, figsize=(10, 2.3 * len(idxs)), constrained_layout=True)
    if len(idxs) == 1:
        axes = axes[None, :]
    for row_i, idx in enumerate(idxs):
        m = mask[idx] > 0.5
        t = true[idx, 0] * m
        p = pred[idx, 0] * m
        e = (p - t) * m
        vmax = max(float(np.max(t)), float(np.max(p)), 1.0e-12)
        panels = [
            (m.astype(float), "mask", "gray", None, None),
            (t, f"true #{idx}", "viridis", 0.0, vmax),
            (p, "pred", "viridis", 0.0, vmax),
            (e, "pred-true", "coolwarm", -max(abs(float(np.min(e))), abs(float(np.max(e)))), max(abs(float(np.min(e))), abs(float(np.max(e))))),
        ]
        for ax, (arr, title, cmap, vmin, vmax_i) in zip(axes[row_i], panels):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax_i)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if title in {"pred-true"}:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = out_dir / "top_g_error_velocity_examples.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
