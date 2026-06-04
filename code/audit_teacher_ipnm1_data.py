#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit teacher 300x200x200 IPNM1 geometry and conductance data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


ROCKS = [
    ("Bead", "Bead"),
    ("Benth", "Benth"),
    ("Berea_ICL", "Berea"),
    ("Font18", "Font"),
]


def parse_cross_section(path: Path) -> np.ndarray | None:
    tokens = path.read_text(encoding="utf-8", errors="ignore").split()
    vals = np.asarray([int(x) for x in tokens], dtype=np.int32)
    if vals.size < 4:
        return None
    flag = int(vals[0])
    ny = int(vals[1])
    nz = int(vals[2])
    if flag <= 0:
        return None
    grid = vals[3:]
    if grid.size != ny * nz:
        raise ValueError(f"Grid size mismatch in {path}: expected {ny*nz}, got {grid.size}")
    solid = grid.reshape((nz, ny))
    return (solid == 0).astype(np.uint8)


def iter_cross_sections(folder: Path) -> List[tuple[int, Path]]:
    out = []
    for path in sorted(folder.glob("Cross_section_sub*.in")):
        if "(" in path.name:
            continue
        local_id = int(path.stem.replace("Cross_section_sub", ""))
        out.append((local_id, path))
    return out


def read_conductivity(path: Path) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        toks = line.replace(",", " ").split()
        if not toks:
            continue
        try:
            values[i] = float(toks[0])
        except ValueError:
            continue
    return values


def stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"min": math.nan, "median": math.nan, "mean": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def audit_one(data_root: Path, rock_dir: str, pnm_dir: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    cs_dir = data_root / rock_dir / "out_throat_minimum_cross"
    cond_path = data_root / "PNM_simulation" / pnm_dir / "conductivity_throats.dat"
    sections = iter_cross_sections(cs_dir)
    cond = read_conductivity(cond_path)
    pore_pixels: List[int] = []
    heights: List[int] = []
    widths: List[int] = []
    aspect: List[float] = []
    matched = 0
    invalid = 0
    for local_id, path in sections:
        mask = parse_cross_section(path)
        if mask is None:
            invalid += 1
            continue
        h, w = mask.shape
        pore = int(mask.sum())
        pore_pixels.append(pore)
        heights.append(h)
        widths.append(w)
        aspect.append(float(w / max(h, 1)))
        g = cond.get(local_id)
        if g is not None:
            matched += 1
        rows.append(
            {
                "rock": rock_dir,
                "pnm_rock": pnm_dir,
                "local_id": local_id,
                "height": h,
                "width": w,
                "pore_pixels": pore,
                "pore_fraction": pore / float(h * w),
                "conductivity_throats_g": "" if g is None else g,
            }
        )
    g_vals = [v for _, v in sorted(cond.items())]
    return {
        "rock": rock_dir,
        "pnm_rock": pnm_dir,
        "n_cross_sections": len(sections),
        "n_valid_cross_sections": len(pore_pixels),
        "n_invalid_or_disabled": invalid,
        "n_conductivity_rows": len(cond),
        "n_matched_by_local_id": matched,
        "all_valid_cross_sections_have_conductivity": bool(matched == len(pore_pixels)),
        "conductivity_rows_include_disabled_sections": bool(len(cond) > len(pore_pixels)),
        "height": stats([float(v) for v in heights]),
        "width": stats([float(v) for v in widths]),
        "pore_pixels": stats([float(v) for v in pore_pixels]),
        "pore_fraction": stats([float(v) / float(h * w) for v, h, w in zip(pore_pixels, heights, widths)]),
        "aspect_width_over_height": stats(aspect),
        "conductivity_g": stats(g_vals),
    }


def plot_summary(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rocks = sorted({str(r["rock"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    data_pore = [[float(r["pore_pixels"]) for r in rows if r["rock"] == rock] for rock in rocks]
    data_g = [[float(r["conductivity_throats_g"]) for r in rows if r["rock"] == rock and r["conductivity_throats_g"] != ""] for rock in rocks]
    axes[0].boxplot(data_pore, labels=rocks, showfliers=False)
    axes[0].set_title("Cross-section pore pixels")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].boxplot(data_g, labels=rocks, showfliers=False)
    axes[1].set_title("Teacher IPNM1 conductance")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_ipnm1_data_audit.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit teacher 300x200x200 IPNM1 data.")
    parser.add_argument("--data-root", type=str, default=r"E:\mhw\1\cup\300x200x200_data")
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\teacher_ipnm1_audit_20260527")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    summary = [audit_one(data_root, rock_dir, pnm_dir, rows) for rock_dir, pnm_dir in ROCKS]

    with (out_dir / "teacher_ipnm1_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "teacher_ipnm1_audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rock",
            "pnm_rock",
            "n_cross_sections",
            "n_valid_cross_sections",
            "n_conductivity_rows",
            "n_matched_by_local_id",
            "all_valid_cross_sections_have_conductivity",
            "conductivity_rows_include_disabled_sections",
            "pore_pixels_median",
            "pore_pixels_mean",
            "pore_fraction_median",
            "conductivity_g_median",
            "conductivity_g_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            writer.writerow(
                {
                    "rock": item["rock"],
                    "pnm_rock": item["pnm_rock"],
                    "n_cross_sections": item["n_cross_sections"],
                    "n_valid_cross_sections": item["n_valid_cross_sections"],
                    "n_conductivity_rows": item["n_conductivity_rows"],
                    "n_matched_by_local_id": item["n_matched_by_local_id"],
                    "all_valid_cross_sections_have_conductivity": item["all_valid_cross_sections_have_conductivity"],
                    "conductivity_rows_include_disabled_sections": item["conductivity_rows_include_disabled_sections"],
                    "pore_pixels_median": item["pore_pixels"]["median"],
                    "pore_pixels_mean": item["pore_pixels"]["mean"],
                    "pore_fraction_median": item["pore_fraction"]["median"],
                    "conductivity_g_median": item["conductivity_g"]["median"],
                    "conductivity_g_mean": item["conductivity_g"]["mean"],
                }
            )
    (out_dir / "teacher_ipnm1_audit_summary.json").write_text(
        json.dumps({"rocks": summary, "n_total_samples": len(rows)}, indent=2),
        encoding="utf-8",
    )
    plot_summary(out_dir, rows)
    print(out_dir / "teacher_ipnm1_audit_summary.csv")


if __name__ == "__main__":
    main()
