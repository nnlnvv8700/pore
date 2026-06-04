#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose where velocity-field and local-conductance errors concentrate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ROCK_NAMES = {
    0: "Bead",
    1: "Benth",
    2: "Berea_ICL",
    3: "Font18",
    4: "rock_type_4",
    5: "rock_type_5",
}


def read_split(split_json: Path, global_ids: np.ndarray) -> np.ndarray:
    split = json.loads(split_json.read_text(encoding="utf-8"))
    gid_to_split: dict[int, str] = {}
    for info in split["by_rock_type"].values():
        for name, key in [("train", "train_global_ids"), ("val", "val_global_ids"), ("test", "test_global_ids")]:
            for gid in info.get(key, []):
                gid_to_split[int(gid)] = name
    return np.asarray([gid_to_split.get(int(gid), "unknown") for gid in global_ids], dtype=object)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    den = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if den <= 1.0e-30:
        return float("nan")
    return 1.0 - float(np.sum((y_pred - y_true) ** 2)) / den


def summary_stats(vals: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantile_bin(values: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values[np.isfinite(values)], qs)
    edges = np.maximum.accumulate(edges)
    labels = np.full(values.shape, -1, dtype=np.int32)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (values >= lo) & (values <= hi)
        else:
            m = (values >= lo) & (values < hi)
        labels[m] = i
    return labels, edges


def load_conductance(path: Path, n: int) -> dict[str, np.ndarray]:
    out = {
        "g_table": np.full(n, np.nan, dtype=np.float64),
        "g_pred": np.full(n, np.nan, dtype=np.float64),
        "g_rel_error_table": np.full(n, np.nan, dtype=np.float64),
        "g_signed_error_table": np.full(n, np.nan, dtype=np.float64),
        "q_table": np.full(n, np.nan, dtype=np.float64),
        "q_pred_abs": np.full(n, np.nan, dtype=np.float64),
        "q_rel_error_table": np.full(n, np.nan, dtype=np.float64),
    }
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = int(float(row["dataset_index"]))
            if not (0 <= i < n):
                continue
            g_pred = float(row["g_pred"])
            out["g_pred"][i] = g_pred
            if row.get("g_table", "") != "":
                g_table = float(row["g_table"])
                out["g_table"][i] = g_table
                out["g_signed_error_table"][i] = (g_pred - g_table) / max(abs(g_table), 1.0e-30)
            if row.get("g_rel_error_table", "") != "":
                out["g_rel_error_table"][i] = float(row["g_rel_error_table"])
            if row.get("q_table", "") != "":
                out["q_table"][i] = float(row["q_table"])
            if row.get("q_pred_abs", "") != "":
                out["q_pred_abs"][i] = float(row["q_pred_abs"])
            if row.get("q_rel_error_table", "") != "":
                out["q_rel_error_table"][i] = float(row["q_rel_error_table"])
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
        "--conductance-csv",
        type=str,
        default=r"E:\mhw\1\cup\runs\lambda_selected_3seed_20260529\flux_0p2__dist_0p005__seed_42\res_ed\conductance_ipnm1_rho1\conductance_results.csv",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default=r"E:\mhw\1\cup\runs\final_ipnm1_flownet_flux010_30e\split_info.json",
    )
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\local_error_diagnosis_seed42_20260530")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_npz = np.load(args.pred_npz)
    pred = pred_npz["pred_ux"].astype(np.float32)
    with h5py.File(args.h5, "r") as f:
        x = f["X"][:].astype(np.float32)
        true = f["Y"][:].astype(np.float32)
        rock_type = f["rock_type"][:].astype(np.int32)
        global_id = f["global_id"][:].astype(np.int64)
        local_id = f["local_id"][:].astype(np.int64)
        scale_s = f["scale_s"][:].astype(np.float64)
        eta = f["eta"][:].astype(np.float64)
        ri = f["RI"][:].astype(np.float64)
        r0_ra = f["R0_RA"][:].astype(np.float64)
        q0 = f["q0"][:].astype(np.float64)

    mask = x[:, 0:1] > 0.5
    dist = x[:, 1:2].astype(np.float32)
    pred = pred * mask
    true = true * mask
    err = pred - true
    abs_err = np.abs(err)
    sq_err = err * err
    n = true.shape[0]
    split_name = read_split(Path(args.split_json), global_id)
    cond = load_conductance(Path(args.conductance_csv), n)

    pixel_rows: list[dict] = []
    q_pred_norm = np.sum(pred, axis=(1, 2, 3))
    q_true_norm = np.sum(true, axis=(1, 2, 3))
    for i in range(n):
        m = mask[i]
        yt = true[i][m]
        yp = pred[i][m]
        ae = np.abs(yp - yt)
        se = (yp - yt) ** 2
        denom = max(float(np.mean(np.abs(yt))), 1.0e-12)
        pixel_rows.append(
            {
                "dataset_index": i,
                "global_id": int(global_id[i]),
                "local_id": int(local_id[i]),
                "rock_type": int(rock_type[i]),
                "rock_name": ROCK_NAMES.get(int(rock_type[i]), str(int(rock_type[i]))),
                "split": str(split_name[i]),
                "pore_pixels": int(np.sum(m)),
                "eta": float(eta[i]),
                "RI": float(ri[i]),
                "R0_RA": float(r0_ra[i]),
                "scale_s": float(scale_s[i]),
                "q0": float(q0[i]),
                "q_true_norm": float(q_true_norm[i]),
                "q_pred_norm": float(q_pred_norm[i]),
                "q_signed_rel_error_norm": float((q_pred_norm[i] - q_true_norm[i]) / max(abs(float(q_true_norm[i])), 1.0e-30)),
                "pixel_mae": float(np.mean(ae)),
                "pixel_rmse": float(math.sqrt(float(np.mean(se)))),
                "pixel_nmae_mean_abs_u": float(np.mean(ae) / denom),
                "pixel_r2_sample": float(safe_r2(yt, yp)),
                "g_table": float(cond["g_table"][i]),
                "g_pred": float(cond["g_pred"][i]),
                "g_rel_error_table": float(cond["g_rel_error_table"][i]),
                "g_signed_error_table": float(cond["g_signed_error_table"][i]),
                "q_rel_error_table": float(cond["q_rel_error_table"][i]),
            }
        )

    write_csv(out_dir / "per_sample_error.csv", pixel_rows)

    # Group summaries.
    group_rows: list[dict] = []
    for group_name, values in [
        ("all", ["all"]),
        ("split", sorted(set(split_name.tolist()))),
        ("rock", sorted(set(int(x) for x in rock_type))),
    ]:
        for value in values:
            if group_name == "all":
                idx = np.ones(n, dtype=bool)
                label = "all"
            elif group_name == "split":
                idx = split_name == value
                label = str(value)
            else:
                idx = rock_type == int(value)
                label = ROCK_NAMES.get(int(value), str(value))
            m = mask[idx]
            yt = true[idx][m]
            yp = pred[idx][m]
            g_re = cond["g_rel_error_table"][idx]
            g_signed = cond["g_signed_error_table"][idx]
            q_re = cond["q_rel_error_table"][idx]
            group_rows.append(
                {
                    "group_type": group_name,
                    "group": label,
                    "n_samples": int(np.sum(idx)),
                    "pixel_r2": safe_r2(yt, yp),
                    "pixel_mae": float(np.mean(np.abs(yp - yt))),
                    "pixel_rmse": float(math.sqrt(float(np.mean((yp - yt) ** 2)))),
                    "g_re_mean": float(np.nanmean(g_re)),
                    "g_re_median": float(np.nanmedian(g_re)),
                    "g_re_p90": float(np.nanquantile(g_re, 0.90)),
                    "g_signed_mean": float(np.nanmean(g_signed)),
                    "g_over_frac": float(np.nanmean(g_signed > 0)),
                    "q_re_mean": float(np.nanmean(q_re)),
                }
            )
    write_csv(out_dir / "group_error_summary.csv", group_rows)

    # Geometry / label quantile bins for conductance errors.
    bin_rows: list[dict] = []
    features = {
        "pore_pixels": np.asarray([r["pore_pixels"] for r in pixel_rows], dtype=np.float64),
        "eta": eta,
        "RI": ri,
        "R0_RA": r0_ra,
        "g_table": cond["g_table"],
        "q0": q0,
        "scale_s": scale_s,
    }
    for feat_name, values in features.items():
        good = np.isfinite(values) & np.isfinite(cond["g_rel_error_table"])
        labels, edges = quantile_bin(values[good], 5)
        good_idx = np.where(good)[0]
        for b in range(5):
            idx_local = labels == b
            idx = good_idx[idx_local]
            if idx.size == 0:
                continue
            g_re = cond["g_rel_error_table"][idx]
            g_signed = cond["g_signed_error_table"][idx]
            sample_mae = np.asarray([pixel_rows[i]["pixel_mae"] for i in idx], dtype=np.float64)
            bin_rows.append(
                {
                    "feature": feat_name,
                    "bin": b + 1,
                    "range_min": float(edges[b]),
                    "range_max": float(edges[b + 1]),
                    "n_samples": int(idx.size),
                    "pixel_mae_mean": float(np.mean(sample_mae)),
                    "g_re_mean": float(np.nanmean(g_re)),
                    "g_re_median": float(np.nanmedian(g_re)),
                    "g_re_p90": float(np.nanquantile(g_re, 0.90)),
                    "g_signed_mean": float(np.nanmean(g_signed)),
                    "g_over_frac": float(np.nanmean(g_signed > 0)),
                }
            )
    write_csv(out_dir / "feature_bin_error_summary.csv", bin_rows)

    # Pixel-region summaries: distance-to-wall and true-velocity magnitude bins.
    region_rows: list[dict] = []
    total_sae = float(np.sum(abs_err[mask]))
    total_sse = float(np.sum(sq_err[mask]))
    dist_edges = [0.0, 0.025, 0.05, 0.10, 0.20, 1.0]
    for lo, hi in zip(dist_edges[:-1], dist_edges[1:]):
        region = mask & (dist >= lo) & (dist < hi if hi < 1.0 else dist <= hi)
        if not np.any(region):
            continue
        region_rows.append(
            {
                "region_type": "distance_to_wall",
                "bin": f"{lo:g}-{hi:g}",
                "n_pixels": int(np.sum(region)),
                "pixel_fraction": float(np.sum(region) / np.sum(mask)),
                "mae": float(np.mean(abs_err[region])),
                "rmse": float(math.sqrt(float(np.mean(sq_err[region])))),
                "share_abs_error": float(np.sum(abs_err[region]) / max(total_sae, 1.0e-30)),
                "share_sq_error": float(np.sum(sq_err[region]) / max(total_sse, 1.0e-30)),
                "true_u_mean": float(np.mean(true[region])),
            }
        )
    u_vals = true[mask].astype(np.float64)
    u_edges = np.quantile(u_vals, [0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
    for j in range(len(u_edges) - 1):
        lo, hi = u_edges[j], u_edges[j + 1]
        if j == len(u_edges) - 2:
            region = mask & (true >= lo) & (true <= hi)
        else:
            region = mask & (true >= lo) & (true < hi)
        if not np.any(region):
            continue
        region_rows.append(
            {
                "region_type": "true_velocity_quantile",
                "bin": f"q{j + 1}",
                "range_min": float(lo),
                "range_max": float(hi),
                "n_pixels": int(np.sum(region)),
                "pixel_fraction": float(np.sum(region) / np.sum(mask)),
                "mae": float(np.mean(abs_err[region])),
                "rmse": float(math.sqrt(float(np.mean(sq_err[region])))),
                "share_abs_error": float(np.sum(abs_err[region]) / max(total_sae, 1.0e-30)),
                "share_sq_error": float(np.sum(sq_err[region]) / max(total_sse, 1.0e-30)),
                "true_u_mean": float(np.mean(true[region])),
            }
        )
    write_csv(out_dir / "pixel_region_error_summary.csv", region_rows)

    # Worst cases by conductance and field error.
    sorted_g = sorted(pixel_rows, key=lambda r: (np.nan_to_num(r["g_rel_error_table"], nan=-1.0)), reverse=True)
    sorted_mae = sorted(pixel_rows, key=lambda r: r["pixel_mae"], reverse=True)
    write_csv(out_dir / "top_g_error_samples.csv", sorted_g[: args.top_k])
    write_csv(out_dir / "top_velocity_mae_samples.csv", sorted_mae[: args.top_k])

    # Plain-English report.
    by_rock = [r for r in group_rows if r["group_type"] == "rock"]
    by_rock_sorted = sorted(by_rock, key=lambda r: r["g_re_mean"], reverse=True)
    g_stats = summary_stats(cond["g_rel_error_table"])
    q_stats = summary_stats(cond["q_rel_error_table"])
    sample_mae = np.asarray([r["pixel_mae"] for r in pixel_rows], dtype=np.float64)
    corr_features = []
    for feat_name, values in features.items():
        good = np.isfinite(values) & np.isfinite(cond["g_rel_error_table"])
        if np.sum(good) > 3:
            corr = float(np.corrcoef(values[good], cond["g_rel_error_table"][good])[0, 1])
            corr_features.append((feat_name, corr))
    corr_features.sort(key=lambda x: abs(x[1]), reverse=True)

    report = [
        "# Local Error Diagnosis",
        "",
        "Scope: seed42 Ours checkpoint on `dataset_all_32.h5`; conductance metrics use the postprocessed table labels.",
        "",
        "## Overall",
        "",
        f"- Local conductance relative error: mean {100*g_stats['mean']:.2f}%, median {100*g_stats['median']:.2f}%, p90 {100*g_stats['p90']:.2f}%.",
        f"- Local flow relative error: mean {100*q_stats['mean']:.2f}%, median {100*q_stats['median']:.2f}%, p90 {100*q_stats['p90']:.2f}%.",
        f"- Per-sample velocity MAE: mean {np.mean(sample_mae):.3e}, p90 {np.quantile(sample_mae, 0.90):.3e}.",
        "",
        "## By Rock/Source",
        "",
        "| Source | n | velocity R2 | g RE mean | g RE p90 | signed g bias | over-pred frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in by_rock_sorted:
        report.append(
            f"| {r['group']} | {r['n_samples']} | {r['pixel_r2']:.4f} | "
            f"{100*r['g_re_mean']:.2f}% | {100*r['g_re_p90']:.2f}% | "
            f"{100*r['g_signed_mean']:.2f}% | {100*r['g_over_frac']:.1f}% |"
        )
    report += [
        "",
        "## Strongest Correlations With g Error",
        "",
    ]
    for name, corr in corr_features[:6]:
        report.append(f"- `{name}` vs g relative error: Pearson r = {corr:.3f}")
    report += [
        "",
        "## Pixel Error Localization",
        "",
        "See `pixel_region_error_summary.csv`. In general, compare `share_sq_error` with `pixel_fraction`: if the former is larger, that region contributes disproportionately to field error.",
        "",
        "## Files",
        "",
        "- `per_sample_error.csv`: per-sample velocity/q/g/geometry metrics",
        "- `group_error_summary.csv`: split and rock/source grouped metrics",
        "- `feature_bin_error_summary.csv`: geometry/label quantile-bin metrics",
        "- `pixel_region_error_summary.csv`: distance-to-wall and velocity-magnitude pixel bins",
        "- `top_g_error_samples.csv`: worst local conductance cases",
        "- `top_velocity_mae_samples.csv`: worst velocity-field cases",
    ]
    (out_dir / "LOCAL_ERROR_DIAGNOSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_dir / "LOCAL_ERROR_DIAGNOSIS.md")


if __name__ == "__main__":
    main()
