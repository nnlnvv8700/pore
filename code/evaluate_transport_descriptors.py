#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate transport-relevant descriptors from predicted and reference velocity fields.

This is a lightweight bridge for the IPNM1 manuscript story:
conductance-only IPNM stores a scalar g, while the neural surrogate preserves a
cross-section velocity distribution that can support later transport modeling.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare transport descriptors derived from predicted and true velocity fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred-file", type=str, required=True, help="NPZ file from infer_unet_h5.py.")
    parser.add_argument("--pred-key", type=str, default="pred_ux", help="Prediction array key.")
    parser.add_argument("--h5", type=str, required=True, help="HDF5 dataset containing Y, X, and metadata.")
    parser.add_argument("--true-key", type=str, default="Y", help="True velocity dataset key in HDF5.")
    parser.add_argument("--mask-key", type=str, default="X", help="Input tensor dataset key containing mask channel.")
    parser.add_argument("--mask-channel", type=int, default=0, help="Mask channel index in X.")
    parser.add_argument("--index-key", type=str, default="index", help="Prediction index key in NPZ.")
    parser.add_argument("--low-frac", type=float, default=0.10, help="Low-velocity threshold as fraction of mean pore velocity.")
    parser.add_argument("--high-frac", type=float, default=2.0, help="High-velocity threshold as fraction of mean pore velocity.")
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Small epsilon for divisions.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    return parser.parse_args()


def ensure_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None, :, :]
    raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def load_predictions(path: Path, key: str, index_key: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Prediction key '{key}' not found in {path}. Available: {list(data.keys())}")
    pred = ensure_nchw(np.asarray(data[key], dtype=np.float32))
    indices = np.asarray(data[index_key], dtype=np.int64) if index_key in data else None
    return pred, indices


def descriptor(u: np.ndarray, mask: np.ndarray, low_frac: float, high_frac: float, eps: float) -> Dict[str, float]:
    pore = mask > 0.5
    vals = np.asarray(u[pore], dtype=np.float64)
    if vals.size == 0:
        return {
            "pore_area": 0.0,
            "q": float("nan"),
            "mean_u": float("nan"),
            "std_u": float("nan"),
            "cv_u": float("nan"),
            "p10_u": float("nan"),
            "p50_u": float("nan"),
            "p90_u": float("nan"),
            "low_velocity_fraction": float("nan"),
            "high_velocity_fraction": float("nan"),
            "residence_proxy": float("nan"),
        }
    q = float(np.sum(vals))
    mean_u = float(np.mean(vals))
    std_u = float(np.std(vals))
    low_thr = low_frac * max(mean_u, eps)
    high_thr = high_frac * max(mean_u, eps)
    return {
        "pore_area": float(vals.size),
        "q": q,
        "mean_u": mean_u,
        "std_u": std_u,
        "cv_u": float(std_u / max(abs(mean_u), eps)),
        "p10_u": float(np.percentile(vals, 10)),
        "p50_u": float(np.percentile(vals, 50)),
        "p90_u": float(np.percentile(vals, 90)),
        "low_velocity_fraction": float(np.mean(vals <= low_thr)),
        "high_velocity_fraction": float(np.mean(vals >= high_thr)),
        "residence_proxy": float(np.mean(1.0 / (np.abs(vals) + eps))),
    }


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> Dict[str, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mean_rel_error": float("nan"), "r2": float("nan")}
    t = y_true[finite].astype(np.float64)
    p = y_pred[finite].astype(np.float64)
    err = p - t
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    return {
        "n": int(t.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_rel_error": float(np.mean(np.abs(err) / (np.abs(t) + eps))),
        "median_rel_error": float(np.median(np.abs(err) / (np.abs(t) + eps))),
        "r2": float(1.0 - ss_res / max(ss_tot, eps)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred, indices = load_predictions(Path(args.pred_file), args.pred_key, args.index_key)
    n_pred = pred.shape[0]

    rows: List[Dict[str, object]] = []
    with h5py.File(args.h5, "r") as f:
        if indices is None:
            indices = np.arange(n_pred, dtype=np.int64)
        y_all = f[args.true_key]
        x_all = f[args.mask_key]
        rock_type = f["rock_type"][:] if "rock_type" in f else None
        local_id = f["local_id"][:] if "local_id" in f else None
        global_id = f["global_id"][:] if "global_id" in f else None

        if len(indices) != n_pred:
            raise ValueError(f"index length {len(indices)} does not match predictions {n_pred}")

        for row_i, h5_i in enumerate(indices.tolist()):
            u_pred = pred[row_i, 0]
            u_true = np.asarray(y_all[h5_i][0], dtype=np.float32)
            mask = np.asarray(x_all[h5_i][args.mask_channel], dtype=np.float32)
            d_true = descriptor(u_true, mask, args.low_frac, args.high_frac, args.eps)
            d_pred = descriptor(u_pred, mask, args.low_frac, args.high_frac, args.eps)

            row: Dict[str, object] = {
                "row": row_i,
                "index": int(h5_i),
                "rock_type": int(rock_type[h5_i]) if rock_type is not None else "",
                "local_id": int(local_id[h5_i]) if local_id is not None else "",
                "global_id": int(global_id[h5_i]) if global_id is not None else "",
            }
            for key, value in d_true.items():
                row[f"true_{key}"] = value
            for key, value in d_pred.items():
                row[f"pred_{key}"] = value
                true_value = d_true.get(key)
                if isinstance(true_value, float) and np.isfinite(true_value) and np.isfinite(value):
                    row[f"abs_err_{key}"] = float(abs(float(value) - true_value))
                    row[f"rel_err_{key}"] = float(abs(float(value) - true_value) / (abs(true_value) + args.eps))
            rows.append(row)

    csv_path = out_dir / "transport_descriptors.csv"
    fieldnames = list(rows[0].keys()) if rows else ["row", "index"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    descriptor_names = [
        "q",
        "mean_u",
        "std_u",
        "cv_u",
        "p10_u",
        "p50_u",
        "p90_u",
        "low_velocity_fraction",
        "high_velocity_fraction",
        "residence_proxy",
    ]
    summary = {
        "pred_file": args.pred_file,
        "h5": args.h5,
        "n_samples": len(rows),
        "low_frac": args.low_frac,
        "high_frac": args.high_frac,
        "metrics": {},
    }
    for name in descriptor_names:
        y_true = np.asarray([float(r.get(f"true_{name}", np.nan)) for r in rows], dtype=np.float64)
        y_pred = np.asarray([float(r.get(f"pred_{name}", np.nan)) for r in rows], dtype=np.float64)
        summary["metrics"][name] = metric_summary(y_true, y_pred, args.eps)

    summary_path = out_dir / "transport_descriptor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved transport descriptors: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
