#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Infer Res-IPNM1-FlowNet on teacher geometry-only HDF5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_architecture_comparison import (
    UnifiedDataset,
    apply_channel_stats,
    build_arrays,
    fit_channel_stats,
    fit_minmax2_stats,
    inverse_minmax2_torch,
    load_split_indices,
    make_model,
    transform_minmax2,
)


def compute_reference_scalars(train_h5: str) -> Dict[str, float]:
    with h5py.File(train_h5, "r") as f:
        eta = f["eta"][:].astype(np.float32) if "eta" in f else np.zeros(f["X"].shape[0], dtype=np.float32)
        ri = f["RI"][:].astype(np.float32) if "RI" in f else np.zeros(f["X"].shape[0], dtype=np.float32)
        r0_ra = f["R0_RA"][:].astype(np.float32) if "R0_RA" in f else np.zeros(f["X"].shape[0], dtype=np.float32)
    return {
        "eta_min": float(eta.min()),
        "eta_rng": max(float(eta.max() - eta.min()), 1.0e-8),
        "ri_min": float(ri.min()),
        "ri_rng": max(float(ri.max() - ri.min()), 1.0e-8),
        "r0_min": float(r0_ra.min()),
        "r0_rng": max(float(r0_ra.max() - r0_ra.min()), 1.0e-8),
    }


def patch_teacher_scalar_raw(arrays: Dict[str, np.ndarray], teacher_h5: str, ref: Dict[str, float]) -> None:
    with h5py.File(teacher_h5, "r") as f:
        eta = f["eta"][:].astype(np.float32) if "eta" in f else np.zeros(arrays["geom_raw"].shape[0], dtype=np.float32)
        ri = f["RI"][:].astype(np.float32) if "RI" in f else np.zeros(arrays["geom_raw"].shape[0], dtype=np.float32)
        r0_ra = f["R0_RA"][:].astype(np.float32) if "R0_RA" in f else np.zeros(arrays["geom_raw"].shape[0], dtype=np.float32)
        local_id = f["local_id"][:].astype(np.int32) if "local_id" in f else f["global_id"][:].astype(np.int32)
    eta_norm = (eta - ref["eta_min"]) / ref["eta_rng"]
    ri_norm = (ri - ref["ri_min"]) / ref["ri_rng"]
    r0_norm = (r0_ra - ref["r0_min"]) / ref["r0_rng"]
    arrays["scalar_raw"] = np.stack([eta_norm, ri_norm, r0_norm], axis=1).astype(np.float32)
    arrays["local_id"] = local_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-h5", type=str, required=True)
    parser.add_argument("--train-h5", type=str, default=r"E:\mhw\1\cup\dataset_all_32.h5")
    parser.add_argument("--split-json", type=str, default=r"E:\mhw\1\cup\runs\final_ipnm1_flownet_flux010_30e\split_info.json")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="res_ed")
    parser.add_argument("--base", type=int, default=20)
    parser.add_argument("--fno-width", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = build_arrays(args.train_h5)
    split = load_split_indices(args.split_json, train_arrays["rock_type"], train_arrays["global_id"])
    geom_stats = fit_channel_stats(train_arrays["geom_raw"], split["train"])
    branch_stats = fit_channel_stats(train_arrays["branch_raw"], split["train"])
    scalar_min = train_arrays["scalar_raw"][split["train"]].min(axis=0, keepdims=True)
    scalar_rng = np.maximum(train_arrays["scalar_raw"][split["train"]].max(axis=0, keepdims=True) - scalar_min, 1.0e-8)
    y_stats = fit_minmax2_stats(train_arrays["y_raw"][split["train"]])
    ref_scalars = compute_reference_scalars(args.train_h5)

    arrays = build_arrays(args.teacher_h5)
    patch_teacher_scalar_raw(arrays, args.teacher_h5, ref_scalars)
    arrays["geom_t"] = apply_channel_stats(arrays["geom_raw"], geom_stats)
    arrays["branch_t"] = apply_channel_stats(arrays["branch_raw"], branch_stats)
    arrays["scalar_t"] = ((arrays["scalar_raw"] - scalar_min) * 2.0 / scalar_rng - 1.0).astype(np.float32)
    arrays["y_t"] = transform_minmax2(arrays["y_raw"], y_stats)

    ds = UnifiedDataset(arrays, np.arange(arrays["y_raw"].shape[0], dtype=np.int64))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    _, _, h, w = arrays["geom_t"].shape
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = make_model(args.model, arrays["geom_t"].shape[1], h, w, args.base, args.fno_width).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_field = np.zeros_like(arrays["y_raw"], dtype=np.float16)
    q_pred_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            geom = batch["geom"].to(device)
            mask = batch["mask"].to(device)
            pred_t, _ = model(geom, batch["branch"].to(device), batch["scalar"].to(device))
            pred_raw = inverse_minmax2_torch(pred_t, y_stats) * mask
            idx = np.asarray(batch["index"], dtype=np.int64)
            pred_np = pred_raw.detach().cpu().numpy()
            pred_field[idx] = pred_np.astype(np.float16)
            q_pred_all.append((pred_raw * mask).sum(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32))

    index = np.arange(pred_field.shape[0], dtype=np.int32)
    y_dummy = arrays["y_raw"].astype(np.float32)
    q_true_dummy = (y_dummy * arrays["mask"]).sum(axis=(1, 2, 3)).astype(np.float32)
    q_pred = np.concatenate(q_pred_all, axis=0)
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred_ux=pred_field,
        q_true=q_true_dummy,
        q_pred=q_pred,
        index=index,
        rock_type=arrays["rock_type"].astype(np.int16),
        global_id=arrays["global_id"].astype(np.int32),
        local_id=arrays["local_id"].astype(np.int32),
    )
    summary = {
        "teacher_h5": args.teacher_h5,
        "train_h5": args.train_h5,
        "ckpt": args.ckpt,
        "n_samples": int(pred_field.shape[0]),
        "model": args.model,
        "device": str(device),
        "predictions": str(out_dir / "predictions.npz"),
    }
    (out_dir / "infer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_dir / "predictions.npz")


if __name__ == "__main__":
    main()
