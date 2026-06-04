#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run field-only inference on a geometry-only HDF5 dataset.

This script is intended for teacher-provided cross-section data that already
matches the training input format but does not contain valid ground-truth Y.
It reuses the trained U-Net and exports predicted velocity fields together with
the sample indices needed by later conductance / PNM steps.

Example:
python infer_unet_geom_only.py ^
  --h5 "E:\\mhw\\1\\pore\\teacher_model_inputs_full\\Bead\\teacher_geom_only.h5" ^
  --ckpt "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\best.pt" ^
  --out_dir "E:\\mhw\\1\\pore\\teacher_infer\\Bead" ^
  --save_field 1 --pred_format npz --field_dtype float16 ^
  --use_softplus --use_rock_type_channel --use_film
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from infer_unet_h5 import (
    PorousH5InferenceDataset,
    discover_meta_fields,
    load_checkpoint_model,
    read_channel_order,
    safe_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry-only U-Net inference and save predicted fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5", type=str, required=True, help="Geometry-only HDF5 path.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")

    parser.add_argument("--use_softplus", action="store_true", help="Use softplus output masking path.")
    parser.add_argument("--use_rock_type_channel", action="store_true", help="Append rock_type map as input channel.")
    parser.add_argument("--use_scale_head", dest="use_scale_head", action="store_true", help="Use scale head.")
    parser.add_argument("--no_scale_head", dest="use_scale_head", action="store_false", help="Disable scale head.")
    parser.set_defaults(use_scale_head=True)
    parser.add_argument("--use_film", dest="use_film", action="store_true", help="Enable FiLM.")
    parser.add_argument("--no_film", dest="use_film", action="store_false", help="Disable FiLM.")
    parser.set_defaults(use_film=None)
    parser.add_argument("--film_dim", type=int, default=None, help="FiLM embedding dimension.")
    parser.add_argument("--num_rock_types", type=int, default=None, help="Number of rock types.")
    parser.add_argument("--base_channels", type=int, default=None, help="U-Net base channels.")

    parser.add_argument("--save_field", type=int, default=1, choices=[0, 1], help="Save full pred_ux field.")
    parser.add_argument("--pred_format", type=str, default="npz", choices=["npz", "h5"], help="Prediction output format.")
    parser.add_argument("--field_dtype", type=str, default="float16", choices=["float16", "float32"], help="Field dtype.")
    return parser.parse_args()


def _load_meta_arrays(h5_path: str, fields: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for field in fields:
            if field in f:
                out[field] = f[field][...]
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    channel_order, mask_idx = read_channel_order(args.h5)
    if mask_idx is None:
        raise SystemExit("[ERROR] Failed to find 'mask' channel in HDF5 attrs.")

    device = safe_device(args.device)
    dataset = PorousH5InferenceDataset(
        h5_path=args.h5,
        mask_idx=mask_idx,
        use_rock_type_channel=args.use_rock_type_channel,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, ckpt_args = load_checkpoint_model(
        ckpt_path=args.ckpt,
        in_channels=dataset.in_channels,
        use_softplus=args.use_softplus,
        base_channels=args.base_channels,
        device=device,
        use_rock_type_channel=args.use_rock_type_channel,
        use_scale_head=args.use_scale_head,
        use_film=args.use_film,
        film_dim=args.film_dim,
        num_rock_types=args.num_rock_types,
    )
    model.to(device)
    model.eval()

    n = len(dataset)
    field_dtype = np.float16 if args.field_dtype == "float16" else np.float32
    pred_field = None
    pred_h5 = None
    pred_path = os.path.join(args.out_dir, f"predictions.{args.pred_format}")

    if args.save_field == 1:
        if args.pred_format == "npz":
            pred_field = np.zeros((n, 1, dataset.patch_size, dataset.patch_size), dtype=field_dtype)
        else:
            pred_h5 = h5py.File(pred_path, "w")
            pred_h5.create_dataset(
                "pred_ux",
                shape=(n, 1, dataset.patch_size, dataset.patch_size),
                dtype=field_dtype,
                compression="gzip",
            )

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            idx = batch["idx"].cpu().numpy().astype(np.int64)
            rock_type = batch["rock_type"]
            rock_type_dev = None
            if getattr(model, "use_film", False):
                rock_type_dev = rock_type.to(device, non_blocking=True)

            pred = model(x, mask=mask if args.use_softplus else None, rock_type=rock_type_dev)
            if not args.use_softplus:
                pred = pred * mask

            if args.save_field == 1:
                pred_np = pred.detach().cpu().numpy().astype(field_dtype, copy=False)
                if args.pred_format == "npz":
                    pred_field[idx] = pred_np
                else:
                    pred_h5["pred_ux"][idx] = pred_np

    meta_fields = ["rock_type", "global_id", "local_id", "scale_s"] + discover_meta_fields(args.h5, n)
    meta_fields = list(dict.fromkeys(meta_fields))
    meta_arrays = _load_meta_arrays(args.h5, meta_fields)
    index = np.arange(n, dtype=np.int64)

    if args.pred_format == "npz":
        save_dict = {"index": index}
        save_dict.update(meta_arrays)
        if args.save_field == 1 and pred_field is not None:
            save_dict["pred_ux"] = pred_field
        np.savez_compressed(pred_path, **save_dict)
    else:
        if pred_h5 is None:
            pred_h5 = h5py.File(pred_path, "w")
        pred_h5.create_dataset("index", data=index)
        for key, value in meta_arrays.items():
            pred_h5.create_dataset(key, data=value)
        if args.save_field == 0:
            pred_h5.flush()
        pred_h5.close()
        pred_h5 = None

    if pred_h5 is not None:
        pred_h5.close()

    summary = {
        "h5": args.h5,
        "ckpt": args.ckpt,
        "predictions": pred_path,
        "n_samples": n,
        "patch_size": int(dataset.patch_size),
        "in_channels": int(dataset.in_channels),
        "channel_order": channel_order,
        "mask_idx": int(mask_idx),
        "save_field": int(args.save_field),
        "pred_format": args.pred_format,
        "field_dtype": args.field_dtype,
        "device": str(device),
        "checkpoint_args": ckpt_args,
        "meta_fields": list(meta_arrays.keys()),
    }
    summary_path = os.path.join(args.out_dir, "infer_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
