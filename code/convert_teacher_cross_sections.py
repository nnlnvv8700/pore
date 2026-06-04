#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert teacher-provided Cross_section_sub*.in files into model-ready HDF5 tensors.

Input convention confirmed by the user:
1. First integer is a run flag.
2. Next two integers are ny, nz.
3. The remaining ny*nz integers are the voxel grid.
4. Grid value 1 means solid, 0 means pore.

This script mirrors the geometry-only part of build_hdf5_dataset.py:
    raw mask -> centroid recenter -> radius normalization -> dist/eta_map

It writes a geometry-only HDF5 file with the same X channel order as the
training dataset:
    X[:, 0] = mask
    X[:, 1] = dist
    X[:, 2] = eta_map

Y is written as zeros only to satisfy existing HDF5 loaders. Do not use those
zero targets for evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom


ROCK_NAME_TO_ID = {
    "Bead": 0,
    "Benth": 1,
    "Berea_ICL": 2,
    "Font18": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert teacher Cross_section_sub*.in files into model-ready HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"E:\mhw\1\pore\results\喉道资料\2025大创\300x200x200_data",
        help="Root directory containing teacher rock folders.",
    )
    parser.add_argument(
        "--rocks",
        nargs="+",
        default=["Bead"],
        help="Rock folders to convert.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"E:\mhw\1\pore\teacher_model_inputs",
        help="Output directory for converted geometry HDF5 files.",
    )
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--raw-patch", type=int, default=64)
    parser.add_argument("--target-r1", type=float, default=10.0)
    parser.add_argument("--min-scale", type=float, default=0.1, help="Lower clamp for geometry scaling factor.")
    parser.add_argument("--max-scale", type=float, default=10.0, help="Upper clamp for geometry scaling factor.")
    parser.add_argument(
        "--adaptive-threshold",
        type=int,
        default=0,
        help="If >0, adaptive target-R1 scaling is applied only when raw pore pixels <= threshold.",
    )
    parser.add_argument(
        "--non-tiny-scale",
        type=float,
        default=0.0,
        help="If >0, pores larger than adaptive-threshold use this fixed scale instead of adaptive target-R1 scaling.",
    )
    parser.add_argument(
        "--non-tiny-max-scale",
        type=float,
        default=0.0,
        help="Optional extra cap for pores larger than adaptive-threshold. 0 disables the cap.",
    )
    parser.add_argument(
        "--piecewise-scale-caps",
        type=str,
        default="",
        help="Optional comma-separated raw_pore_pixels:max_scale caps, e.g. '10:8,30:3,1000000:1.5'. "
             "Each cap applies to pores with raw_pore_pixels <= threshold, using the first matching rule.",
    )
    parser.add_argument(
        "--tiny-pore-threshold",
        type=int,
        default=0,
        help="If >0, pores with raw pore pixels <= threshold are treated as tiny pores.",
    )
    parser.add_argument(
        "--tiny-scale-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "fixed"],
        help="Scaling strategy applied to tiny pores only.",
    )
    parser.add_argument(
        "--fixed-scale",
        type=float,
        default=1.0,
        help="Fixed scaling factor used when tiny-scale-mode=fixed.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional limit per rock.")
    parser.add_argument("--viz-samples", type=int, default=9, help="How many samples to render as a preview grid.")
    return parser.parse_args()


def parse_cross_section_file(path: Path) -> np.ndarray | None:
    tokens = path.read_text(encoding="utf-8", errors="ignore").split()
    data = np.asarray([int(tok) for tok in tokens], dtype=np.int32)
    if data.ndim != 1 or data.size < 4:
        raise ValueError(f"Unexpected file content: {path}")
    flag = int(data[0])
    ny = int(data[1])
    nz = int(data[2])
    if flag <= 0:
        return None
    grid = data[3:]
    if grid.size != ny * nz:
        raise ValueError(f"Grid size mismatch in {path}: expected {ny*nz}, got {grid.size}")
    solid = grid.reshape((nz, ny))
    if not np.all(np.isin(solid, [0, 1])):
        raise ValueError(f"Found values other than 0/1 in {path}")
    pore_mask = (solid == 0).astype(np.uint8)
    return pore_mask


def center_crop_or_pad(img: np.ndarray, out_hw: Tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    h, w = img.shape
    out_h, out_w = out_hw
    y0 = max((h - out_h) // 2, 0)
    x0 = max((w - out_w) // 2, 0)
    cropped = img[y0:y0 + min(out_h, h), x0:x0 + min(out_w, w)]
    ch, cw = cropped.shape
    pad_top = max((out_h - ch) // 2, 0)
    pad_bottom = max(out_h - ch - pad_top, 0)
    pad_left = max((out_w - cw) // 2, 0)
    pad_right = max(out_w - cw - pad_left, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    return cropped.astype(img.dtype, copy=False)


def compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return h / 2.0, w / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def shift_with_zero_pad(img: np.ndarray, dy: int, dx: int, pad_value: float = 0.0) -> np.ndarray:
    h, w = img.shape
    out = np.full((h, w), pad_value, dtype=img.dtype)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def recenter_mask(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    cy, cx = compute_centroid(mask)
    center_y, center_x = mask.shape[0] / 2.0, mask.shape[1] / 2.0
    dy = int(round(center_y - cy))
    dx = int(round(center_x - cx))
    shifted = shift_with_zero_pad(mask.astype(np.float32), dy, dx, pad_value=0.0)
    return (shifted > 0.5).astype(np.uint8), dy, dx


def normalize_geometry(
    mask: np.ndarray,
    target_r1: float,
    patch_size: int,
    min_scale: float,
    max_scale: float,
    override_scale: float | None = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    area = float(mask.sum())
    if area < 1.0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8), 0.0, 0.0, 0.0, 1.0
    ra = math.sqrt(area / math.pi)
    ri = float(distance_transform_edt(mask.astype(bool)).max())
    eta = float(ri / (ra + 1.0e-8))
    if override_scale is not None:
        scale_s = float(override_scale)
    else:
        scale_s = float(target_r1 / max(ra, 1.0e-8))
    scale_s = max(min(scale_s, float(max_scale)), float(min_scale))
    mask_zoom = zoom(mask.astype(np.float32), zoom=(scale_s, scale_s), order=0, mode="constant", cval=0.0)
    mask_norm = center_crop_or_pad(mask_zoom, (patch_size, patch_size), pad_value=0.0)
    mask_norm = (mask_norm > 0.5).astype(np.uint8)
    return mask_norm, ra, ri, eta, scale_s


def parse_piecewise_scale_caps(spec: str) -> List[Tuple[int, float]]:
    rules: List[Tuple[int, float]] = []
    text = str(spec).strip()
    if not text:
        return rules
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        threshold_raw, cap_raw = item.split(":", 1)
        threshold = int(threshold_raw.strip())
        cap = float(cap_raw.strip())
        rules.append((threshold, cap))
    rules.sort(key=lambda x: x[0])
    return rules


def lookup_piecewise_cap(raw_pore_pixels: int, rules: Sequence[Tuple[int, float]]) -> float | None:
    for threshold, cap in rules:
        if int(raw_pore_pixels) <= int(threshold):
            return float(cap)
    return None


def build_x_tensor(mask_norm: np.ndarray, eta: float, target_r1: float) -> np.ndarray:
    dist = distance_transform_edt(mask_norm.astype(bool)).astype(np.float32)
    dist = dist / max(target_r1, 1.0e-6)
    dist = np.clip(dist, 0.0, 2.0)
    eta_map = np.full(mask_norm.shape, eta, dtype=np.float32)
    return np.stack(
        [
            mask_norm.astype(np.float32),
            dist.astype(np.float32),
            eta_map,
        ],
        axis=0,
    )


def maybe_render_preview(samples: Sequence[Dict[str, object]], out_path: Path) -> None:
    if not samples:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row_idx, item in enumerate(samples):
        x = item["X"]
        axes[row_idx, 0].imshow(x[0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx, 0].set_title(f"mask id={item['local_id']}")
        axes[row_idx, 1].imshow(x[1], cmap="viridis")
        axes[row_idx, 1].set_title("dist")
        axes[row_idx, 2].imshow(x[2], cmap="magma")
        axes[row_idx, 2].set_title(f"eta={item['eta']:.3f}")
        for col in range(3):
            axes[row_idx, col].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def iter_cross_section_files(folder: Path) -> Iterable[Tuple[int, Path]]:
    files = sorted(folder.glob("Cross_section_sub*.in"))
    for path in files:
        if "(" in path.name:
            continue
        sub_id = int(path.stem.replace("Cross_section_sub", ""))
        yield sub_id, path


def convert_one_rock(
    data_root: Path,
    out_root: Path,
    rock: str,
    patch_size: int,
    raw_patch: int,
    target_r1: float,
    min_scale: float,
    max_scale: float,
    adaptive_threshold: int,
    non_tiny_scale: float,
    non_tiny_max_scale: float,
    piecewise_scale_caps: Sequence[Tuple[int, float]],
    tiny_pore_threshold: int,
    tiny_scale_mode: str,
    fixed_scale: float,
    max_samples: int,
    viz_samples: int,
) -> Dict[str, object]:
    rock_id = ROCK_NAME_TO_ID.get(rock, 0)
    src_dir = data_root / rock / "out_throat_minimum_cross"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source dir: {src_dir}")
    out_dir = out_root / rock
    out_dir.mkdir(parents=True, exist_ok=True)

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rock_type_list: List[int] = []
    global_id_list: List[int] = []
    local_id_list: List[int] = []
    eta_list: List[float] = []
    ri_list: List[float] = []
    ra_list: List[float] = []
    scale_list: List[float] = []
    raw_pore_pixels_list: List[int] = []
    tiny_scaled_flag_list: List[int] = []
    dims_list: List[Tuple[int, int]] = []
    preview_items: List[Dict[str, object]] = []

    processed = 0
    for sub_id, path in iter_cross_section_files(src_dir):
        if max_samples > 0 and processed >= max_samples:
            break
        mask = parse_cross_section_file(path)
        if mask is None:
            continue
        dims_list.append((int(mask.shape[0]), int(mask.shape[1])))
        raw_pore_pixels = int(mask.sum())
        mask_raw = center_crop_or_pad(mask, (max(raw_patch, mask.shape[0]), max(raw_patch, mask.shape[1])), pad_value=0).astype(np.uint8)
        mask_centered, _, _ = recenter_mask(mask_raw)
        use_fixed_tiny_scale = (
            int(tiny_pore_threshold) > 0
            and raw_pore_pixels <= int(tiny_pore_threshold)
            and str(tiny_scale_mode).lower() == "fixed"
        )
        use_adaptive_gate = (
            int(adaptive_threshold) > 0
            and raw_pore_pixels > int(adaptive_threshold)
        )
        override_scale = None
        local_max_scale = float(max_scale)
        piecewise_cap = lookup_piecewise_cap(raw_pore_pixels, piecewise_scale_caps)
        if piecewise_cap is not None and float(piecewise_cap) > 0.0:
            local_max_scale = min(local_max_scale, float(piecewise_cap))
        if use_fixed_tiny_scale:
            override_scale = float(fixed_scale)
        elif use_adaptive_gate:
            if float(non_tiny_scale) > 0.0:
                override_scale = float(non_tiny_scale)
            if float(non_tiny_max_scale) > 0.0:
                local_max_scale = min(local_max_scale, float(non_tiny_max_scale))
        mask_norm, ra, ri, eta, scale_s = normalize_geometry(
            mask_centered,
            target_r1=target_r1,
            patch_size=patch_size,
            min_scale=min_scale,
            max_scale=local_max_scale,
            override_scale=override_scale,
        )
        x_tensor = build_x_tensor(mask_norm, eta=eta, target_r1=target_r1)
        y_dummy = np.zeros((1, patch_size, patch_size), dtype=np.float32)

        x_list.append(x_tensor)
        y_list.append(y_dummy)
        rock_type_list.append(int(rock_id))
        global_id_list.append(int(sub_id))
        local_id_list.append(int(sub_id))
        eta_list.append(float(eta))
        ri_list.append(float(ri))
        ra_list.append(float(ra))
        scale_list.append(float(scale_s))
        raw_pore_pixels_list.append(int(raw_pore_pixels))
        tiny_scaled_flag_list.append(1 if use_fixed_tiny_scale else 0)

        if len(preview_items) < viz_samples:
            preview_items.append(
                {
                    "local_id": sub_id,
                    "eta": eta,
                    "X": x_tensor,
                }
            )
        processed += 1

    if not x_list:
        raise ValueError(f"No valid cross sections found for {rock}")

    x_arr = np.stack(x_list, axis=0).astype(np.float32)
    y_arr = np.stack(y_list, axis=0).astype(np.float32)
    with h5py.File(out_dir / "teacher_geom_only.h5", "w") as f:
        f.create_dataset("X", data=x_arr, compression="gzip")
        f.create_dataset("Y", data=y_arr, compression="gzip")
        f.create_dataset("rock_type", data=np.asarray(rock_type_list, dtype=np.int16))
        f.create_dataset("global_id", data=np.asarray(global_id_list, dtype=np.int32))
        f.create_dataset("local_id", data=np.asarray(local_id_list, dtype=np.int32))
        f.create_dataset("eta", data=np.asarray(eta_list, dtype=np.float32))
        f.create_dataset("RI", data=np.asarray(ri_list, dtype=np.float32))
        f.create_dataset("R0_RA", data=np.asarray(ra_list, dtype=np.float32))
        f.create_dataset("scale_s", data=np.asarray(scale_list, dtype=np.float32))
        f.create_dataset("raw_pore_pixels", data=np.asarray(raw_pore_pixels_list, dtype=np.int32))
        f.create_dataset("tiny_scaled", data=np.asarray(tiny_scaled_flag_list, dtype=np.int8))
        f.attrs["channel_order"] = "mask,dist,eta_map"
        f.attrs["channels"] = 3
        f.attrs["patch_size"] = int(patch_size)
        f.attrs["raw_patch"] = int(raw_patch)
        f.attrs["target_R1"] = float(target_r1)
        f.attrs["min_scale"] = float(min_scale)
        f.attrs["max_scale"] = float(max_scale)
        f.attrs["adaptive_threshold"] = int(adaptive_threshold)
        f.attrs["non_tiny_scale"] = float(non_tiny_scale)
        f.attrs["non_tiny_max_scale"] = float(non_tiny_max_scale)
        f.attrs["piecewise_scale_caps"] = str(",".join(f"{threshold}:{cap}" for threshold, cap in piecewise_scale_caps))
        f.attrs["tiny_pore_threshold"] = int(tiny_pore_threshold)
        f.attrs["tiny_scale_mode"] = str(tiny_scale_mode)
        f.attrs["fixed_scale"] = float(fixed_scale)
        f.attrs["rock_type_name"] = rock
        f.attrs["geometry_only"] = 1
        f.attrs["y_is_dummy_zero"] = 1

    maybe_render_preview(preview_items, out_dir / "preview.png")

    summary = {
        "rock": rock,
        "rock_id": int(rock_id),
        "n_samples": int(x_arr.shape[0]),
        "source_dir": str(src_dir),
        "output_h5": str(out_dir / "teacher_geom_only.h5"),
        "patch_size": int(patch_size),
        "raw_patch": int(raw_patch),
        "target_R1": float(target_r1),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "adaptive_threshold": int(adaptive_threshold),
        "non_tiny_scale": float(non_tiny_scale),
        "non_tiny_max_scale": float(non_tiny_max_scale),
        "piecewise_scale_caps": [f"{threshold}:{cap}" for threshold, cap in piecewise_scale_caps],
        "tiny_pore_threshold": int(tiny_pore_threshold),
        "tiny_scale_mode": str(tiny_scale_mode),
        "fixed_scale": float(fixed_scale),
        "raw_shape_min": [int(min(h for h, _ in dims_list)), int(min(w for _, w in dims_list))],
        "raw_shape_max": [int(max(h for h, _ in dims_list)), int(max(w for _, w in dims_list))],
        "raw_pore_pixels_min": int(min(raw_pore_pixels_list)),
        "raw_pore_pixels_median": float(np.median(raw_pore_pixels_list)),
        "raw_pore_pixels_max": int(max(raw_pore_pixels_list)),
        "tiny_pores_le_3px": int(sum(v <= 3 for v in raw_pore_pixels_list)),
        "tiny_scaled_count": int(sum(tiny_scaled_flag_list)),
        "mask_pore_frac_mean": float(x_arr[:, 0].mean()),
        "dist_max": float(x_arr[:, 1].max()),
        "eta_mean": float(np.mean(eta_list)),
        "scale_s_min": float(np.min(scale_list)),
        "scale_s_mean": float(np.mean(scale_list)),
        "scale_s_median": float(np.median(scale_list)),
        "scale_s_max": float(np.max(scale_list)),
        "note": "Y is dummy zeros. This HDF5 is for geometry-only inference preparation, not metric evaluation.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    piecewise_scale_caps = parse_piecewise_scale_caps(args.piecewise_scale_caps)
    for rock in args.rocks:
        summary = convert_one_rock(
            data_root=data_root,
            out_root=out_root,
            rock=rock,
            patch_size=args.patch_size,
            raw_patch=args.raw_patch,
            target_r1=args.target_r1,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            adaptive_threshold=args.adaptive_threshold,
            non_tiny_scale=args.non_tiny_scale,
            non_tiny_max_scale=args.non_tiny_max_scale,
            piecewise_scale_caps=piecewise_scale_caps,
            tiny_pore_threshold=args.tiny_pore_threshold,
            tiny_scale_mode=args.tiny_scale_mode,
            fixed_scale=args.fixed_scale,
            max_samples=args.max_samples,
            viz_samples=args.viz_samples,
        )
        summaries.append(summary)
        print(
            f"[OK] {rock}: n_samples={summary['n_samples']} "
            f"mask_mean={summary['mask_pore_frac_mean']:.4f} "
            f"dist_max={summary['dist_max']:.4f} eta_mean={summary['eta_mean']:.4f}"
        )
    (out_root / "summary_all.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[DONE] Saved converted geometry datasets to: {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
