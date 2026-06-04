#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Postprocess local conductance from predicted velocity fields.

This script is the local surrogate bridge:
    geometry -> predicted velocity -> integrated flow rate q -> conductance g

It reuses the same field alignment, mask handling, and velocity de-normalization
logic as postprocess_permeability.py, but targets local conductance rather than
sample-scale permeability.

Optional references:
1. True velocity fields from HDF5/NPZ, used to compute q_true and g_true_velocity.
2. Zhao-style per-rock conductivity tables, e.g. data/<rock>/conductivity.dat,
   used to compare the surrogate conductance against tabulated conductance labels.

Example:
python postprocess_conductance.py ^
  --pred-file "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\infer\\predictions.npz" ^
  --pred-key "pred_ux" ^
  --true-file "E:\\mhw\\1\\pore\\dataset_all_32.h5" ^
  --true-key "Y" ^
  --mask-file "E:\\mhw\\1\\pore\\dataset_all_32.h5" ^
  --mask-key "X" ^
  --scale-file "E:\\mhw\\1\\pore\\dataset_all_32.h5" ^
  --scale-key "scale_s" ^
  --index-file "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\infer\\predictions.npz" ^
  --index-key "index" ^
  --meta-file "E:\\mhw\\1\\pore\\dataset_all_32.h5" ^
  --spatial-axis-order yz ^
  --flow-axis x ^
  --flow-mode pressure ^
  --delta-p 1.0 ^
  --undo-velocity-normalization ^
  --conductivity-root "E:\\mhw\\1\\pore\\data" ^
  --output-dir "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\infer\\conductance_post"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from postprocess_permeability import (
    ensure_output_dir,
    get_n_prediction_rows,
    infer_mask_channel_index,
    integrate_flow_rate,
    load_array,
    load_optional_metadata,
    normalize_spacing,
    parse_index_spec,
    save_results_csv,
    save_summary_json,
    select_field,
    select_mask,
    stats_summary,
    undo_velocity_normalization,
    verify_sample_counts,
)


DEFAULT_DYNAMIC_VISCOSITY = 1.0
DEFAULT_DELTA_P = 1.0
DEFAULT_FLUID_DENSITY = 1000.0
DEFAULT_BODY_ACCELERATION = 1.0
DEFAULT_VELOCITY_UNIT_SCALE = 1.0
DEFAULT_NORMALIZATION_EXPONENT = 2.0
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_VOXEL_SIZE = 1.0
DEFAULT_OUT_OF_PLANE_THICKNESS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local flow rate and conductance from predicted velocity fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pred-file", type=str, required=True, help="NPZ or HDF5 file containing predicted velocity.")
    parser.add_argument("--pred-key", type=str, required=True, help="Dataset/key for predicted velocity.")
    parser.add_argument(
        "--pred-component",
        type=int,
        default=0,
        help="Velocity component index when the prediction dataset contains a channel dimension.",
    )

    parser.add_argument("--true-file", type=str, default="", help="Optional NPZ or HDF5 file containing true velocity.")
    parser.add_argument("--true-key", type=str, default="", help="Dataset/key for true velocity.")
    parser.add_argument(
        "--true-component",
        type=int,
        default=0,
        help="Velocity component index when the true dataset contains a channel dimension.",
    )

    parser.add_argument("--mask-file", type=str, required=True, help="NPZ or HDF5 file containing the pore mask.")
    parser.add_argument("--mask-key", type=str, required=True, help="Dataset/key for the pore mask.")
    parser.add_argument(
        "--mask-channel-index",
        type=int,
        default=None,
        help="Channel index for the mask when mask-key points to a multi-channel tensor such as X.",
    )
    parser.add_argument(
        "--mask-channel-name",
        type=str,
        default="mask",
        help="Channel name to resolve from HDF5 attrs when mask-channel-index is omitted.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=DEFAULT_MASK_THRESHOLD,
        help="Threshold used to convert mask values into pore/solid boolean regions.",
    )

    parser.add_argument("--scale-file", type=str, default="", help="Optional file containing normalization scales.")
    parser.add_argument("--scale-key", type=str, default="scale_s", help="Dataset/key for the normalization scale.")
    parser.add_argument(
        "--undo-velocity-normalization",
        action="store_true",
        help="Undo project normalization, e.g. u_raw = u_norm / scale_s^exp.",
    )
    parser.add_argument(
        "--normalization-exponent",
        type=float,
        default=DEFAULT_NORMALIZATION_EXPONENT,
        help="Exponent used during velocity normalization; this project uses 2.0.",
    )
    parser.add_argument(
        "--undo-area-normalization",
        action="store_true",
        help="Undo the geometric area scaling introduced by sample normalization. "
             "For the current yz cross-section dataset this typically divides q by scale_s^2.",
    )

    parser.add_argument(
        "--index-file",
        type=str,
        default="",
        help="Optional file containing dataset indices that align prediction rows with HDF5 samples.",
    )
    parser.add_argument("--index-key", type=str, default="index", help="Dataset/key for the sample index array.")

    parser.add_argument("--meta-file", type=str, default="", help="Optional metadata file. Defaults to mask-file.")
    parser.add_argument("--global-id-key", type=str, default="global_id", help="Metadata key for global_id.")
    parser.add_argument("--local-id-key", type=str, default="local_id", help="Metadata key for local_id.")
    parser.add_argument("--rock-type-key", type=str, default="rock_type", help="Metadata key for rock_type.")

    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Prediction row indices to process, e.g. '0,5,10-20'. Empty means all rows in pred-file.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional hard limit after row-index filtering.")

    parser.add_argument(
        "--spatial-axis-order",
        type=str,
        required=True,
        help="Spatial axis labels in array order, e.g. 'yz' or 'xyz'.",
    )
    parser.add_argument(
        "--flow-axis",
        type=str,
        choices=("x", "y", "z"),
        required=True,
        help="Physical flow direction aligned with the predicted velocity component.",
    )
    parser.add_argument(
        "--flow-mode",
        type=str,
        choices=("pressure", "body-force"),
        default="pressure",
        help="Driving condition used to convert q into conductance.",
    )
    parser.add_argument(
        "--conductance-mode",
        type=str,
        choices=("simple", "ipnm2"),
        default="simple",
        help="Conductance definition. "
             "'simple' uses g=q/driver. "
             "'ipnm2' uses the body-force-based IPNM2 formula inferred from the teacher dataset.",
    )
    parser.add_argument(
        "--delta-p",
        type=float,
        default=DEFAULT_DELTA_P,
        help="Local pressure drop used when flow-mode=pressure.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=DEFAULT_FLUID_DENSITY,
        help="Fluid density used when flow-mode=body-force.",
    )
    parser.add_argument(
        "--body-acceleration",
        type=float,
        default=DEFAULT_BODY_ACCELERATION,
        help="Uniform body-force acceleration used when flow-mode=body-force.",
    )
    parser.add_argument(
        "--ax",
        type=float,
        default=1.0e-4,
        help="IPNM2 body-force acceleration a_x used in local LBM simulations.",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=4.0,
        help="IPNM2 sub-throat length L_k or L_sub.",
    )
    parser.add_argument(
        "--mu-lbm",
        type=float,
        default=0.5,
        help="Dynamic viscosity used in the local LBM simulations.",
    )
    parser.add_argument(
        "--target-mu",
        type=float,
        default=1.0,
        help="Target viscosity used when converting local q into the IPNM2 conductance stored in PNM.",
    )
    parser.add_argument(
        "--velocity-unit-scale",
        type=float,
        default=DEFAULT_VELOCITY_UNIT_SCALE,
        help="Multiply by this factor to convert lattice-unit velocity into physical velocity.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        default=[DEFAULT_VOXEL_SIZE],
        help="Spatial spacing for each axis in spatial-axis-order. Use one value for isotropic spacing.",
    )
    parser.add_argument(
        "--out-of-plane-thickness",
        type=float,
        default=DEFAULT_OUT_OF_PLANE_THICKNESS,
        help="Only used when spatial-axis-order has 2 axes and still contains the flow axis.",
    )

    parser.add_argument(
        "--conductivity-root",
        type=str,
        default="",
        help="Optional root directory containing per-rock conductivity.dat files.",
    )
    parser.add_argument(
        "--conductivity-filename",
        type=str,
        default="conductivity.dat",
        help="Filename under each rock folder that stores reference conductance values.",
    )
    parser.add_argument(
        "--table-g-column",
        type=str,
        default="conductivity",
        help="Column name in conductivity.dat interpreted as reference conductance.",
    )
    parser.add_argument(
        "--table-q-column",
        type=str,
        default="rho_Q",
        help="Optional column in conductivity.dat interpreted as reference flow rate.",
    )
    parser.add_argument(
        "--rock-folder-offset",
        type=int,
        default=1,
        help="Folder name = rock_type + offset. Current dataset uses rock_type in [0,5] and folders [1,6].",
    )
    parser.add_argument(
        "--permeability-root",
        type=str,
        default="",
        help="Optional root directory containing per-sample permeability_<local_id>.dat files. "
             "If provided, these per-sample steady values are preferred over conductivity.dat.",
    )
    parser.add_argument(
        "--permeability-prefix",
        type=str,
        default="permeability_",
        help="Filename prefix for per-sample steady-state tables under each rock folder.",
    )
    parser.add_argument(
        "--permeability-suffix",
        type=str,
        default=".dat",
        help="Filename suffix for per-sample steady-state tables.",
    )

    parser.add_argument("--output-dir", type=str, required=True, help="Directory for CSV and JSON outputs.")
    parser.add_argument("--csv-name", type=str, default="conductance_results.csv", help="Per-sample CSV output name.")
    parser.add_argument("--summary-name", type=str, default="conductance_summary.json", help="Summary JSON output name.")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="Small epsilon for stable division.")
    return parser.parse_args()


def compute_conductance(
    q_abs: float,
    conductance_mode: str,
    flow_mode: str,
    delta_p: float,
    rho: float,
    body_acceleration: float,
    ax: float,
    segment_length: float,
    mu_lbm: float,
    target_mu: float,
    eps: float,
) -> float:
    if conductance_mode == "ipnm2":
        driver = abs(float(rho) * float(ax) * float(segment_length) * float(mu_lbm))
        if driver <= eps:
            raise ValueError("rho * a_x * segment_length * mu_lbm must be non-zero when conductance-mode=ipnm2.")
        return float(q_abs * float(target_mu) / driver)
    if flow_mode == "pressure":
        driver = abs(float(delta_p))
        if driver <= eps:
            raise ValueError("delta-p must be non-zero when flow-mode=pressure.")
        return float(q_abs / driver)
    if flow_mode == "body-force":
        driver = abs(float(rho) * float(body_acceleration))
        if driver <= eps:
            raise ValueError("rho * body-acceleration must be non-zero when flow-mode=body-force.")
        return float(q_abs / driver)
    raise ValueError(f"Unsupported flow-mode: {flow_mode}")


def infer_area_scale_exponent(spatial_axis_order: str, flow_axis: str) -> int:
    if flow_axis in spatial_axis_order:
        return max(len(spatial_axis_order) - 1, 1)
    return len(spatial_axis_order)


def apply_area_correction(
    q_signed: float,
    q_abs: float,
    q_std: float,
    scale_s: Optional[float],
    enabled: bool,
    exponent: int,
    eps: float,
) -> Tuple[float, float, float]:
    if not enabled:
        return float(q_signed), float(q_abs), float(q_std)
    if scale_s is None:
        raise ValueError("undo-area-normalization was requested, but no scale_s value is available.")
    denom = max(float(scale_s) ** int(exponent), float(eps))
    return float(q_signed / denom), float(q_abs / denom), float(q_std / denom)


def read_whitespace_table(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Table not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        row = {header[i]: parts[i] for i in range(len(header))}
        rows.append(row)
    return rows


PERM_LINE_RE = re.compile(
    r"^\s*(?P<t>\d+)\s+"
    r"(?P<rho_Q>[+-]?\d+\.\d+)\s+"
    r"(?P<permeability>[+-]?\d+\.\d{10})\s*"
    r"(?P<conductivity>[+-]?\d+\.\d+)\s*$"
)


def parse_permeability_file(path: Path) -> Optional[Dict[str, float]]:
    if not path.is_file():
        return None
    rows: List[Dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("t"):
            continue
        match = PERM_LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "t": float(match.group("t")),
                "q_table": float(match.group("rho_Q")),
                "permeability_table": float(match.group("permeability")),
                "g_table": float(match.group("conductivity")),
            }
        )
    if not rows:
        return None
    last = rows[-1].copy()
    last["n_steps"] = float(len(rows))
    return last


def build_conductivity_lookup(
    root: str,
    filename: str,
    g_column: str,
    q_column: str,
    rock_folder_offset: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"conductivity-root is not a directory: {root}")

    for rock_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        try:
            folder_number = int(rock_dir.name)
        except ValueError:
            continue
        rock_type = folder_number - int(rock_folder_offset)
        table_path = rock_dir / filename
        if not table_path.is_file():
            continue
        rows = read_whitespace_table(table_path)
        for row in rows:
            if "sub" not in row:
                continue
            local_id = int(row["sub"])
            item: Dict[str, float] = {}
            if g_column in row:
                item["g_table"] = float(row[g_column])
            if q_column and q_column in row:
                item["q_table"] = float(row[q_column])
            if "permeability" in row:
                item["permeability_table"] = float(row["permeability"])
            if item:
                lookup[(rock_type, local_id)] = item
    return lookup


def build_permeability_lookup(
    root: str,
    prefix: str,
    suffix: str,
    rock_folder_offset: int,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"permeability-root is not a directory: {root}")

    for rock_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        try:
            folder_number = int(rock_dir.name)
        except ValueError:
            continue
        rock_type = folder_number - int(rock_folder_offset)
        for file_path in rock_dir.glob(f"{prefix}*{suffix}"):
            stem = file_path.stem
            if not stem.startswith(prefix):
                continue
            local_raw = stem[len(prefix):]
            try:
                local_id = int(local_raw)
            except ValueError:
                continue
            parsed = parse_permeability_file(file_path)
            if parsed is None:
                continue
            lookup[(rock_type, local_id)] = parsed
    return lookup


def append_metric(summary: Dict[str, object], name: str, y_true: Sequence[float], y_pred: Sequence[float], eps: float) -> None:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        summary[name] = None
        return
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Metric arrays for {name} have mismatched shapes: {y_true_arr.shape} vs {y_pred_arr.shape}")
    summary[name] = stats_summary(y_true_arr, y_pred_arr, eps=float(eps))


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)

    spatial_ndim = len(args.spatial_axis_order)
    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial-axis-order must describe 2D or 3D spatial fields, got {args.spatial_axis_order}")
    spacing = normalize_spacing(args.spacing, spatial_ndim)

    n_pred_rows = get_n_prediction_rows(args.pred_file, args.pred_key)
    row_indices = parse_index_spec(args.indices, n_pred_rows)
    if args.max_samples > 0:
        row_indices = row_indices[: args.max_samples]

    index_file = args.index_file or args.pred_file
    dataset_indices = load_array(index_file, args.index_key, row_indices=row_indices).astype(np.int64)
    pred_raw = load_array(args.pred_file, args.pred_key, row_indices=row_indices)
    mask_raw = load_array(args.mask_file, args.mask_key, row_indices=dataset_indices)

    true_raw = None
    if args.true_file and args.true_key:
        true_raw = load_array(args.true_file, args.true_key, row_indices=dataset_indices)

    scale_raw = None
    if args.scale_file and args.scale_key:
        scale_raw = load_array(args.scale_file, args.scale_key, row_indices=dataset_indices).astype(np.float64)

    meta_source = args.meta_file or args.mask_file or args.true_file or args.pred_file
    global_ids = load_optional_metadata(meta_source, args.global_id_key, dataset_indices)
    local_ids = load_optional_metadata(meta_source, args.local_id_key, dataset_indices)
    rock_types = load_optional_metadata(meta_source, args.rock_type_key, dataset_indices)

    mask_channel_index = args.mask_channel_index
    if mask_channel_index is None:
        mask_channel_index = infer_mask_channel_index(args.mask_file, args.mask_key, args.mask_channel_name)

    verify_sample_counts(
        n_pred_rows,
        row_indices,
        {
            "dataset_indices": dataset_indices,
            "pred_raw": pred_raw,
            "mask_raw": mask_raw,
            "true_raw": true_raw,
            "scale_raw": scale_raw,
            "global_ids": global_ids,
            "local_ids": local_ids,
            "rock_types": rock_types,
        },
    )

    conductivity_lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    if args.conductivity_root:
        conductivity_lookup = build_conductivity_lookup(
            root=args.conductivity_root,
            filename=args.conductivity_filename,
            g_column=args.table_g_column,
            q_column=args.table_q_column,
            rock_folder_offset=int(args.rock_folder_offset),
        )
    permeability_lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    if args.permeability_root:
        permeability_lookup = build_permeability_lookup(
            root=args.permeability_root,
            prefix=args.permeability_prefix,
            suffix=args.permeability_suffix,
            rock_folder_offset=int(args.rock_folder_offset),
        )

    rows: List[Dict[str, object]] = []
    q_true_list: List[float] = []
    q_pred_list: List[float] = []
    g_true_velocity_list: List[float] = []
    g_pred_list: List[float] = []
    g_table_true_list: List[float] = []
    g_table_pred_list: List[float] = []
    q_table_true_list: List[float] = []
    q_table_pred_list: List[float] = []
    area_scale_exponent = infer_area_scale_exponent(args.spatial_axis_order, args.flow_axis)

    for local_row_idx in range(row_indices.size):
        pred_field_norm = select_field(
            pred_raw[local_row_idx],
            spatial_ndim=spatial_ndim,
            component_index=args.pred_component,
            role="predicted velocity",
        )
        mask_field = select_mask(mask_raw[local_row_idx], spatial_ndim=spatial_ndim, mask_channel_index=mask_channel_index)

        scale_s = None if scale_raw is None else float(scale_raw[local_row_idx])
        pred_field_phys = undo_velocity_normalization(
            pred_field_norm,
            scale_s=scale_s,
            enabled=args.undo_velocity_normalization,
            exponent=float(args.normalization_exponent),
            velocity_unit_scale=float(args.velocity_unit_scale),
            eps=float(args.eps),
        )
        pred_flow = integrate_flow_rate(
            velocity=pred_field_phys,
            mask=mask_field,
            axis_order=args.spatial_axis_order,
            flow_axis=args.flow_axis,
            spacing=spacing,
            out_of_plane_thickness=float(args.out_of_plane_thickness),
            mask_threshold=float(args.mask_threshold),
        )
        q_pred_signed, q_pred_abs, q_pred_std = apply_area_correction(
            q_signed=pred_flow.q_signed,
            q_abs=pred_flow.q_abs,
            q_std=pred_flow.q_std,
            scale_s=scale_s,
            enabled=bool(args.undo_area_normalization),
            exponent=area_scale_exponent,
            eps=float(args.eps),
        )
        g_pred = compute_conductance(
            q_abs=q_pred_abs,
            conductance_mode=args.conductance_mode,
            flow_mode=args.flow_mode,
            delta_p=float(args.delta_p),
            rho=float(args.rho),
            body_acceleration=float(args.body_acceleration),
            ax=float(args.ax),
            segment_length=float(args.segment_length),
            mu_lbm=float(args.mu_lbm),
            target_mu=float(args.target_mu),
            eps=float(args.eps),
        )

        global_id = int(global_ids[local_row_idx]) if global_ids is not None else -1
        rock_type = int(rock_types[local_row_idx]) if rock_types is not None else -1
        local_id = int(local_ids[local_row_idx]) if local_ids is not None else (global_id % 1_000_000 if global_id >= 0 else -1)
        driver_value = abs(float(args.delta_p)) if args.flow_mode == "pressure" else abs(float(args.rho) * float(args.body_acceleration))

        row: Dict[str, object] = {
            "pred_row_index": int(row_indices[local_row_idx]),
            "dataset_index": int(dataset_indices[local_row_idx]),
            "global_id": global_id,
            "local_id": local_id,
            "rock_type": rock_type,
            "scale_s": float(scale_s) if scale_s is not None else float("nan"),
            "flow_mode": args.flow_mode,
            "conductance_mode": args.conductance_mode,
            "driver_value": float(driver_value),
            "ax": float(args.ax),
            "segment_length": float(args.segment_length),
            "mu_lbm": float(args.mu_lbm),
            "target_mu": float(args.target_mu),
            "area_scale_exponent": int(area_scale_exponent),
            "q_pred_signed": float(q_pred_signed),
            "q_pred_abs": float(q_pred_abs),
            "q_pred_std": float(q_pred_std),
            "g_pred": float(g_pred),
            "bulk_area": float(pred_flow.bulk_area),
            "mean_pore_area": float(pred_flow.pore_area_mean),
            "n_sections": int(pred_flow.n_sections),
        }

        q_pred_list.append(float(q_pred_abs))
        g_pred_list.append(float(g_pred))

        if true_raw is not None:
            true_field_norm = select_field(
                true_raw[local_row_idx],
                spatial_ndim=spatial_ndim,
                component_index=args.true_component,
                role="true velocity",
            )
            true_field_phys = undo_velocity_normalization(
                true_field_norm,
                scale_s=scale_s,
                enabled=args.undo_velocity_normalization,
                exponent=float(args.normalization_exponent),
                velocity_unit_scale=float(args.velocity_unit_scale),
                eps=float(args.eps),
            )
            true_flow = integrate_flow_rate(
                velocity=true_field_phys,
                mask=mask_field,
                axis_order=args.spatial_axis_order,
                flow_axis=args.flow_axis,
                spacing=spacing,
                out_of_plane_thickness=float(args.out_of_plane_thickness),
                mask_threshold=float(args.mask_threshold),
            )
            q_true_signed, q_true_abs, q_true_std = apply_area_correction(
                q_signed=true_flow.q_signed,
                q_abs=true_flow.q_abs,
                q_std=true_flow.q_std,
                scale_s=scale_s,
                enabled=bool(args.undo_area_normalization),
                exponent=area_scale_exponent,
                eps=float(args.eps),
            )
            g_true_velocity = compute_conductance(
                q_abs=q_true_abs,
                conductance_mode=args.conductance_mode,
                flow_mode=args.flow_mode,
                delta_p=float(args.delta_p),
                rho=float(args.rho),
                body_acceleration=float(args.body_acceleration),
                ax=float(args.ax),
                segment_length=float(args.segment_length),
                mu_lbm=float(args.mu_lbm),
                target_mu=float(args.target_mu),
                eps=float(args.eps),
            )

            row.update(
                {
                    "q_true_signed": float(q_true_signed),
                    "q_true_abs": float(q_true_abs),
                    "q_true_std": float(q_true_std),
                    "g_true_velocity": float(g_true_velocity),
                    "q_abs_error": float(abs(q_pred_abs - q_true_abs)),
                    "q_rel_error": float(abs(q_pred_abs - q_true_abs) / max(abs(q_true_abs), float(args.eps))),
                    "g_abs_error_velocity": float(abs(g_pred - g_true_velocity)),
                    "g_rel_error_velocity": float(abs(g_pred - g_true_velocity) / max(abs(g_true_velocity), float(args.eps))),
                }
            )

            q_true_list.append(float(q_true_abs))
            g_true_velocity_list.append(float(g_true_velocity))

        ref = None
        ref_source = ""
        if permeability_lookup:
            ref = permeability_lookup.get((rock_type, local_id))
            ref_source = "permeability_file" if ref is not None else ""
        if ref is None and conductivity_lookup:
            ref = conductivity_lookup.get((rock_type, local_id))
            ref_source = "conductivity_table" if ref is not None else ""
        if ref is not None:
            row["reference_source"] = ref_source
            if "t" in ref:
                row["table_last_t"] = float(ref["t"])
            if "n_steps" in ref:
                row["table_n_steps"] = int(ref["n_steps"])
            if "q_table" in ref:
                q_table = float(ref["q_table"])
                row["q_table"] = q_table
                row["q_abs_error_table"] = float(abs(q_pred_abs - q_table))
                row["q_rel_error_table"] = float(abs(q_pred_abs - q_table) / max(abs(q_table), float(args.eps)))
                q_table_true_list.append(q_table)
                q_table_pred_list.append(float(q_pred_abs))
            if "g_table" in ref:
                g_table = float(ref["g_table"])
                row["g_table"] = g_table
                row["g_abs_error_table"] = float(abs(g_pred - g_table))
                row["g_rel_error_table"] = float(abs(g_pred - g_table) / max(abs(g_table), float(args.eps)))
                g_table_true_list.append(g_table)
                g_table_pred_list.append(float(g_pred))
            if "permeability_table" in ref:
                row["permeability_table"] = float(ref["permeability_table"])

        rows.append(row)

    csv_path = out_dir / args.csv_name
    save_results_csv(csv_path, rows)

    summary: Dict[str, object] = {
        "n_samples": len(rows),
        "pred_file": args.pred_file,
        "pred_key": args.pred_key,
        "true_file": args.true_file,
        "true_key": args.true_key,
        "mask_file": args.mask_file,
        "mask_key": args.mask_key,
        "scale_file": args.scale_file,
        "scale_key": args.scale_key,
        "meta_file": meta_source,
        "spatial_axis_order": args.spatial_axis_order,
        "flow_axis": args.flow_axis,
        "flow_mode": args.flow_mode,
        "conductance_mode": args.conductance_mode,
        "delta_p": float(args.delta_p),
        "rho": float(args.rho),
        "body_acceleration": float(args.body_acceleration),
        "ax": float(args.ax),
        "segment_length": float(args.segment_length),
        "mu_lbm": float(args.mu_lbm),
        "target_mu": float(args.target_mu),
        "spacing": list(spacing),
        "out_of_plane_thickness": float(args.out_of_plane_thickness),
        "velocity_unit_scale": float(args.velocity_unit_scale),
        "undo_velocity_normalization": bool(args.undo_velocity_normalization),
        "undo_area_normalization": bool(args.undo_area_normalization),
        "normalization_exponent": float(args.normalization_exponent),
        "area_scale_exponent": int(area_scale_exponent),
        "conductivity_root": args.conductivity_root,
        "permeability_root": args.permeability_root,
        "permeability_prefix": args.permeability_prefix,
        "permeability_suffix": args.permeability_suffix,
        "table_g_column": args.table_g_column,
        "table_q_column": args.table_q_column,
    }

    append_metric(summary, "q_metrics_vs_true_velocity", q_true_list, q_pred_list, float(args.eps))
    append_metric(summary, "g_metrics_vs_true_velocity", g_true_velocity_list, g_pred_list[: len(g_true_velocity_list)], float(args.eps))
    append_metric(summary, "q_metrics_vs_table", q_table_true_list, q_table_pred_list, float(args.eps))
    append_metric(summary, "g_metrics_vs_table", g_table_true_list, g_table_pred_list, float(args.eps))

    summary_path = out_dir / args.summary_name
    save_summary_json(summary_path, summary)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")

    if summary["g_metrics_vs_true_velocity"] is not None:
        metrics = summary["g_metrics_vs_true_velocity"]
        print(
            "Velocity-derived g metrics: "
            f"MAE={metrics['mean_abs_error']:.6e}, "
            f"RMSE={metrics['rmse']:.6e}, "
            f"R2={metrics['r2']:.6f}"
        )
    if summary["g_metrics_vs_table"] is not None:
        metrics = summary["g_metrics_vs_table"]
        print(
            "Table g metrics: "
            f"MAE={metrics['mean_abs_error']:.6e}, "
            f"RMSE={metrics['rmse']:.6e}, "
            f"R2={metrics['r2']:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
