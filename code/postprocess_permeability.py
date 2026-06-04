#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Postprocess permeability from predicted velocity fields.

Overall logic:
1. Read predicted velocity fields from NPZ/HDF5.
2. Read mask, optional true velocity, and optional normalization scale from HDF5/NPZ.
3. Undo project-specific normalization and convert lattice-unit velocity to physical velocity.
4. Integrate velocity only over pore voxels/pixels to obtain volumetric flow rate Q.
5. Convert Q to permeability k with Darcy's law for either pressure-driven or body-force-driven flow.
6. Save per-sample CSV, summary JSON, and evaluation plots.

Important physical note:
This script assumes each sample represents a full cross-section (2D) or a full volume (3D)
of the porous medium. If you feed local patches, the result is only a patch-scale apparent
permeability, not a rock-sample permeability suitable for direct reporting.

Example for the current project:
python postprocess_permeability.py ^
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
  --meta-file "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\infer\\predictions.npz" ^
  --spatial-axis-order yz ^
  --flow-axis x ^
  --flow-mode pressure ^
  --output-dir "E:\\mhw\\1\\pore\\runs\\unet_all32_v2_20260212_215344\\infer\\permeability_post"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


# Replace these placeholders with your real physical parameters before using
# the permeability values in a paper.
DEFAULT_DYNAMIC_VISCOSITY = 1.0e-3
DEFAULT_DELTA_P = 1.0
DEFAULT_SAMPLE_LENGTH = 1.0
DEFAULT_FLUID_DENSITY = 1000.0
DEFAULT_BODY_ACCELERATION = 1.0
DEFAULT_VOXEL_SIZE = 1.0
DEFAULT_OUT_OF_PLANE_THICKNESS = 1.0
DEFAULT_VELOCITY_UNIT_SCALE = 1.0
DEFAULT_NORMALIZATION_EXPONENT = 2.0
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_BULK_AREA_OVERRIDE = 0.0


@dataclass
class PhysicalParameters:
    """Physical constants used in Darcy postprocessing."""

    mu: float
    delta_p: float
    sample_length: float
    rho: float
    body_acceleration: float
    velocity_unit_scale: float
    spacing: Tuple[float, ...]
    out_of_plane_thickness: float
    bulk_area_override: float


@dataclass
class FlowIntegrationResult:
    """Flow integration outputs for one sample."""

    q_signed: float
    q_abs: float
    q_std: float
    bulk_area: float
    pore_area_mean: float
    n_sections: int
    section_q_values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute permeability from predicted velocity fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pred-file", type=str, required=True, help="NPZ or HDF5 file containing predicted velocity.")
    parser.add_argument("--pred-key", type=str, required=True, help="Dataset/key for predicted velocity.")
    parser.add_argument(
        "--pred-component",
        type=int,
        default=0,
        help="Velocity component index when the prediction dataset contains a channel/component dimension.",
    )

    parser.add_argument("--true-file", type=str, default="", help="Optional NPZ or HDF5 file containing true velocity.")
    parser.add_argument("--true-key", type=str, default="", help="Dataset/key for true velocity.")
    parser.add_argument(
        "--true-component",
        type=int,
        default=0,
        help="Velocity component index when the true dataset contains a channel/component dimension.",
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
        help="Channel name to resolve from HDF5 channel_order attrs when mask-channel-index is omitted.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=DEFAULT_MASK_THRESHOLD,
        help="Threshold used to convert mask values to pore/solid boolean regions.",
    )

    parser.add_argument("--scale-file", type=str, default="", help="Optional file containing per-sample normalization scale.")
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
        "--index-file",
        type=str,
        default="",
        help="Optional file containing dataset indices that align predictions with HDF5 samples.",
    )
    parser.add_argument("--index-key", type=str, default="index", help="Dataset/key for the sample index array.")
    parser.add_argument("--meta-file", type=str, default="", help="Optional file with metadata such as global_id/rock_type.")
    parser.add_argument("--global-id-key", type=str, default="global_id", help="Metadata key for global_id.")
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
        help="Spatial axis labels in array order, e.g. 'yz' for a 2D yz cross-section or 'xyz' for a 3D volume.",
    )
    parser.add_argument(
        "--flow-axis",
        type=str,
        choices=("x", "y", "z"),
        required=True,
        help="Physical flow direction used in Darcy's law.",
    )

    parser.add_argument(
        "--flow-mode",
        type=str,
        choices=("pressure", "body-force"),
        default="pressure",
        help="Driving mechanism used to convert Q into permeability.",
    )
    parser.add_argument("--mu", type=float, default=DEFAULT_DYNAMIC_VISCOSITY, help="Dynamic viscosity [Pa*s]. Replace placeholder.")
    parser.add_argument("--delta-p", type=float, default=DEFAULT_DELTA_P, help="Pressure drop deltaP [Pa]. Replace placeholder.")
    parser.add_argument("--sample-length", type=float, default=DEFAULT_SAMPLE_LENGTH, help="Sample length L along flow direction [m]. Replace placeholder.")
    parser.add_argument("--rho", type=float, default=DEFAULT_FLUID_DENSITY, help="Fluid density rho [kg/m^3]. Replace placeholder.")
    parser.add_argument(
        "--body-acceleration",
        type=float,
        default=DEFAULT_BODY_ACCELERATION,
        help="Uniform body-force acceleration a_flow [m/s^2]. Replace placeholder.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs="+",
        default=[DEFAULT_VOXEL_SIZE],
        help="Spatial spacing [m] for each axis in spatial-axis-order. Use one value for isotropic voxels.",
    )
    parser.add_argument(
        "--out-of-plane-thickness",
        type=float,
        default=DEFAULT_OUT_OF_PLANE_THICKNESS,
        help="Only used when spatial-axis-order has 2 axes and still contains the flow axis.",
    )
    parser.add_argument(
        "--velocity-unit-scale",
        type=float,
        default=DEFAULT_VELOCITY_UNIT_SCALE,
        help="Multiply by this factor to convert lattice-unit velocity into physical velocity [m/s]. Replace placeholder.",
    )
    parser.add_argument(
        "--bulk-area-override",
        type=float,
        default=DEFAULT_BULK_AREA_OVERRIDE,
        help="Optional external bulk cross-sectional area A [m^2]. Use 0 to compute A from geometry.",
    )

    parser.add_argument("--output-dir", type=str, required=True, help="Directory for CSV, JSON, and plots.")
    parser.add_argument("--csv-name", type=str, default="permeability_results.csv", help="Per-sample CSV output name.")
    parser.add_argument("--summary-name", type=str, default="permeability_summary.json", help="Summary JSON output name.")
    parser.add_argument("--scatter-name", type=str, default="k_pred_vs_true.png", help="Scatter plot output name.")
    parser.add_argument("--hist-name", type=str, default="k_relative_error_hist.png", help="Relative-error histogram output name.")
    parser.add_argument("--dpi", type=int, default=200, help="Output plot DPI.")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="Small epsilon for stable division.")

    return parser.parse_args()


def ensure_output_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_spacing(spacing_values: Sequence[float], ndim: int) -> Tuple[float, ...]:
    if len(spacing_values) == 1:
        return tuple(float(spacing_values[0]) for _ in range(ndim))
    if len(spacing_values) != ndim:
        raise ValueError(
            f"spacing must have length 1 or {ndim}; got {len(spacing_values)} values for ndim={ndim}."
        )
    return tuple(float(v) for v in spacing_values)


def parse_index_spec(spec: str, n_rows: int) -> np.ndarray:
    if not spec:
        return np.arange(n_rows, dtype=np.int64)

    selected: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: {token}")
            start = int(parts[0].strip())
            stop = int(parts[1].strip())
            if stop < start:
                raise ValueError(f"Invalid descending range: {token}")
            selected.extend(range(start, stop + 1))
        else:
            selected.append(int(token))

    selected_arr = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected_arr.size == 0:
        raise ValueError("No valid indices were selected.")
    if selected_arr[0] < 0 or selected_arr[-1] >= n_rows:
        raise IndexError(f"Selected indices must be in [0, {n_rows - 1}]")
    return selected_arr


def _strip_leading_slash(key: str) -> str:
    return key[1:] if key.startswith("/") else key


def load_array(path: str, key: str, row_indices: Optional[np.ndarray] = None) -> np.ndarray:
    path_obj = Path(path)
    key = _strip_leading_slash(key)

    if path_obj.suffix.lower() == ".npz":
        with np.load(path_obj, allow_pickle=True) as data:
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in NPZ file: {path}")
            arr = data[key]
            if row_indices is not None:
                return np.asarray(arr[row_indices])
            return np.asarray(arr)

    if path_obj.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(path_obj, "r") as handle:
            if key not in handle:
                raise KeyError(f"Key '{key}' not found in HDF5 file: {path}")
            ds = handle[key]
            if row_indices is not None:
                return np.asarray(ds[row_indices])
            return np.asarray(ds[...])

    raise ValueError(f"Unsupported file type for {path}. Use NPZ or HDF5.")


def infer_mask_channel_index(path: str, dataset_key: str, channel_name: str) -> Optional[int]:
    path_obj = Path(path)
    if path_obj.suffix.lower() not in (".h5", ".hdf5"):
        return None

    dataset_key = _strip_leading_slash(dataset_key)
    with h5py.File(path_obj, "r") as handle:
        if dataset_key not in handle:
            return None
        ds = handle[dataset_key]
        raw_order = ds.attrs.get("channel_order", handle.attrs.get("channel_order"))
        if raw_order is None:
            return None
        if isinstance(raw_order, bytes):
            raw_order = raw_order.decode("utf-8")
        if isinstance(raw_order, np.ndarray):
            order_list = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in raw_order.tolist()
            ]
        else:
            order_list = [item.strip() for item in str(raw_order).split(",") if item.strip()]
        for idx, name in enumerate(order_list):
            if name.lower() == channel_name.lower():
                return idx
    return None


def get_n_prediction_rows(pred_file: str, pred_key: str) -> int:
    arr = load_array(pred_file, pred_key, row_indices=None)
    if arr.ndim == 0:
        raise ValueError("Prediction array is scalar; expected batched data.")
    return int(arr.shape[0])


def select_field(sample: np.ndarray, spatial_ndim: int, component_index: int, role: str) -> np.ndarray:
    sample = np.asarray(sample)
    if sample.ndim == spatial_ndim:
        return sample.astype(np.float64, copy=False)
    if sample.ndim == spatial_ndim + 1:
        if component_index < 0 or component_index >= sample.shape[0]:
            raise IndexError(
                f"{role} component_index={component_index} out of bounds for leading dimension {sample.shape[0]}"
            )
        return sample[component_index].astype(np.float64, copy=False)
    raise ValueError(
        f"{role} sample has ndim={sample.ndim}, but expected spatial_ndim={spatial_ndim} "
        f"or spatial_ndim+1={spatial_ndim + 1}. Shape={sample.shape}"
    )


def select_mask(sample: np.ndarray, spatial_ndim: int, mask_channel_index: Optional[int]) -> np.ndarray:
    sample = np.asarray(sample)
    if sample.ndim == spatial_ndim:
        return sample.astype(np.float64, copy=False)
    if sample.ndim == spatial_ndim + 1:
        if mask_channel_index is None:
            raise ValueError(
                "Mask sample has a channel dimension, but mask-channel-index could not be inferred. "
                "Provide --mask-channel-index explicitly."
            )
        if mask_channel_index < 0 or mask_channel_index >= sample.shape[0]:
            raise IndexError(
                f"mask-channel-index={mask_channel_index} out of bounds for leading dimension {sample.shape[0]}"
            )
        return sample[mask_channel_index].astype(np.float64, copy=False)
    raise ValueError(
        f"Mask sample has ndim={sample.ndim}, but expected spatial_ndim={spatial_ndim} "
        f"or spatial_ndim+1={spatial_ndim + 1}. Shape={sample.shape}"
    )


def undo_velocity_normalization(
    velocity: np.ndarray,
    scale_s: Optional[float],
    enabled: bool,
    exponent: float,
    velocity_unit_scale: float,
    eps: float,
) -> np.ndarray:
    """Undo training-time normalization and convert to physical units."""
    vel = np.asarray(velocity, dtype=np.float64)
    if enabled:
        if scale_s is None:
            raise ValueError("undo-velocity-normalization was requested, but no scale_s value is available.")
        vel = vel / max(float(scale_s) ** exponent, eps)
    vel = vel * float(velocity_unit_scale)
    return vel


def build_axis_index(axis_order: str) -> Dict[str, int]:
    if len(axis_order) != len(set(axis_order)):
        raise ValueError(f"spatial-axis-order must not repeat axis labels: {axis_order}")
    valid = {"x", "y", "z"}
    labels = set(axis_order)
    if not labels.issubset(valid):
        raise ValueError(f"spatial-axis-order may only contain x/y/z, got: {axis_order}")
    return {label: idx for idx, label in enumerate(axis_order)}


def integrate_flow_rate(
    velocity: np.ndarray,
    mask: np.ndarray,
    axis_order: str,
    flow_axis: str,
    spacing: Tuple[float, ...],
    out_of_plane_thickness: float,
    mask_threshold: float,
) -> FlowIntegrationResult:
    """
    Integrate velocity over pore space only.

    Physical meaning:
    - Q is the volumetric flow rate through a plane perpendicular to the flow direction.
    - Velocity in the solid phase is forced to zero before integration.
    - Darcy's law uses the bulk cross-sectional area of the sample, not the pore area.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=np.float64) > mask_threshold
    if velocity.shape != mask_bool.shape:
        raise ValueError(f"velocity shape {velocity.shape} does not match mask shape {mask_bool.shape}")

    axis_to_idx = build_axis_index(axis_order)
    masked_velocity = np.where(mask_bool, velocity, 0.0)
    ndim = velocity.ndim

    if ndim == 2 and flow_axis not in axis_to_idx:
        d_area = spacing[0] * spacing[1]
        q_value = float(masked_velocity.sum() * d_area)
        pore_area = float(mask_bool.sum() * d_area)
        bulk_area = float((velocity.shape[0] * spacing[0]) * (velocity.shape[1] * spacing[1]))
        return FlowIntegrationResult(
            q_signed=q_value,
            q_abs=abs(q_value),
            q_std=0.0,
            bulk_area=bulk_area,
            pore_area_mean=pore_area,
            n_sections=1,
            section_q_values=np.asarray([q_value], dtype=np.float64),
        )

    if ndim == 2 and flow_axis in axis_to_idx:
        flow_dim = axis_to_idx[flow_axis]
        other_dim = 1 - flow_dim
        d_area = spacing[other_dim] * float(out_of_plane_thickness)
        section_q_values = []
        pore_areas = []
        for section_idx in range(velocity.shape[flow_dim]):
            vel_slice = np.take(masked_velocity, section_idx, axis=flow_dim)
            mask_slice = np.take(mask_bool, section_idx, axis=flow_dim)
            q_slice = float(vel_slice.sum() * d_area)
            pore_area_slice = float(mask_slice.sum() * d_area)
            section_q_values.append(q_slice)
            pore_areas.append(pore_area_slice)
        section_q_arr = np.asarray(section_q_values, dtype=np.float64)
        bulk_area = float((velocity.shape[other_dim] * spacing[other_dim]) * float(out_of_plane_thickness))
        return FlowIntegrationResult(
            q_signed=float(section_q_arr.mean()),
            q_abs=float(np.abs(section_q_arr.mean())),
            q_std=float(section_q_arr.std(ddof=0)),
            bulk_area=bulk_area,
            pore_area_mean=float(np.mean(pore_areas)),
            n_sections=int(section_q_arr.size),
            section_q_values=section_q_arr,
        )

    if ndim == 3:
        if flow_axis not in axis_to_idx:
            raise ValueError(
                f"For 3D data the flow axis must appear in spatial-axis-order. "
                f"Got flow-axis={flow_axis}, spatial-axis-order={axis_order}"
            )
        flow_dim = axis_to_idx[flow_axis]
        section_dims = [idx for idx in range(3) if idx != flow_dim]
        d_area = spacing[section_dims[0]] * spacing[section_dims[1]]
        section_q_values = []
        pore_areas = []
        for section_idx in range(velocity.shape[flow_dim]):
            vel_slice = np.take(masked_velocity, section_idx, axis=flow_dim)
            mask_slice = np.take(mask_bool, section_idx, axis=flow_dim)
            q_slice = float(vel_slice.sum() * d_area)
            pore_area_slice = float(mask_slice.sum() * d_area)
            section_q_values.append(q_slice)
            pore_areas.append(pore_area_slice)
        section_q_arr = np.asarray(section_q_values, dtype=np.float64)
        bulk_area = float(
            (velocity.shape[section_dims[0]] * spacing[section_dims[0]])
            * (velocity.shape[section_dims[1]] * spacing[section_dims[1]])
        )
        return FlowIntegrationResult(
            q_signed=float(section_q_arr.mean()),
            q_abs=float(np.abs(section_q_arr.mean())),
            q_std=float(section_q_arr.std(ddof=0)),
            bulk_area=bulk_area,
            pore_area_mean=float(np.mean(pore_areas)),
            n_sections=int(section_q_arr.size),
            section_q_values=section_q_arr,
        )

    raise ValueError(f"Only 2D or 3D spatial fields are supported, got ndim={ndim}")


def compute_permeability(
    q_abs: float,
    bulk_area: float,
    flow_mode: str,
    physical: PhysicalParameters,
    eps: float,
) -> float:
    """
    Convert volumetric flow rate Q into permeability k.

    Pressure-driven Darcy:
        Q / A = (k / mu) * (DeltaP / L)
        k = mu * Q * L / (A * DeltaP)

    Body-force-driven Darcy:
        gradP_equiv = rho * a
        Q / A = (k / mu) * (rho * a)
        k = mu * Q / (A * rho * a)

    We use |Q| and the magnitude of the driving term because permeability is reported as
    a positive material property, while the sign is controlled by the chosen coordinate system.
    """
    area_used = float(physical.bulk_area_override) if physical.bulk_area_override > 0.0 else float(bulk_area)
    if area_used <= eps:
        raise ValueError(f"bulk_area must be positive, got {bulk_area}")

    superficial_velocity = q_abs / area_used
    if flow_mode == "pressure":
        if abs(physical.delta_p) <= eps:
            raise ValueError("delta_p must be non-zero for pressure-driven permeability.")
        if physical.sample_length <= eps:
            raise ValueError("sample_length must be positive for pressure-driven permeability.")
        pressure_gradient = abs(physical.delta_p) / physical.sample_length
        return float(physical.mu * superficial_velocity / max(pressure_gradient, eps))

    if flow_mode == "body-force":
        equivalent_gradient = abs(physical.rho * physical.body_acceleration)
        if equivalent_gradient <= eps:
            raise ValueError("rho * body_acceleration must be non-zero for body-force-driven permeability.")
        return float(physical.mu * superficial_velocity / max(equivalent_gradient, eps))

    raise ValueError(f"Unsupported flow_mode: {flow_mode}")


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= eps:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def stats_summary(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(y_true), eps)
    rmse = math.sqrt(float(np.mean(diff ** 2))) if diff.size else float("nan")
    return {
        "n_samples": int(y_true.size),
        "mean_true": float(np.mean(y_true)) if y_true.size else float("nan"),
        "mean_pred": float(np.mean(y_pred)) if y_pred.size else float("nan"),
        "mean_abs_error": float(np.mean(abs_err)) if abs_err.size else float("nan"),
        "median_abs_error": float(np.median(abs_err)) if abs_err.size else float("nan"),
        "mean_rel_error": float(np.mean(rel_err)) if rel_err.size else float("nan"),
        "median_rel_error": float(np.median(rel_err)) if rel_err.size else float("nan"),
        "rmse": rmse,
        "r2": float(compute_r2(y_true, y_pred, eps)),
    }


def save_results_csv(out_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No result rows to save.")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_json(out_path: Path, summary: Dict[str, object]) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] matplotlib is unavailable, plots will be skipped: {exc}")
        return None


def plot_pred_vs_true(
    out_path: Path,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    summary: Dict[str, float],
    dpi: int,
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    finite = np.isfinite(true_values) & np.isfinite(pred_values)
    if not np.any(finite):
        print("[WARN] No finite values for scatter plot.")
        return

    true_values = true_values[finite]
    pred_values = pred_values[finite]
    v_min = float(min(np.min(true_values), np.min(pred_values)))
    v_max = float(max(np.max(true_values), np.max(pred_values)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_values, pred_values, s=18, alpha=0.7, edgecolor="none", color="#1f77b4")
    ax.plot([v_min, v_max], [v_min, v_max], "--", color="black", linewidth=1.2, label="y = x")
    ax.set_xlabel("True permeability")
    ax.set_ylabel("Predicted permeability")
    ax.set_title("Predicted vs True Permeability")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(
        0.03,
        0.97,
        f"MAE = {summary['mean_abs_error']:.6e}\nRMSE = {summary['rmse']:.6e}\nR2 = {summary['r2']:.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_relative_error_hist(out_path: Path, true_values: np.ndarray, pred_values: np.ndarray, eps: float, dpi: int) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return

    true_values = np.asarray(true_values, dtype=np.float64)
    pred_values = np.asarray(pred_values, dtype=np.float64)
    finite = np.isfinite(true_values) & np.isfinite(pred_values)
    if not np.any(finite):
        print("[WARN] No finite values for relative-error histogram.")
        return

    rel_err = np.abs(pred_values[finite] - true_values[finite]) / np.maximum(np.abs(true_values[finite]), eps)
    rel_err_pct = rel_err * 100.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rel_err_pct, bins=40, color="#d62728", alpha=0.8)
    ax.set_xlabel("Relative error [%]")
    ax.set_ylabel("Count")
    ax.set_title("Permeability Relative Error Histogram")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_optional_metadata(path: str, key: str, row_indices: np.ndarray) -> Optional[np.ndarray]:
    if not path or not key:
        return None
    try:
        return load_array(path, key, row_indices=row_indices)
    except Exception as exc:
        print(f"[WARN] Failed to read optional metadata '{key}' from {path}: {exc}")
        return None


def verify_sample_counts(pred_rows: int, row_indices: np.ndarray, arrays: Dict[str, Optional[np.ndarray]]) -> None:
    expected = int(row_indices.size)
    for name, arr in arrays.items():
        if arr is None:
            continue
        if arr.shape[0] != expected:
            raise ValueError(
                f"{name} has {arr.shape[0]} selected rows, expected {expected}. "
                f"Check index alignment between files."
            )
    if expected <= 0 or pred_rows <= 0:
        raise ValueError("No samples available for processing.")


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

    global_ids = load_optional_metadata(args.meta_file or args.pred_file, args.global_id_key, row_indices)
    rock_types = load_optional_metadata(args.meta_file or args.pred_file, args.rock_type_key, row_indices)

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
            "rock_types": rock_types,
        },
    )

    physical = PhysicalParameters(
        mu=float(args.mu),
        delta_p=float(args.delta_p),
        sample_length=float(args.sample_length),
        rho=float(args.rho),
        body_acceleration=float(args.body_acceleration),
        velocity_unit_scale=float(args.velocity_unit_scale),
        spacing=spacing,
        out_of_plane_thickness=float(args.out_of_plane_thickness),
        bulk_area_override=float(args.bulk_area_override),
    )

    rows: List[Dict[str, object]] = []
    k_pred_values: List[float] = []
    k_true_values: List[float] = []

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
            spacing=physical.spacing,
            out_of_plane_thickness=physical.out_of_plane_thickness,
            mask_threshold=float(args.mask_threshold),
        )
        k_pred = compute_permeability(
            q_abs=pred_flow.q_abs,
            bulk_area=pred_flow.bulk_area,
            flow_mode=args.flow_mode,
            physical=physical,
            eps=float(args.eps),
        )

        row: Dict[str, object] = {
            "pred_row_index": int(row_indices[local_row_idx]),
            "dataset_index": int(dataset_indices[local_row_idx]),
            "global_id": int(global_ids[local_row_idx]) if global_ids is not None else -1,
            "rock_type": int(rock_types[local_row_idx]) if rock_types is not None else -1,
            "scale_s": float(scale_s) if scale_s is not None else float("nan"),
            "q_pred_signed": float(pred_flow.q_signed),
            "q_pred_abs": float(pred_flow.q_abs),
            "q_pred_std": float(pred_flow.q_std),
            "bulk_area": float(pred_flow.bulk_area),
            "bulk_area_used": float(physical.bulk_area_override) if physical.bulk_area_override > 0.0 else float(pred_flow.bulk_area),
            "mean_pore_area": float(pred_flow.pore_area_mean),
            "n_sections": int(pred_flow.n_sections),
            "k_pred": float(k_pred),
        }

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
                spacing=physical.spacing,
                out_of_plane_thickness=physical.out_of_plane_thickness,
                mask_threshold=float(args.mask_threshold),
            )
            k_true = compute_permeability(
                q_abs=true_flow.q_abs,
                bulk_area=true_flow.bulk_area,
                flow_mode=args.flow_mode,
                physical=physical,
                eps=float(args.eps),
            )

            abs_err_k = abs(k_pred - k_true)
            rel_err_k = abs_err_k / max(abs(k_true), float(args.eps))

            row.update(
                {
                    "q_true_signed": float(true_flow.q_signed),
                    "q_true_abs": float(true_flow.q_abs),
                    "q_true_std": float(true_flow.q_std),
                    "k_true": float(k_true),
                    "k_abs_error": float(abs_err_k),
                    "k_rel_error": float(rel_err_k),
                }
            )
            k_true_values.append(float(k_true))

        k_pred_values.append(float(k_pred))
        rows.append(row)

    csv_path = out_dir / args.csv_name
    save_results_csv(csv_path, rows)

    summary: Dict[str, object] = {
        "flow_mode": args.flow_mode,
        "flow_axis": args.flow_axis,
        "spatial_axis_order": args.spatial_axis_order,
        "n_samples": len(rows),
        "mu": physical.mu,
        "delta_p": physical.delta_p,
        "sample_length": physical.sample_length,
        "rho": physical.rho,
        "body_acceleration": physical.body_acceleration,
        "spacing": list(physical.spacing),
        "out_of_plane_thickness": physical.out_of_plane_thickness,
        "velocity_unit_scale": physical.velocity_unit_scale,
        "bulk_area_override": physical.bulk_area_override,
        "undo_velocity_normalization": bool(args.undo_velocity_normalization),
        "normalization_exponent": float(args.normalization_exponent),
        "pred_file": args.pred_file,
        "pred_key": args.pred_key,
        "mask_file": args.mask_file,
        "mask_key": args.mask_key,
        "true_file": args.true_file,
        "true_key": args.true_key,
    }

    if k_true_values:
        k_true_arr = np.asarray(k_true_values, dtype=np.float64)
        k_pred_arr = np.asarray(k_pred_values, dtype=np.float64)
        metrics = stats_summary(k_true_arr, k_pred_arr, eps=float(args.eps))
        summary["metrics"] = metrics
        plot_pred_vs_true(out_dir / args.scatter_name, k_true_arr, k_pred_arr, metrics, dpi=int(args.dpi))
        plot_relative_error_hist(
            out_dir / args.hist_name,
            k_true_arr,
            k_pred_arr,
            eps=float(args.eps),
            dpi=int(args.dpi),
        )
        print(
            f"Permeability metrics: MAE={metrics['mean_abs_error']:.6e}, "
            f"RMSE={metrics['rmse']:.6e}, R2={metrics['r2']:.6f}"
        )
    else:
        summary["metrics"] = None
        print("[WARN] No true velocity was provided; scatter/histogram and accuracy metrics were skipped.")

    summary_path = out_dir / args.summary_name
    save_summary_json(summary_path, summary)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
