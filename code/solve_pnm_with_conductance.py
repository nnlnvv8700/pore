#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solve a resistor-style pore network for one or more conductance columns.

Required inputs:
1. Edge CSV with at least:
       edge_id,node_i,node_j,<conductance columns>
2. Boundary CSV with at least:
       node_id,pressure
   Optional:
       label

The solver treats each conductance column as a separate network realization,
solves node pressures under Dirichlet boundary conditions, computes edge flows,
and optionally compares all runs against a reference conductance column.

Example:
python solve_pnm_with_conductance.py ^
  --edge-csv "E:\\mhw\\1\\pore\\pnm\\edges_with_conductance.csv" ^
  --boundary-csv "E:\\mhw\\1\\pore\\pnm\\boundary_nodes.csv" ^
  --conductance-cols g_true_velocity g_pred g_table ^
  --reference-col g_true_velocity ^
  --inlet-label inlet ^
  --outlet-label outlet ^
  --mu 1.0 ^
  --sample-length 1.0 ^
  --bulk-area 1.0 ^
  --out-dir "E:\\mhw\\1\\pore\\pnm\\solve_runs"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a PNM graph for one or more conductance columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--edge-csv", type=str, required=True, help="Edge CSV with node_i, node_j, and conductance columns.")
    parser.add_argument("--boundary-csv", type=str, required=True, help="Boundary CSV with node_id and pressure.")
    parser.add_argument(
        "--conductance-cols",
        type=str,
        nargs="+",
        required=True,
        help="Conductance columns to solve, e.g. g_pred g_true_velocity g_table.",
    )
    parser.add_argument("--edge-id-col", type=str, default="edge_id", help="Edge id column name.")
    parser.add_argument("--node-i-col", type=str, default="node_i", help="Start node column name.")
    parser.add_argument("--node-j-col", type=str, default="node_j", help="End node column name.")
    parser.add_argument("--node-id-col", type=str, default="node_id", help="Boundary node id column name.")
    parser.add_argument("--pressure-col", type=str, default="pressure", help="Boundary pressure column name.")
    parser.add_argument("--label-col", type=str, default="label", help="Optional boundary label column name.")
    parser.add_argument("--reference-col", type=str, default="", help="Optional conductance column used as reference run.")
    parser.add_argument("--inlet-label", type=str, default="inlet", help="Boundary label used as inlet.")
    parser.add_argument("--outlet-label", type=str, default="outlet", help="Boundary label used as outlet.")
    parser.add_argument("--mu", type=float, default=1.0, help="Dynamic viscosity for effective permeability.")
    parser.add_argument("--sample-length", type=float, default=0.0, help="Sample length along the flow direction.")
    parser.add_argument("--bulk-area", type=float, default=0.0, help="Bulk cross-sectional area.")
    parser.add_argument("--eps", type=float, default=1.0e-12, help="Small epsilon for stable division.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fields


def to_int(value: str, name: str) -> int:
    try:
        return int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Failed to parse integer {name}={value!r}") from exc


def to_float(value: str, name: str) -> float:
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Failed to parse float {name}={value!r}") from exc


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= eps:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def solve_network(
    edge_rows: Sequence[Dict[str, str]],
    boundary_rows: Sequence[Dict[str, str]],
    conductance_col: str,
    edge_id_col: str,
    node_i_col: str,
    node_j_col: str,
    node_id_col: str,
    pressure_col: str,
    label_col: str,
    eps: float,
) -> Dict[str, object]:
    boundary_pressure_by_node: Dict[int, float] = {}
    boundary_label_by_node: Dict[int, str] = {}
    node_ids = set()

    for row in boundary_rows:
        node_id = to_int(row[node_id_col], node_id_col)
        pressure = to_float(row[pressure_col], pressure_col)
        boundary_pressure_by_node[node_id] = pressure
        boundary_label_by_node[node_id] = str(row.get(label_col, "")).strip()
        node_ids.add(node_id)

    edge_records: List[Dict[str, object]] = []
    for row in edge_rows:
        if conductance_col not in row:
            raise KeyError(f"Conductance column '{conductance_col}' not found in edge CSV.")
        node_i = to_int(row[node_i_col], node_i_col)
        node_j = to_int(row[node_j_col], node_j_col)
        g = to_float(row[conductance_col], conductance_col)
        if g < -eps:
            raise ValueError(f"Negative conductance detected in column '{conductance_col}': {g}")
        if abs(g) <= eps:
            g = 0.0
        edge_id = str(row.get(edge_id_col, len(edge_records)))
        edge_records.append({"edge_id": edge_id, "node_i": node_i, "node_j": node_j, "g": g, "raw": row})
        node_ids.add(node_i)
        node_ids.add(node_j)

    node_list = sorted(node_ids)
    node_to_pos = {node_id: pos for pos, node_id in enumerate(node_list)}
    n_nodes = len(node_list)
    lap = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for rec in edge_records:
        i = node_to_pos[int(rec["node_i"])]
        j = node_to_pos[int(rec["node_j"])]
        g = float(rec["g"])
        if g <= eps:
            continue
        lap[i, i] += g
        lap[j, j] += g
        lap[i, j] -= g
        lap[j, i] -= g

    dirichlet_nodes = sorted(boundary_pressure_by_node.keys())
    dirichlet_mask = np.asarray([node_id in boundary_pressure_by_node for node_id in node_list], dtype=bool)
    unknown_mask = ~dirichlet_mask

    pressures = np.zeros(n_nodes, dtype=np.float64)
    for node_id, pressure in boundary_pressure_by_node.items():
        pressures[node_to_pos[node_id]] = pressure

    if unknown_mask.any():
        luu = lap[np.ix_(unknown_mask, unknown_mask)]
        lud = lap[np.ix_(unknown_mask, dirichlet_mask)]
        p_dir = pressures[dirichlet_mask]
        rhs = -lud @ p_dir
        try:
            p_unknown = np.linalg.solve(luu, rhs)
        except np.linalg.LinAlgError:
            p_unknown = np.linalg.lstsq(luu, rhs, rcond=None)[0]
        pressures[unknown_mask] = p_unknown

    edge_flow = np.zeros(len(edge_records), dtype=np.float64)
    edge_dp = np.zeros(len(edge_records), dtype=np.float64)
    for idx, rec in enumerate(edge_records):
        i = node_to_pos[int(rec["node_i"])]
        j = node_to_pos[int(rec["node_j"])]
        dp = pressures[i] - pressures[j]
        q = float(rec["g"]) * dp
        edge_dp[idx] = dp
        edge_flow[idx] = q

    boundary_flux_by_label: Dict[str, float] = {}
    for node_id, label in boundary_label_by_node.items():
        if not label:
            continue
        pos = node_to_pos[node_id]
        flux_out = 0.0
        for rec in edge_records:
            if int(rec["node_i"]) == node_id:
                other = node_to_pos[int(rec["node_j"])]
                flux_out += float(rec["g"]) * (pressures[pos] - pressures[other])
            elif int(rec["node_j"]) == node_id:
                other = node_to_pos[int(rec["node_i"])]
                flux_out += float(rec["g"]) * (pressures[pos] - pressures[other])
        boundary_flux_by_label[label] = boundary_flux_by_label.get(label, 0.0) + flux_out

    node_rows: List[Dict[str, object]] = []
    for node_id, pressure in zip(node_list, pressures):
        node_rows.append(
            {
                "node_id": int(node_id),
                "pressure": float(pressure),
                "is_dirichlet": 1 if node_id in boundary_pressure_by_node else 0,
                "label": boundary_label_by_node.get(node_id, ""),
            }
        )

    edge_rows_out: List[Dict[str, object]] = []
    for idx, rec in enumerate(edge_records):
        raw = dict(rec["raw"])
        raw["conductance_used"] = float(rec["g"])
        raw["delta_p"] = float(edge_dp[idx])
        raw["flow_ij"] = float(edge_flow[idx])
        raw["abs_flow"] = float(abs(edge_flow[idx]))
        edge_rows_out.append(raw)

    return {
        "conductance_col": conductance_col,
        "node_list": node_list,
        "pressures": pressures,
        "edge_flow": edge_flow,
        "edge_dp": edge_dp,
        "node_rows": node_rows,
        "edge_rows": edge_rows_out,
        "boundary_flux_by_label": boundary_flux_by_label,
    }


def maybe_compute_keff(
    boundary_flux_by_label: Dict[str, float],
    inlet_label: str,
    outlet_label: str,
    boundary_rows: Sequence[Dict[str, str]],
    node_id_col: str,
    pressure_col: str,
    label_col: str,
    mu: float,
    sample_length: float,
    bulk_area: float,
    eps: float,
) -> Dict[str, float]:
    out = {
        "q_inlet": float("nan"),
        "q_outlet": float("nan"),
        "q_balance_rel": float("nan"),
        "delta_p": float("nan"),
        "k_eff": float("nan"),
    }
    q_inlet = boundary_flux_by_label.get(inlet_label)
    q_outlet = boundary_flux_by_label.get(outlet_label)
    if q_inlet is None or q_outlet is None:
        return out

    inlet_pressures: List[float] = []
    outlet_pressures: List[float] = []
    for row in boundary_rows:
        label = str(row.get(label_col, "")).strip()
        if label == inlet_label:
            inlet_pressures.append(to_float(row[pressure_col], pressure_col))
        elif label == outlet_label:
            outlet_pressures.append(to_float(row[pressure_col], pressure_col))
    if not inlet_pressures or not outlet_pressures:
        return out

    delta_p = abs(float(np.mean(inlet_pressures) - np.mean(outlet_pressures)))
    q_in = abs(float(q_inlet))
    q_out = abs(float(q_outlet))
    q_ref = max(q_in, q_out, eps)
    q_balance_rel = abs(q_in - q_out) / q_ref

    out["q_inlet"] = q_in
    out["q_outlet"] = q_out
    out["q_balance_rel"] = q_balance_rel
    out["delta_p"] = delta_p

    if bulk_area > eps and sample_length > eps and delta_p > eps:
        out["k_eff"] = float(mu * q_in * sample_length / (bulk_area * delta_p))
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_rows, _ = read_csv(Path(args.edge_csv))
    boundary_rows, _ = read_csv(Path(args.boundary_csv))

    run_results: Dict[str, Dict[str, object]] = {}
    aggregate_rows: List[Dict[str, object]] = []

    for conductance_col in args.conductance_cols:
        result = solve_network(
            edge_rows=edge_rows,
            boundary_rows=boundary_rows,
            conductance_col=conductance_col,
            edge_id_col=args.edge_id_col,
            node_i_col=args.node_i_col,
            node_j_col=args.node_j_col,
            node_id_col=args.node_id_col,
            pressure_col=args.pressure_col,
            label_col=args.label_col,
            eps=float(args.eps),
        )
        keff_info = maybe_compute_keff(
            boundary_flux_by_label=result["boundary_flux_by_label"],
            inlet_label=args.inlet_label,
            outlet_label=args.outlet_label,
            boundary_rows=boundary_rows,
            node_id_col=args.node_id_col,
            pressure_col=args.pressure_col,
            label_col=args.label_col,
            mu=float(args.mu),
            sample_length=float(args.sample_length),
            bulk_area=float(args.bulk_area),
            eps=float(args.eps),
        )
        summary = {
            "conductance_col": conductance_col,
            "n_nodes": len(result["node_list"]),
            "n_edges": len(result["edge_rows"]),
            "boundary_flux_by_label": result["boundary_flux_by_label"],
            "q_inlet": keff_info["q_inlet"],
            "q_outlet": keff_info["q_outlet"],
            "q_balance_rel": keff_info["q_balance_rel"],
            "delta_p": keff_info["delta_p"],
            "k_eff": keff_info["k_eff"],
        }

        node_csv = out_dir / f"nodes_{conductance_col}.csv"
        edge_csv = out_dir / f"edges_{conductance_col}.csv"
        summary_json = out_dir / f"summary_{conductance_col}.json"
        write_csv(node_csv, result["node_rows"])
        write_csv(edge_csv, result["edge_rows"])
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        run_results[conductance_col] = {
            "summary": summary,
            "pressures": np.asarray(result["pressures"], dtype=np.float64),
            "edge_flow": np.asarray(result["edge_flow"], dtype=np.float64),
            "node_list": list(result["node_list"]),
        }
        aggregate_rows.append(summary)
        print(f"Solved conductance column: {conductance_col}")

    comparison_rows: List[Dict[str, object]] = []
    if args.reference_col:
        if args.reference_col not in run_results:
            raise KeyError(f"reference-col '{args.reference_col}' was not solved.")
        ref = run_results[args.reference_col]
        ref_pressure = np.asarray(ref["pressures"], dtype=np.float64)
        ref_flow = np.asarray(ref["edge_flow"], dtype=np.float64)
        ref_q = float(ref["summary"]["q_inlet"]) if math.isfinite(float(ref["summary"]["q_inlet"])) else float("nan")
        ref_k = float(ref["summary"]["k_eff"]) if math.isfinite(float(ref["summary"]["k_eff"])) else float("nan")

        for conductance_col, run in run_results.items():
            if conductance_col == args.reference_col:
                continue
            p = np.asarray(run["pressures"], dtype=np.float64)
            q_edge = np.asarray(run["edge_flow"], dtype=np.float64)
            row = {
                "conductance_col": conductance_col,
                "reference_col": args.reference_col,
                "pressure_mae": float(np.mean(np.abs(p - ref_pressure))),
                "pressure_rmse": float(np.sqrt(np.mean((p - ref_pressure) ** 2))),
                "edge_flow_mae": float(np.mean(np.abs(q_edge - ref_flow))),
                "edge_flow_rmse": float(np.sqrt(np.mean((q_edge - ref_flow) ** 2))),
                "edge_flow_r2": float(compute_r2(ref_flow, q_edge, float(args.eps))),
                "q_inlet_rel_error": float(
                    abs(float(run["summary"]["q_inlet"]) - ref_q) / max(abs(ref_q), float(args.eps))
                ) if math.isfinite(ref_q) and math.isfinite(float(run["summary"]["q_inlet"])) else float("nan"),
                "k_eff_rel_error": float(
                    abs(float(run["summary"]["k_eff"]) - ref_k) / max(abs(ref_k), float(args.eps))
                ) if math.isfinite(ref_k) and math.isfinite(float(run["summary"]["k_eff"])) else float("nan"),
            }
            comparison_rows.append(row)

    aggregate_csv = out_dir / "aggregate_summary.csv"
    write_csv(aggregate_csv, aggregate_rows)
    if comparison_rows:
        comparison_csv = out_dir / "comparison_to_reference.csv"
        write_csv(comparison_csv, comparison_rows)
        print(f"Saved comparison CSV: {comparison_csv}")

    aggregate_json = out_dir / "aggregate_summary.json"
    aggregate_json.write_text(
        json.dumps(
            {
                "edge_csv": args.edge_csv,
                "boundary_csv": args.boundary_csv,
                "conductance_cols": args.conductance_cols,
                "reference_col": args.reference_col,
                "mu": float(args.mu),
                "sample_length": float(args.sample_length),
                "bulk_area": float(args.bulk_area),
                "runs": aggregate_rows,
                "comparisons": comparison_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved aggregate summary: {aggregate_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
