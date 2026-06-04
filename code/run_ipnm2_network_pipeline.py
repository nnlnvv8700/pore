#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the network-scale IPNM2 pipeline after local sub-throat conductance is ready.

Pipeline:
1. Aggregate sub-throat conductance into throat conductance by series combination.
2. Backfill throat conductance into the edge table.
3. Solve the pore network for one or more conductance columns.
4. Optionally validate mapping inputs before running.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IPNM2 network-scale pipeline from local sub-throat conductance to PNM solve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sub-csv", type=str, required=True, help="Sub-throat conductance CSV from postprocess_conductance.py")
    parser.add_argument("--sub-map-csv", type=str, required=True, help="subthroat_to_throat.csv")
    parser.add_argument("--edges-csv", type=str, required=True, help="Base edge table.")
    parser.add_argument("--boundary-csv", type=str, required=True, help="Boundary node CSV.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for the full pipeline.")
    parser.add_argument("--group-cols", type=str, default="rock_type", help="Grouping columns shared across mapping files.")
    parser.add_argument("--sub-key-col", type=str, default="local_id", help="Sub-throat key in sub CSV.")
    parser.add_argument("--map-sub-key-col", type=str, default="sub_id", help="Sub-throat key in mapping CSV.")
    parser.add_argument("--map-throat-col", type=str, default="edge_id", help="Target throat/edge id in mapping CSV.")
    parser.add_argument(
        "--conductance-cols",
        type=str,
        nargs="+",
        default=["g_pred", "g_true_velocity", "g_table"],
        help="Throat conductance columns to aggregate and solve.",
    )
    parser.add_argument("--reference-col", type=str, default="g_true_velocity", help="Reference conductance column for comparison.")
    parser.add_argument("--inlet-label", type=str, default="inlet")
    parser.add_argument("--outlet-label", type=str, default="outlet")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--sample-length", type=float, default=1.0)
    parser.add_argument("--bulk-area", type=float, default=1.0)
    parser.add_argument("--validate-first", action="store_true", help="Run mapping validation before the pipeline.")
    return parser.parse_args()


def run_cmd(args: List[str]) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_dir = Path(__file__).resolve().parent

    if args.validate_first:
        run_cmd(
            [
                sys.executable,
                str(code_dir / "validate_ipnm2_mapping.py"),
                "--edges-csv",
                args.edges_csv,
                "--sub-map-csv",
                args.sub_map_csv,
                "--boundary-csv",
                args.boundary_csv,
                "--sub-csv",
                args.sub_csv,
                "--group-cols",
                args.group_cols,
                "--summary-json",
                str(out_dir / "mapping_validation.json"),
            ]
        )

    throat_csv = out_dir / "throat_conductance_ipnm2.csv"
    run_cmd(
        [
            sys.executable,
            str(code_dir / "aggregate_ipnm2_subthroats.py"),
            "--sub-csv",
            args.sub_csv,
            "--map-csv",
            args.sub_map_csv,
            "--sub-key-col",
            args.sub_key_col,
            "--map-sub-key-col",
            args.map_sub_key_col,
            "--map-throat-col",
            args.map_throat_col,
            "--group-cols",
            args.group_cols,
            "--conductance-cols",
            *args.conductance_cols,
            "--out-csv",
            str(throat_csv),
        ]
    )

    backfilled_edges = out_dir / "edges_with_ipnm2_conductance.csv"
    run_cmd(
        [
            sys.executable,
            str(code_dir / "backfill_conductance_to_pnm.py"),
            "--edge-csv",
            args.edges_csv,
            "--conductance-csv",
            str(throat_csv),
            "--edge-match-cols",
            ",".join([c for c in [*([col for col in args.group_cols.split(",") if col.strip()]), args.map_throat_col] if c]),
            "--conductance-match-cols",
            ",".join([c for c in [*([col for col in args.group_cols.split(",") if col.strip()]), args.map_throat_col] if c]),
            "--copy-cols",
            ",".join(args.conductance_cols),
            "--out-csv",
            str(backfilled_edges),
            "--missing-policy",
            "error",
        ]
    )

    solve_out_dir = out_dir / "network_solve"
    run_cmd(
        [
            sys.executable,
            str(code_dir / "solve_pnm_with_conductance.py"),
            "--edge-csv",
            str(backfilled_edges),
            "--boundary-csv",
            args.boundary_csv,
            "--conductance-cols",
            *args.conductance_cols,
            "--reference-col",
            args.reference_col,
            "--inlet-label",
            args.inlet_label,
            "--outlet-label",
            args.outlet_label,
            "--mu",
            str(args.mu),
            "--sample-length",
            str(args.sample_length),
            "--bulk-area",
            str(args.bulk_area),
            "--out-dir",
            str(solve_out_dir),
        ]
    )

    print(f"Finished IPNM2 network pipeline. Outputs: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
