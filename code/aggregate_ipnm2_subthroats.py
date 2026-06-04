#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate IPNM2 sub-throat conductance into full-throat conductance by series combination.

IPNM2 core relation:
    1 / g_ij = sum_k (1 / g_k)

This script expects:
1. A sub-throat conductance CSV, e.g. from postprocess_conductance.py
2. A mapping CSV that tells which sub-throats belong to the same throat

Required mapping columns by default:
    throat_id, sub_id

Required sub-throat CSV column by default:
    local_id

Example:
python code\\aggregate_ipnm2_subthroats.py ^
  --sub-csv "E:\\mhw\\1\\pore\\runs\\conductance_post\\conductance_results.csv" ^
  --map-csv "E:\\mhw\\1\\pore\\pnm\\subthroat_to_throat.csv" ^
  --sub-key-col local_id ^
  --map-sub-key-col sub_id ^
  --map-throat-col throat_id ^
  --conductance-cols g_pred g_true_velocity g_table ^
  --out-csv "E:\\mhw\\1\\pore\\pnm\\throat_conductance_ipnm2.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate IPNM2 sub-throat conductance by series combination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sub-csv", type=str, required=True, help="Sub-throat conductance CSV.")
    parser.add_argument("--map-csv", type=str, required=True, help="Mapping CSV from sub-throat to throat.")
    parser.add_argument("--out-csv", type=str, required=True, help="Output throat-level conductance CSV.")
    parser.add_argument("--summary-json", type=str, default="", help="Optional summary JSON path.")
    parser.add_argument("--sub-key-col", type=str, default="local_id", help="Key column in sub-throat CSV.")
    parser.add_argument("--map-sub-key-col", type=str, default="sub_id", help="Sub-throat key column in mapping CSV.")
    parser.add_argument("--map-throat-col", type=str, default="throat_id", help="Throat id column in mapping CSV.")
    parser.add_argument(
        "--group-cols",
        type=str,
        default="rock_type",
        help="Optional extra columns copied into the aggregation key to avoid cross-rock collisions.",
    )
    parser.add_argument(
        "--conductance-cols",
        type=str,
        nargs="+",
        required=True,
        help="Conductance columns aggregated via harmonic sum, e.g. g_pred g_true_velocity g_table.",
    )
    parser.add_argument("--eps", type=float, default=1.0e-12, help="Small epsilon for stable division.")
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fields


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_key(row: Dict[str, str], cols: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row[col]).strip() for col in cols)


def harmonic_series(values: Sequence[float], eps: float) -> float:
    vals = [float(v) for v in values if float(v) > eps]
    if not vals:
        return 0.0
    return float(1.0 / np.sum(1.0 / np.asarray(vals, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    sub_rows, _ = read_csv(Path(args.sub_csv))
    map_rows, _ = read_csv(Path(args.map_csv))
    group_cols = parse_list(args.group_cols)

    lookup: Dict[Tuple[str, ...], Dict[str, str]] = {}
    sub_key_cols = group_cols + [args.sub_key_col]
    for row in sub_rows:
        key = build_key(row, sub_key_cols)
        if key in lookup:
            raise ValueError(f"Duplicate sub-throat key in sub CSV: {key}")
        lookup[key] = row

    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    throat_key_cols = group_cols + [args.map_throat_col]
    map_sub_key_cols = group_cols + [args.map_sub_key_col]
    for row in map_rows:
        throat_key = build_key(row, throat_key_cols)
        sub_key = build_key(row, map_sub_key_cols)
        if sub_key not in lookup:
            raise KeyError(f"Mapping references missing sub-throat key: {sub_key}")
        grouped.setdefault(throat_key, []).append(lookup[sub_key])

    out_rows: List[Dict[str, object]] = []
    for throat_key, members in grouped.items():
        out: Dict[str, object] = {}
        for col_name, value in zip(throat_key_cols, throat_key):
            out[col_name] = value
        out["n_subthroats"] = len(members)

        for g_col in args.conductance_cols:
            values = []
            for member in members:
                raw = str(member.get(g_col, "")).strip()
                if raw == "":
                    continue
                values.append(float(raw))
            out[g_col] = harmonic_series(values, eps=float(args.eps))
        out_rows.append(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else throat_key_cols + ["n_subthroats"] + list(args.conductance_cols)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "sub_csv": args.sub_csv,
        "map_csv": args.map_csv,
        "out_csv": args.out_csv,
        "group_cols": group_cols,
        "sub_key_col": args.sub_key_col,
        "map_sub_key_col": args.map_sub_key_col,
        "map_throat_col": args.map_throat_col,
        "conductance_cols": args.conductance_cols,
        "n_sub_rows": len(sub_rows),
        "n_map_rows": len(map_rows),
        "n_throats": len(out_rows),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved throat-level IPNM2 conductance CSV: {out_path}")
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
