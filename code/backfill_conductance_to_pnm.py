#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Join surrogate conductance results back into a PNM edge table.

Typical use:
1. Run postprocess_conductance.py to obtain per-element conductance rows.
2. Prepare an edge table that contains the same matching key(s), for example
   global_id or dataset_index.
3. Use this script to copy g_pred / g_true / g_table into the edge table.

The edge CSV is intentionally generic. At minimum it should contain:
    edge_id,node_i,node_j,<match columns>

Example:
python backfill_conductance_to_pnm.py ^
  --edge-csv "E:\\mhw\\1\\pore\\pnm\\edges.csv" ^
  --conductance-csv "E:\\mhw\\1\\pore\\runs\\unet\\infer\\conductance_post\\conductance_results.csv" ^
  --edge-match-cols global_id ^
  --conductance-match-cols global_id ^
  --copy-cols g_pred,g_true_velocity,g_table,q_pred_abs,local_id,rock_type ^
  --active-source-col g_pred ^
  --active-target-col conductance_dl ^
  --out-csv "E:\\mhw\\1\\pore\\pnm\\edges_with_conductance.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join local conductance results into a PNM edge CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--edge-csv", type=str, required=True, help="Input PNM edge CSV.")
    parser.add_argument("--conductance-csv", type=str, required=True, help="Per-element conductance CSV.")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV with copied conductance columns.")
    parser.add_argument(
        "--edge-match-cols",
        type=str,
        required=True,
        help="Comma-separated key columns in the edge CSV, e.g. global_id or rock_type,local_id.",
    )
    parser.add_argument(
        "--conductance-match-cols",
        type=str,
        default="",
        help="Comma-separated key columns in the conductance CSV. Defaults to edge-match-cols.",
    )
    parser.add_argument(
        "--copy-cols",
        type=str,
        default="g_pred,g_true_velocity,g_table,q_pred_abs,local_id,rock_type,dataset_index,global_id",
        help="Comma-separated columns copied from conductance CSV into edge CSV when available.",
    )
    parser.add_argument(
        "--active-source-col",
        type=str,
        default="",
        help="Optional conductance column from conductance CSV to copy into active-target-col.",
    )
    parser.add_argument(
        "--active-target-col",
        type=str,
        default="",
        help="Optional target column name written into the output edge CSV.",
    )
    parser.add_argument(
        "--missing-policy",
        type=str,
        choices=("error", "keep", "empty"),
        default="error",
        help="How to handle unmatched edges.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional summary JSON path. Defaults to <out-csv stem>_summary.json.",
    )
    parser.add_argument(
        "--missing-csv",
        type=str,
        default="",
        help="Optional CSV path for unmatched edge rows.",
    )
    return parser.parse_args()


def parse_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fieldnames


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def key_of(row: Dict[str, str], cols: Sequence[str]) -> Tuple[str, ...]:
    missing = [col for col in cols if col not in row]
    if missing:
        raise KeyError(f"Missing key columns {missing} in row with columns {list(row.keys())}")
    return tuple(str(row[col]).strip() for col in cols)


def main() -> None:
    args = parse_args()

    edge_path = Path(args.edge_csv)
    conductance_path = Path(args.conductance_csv)
    out_path = Path(args.out_csv)
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_name(out_path.stem + "_summary.json")
    missing_path = Path(args.missing_csv) if args.missing_csv else None

    edge_rows, edge_fields = parse_csv(edge_path)
    conductance_rows, conductance_fields = parse_csv(conductance_path)

    edge_key_cols = parse_list(args.edge_match_cols)
    if not edge_key_cols:
        raise ValueError("edge-match-cols must not be empty.")
    conductance_key_cols = parse_list(args.conductance_match_cols) or edge_key_cols
    copy_cols = parse_list(args.copy_cols)

    conductance_lookup: Dict[Tuple[str, ...], Dict[str, str]] = {}
    duplicate_keys: List[Tuple[str, ...]] = []
    for row in conductance_rows:
        key = key_of(row, conductance_key_cols)
        if key in conductance_lookup:
            duplicate_keys.append(key)
        conductance_lookup[key] = row
    if duplicate_keys:
        dup_preview = ", ".join(str(key) for key in duplicate_keys[:5])
        raise ValueError(f"Duplicate conductance keys detected, for example: {dup_preview}")

    output_fields = list(edge_fields)
    for col in copy_cols:
        if col in conductance_fields and col not in output_fields:
            output_fields.append(col)
    if args.active_source_col and args.active_target_col and args.active_target_col not in output_fields:
        output_fields.append(args.active_target_col)

    matched = 0
    missing = 0
    unmatched_rows: List[Dict[str, str]] = []
    used_keys = set()
    out_rows: List[Dict[str, str]] = []

    for edge_row in edge_rows:
        out_row = dict(edge_row)
        edge_key = key_of(edge_row, edge_key_cols)
        matched_row = conductance_lookup.get(edge_key)
        if matched_row is None:
            missing += 1
            unmatched_rows.append(out_row)
            if args.missing_policy == "error":
                raise KeyError(f"No conductance row found for edge key {edge_key}")
            if args.missing_policy == "empty":
                for col in copy_cols:
                    if col in conductance_fields:
                        out_row[col] = ""
                if args.active_source_col and args.active_target_col:
                    out_row[args.active_target_col] = ""
            out_rows.append(out_row)
            continue

        matched += 1
        used_keys.add(edge_key)
        for col in copy_cols:
            if col in conductance_fields:
                out_row[col] = matched_row.get(col, "")
        if args.active_source_col and args.active_target_col:
            if args.active_source_col not in conductance_fields:
                raise KeyError(f"active-source-col '{args.active_source_col}' not present in conductance CSV.")
            out_row[args.active_target_col] = matched_row.get(args.active_source_col, "")
        out_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    if missing_path is not None and unmatched_rows:
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with missing_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=edge_fields)
            writer.writeheader()
            writer.writerows(unmatched_rows)

    summary = {
        "edge_csv": str(edge_path),
        "conductance_csv": str(conductance_path),
        "out_csv": str(out_path),
        "edge_match_cols": edge_key_cols,
        "conductance_match_cols": conductance_key_cols,
        "copy_cols": [col for col in copy_cols if col in conductance_fields],
        "active_source_col": args.active_source_col,
        "active_target_col": args.active_target_col,
        "n_edge_rows": len(edge_rows),
        "n_conductance_rows": len(conductance_rows),
        "matched_edges": matched,
        "missing_edges": missing,
        "unused_conductance_rows": len(conductance_lookup) - len(used_keys),
        "missing_policy": args.missing_policy,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved backfilled edge CSV: {out_path}")
    print(f"Saved summary JSON: {summary_path}")
    if missing_path is not None and unmatched_rows:
        print(f"Saved unmatched edges CSV: {missing_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
