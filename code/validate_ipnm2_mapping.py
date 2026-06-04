#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate the consistency of the IPNM2 mapping layer.

Checks:
1. Every sub-throat row points to an existing edge.
2. Every boundary pore exists in the edge graph.
3. Optional: every mapped sub_id exists in the sub-throat conductance CSV.
4. Optional: every conductance row is either used or reported as unused.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IPNM2 sub-throat to throat to network mapping files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--edges-csv", type=str, required=True)
    parser.add_argument("--sub-map-csv", type=str, required=True)
    parser.add_argument("--boundary-csv", type=str, required=True)
    parser.add_argument("--sub-csv", type=str, default="", help="Optional sub-throat conductance CSV.")
    parser.add_argument("--edge-id-col", type=str, default="edge_id")
    parser.add_argument("--node-i-col", type=str, default="pore_i")
    parser.add_argument("--node-j-col", type=str, default="pore_j")
    parser.add_argument("--map-edge-col", type=str, default="edge_id")
    parser.add_argument("--map-sub-col", type=str, default="sub_id")
    parser.add_argument("--map-pore-i-col", type=str, default="pore_i")
    parser.add_argument("--map-pore-j-col", type=str, default="pore_j")
    parser.add_argument("--boundary-pore-col", type=str, default="pore_id")
    parser.add_argument("--sub-key-col", type=str, default="local_id", help="Key column in sub-throat conductance CSV.")
    parser.add_argument("--group-cols", type=str, default="rock_type", help="Optional shared grouping columns.")
    parser.add_argument("--summary-json", type=str, required=True)
    return parser.parse_args()


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fields


def key_of(row: Dict[str, str], cols: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row[col]).strip() for col in cols)


def main() -> None:
    args = parse_args()
    group_cols = parse_list(args.group_cols)

    edge_rows, _ = read_csv(Path(args.edges_csv))
    map_rows, _ = read_csv(Path(args.sub_map_csv))
    boundary_rows, _ = read_csv(Path(args.boundary_csv))

    edge_key_cols = group_cols + [args.edge_id_col]
    edge_lookup = {key_of(row, edge_key_cols): row for row in edge_rows}
    pore_ids = set()
    for row in edge_rows:
        pore_ids.add(str(row[args.node_i_col]).strip())
        pore_ids.add(str(row[args.node_j_col]).strip())

    missing_edge_refs: List[Tuple[str, ...]] = []
    inconsistent_pore_pairs: List[Dict[str, object]] = []
    throat_counts: Dict[Tuple[str, ...], int] = {}
    for row in map_rows:
        edge_key = key_of(row, group_cols + [args.map_edge_col])
        if edge_key not in edge_lookup:
            missing_edge_refs.append(edge_key)
        else:
            edge_row = edge_lookup[edge_key]
            if args.map_pore_i_col in row and args.map_pore_j_col in row:
                map_pair = (str(row[args.map_pore_i_col]).strip(), str(row[args.map_pore_j_col]).strip())
                edge_pair = (str(edge_row[args.node_i_col]).strip(), str(edge_row[args.node_j_col]).strip())
                edge_pair_rev = (edge_pair[1], edge_pair[0])
                if map_pair != edge_pair and map_pair != edge_pair_rev:
                    inconsistent_pore_pairs.append(
                        {
                            "edge_key": list(edge_key),
                            "map_pair": list(map_pair),
                            "edge_pair": list(edge_pair),
                        }
                    )
        throat_counts[edge_key] = throat_counts.get(edge_key, 0) + 1

    missing_boundary_pores: List[str] = []
    for row in boundary_rows:
        pore_id = str(row[args.boundary_pore_col]).strip()
        if pore_id not in pore_ids:
            missing_boundary_pores.append(pore_id)

    unused_sub_rows = None
    missing_sub_rows = None
    if args.sub_csv:
        sub_rows, _ = read_csv(Path(args.sub_csv))
        sub_key_cols = group_cols + [args.sub_key_col]
        sub_lookup = {key_of(row, sub_key_cols): row for row in sub_rows}
        mapped_sub_keys = {key_of(row, group_cols + [args.map_sub_col]) for row in map_rows}
        missing_sub_rows = [list(k) for k in sorted(mapped_sub_keys) if k not in sub_lookup]
        unused_sub_rows = [list(k) for k in sorted(sub_lookup.keys()) if k not in mapped_sub_keys]

    summary = {
        "edges_csv": args.edges_csv,
        "sub_map_csv": args.sub_map_csv,
        "boundary_csv": args.boundary_csv,
        "sub_csv": args.sub_csv,
        "group_cols": group_cols,
        "n_edges": len(edge_rows),
        "n_sub_map_rows": len(map_rows),
        "n_boundary_rows": len(boundary_rows),
        "n_unique_throats_in_map": len(throat_counts),
        "missing_edge_refs": [list(k) for k in missing_edge_refs],
        "n_missing_edge_refs": len(missing_edge_refs),
        "missing_boundary_pores": missing_boundary_pores,
        "n_missing_boundary_pores": len(missing_boundary_pores),
        "inconsistent_pore_pairs": inconsistent_pore_pairs,
        "n_inconsistent_pore_pairs": len(inconsistent_pore_pairs),
        "sub_rows_missing_from_map": missing_sub_rows,
        "sub_rows_unused_by_map": unused_sub_rows,
        "is_valid": len(missing_edge_refs) == 0
                    and len(missing_boundary_pores) == 0
                    and len(inconsistent_pore_pairs) == 0
                    and (missing_sub_rows in (None, [])),
    }

    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mapping validation summary: {out_path}")
    print(f"Mapping valid: {summary['is_valid']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
