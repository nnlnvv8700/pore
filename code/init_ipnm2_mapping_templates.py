#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate template CSVs for the missing IPNM2 network mapping layer.

This script creates empty-but-documented CSV templates for:
1. edges.csv
2. subthroat_to_throat.csv
3. boundary_nodes.csv

These files are not provided by the paper directly, but they are the natural
materialization of the paper's mapping rules:
    sub-throat k -> throat (i, j) -> pore-network edge between pore_i and pore_j
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


EDGES_FIELDS = [
    "edge_id",
    "pore_i",
    "pore_j",
    "L_ij",
    "rock_type",
    "note",
]

SUBTHROAT_FIELDS = [
    "sub_id",
    "edge_id",
    "pore_i",
    "pore_j",
    "k",
    "L_k",
    "rock_type",
    "local_id",
    "global_id",
    "note",
]

BOUNDARY_FIELDS = [
    "pore_id",
    "pressure",
    "label",
    "rock_type",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create template CSVs for the IPNM2 mapping layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for template CSVs.")
    parser.add_argument(
        "--rock-type-aware",
        action="store_true",
        help="Keep rock_type as part of all templates. Recommended for multi-rock workflows.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: List[str], example_row: Dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(example_row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_fields = list(EDGES_FIELDS)
    sub_fields = list(SUBTHROAT_FIELDS)
    boundary_fields = list(BOUNDARY_FIELDS)

    if not args.rock_type_aware:
        edges_fields.remove("rock_type")
        sub_fields.remove("rock_type")
        boundary_fields.remove("rock_type")

    edges_example = {
        "edge_id": "12",
        "pore_i": "101",
        "pore_j": "205",
        "L_ij": "28.0",
        "note": "one physical throat between pore_i and pore_j",
    }
    sub_example = {
        "sub_id": "12_0",
        "edge_id": "12",
        "pore_i": "101",
        "pore_j": "205",
        "k": "0",
        "L_k": "4.0",
        "local_id": "0",
        "global_id": "0",
        "note": "first sub-throat on edge 12",
    }
    boundary_example = {
        "pore_id": "101",
        "pressure": "1.0",
        "label": "inlet",
        "note": "Dirichlet boundary node",
    }
    if args.rock_type_aware:
        edges_example["rock_type"] = "0"
        sub_example["rock_type"] = "0"
        boundary_example["rock_type"] = "0"

    write_csv(out_dir / "edges.csv", edges_fields, edges_example)
    write_csv(out_dir / "subthroat_to_throat.csv", sub_fields, sub_example)
    write_csv(out_dir / "boundary_nodes.csv", boundary_fields, boundary_example)

    schema = {
        "edges.csv": {
            "required": ["edge_id", "pore_i", "pore_j"],
            "optional": [col for col in edges_fields if col not in {"edge_id", "pore_i", "pore_j"}],
            "meaning": "One row per physical throat edge in the extracted PNM.",
        },
        "subthroat_to_throat.csv": {
            "required": ["sub_id", "edge_id", "k"],
            "optional": [col for col in sub_fields if col not in {"sub_id", "edge_id", "k"}],
            "meaning": "Maps each IPNM2 sub-throat segment to the parent throat edge. "
                       "pore_i/pore_j are recommended so the throat identity is explicit as (i,j).",
        },
        "boundary_nodes.csv": {
            "required": ["pore_id", "pressure"],
            "optional": [col for col in boundary_fields if col not in {"pore_id", "pressure"}],
            "meaning": "Dirichlet boundary pores used by the network solver.",
        },
    }
    (out_dir / "mapping_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Created template files in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
