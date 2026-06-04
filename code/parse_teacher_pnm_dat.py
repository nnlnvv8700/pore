#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse teacher-provided PNM .dat exports into CSV tables that match this repo's
network-pipeline expectations.

The script targets the directory layout under:
    results/喉道资料/2025大创/300x200x200_data

For each rock directory such as Bead/Benth/Berea_ICL/Font18, it reads:
    out_throat_minimum_cross/PNM_300x200x200_link1.dat
    out_throat_minimum_cross/PNM_300x200x200_link2.dat
    out_throat_minimum_cross/PNM_300x200x200_node1.dat
    out_throat_minimum_cross/PNM_300x200x200_node2.dat

and writes:
    edges.csv
    nodes.csv
    boundary_nodes.csv
    subthroat_identity_map.csv
    parse_summary.json

Important:
1. The exact physical meaning of several numeric columns is not documented in
   the raw files. Those fields are therefore exported with conservative names
   like link1_value_1 / node2_value_1.
2. The script also emits an identity sub-map under the explicit assumption that
   Cross_section_sub{id}.in corresponds directly to edge_id == id. This is
   consistent with the file counts in several rock folders, but should still be
   treated as a working hypothesis until confirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_ROCKS = ["Bead", "Benth", "Berea_ICL", "Font18"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse teacher PNM .dat exports into CSV tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"E:\mhw\1\pore\results\喉道资料\2025大创\300x200x200_data",
        help="Root directory containing teacher-provided rock folders.",
    )
    parser.add_argument(
        "--rocks",
        nargs="+",
        default=DEFAULT_ROCKS,
        help="Rock folders to parse.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"E:\mhw\1\pore\pnm_from_teacher",
        help="Output directory for parsed CSVs.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_link_table(path: Path, expected_len: int, has_count_header: bool) -> Tuple[List[Dict[str, object]], List[float]]:
    lines = read_lines(path)
    if not lines:
        raise ValueError(f"Empty file: {path}")
    header = [float(tok) for tok in lines[0].split()] if has_count_header else []
    rows: List[Dict[str, object]] = []
    data_lines = lines[1:] if has_count_header else lines
    for line in data_lines:
        tokens = line.split()
        if len(tokens) != expected_len:
            raise ValueError(f"Unexpected column count in {path}: expected {expected_len}, got {len(tokens)}")
        values = [float(tok) for tok in tokens]
        row = {"edge_id": int(values[0])}
        for idx, value in enumerate(values[1:], start=1):
            row[f"value_{idx}"] = value
        rows.append(row)
    return rows, header


def parse_node1(path: Path) -> Tuple[List[Dict[str, object]], List[float]]:
    lines = read_lines(path)
    if not lines:
        raise ValueError(f"Empty file: {path}")
    header = [float(tok) for tok in lines[0].split()]
    rows: List[Dict[str, object]] = []
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) < 8:
            raise ValueError(f"Too few columns in {path}: {line}")
        node_id = int(float(tokens[0]))
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])
        degree = int(float(tokens[4]))
        min_expected = 5 + degree + 2 + degree
        if len(tokens) != min_expected:
            raise ValueError(
                f"Unexpected node row length in {path} for node {node_id}: "
                f"degree={degree}, expected {min_expected}, got {len(tokens)}"
            )
        neighbor_tokens = tokens[5:5 + degree]
        inlet_flag = int(float(tokens[5 + degree]))
        outlet_flag = int(float(tokens[6 + degree]))
        edge_tokens = tokens[7 + degree:]
        rows.append(
            {
                "node_id": node_id,
                "x": x,
                "y": y,
                "z": z,
                "degree": degree,
                "neighbor_ids": "|".join(str(int(float(tok))) for tok in neighbor_tokens),
                "inlet_flag": inlet_flag,
                "outlet_flag": outlet_flag,
                "edge_ids": "|".join(str(int(float(tok))) for tok in edge_tokens),
            }
        )
    return rows, header


def parse_node2(path: Path) -> List[Dict[str, object]]:
    lines = read_lines(path)
    rows: List[Dict[str, object]] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"Unexpected node2 column count in {path}: {len(tokens)}")
        values = [float(tok) for tok in tokens]
        rows.append(
            {
                "node_id": int(values[0]),
                "node2_value_1": values[1],
                "node2_value_2": values[2],
                "node2_value_3": values[3],
                "node2_value_4": values[4],
            }
        )
    return rows


def index_rows(rows: Iterable[Dict[str, object]], key: str) -> Dict[int, Dict[str, object]]:
    indexed: Dict[int, Dict[str, object]] = {}
    for row in rows:
        row_key = int(row[key])
        if row_key in indexed:
            raise ValueError(f"Duplicate key {row_key} for '{key}'")
        indexed[row_key] = dict(row)
    return indexed


def collect_cross_section_ids(path: Path) -> List[int]:
    ids: List[int] = []
    for file_path in sorted(path.glob("Cross_section_sub*.in")):
        name = file_path.name
        if "(" in name:
            continue
        stem = file_path.stem
        sub_id = int(stem.replace("Cross_section_sub", ""))
        ids.append(sub_id)
    return ids


def summarize_id_alignment(edge_ids: Sequence[int], sub_ids: Sequence[int]) -> Dict[str, object]:
    edge_set = set(edge_ids)
    sub_set = set(sub_ids)
    return {
        "n_edge_ids": len(edge_ids),
        "n_cross_section_ids": len(sub_ids),
        "edge_ids_match_cross_section_ids": edge_set == sub_set,
        "min_edge_id": min(edge_ids) if edge_ids else None,
        "max_edge_id": max(edge_ids) if edge_ids else None,
        "min_cross_section_id": min(sub_ids) if sub_ids else None,
        "max_cross_section_id": max(sub_ids) if sub_ids else None,
        "edge_ids_missing_cross_section": sorted(edge_set - sub_set)[:20],
        "cross_section_ids_missing_edge": sorted(sub_set - edge_set)[:20],
    }


def build_boundary_rows(nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in nodes:
        if int(row["inlet_flag"]) != 0:
            out.append(
                {
                    "pore_id": int(row["node_id"]),
                    "pressure": 1.0,
                    "label": "inlet",
                }
            )
        if int(row["outlet_flag"]) != 0:
            out.append(
                {
                    "pore_id": int(row["node_id"]),
                    "pressure": 0.0,
                    "label": "outlet",
                }
            )
    return out


def parse_one_rock(data_root: Path, out_root: Path, rock: str) -> Dict[str, object]:
    rock_dir = data_root / rock / "out_throat_minimum_cross"
    if not rock_dir.exists():
        raise FileNotFoundError(f"Rock directory not found: {rock_dir}")

    link1_rows, link1_header = parse_link_table(
        rock_dir / "PNM_300x200x200_link1.dat",
        expected_len=6,
        has_count_header=True,
    )
    link2_rows, link2_header = parse_link_table(
        rock_dir / "PNM_300x200x200_link2.dat",
        expected_len=8,
        has_count_header=False,
    )
    node1_rows, node1_header = parse_node1(rock_dir / "PNM_300x200x200_node1.dat")
    node2_rows = parse_node2(rock_dir / "PNM_300x200x200_node2.dat")
    cross_section_ids = collect_cross_section_ids(rock_dir)

    link1_by_id = index_rows(link1_rows, "edge_id")
    link2_by_id = index_rows(link2_rows, "edge_id")
    node2_by_id = index_rows(node2_rows, "node_id")

    edge_ids = sorted(link1_by_id.keys())
    if set(edge_ids) != set(link2_by_id.keys()):
        raise ValueError(f"link1/link2 edge_id mismatch for {rock}")

    edges_out: List[Dict[str, object]] = []
    for edge_id in edge_ids:
        l1 = link1_by_id[edge_id]
        l2 = link2_by_id[edge_id]
        edges_out.append(
            {
                "edge_id": edge_id,
                "pore_i": int(l1["value_1"]),
                "pore_j": int(l1["value_2"]),
                "link1_value_1": l1["value_3"],
                "link1_value_2": l1["value_4"],
                "link1_value_3": l1["value_5"],
                "link2_pore_i": int(l2["value_1"]),
                "link2_pore_j": int(l2["value_2"]),
                "link2_value_1": l2["value_3"],
                "link2_value_2": l2["value_4"],
                "link2_value_3": l2["value_5"],
                "link2_value_4": l2["value_6"],
                "link2_value_5": l2["value_7"],
                "rock_type_name": rock,
            }
        )

    nodes_out: List[Dict[str, object]] = []
    for node in node1_rows:
        merged = dict(node)
        node2 = node2_by_id.get(int(node["node_id"]))
        if node2 is not None:
            merged.update(
                {
                    "node2_value_1": node2["node2_value_1"],
                    "node2_value_2": node2["node2_value_2"],
                    "node2_value_3": node2["node2_value_3"],
                    "node2_value_4": node2["node2_value_4"],
                }
            )
        merged["rock_type_name"] = rock
        nodes_out.append(merged)

    boundaries_out = build_boundary_rows(nodes_out)

    edge_lookup = {int(row["edge_id"]): row for row in edges_out}
    submap_out: List[Dict[str, object]] = []
    for sub_id in sorted(cross_section_ids):
        edge_row = edge_lookup.get(sub_id)
        if edge_row is None:
            continue
        submap_out.append(
            {
                "sub_id": sub_id,
                "edge_id": sub_id,
                "pore_i": edge_row["pore_i"],
                "pore_j": edge_row["pore_j"],
                "k": 0,
                "L_k": "",
                "local_id": sub_id,
                "global_id": "",
                "rock_type_name": rock,
                "note": "Identity map generated from Cross_section_sub{id}.in -> edge_id=id. Confirm with teacher before publication.",
            }
        )

    rock_out_dir = out_root / rock
    write_csv(
        rock_out_dir / "edges.csv",
        edges_out,
        [
            "edge_id",
            "pore_i",
            "pore_j",
            "link1_value_1",
            "link1_value_2",
            "link1_value_3",
            "link2_pore_i",
            "link2_pore_j",
            "link2_value_1",
            "link2_value_2",
            "link2_value_3",
            "link2_value_4",
            "link2_value_5",
            "rock_type_name",
        ],
    )
    write_csv(
        rock_out_dir / "nodes.csv",
        nodes_out,
        [
            "node_id",
            "x",
            "y",
            "z",
            "degree",
            "neighbor_ids",
            "inlet_flag",
            "outlet_flag",
            "edge_ids",
            "node2_value_1",
            "node2_value_2",
            "node2_value_3",
            "node2_value_4",
            "rock_type_name",
        ],
    )
    write_csv(
        rock_out_dir / "boundary_nodes.csv",
        boundaries_out,
        ["pore_id", "pressure", "label"],
    )
    write_csv(
        rock_out_dir / "subthroat_identity_map.csv",
        submap_out,
        [
            "sub_id",
            "edge_id",
            "pore_i",
            "pore_j",
            "k",
            "L_k",
            "local_id",
            "global_id",
            "rock_type_name",
            "note",
        ],
    )

    summary = {
        "rock": rock,
        "source_dir": str(rock_dir),
        "link1_header": link1_header,
        "link2_header": link2_header,
        "node1_header": node1_header,
        "n_edges": len(edges_out),
        "n_nodes": len(nodes_out),
        "n_boundary_nodes": len(boundaries_out),
        "n_cross_sections": len(cross_section_ids),
        "identity_submap_rows": len(submap_out),
        "id_alignment": summarize_id_alignment(edge_ids=edge_ids, sub_ids=cross_section_ids),
        "column_notes": {
            "link1_value_*": "Undocumented raw numeric columns from PNM_..._link1.dat",
            "link2_value_*": "Undocumented raw numeric columns from PNM_..._link2.dat",
            "node2_value_*": "Undocumented raw numeric columns from PNM_..._node2.dat",
            "inlet_flag/outlet_flag": "Parsed from node1 variable-length rows and used to build boundary_nodes.csv",
        },
        "warning": (
            "subthroat_identity_map.csv assumes Cross_section_sub{id}.in corresponds directly to edge_id=id. "
            "Use as a working map for code integration, not as a final published claim without confirmation."
        ),
    }
    (rock_out_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, object]] = []
    for rock in args.rocks:
        summary = parse_one_rock(data_root=data_root, out_root=out_root, rock=rock)
        all_summaries.append(summary)
        print(
            f"[OK] {rock}: edges={summary['n_edges']} nodes={summary['n_nodes']} "
            f"boundary_nodes={summary['n_boundary_nodes']} cross_sections={summary['n_cross_sections']}"
        )

    top_summary = {
        "data_root": str(data_root),
        "out_dir": str(out_root),
        "rocks": [summary["rock"] for summary in all_summaries],
        "summaries": all_summaries,
    }
    (out_root / "parse_summary_all.json").write_text(json.dumps(top_summary, indent=2), encoding="utf-8")
    print(f"[DONE] Parsed teacher PNM exports to: {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}")
