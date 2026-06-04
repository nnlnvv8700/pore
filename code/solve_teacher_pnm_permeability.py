#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Solve teacher PNM pressure/permeability with optional predicted conductance backfill."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


ROCK_MAP = {
    "Bead": ("Bead", 0),
    "Benth": ("Benth", 1),
    "Berea": ("Berea_ICL", 2),
    "Font": ("Font18", 3),
}


def read_node1(path: Path) -> Tuple[List[dict], Tuple[float, float, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    first = lines[0].split()
    n = int(first[0])
    resolution = tuple(float(x) for x in first[1:4])
    nodes = [None] * n
    for line in lines[1:]:
        toks = line.split()
        node_id = int(toks[0])
        degree = int(toks[4])
        neigh = [int(x) for x in toks[5:5 + degree]]
        inlet = int(toks[5 + degree])
        outlet = int(toks[6 + degree])
        edge_ids = [int(x) for x in toks[7 + degree:7 + degree + degree]]
        nodes[node_id] = {
            "node_id": node_id,
            "x": float(toks[1]),
            "y": float(toks[2]),
            "z": float(toks[3]),
            "degree": degree,
            "neighbors": neigh,
            "inlet": inlet,
            "outlet": outlet,
            "edge_ids": edge_ids,
        }
    if any(x is None for x in nodes):
        raise ValueError(f"Missing node rows in {path}")
    return nodes, resolution


def read_node2(path: Path, nodes: List[dict]) -> None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = line.split()
        if not toks:
            continue
        node_id = int(toks[0])
        nodes[node_id]["volume"] = float(toks[1])
        nodes[node_id]["radius"] = float(toks[2])
        nodes[node_id]["shape_factor"] = float(toks[3])
        nodes[node_id]["clay_volume"] = float(toks[4])


def read_link1(path: Path) -> List[dict]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    n = int(lines[0].split()[0])
    throats = [None] * n
    for line in lines[1:]:
        toks = line.split()
        edge_id = int(toks[0])
        throats[edge_id] = {
            "edge_id": edge_id,
            "pore1": int(toks[1]),
            "pore2": int(toks[2]),
            "radius": float(toks[3]),
            "shape_factor": float(toks[4]),
            "length_total": float(toks[5]),
        }
    if any(x is None for x in throats):
        raise ValueError(f"Missing throat rows in {path}")
    return throats


def read_link2(path: Path, throats: List[dict]) -> None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = line.split()
        if not toks:
            continue
        edge_id = int(toks[0])
        throats[edge_id]["pore1_link2"] = int(toks[1])
        throats[edge_id]["pore2_link2"] = int(toks[2])
        throats[edge_id]["len_pore1"] = float(toks[3])
        throats[edge_id]["len_pore2"] = float(toks[4])
        throats[edge_id]["length_throat"] = float(toks[5])
        throats[edge_id]["volume"] = float(toks[6])
        throats[edge_id]["clay_volume"] = float(toks[7])


def read_teacher_conductivity(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = line.split()
        if toks:
            vals.append(float(toks[0]))
    return np.asarray(vals, dtype=np.float64)


def read_predicted_g(path: Path, rock_type: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not path:
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["rock_type"])) != int(rock_type):
                continue
            local_id = int(float(row["local_id"]))
            out[local_id] = float(row["g_pred"])
    return out


def compute_geometry(nodes: List[dict]) -> Tuple[float, float]:
    inlet_nodes = [n for n in nodes if int(n["inlet"]) == 1]
    outlet_nodes = [n for n in nodes if int(n["outlet"]) == 1]
    xin = np.mean([[n["x"], n["y"], n["z"]] for n in inlet_nodes], axis=0)
    xout = np.mean([[n["x"], n["y"], n["z"]] for n in outlet_nodes], axis=0)
    mins = np.min([[n["x"], n["y"], n["z"]] for n in nodes], axis=0)
    maxs = np.max([[n["x"], n["y"], n["z"]] for n in nodes], axis=0)
    d = np.abs(xin - xout)
    axis = int(np.argmax(d))
    length = float(d[axis])
    spans = maxs - mins
    if axis == 0:
        area = float(spans[1] * spans[2])
    elif axis == 1:
        area = float(spans[0] * spans[2])
    else:
        area = float(spans[0] * spans[1])
    return length, area


def solve_pnm(
    nodes: List[dict],
    throats: List[dict],
    conduct_base: np.ndarray,
    pred_g: Dict[int, float] | None = None,
    mu: float = 1.0e-6,
    p_in: float = 1.01,
    p_out: float = 1.0,
) -> dict:
    pred_g = pred_g or {}
    n = len(nodes)
    conduct = np.zeros(len(throats), dtype=np.float64)
    replaced = 0
    for i, throat in enumerate(throats):
        if i in pred_g:
            # Teacher stores conductivity_throats as g * mu * LT in the C++ code.
            conduct[i] = float(pred_g[i]) / float(mu) / float(throat["length_total"])
            replaced += 1
        else:
            conduct[i] = float(conduct_base[i]) / float(mu) / float(throat["length_total"])

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    b = np.zeros(n, dtype=np.float64)

    for node in nodes:
        i = int(node["node_id"])
        if int(node["inlet"]) == 1:
            rows.append(i); cols.append(i); data.append(1.0)
            b[i] = p_in
            continue
        if int(node["outlet"]) == 1:
            rows.append(i); cols.append(i); data.append(1.0)
            b[i] = p_out
            continue
        diag = 0.0
        for neigh, edge_id in zip(node["neighbors"], node["edge_ids"]):
            g = float(conduct[int(edge_id)])
            diag += g
            if int(nodes[int(neigh)]["inlet"]) == 0 and int(nodes[int(neigh)]["outlet"]) == 0:
                rows.append(i); cols.append(int(neigh)); data.append(-g)
            else:
                b[i] += g * (p_in if int(nodes[int(neigh)]["inlet"]) == 1 else p_out)
        rows.append(i); cols.append(i); data.append(diag)

    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    pressure = spsolve(mat, b)

    inlet_ids = [int(x["node_id"]) for x in nodes if int(x["inlet"]) == 1]
    outlet_ids = [int(x["node_id"]) for x in nodes if int(x["outlet"]) == 1]
    q_in = 0.0
    for i in inlet_ids:
        node = nodes[i]
        for neigh, edge_id in zip(node["neighbors"], node["edge_ids"]):
            q_in += float(conduct[int(edge_id)]) * (p_in - pressure[int(neigh)])
    q_out = 0.0
    for i in outlet_ids:
        node = nodes[i]
        for neigh, edge_id in zip(node["neighbors"], node["edge_ids"]):
            q_out += float(conduct[int(edge_id)]) * (pressure[int(neigh)] - p_out)

    length, area = compute_geometry(nodes)
    perm = ((q_in + q_out) / 2.0) * float(mu) * length / area / (p_in - p_out)
    return {
        "n_pores": n,
        "n_throats": len(throats),
        "n_pred_replaced": int(replaced),
        "coverage": float(replaced / max(len(throats), 1)),
        "q_inlet": float(q_in),
        "q_outlet": float(q_out),
        "length": float(length),
        "area": float(area),
        "permeability": float(perm),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
    }


def solve_one(rock_dir: Path, pred_csv: Path | None, rock_type: int) -> Tuple[dict, dict]:
    nodes, _ = read_node1(rock_dir / "PNM_node1.dat")
    read_node2(rock_dir / "PNM_node2.dat", nodes)
    throats = read_link1(rock_dir / "PNM_link1.dat")
    read_link2(rock_dir / "PNM_link2.dat", throats)
    teacher_cond = read_teacher_conductivity(rock_dir / "conductivity_throats.dat")
    teacher = solve_pnm(nodes, throats, teacher_cond, pred_g=None)
    pred = {}
    if pred_csv is not None:
        pred_g = read_predicted_g(pred_csv, rock_type)
        pred = solve_pnm(nodes, throats, teacher_cond, pred_g=pred_g)
    return teacher, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnm-root", type=str, default=r"E:\mhw\1\cup\300x200x200_data\PNM_simulation")
    parser.add_argument("--pred-csv", type=str, default="")
    parser.add_argument("--rocks", nargs="+", default=["Bead", "Benth", "Berea", "Font"])
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\global_pnm_permeability_20260529")
    args = parser.parse_args()

    pnm_root = Path(args.pnm_root)
    pred_csv = Path(args.pred_csv) if args.pred_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rock in args.rocks:
        rock_folder, rock_type = ROCK_MAP[rock]
        teacher, pred = solve_one(pnm_root / rock, pred_csv, rock_type)
        row = {
            "rock": rock,
            "teacher_perm": teacher["permeability"],
            "teacher_q_inlet": teacher["q_inlet"],
            "teacher_q_outlet": teacher["q_outlet"],
            "n_throats": teacher["n_throats"],
        }
        if pred:
            row.update(
                {
                    "pred_perm": pred["permeability"],
                    "perm_rel_error": abs(pred["permeability"] - teacher["permeability"]) / max(abs(teacher["permeability"]), 1e-30),
                    "pred_q_inlet": pred["q_inlet"],
                    "pred_q_outlet": pred["q_outlet"],
                    "n_pred_replaced": pred["n_pred_replaced"],
                    "pred_coverage": pred["coverage"],
                }
            )
        rows.append(row)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "global_permeability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "global_permeability_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out_dir / "global_permeability_summary.csv")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
