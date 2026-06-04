#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Ours on teacher IPNM1 geometries for three seeds and solve global K."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(r"E:\mhw\1\cup")
PYTHON = Path(r"D:\anaconda\envs\nnlnvv\python.exe")
INFER = ROOT / "code" / "infer_arch_teacher_geom.py"
POST = ROOT / "code" / "postprocess_conductance.py"
SOLVE = ROOT / "code" / "solve_teacher_pnm_permeability.py"
ROCKS = ["Bead", "Benth", "Berea_ICL", "Font18"]


def run_cmd(cmd: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{tail}")
    return elapsed


def run_postprocess(run_dir: Path, teacher_h5: Path) -> float:
    cmd = [
        str(PYTHON),
        str(POST),
        "--pred-file",
        str(run_dir / "predictions.npz"),
        "--pred-key",
        "pred_ux",
        "--true-file",
        str(teacher_h5),
        "--true-key",
        "Y",
        "--mask-file",
        str(teacher_h5),
        "--mask-key",
        "X",
        "--scale-file",
        str(teacher_h5),
        "--scale-key",
        "scale_s",
        "--index-file",
        str(run_dir / "predictions.npz"),
        "--index-key",
        "index",
        "--meta-file",
        str(teacher_h5),
        "--spatial-axis-order",
        "yz",
        "--flow-axis",
        "x",
        "--flow-mode",
        "pressure",
        "--conductance-mode",
        "ipnm2",
        "--delta-p",
        "1.0",
        "--rho",
        "1.0",
        "--ax",
        "1.0e-4",
        "--segment-length",
        "4.0",
        "--mu-lbm",
        "0.5",
        "--target-mu",
        "1.0",
        "--undo-velocity-normalization",
        "--undo-area-normalization",
        "--output-dir",
        str(run_dir / "conductance_ipnm1_rho1"),
    ]
    return run_cmd(cmd, run_dir / "postprocess.log")


def merge_conductance(seed_dir: Path) -> Path:
    out = seed_dir / "all_teacher_pred_conductance_results.csv"
    writer = None
    fields = None
    with out.open("w", newline="", encoding="utf-8") as fo:
        for rock in ROCKS:
            path = seed_dir / rock / "conductance_ipnm1_rho1" / "conductance_results.csv"
            with path.open("r", newline="", encoding="utf-8") as fi:
                reader = csv.DictReader(fi)
                if writer is None:
                    fields = list(reader.fieldnames or [])
                    writer = csv.DictWriter(fo, fieldnames=fields)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    return out


def read_rows(path: Path, seed: int) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["seed"] = seed
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    import math

    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-geom-root", type=str, default=str(ROOT / "runs" / "teacher_geom_full_all_20260529"))
    parser.add_argument("--ckpt-root", type=str, default=str(ROOT / "runs" / "lambda_selected_3seed_20260529"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "ours_global_3seed_20260530"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_rows = []
    all_rows = []

    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        ckpt = Path(args.ckpt_root) / f"flux_0p2__dist_0p005__seed_{seed}" / "res_ed" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

        for rock in ROCKS:
            run_dir = seed_dir / rock
            teacher_h5 = Path(args.teacher_geom_root) / rock / "teacher_geom_only.h5"
            pred_path = run_dir / "predictions.npz"
            infer_seconds = None
            post_seconds = None
            if args.overwrite or not pred_path.exists():
                cmd = [
                    str(PYTHON),
                    str(INFER),
                    "--teacher-h5",
                    str(teacher_h5),
                    "--ckpt",
                    str(ckpt),
                    "--out-dir",
                    str(run_dir),
                    "--model",
                    "res_ed",
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ]
                infer_seconds = run_cmd(cmd, run_dir / "infer.log")
            if args.overwrite or not (run_dir / "conductance_ipnm1_rho1" / "conductance_results.csv").exists():
                post_seconds = run_postprocess(run_dir, teacher_h5)
            n_samples = 0
            summary_path = run_dir / "infer_summary.json"
            if summary_path.exists():
                n_samples = int(json.loads(summary_path.read_text(encoding="utf-8")).get("n_samples", 0))
            timing_rows.append(
                {
                    "seed": seed,
                    "rock": rock,
                    "n_samples": n_samples,
                    "infer_seconds": "" if infer_seconds is None else f"{infer_seconds:.6f}",
                    "postprocess_seconds": "" if post_seconds is None else f"{post_seconds:.6f}",
                    "infer_ms_per_sample": ""
                    if infer_seconds is None or n_samples <= 0
                    else f"{1000.0 * infer_seconds / n_samples:.6f}",
                }
            )

        pred_csv = merge_conductance(seed_dir)
        solve_dir = seed_dir / "global_pnm"
        if args.overwrite or not (solve_dir / "global_permeability_summary.csv").exists():
            cmd = [
                str(PYTHON),
                str(SOLVE),
                "--pred-csv",
                str(pred_csv),
                "--out-dir",
                str(solve_dir),
            ]
            run_cmd(cmd, seed_dir / "solve_global.log")
        all_rows.extend(read_rows(solve_dir / "global_permeability_summary.csv", seed))
        print(f"[OK] seed {seed}", flush=True)

    write_csv(out_dir / "ours_global_3seed_rows.csv", all_rows)
    write_csv(out_dir / "ours_global_3seed_timing.csv", timing_rows)

    grouped = []
    for rock in ["Bead", "Benth", "Berea", "Font"]:
        vals = [float(r["perm_rel_error"]) for r in all_rows if r["rock"] == rock]
        perms = [float(r["pred_perm"]) for r in all_rows if r["rock"] == rock]
        teacher = [float(r["teacher_perm"]) for r in all_rows if r["rock"] == rock]
        err_mean, err_std = mean_std(vals)
        perm_mean, perm_std = mean_std(perms)
        grouped.append(
            {
                "rock": rock,
                "teacher_perm": teacher[0],
                "pred_perm_mean": perm_mean,
                "pred_perm_std": perm_std,
                "perm_rel_error_mean": err_mean,
                "perm_rel_error_std": err_std,
                "n_seeds": len(vals),
            }
        )
    vals = [float(r["perm_rel_error"]) for r in all_rows]
    err_mean, err_std = mean_std(vals)
    grouped.append(
        {
            "rock": "Mean",
            "teacher_perm": "",
            "pred_perm_mean": "",
            "pred_perm_std": "",
            "perm_rel_error_mean": err_mean,
            "perm_rel_error_std": err_std,
            "n_seeds": len(args.seeds),
        }
    )
    write_csv(out_dir / "ours_global_3seed_summary.csv", grouped)
    (out_dir / "ours_global_3seed_summary.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")

    lines = [
        "# Ours Global Permeability, Three Seeds",
        "",
        "| Rock | Teacher K | Predicted K | K relative error |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in grouped:
        if row["rock"] == "Mean":
            lines.append(
                f"| Mean | - | - | {100.0 * float(row['perm_rel_error_mean']):.2f}% +/- "
                f"{100.0 * float(row['perm_rel_error_std']):.2f}% |"
            )
        else:
            lines.append(
                f"| {row['rock']} | {float(row['teacher_perm']):.6g} | "
                f"{float(row['pred_perm_mean']):.6g} +/- {float(row['pred_perm_std']):.2g} | "
                f"{100.0 * float(row['perm_rel_error_mean']):.2f}% +/- "
                f"{100.0 * float(row['perm_rel_error_std']):.2f}% |"
            )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "ours_global_3seed_summary.csv")


if __name__ == "__main__":
    main()
