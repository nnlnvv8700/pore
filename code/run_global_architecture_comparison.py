#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare trained architectures by global PNM permeability after conductance backfill."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(r"E:\mhw\1\cup")
PYTHON = Path(r"D:\anaconda\envs\nnlnvv\python.exe")
INFER = ROOT / "code" / "infer_arch_teacher_geom.py"
POST = ROOT / "code" / "postprocess_conductance.py"
SOLVE = ROOT / "code" / "solve_teacher_pnm_permeability.py"

ROCKS = ["Bead", "Benth", "Berea_ICL", "Font18"]


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{tail}")


def run_postprocess(run_dir: Path, teacher_h5: Path) -> None:
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
    run_cmd(cmd, run_dir / "postprocess.log")


def merge_conductance(model_dir: Path) -> Path:
    out = model_dir / "all_teacher_pred_conductance_results.csv"
    writer = None
    fields = None
    with out.open("w", newline="", encoding="utf-8") as fo:
        for rock in ROCKS:
            path = model_dir / rock / "conductance_ipnm1_rho1" / "conductance_results.csv"
            with path.open("r", newline="", encoding="utf-8") as fi:
                reader = csv.DictReader(fi)
                if writer is None:
                    fields = list(reader.fieldnames or [])
                    writer = csv.DictWriter(fo, fieldnames=fields)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    return out


def read_global_summary(path: Path, model_name: str) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["model"] = model_name
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-geom-root", type=str, default=str(ROOT / "runs" / "teacher_geom_full_all_20260529"))
    parser.add_argument("--arch-root", type=str, default=str(ROOT / "runs" / "arch_compare_100e_3seed_20260527" / "seed_42"))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "global_arch_compare_seed42_20260529"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ipnm1_flownet", "unet", "convnext_ed", "fno", "deeponet_lite", "multitask_cnn"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    teacher_geom_root = Path(args.teacher_geom_root)
    arch_root = Path(args.arch_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model in args.models:
        model_dir = out_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt = arch_root / model / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        for rock in ROCKS:
            run_dir = model_dir / rock
            teacher_h5 = teacher_geom_root / rock / "teacher_geom_only.h5"
            pred_path = run_dir / "predictions.npz"
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
                    model,
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ]
                run_cmd(cmd, run_dir / "infer.log")
            if args.overwrite or not (run_dir / "conductance_ipnm1_rho1" / "conductance_results.csv").exists():
                run_postprocess(run_dir, teacher_h5)
        pred_csv = merge_conductance(model_dir)
        solve_dir = model_dir / "global_pnm"
        if args.overwrite or not (solve_dir / "global_permeability_summary.csv").exists():
            cmd = [
                str(PYTHON),
                str(SOLVE),
                "--pred-csv",
                str(pred_csv),
                "--out-dir",
                str(solve_dir),
            ]
            run_cmd(cmd, model_dir / "solve_global.log")
        rows = read_global_summary(solve_dir / "global_permeability_summary.csv", model)
        all_rows.extend(rows)
        print(f"[OK] {model}", flush=True)

    write_csv(out_dir / "global_architecture_permeability_rows.csv", all_rows)
    grouped = []
    for model in args.models:
        rows = [r for r in all_rows if r["model"] == model]
        errs = [float(r["perm_rel_error"]) for r in rows]
        grouped.append(
            {
                "model": model,
                "mean_perm_rel_error": float(sum(errs) / len(errs)),
                "max_perm_rel_error": float(max(errs)),
                "n_rocks": len(rows),
            }
        )
    grouped.sort(key=lambda x: x["mean_perm_rel_error"])
    write_csv(out_dir / "global_architecture_permeability_summary.csv", grouped)
    (out_dir / "global_architecture_permeability_summary.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
    lines = [
        "# Global Architecture Permeability Comparison",
        "",
        "| Rank | Model | Mean K error | Max K error |",
        "| ---: | --- | ---: | ---: |",
    ]
    for i, row in enumerate(grouped, 1):
        lines.append(
            f"| {i} | {row['model']} | {100.0 * row['mean_perm_rel_error']:.2f}% | "
            f"{100.0 * row['max_perm_rel_error']:.2f}% |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "global_architecture_permeability_summary.csv")


if __name__ == "__main__":
    main()
