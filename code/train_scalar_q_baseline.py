#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train direct scalar q/g baselines for the IPNM1 manuscript experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from train_poreflownet_baseline import compute_r2, load_split_indices, set_seed


def ensure_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None]
    raise ValueError(f"Expected 3D/4D array, got {arr.shape}")


def load_table_q(h5_path: str, data_root: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        rock_type = f["rock_type"][:].astype(int)
        local_id = f["local_id"][:].astype(int) if "local_id" in f else f["global_id"][:].astype(int)
    out = np.full(rock_type.shape[0], np.nan, dtype=np.float64)
    root = Path(data_root)
    for i, (rt, lid) in enumerate(zip(rock_type, local_id)):
        path = root / str(int(rt) + 1) / f"permeability_{int(lid)}.dat"
        if not path.exists():
            continue
        vals: List[float] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for token in line.replace(",", " ").split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
        if vals:
            out[i] = vals[-1] / 5000.0
    if not np.isfinite(out).all():
        missing = int(np.sum(~np.isfinite(out)))
        raise ValueError(f"Could not load all table q values, missing={missing}")
    return out.astype(np.float32)


def build_arrays(h5_path: str, data_root: str, porosity_kernel: int) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        x = f["X"][:].astype(np.float32)
        y = ensure_nchw(f["Y"][:].astype(np.float32))
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
        eta = f["eta"][:].astype(np.float32)
        ri = f["RI"][:].astype(np.float32)
        r0_ra = f["R0_RA"][:].astype(np.float32)
    mask = x[:, 0:1].astype(np.float32)
    dist = x[:, 1:2].astype(np.float32) if x.shape[1] > 1 else np.zeros_like(mask)
    eta_map = x[:, 2:3].astype(np.float32) if x.shape[1] > 2 else np.zeros_like(mask)
    local_porosity = uniform_filter(mask[:, 0], size=(1, porosity_kernel, porosity_kernel), mode="nearest")[:, None]
    geom = np.concatenate([mask, dist, eta_map, local_porosity], axis=1).astype(np.float32)
    scalar = np.stack([eta, ri, r0_ra, mask[:, 0].mean(axis=(1, 2)), dist[:, 0].max(axis=(1, 2))], axis=1).astype(np.float32)
    q_true = (y * mask).sum(axis=(1, 2, 3)).astype(np.float32)
    q_table = load_table_q(h5_path, data_root)
    return {
        "geom_raw": geom,
        "scalar_raw": scalar,
        "q_true": q_true,
        "q_table": q_table,
        "rock_type": rock_type,
        "global_id": global_id,
    }


class ScalarDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray, target: str):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.geom = torch.from_numpy(arrays["geom_t"][self.indices])
        self.scalar = torch.from_numpy(arrays["scalar_t"][self.indices])
        self.y = torch.from_numpy(arrays[f"{target}_log_t"][self.indices])

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"geom": self.geom[i], "scalar": self.scalar[i], "y": self.y[i], "index": int(self.indices[i])}


class ScalarCNN(nn.Module):
    def __init__(self, in_ch: int, scalar_dim: int, base: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.GELU(),
            nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(base * 4 + scalar_dim, base * 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(base * 4, base * 2),
            nn.GELU(),
            nn.Linear(base * 2, 1),
        )

    def forward(self, geom: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(geom), scalar], dim=1)).view(-1)


def fit_minmax(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn = train.min(axis=0, keepdims=True)
    rng = np.maximum(train.max(axis=0, keepdims=True) - mn, 1.0e-8)
    return mn.astype(np.float32), rng.astype(np.float32)


def transform_minmax(x: np.ndarray, mn: np.ndarray, rng: np.ndarray) -> np.ndarray:
    return ((x - mn) * 2.0 / rng - 1.0).astype(np.float32)


def fit_log_stats(values: np.ndarray, train_idx: np.ndarray) -> tuple[float, float]:
    logv = np.log(np.maximum(values.astype(np.float64), 1.0e-12))
    mu = float(logv[train_idx].mean())
    sd = float(max(logv[train_idx].std(), 1.0e-8))
    return mu, sd


def apply_log_stats(values: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return ((np.log(np.maximum(values.astype(np.float64), 1.0e-12)) - mu) / sd).astype(np.float32)


def invert_log_stats(values_t: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(values_t.astype(np.float64) * sd + mu).astype(np.float64)


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    for batch in loader:
        geom = batch["geom"].to(device)
        scalar = batch["scalar"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(geom, scalar)
            loss = F.smooth_l1_loss(pred, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device, mu: float, sd: float) -> Dict[str, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred_t = model(batch["geom"].to(device), batch["scalar"].to(device)).cpu().numpy()
            preds.append(invert_log_stats(pred_t, mu, sd))
            indices.append(np.asarray(batch["index"], dtype=np.int64))
    return {"index": np.concatenate(indices), "pred": np.concatenate(preds)}


def metrics(pred: Dict[str, np.ndarray], truth: np.ndarray, keep_idx: np.ndarray | None = None) -> Dict[str, float]:
    idx = pred["index"].astype(np.int64)
    values = pred["pred"].astype(np.float64)
    if keep_idx is not None:
        keep = np.isin(idx, np.asarray(keep_idx, dtype=np.int64))
        idx = idx[keep]
        values = values[keep]
    y = truth[idx].astype(np.float64)
    rel = np.abs(values - y) / (np.abs(y) + 1.0e-12)
    return {
        "n": int(idx.size),
        "mean_re": float(np.mean(rel)),
        "median_re": float(np.median(rel)),
        "p90_re": float(np.percentile(rel, 90)),
        "r2": compute_r2(values, y),
    }


def run_one(args: argparse.Namespace, seed: int, target: str, root: Path) -> Dict[str, object]:
    set_seed(seed)
    arrays = build_arrays(args.h5, args.data_root, args.porosity_kernel)
    split = load_split_indices(args.split_json, arrays["rock_type"], arrays["global_id"])
    # Channel-wise image min/max is fitted on the training split only.
    ch_min = arrays["geom_raw"][split["train"]].min(axis=(0, 2, 3), keepdims=True)
    ch_rng = np.maximum(arrays["geom_raw"][split["train"]].max(axis=(0, 2, 3), keepdims=True) - ch_min, 1.0e-8)
    scalar_min, scalar_rng = fit_minmax(arrays["scalar_raw"][split["train"]])
    mu, sd = fit_log_stats(arrays[target], split["train"])
    arrays["geom_t"] = transform_minmax(arrays["geom_raw"], ch_min, ch_rng)
    arrays["scalar_t"] = transform_minmax(arrays["scalar_raw"], scalar_min, scalar_rng)
    arrays[f"{target}_log_t"] = apply_log_stats(arrays[target], mu, sd)

    run_dir = root / f"seed_{seed}" / target
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(ScalarDataset(arrays, split["train"], target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ScalarDataset(arrays, split["val"], target), batch_size=args.batch_size, shuffle=False)
    all_loader = DataLoader(ScalarDataset(arrays, np.arange(arrays[target].shape[0]), target), batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ScalarCNN(arrays["geom_t"].shape[1], arrays["scalar_t"].shape[1], args.base).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: List[Dict[str, float]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, device, optimizer)
        val_loss = train_epoch(model, val_loader, device, None)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": float(optimizer.param_groups[0]["lr"])})
        print(f"scalar {target} seed {seed} epoch {epoch:03d} train {train_loss:.6f} val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save({"model_state": model.state_dict(), "best_epoch": best_epoch, "best_val": best_val}, run_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pred = predict(model, all_loader, device, mu, sd)
    np.savez_compressed(run_dir / "predictions_scalar.npz", index=pred["index"], pred=pred["pred"], target=arrays[target])
    summary = {
        "seed": int(seed),
        "target": target,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs_run": int(len(history)),
        "elapsed_seconds": float(time.time() - start),
        "all": metrics(pred, arrays[target]),
        "val": metrics(pred, arrays[target], split["val"]),
        "test": metrics(pred, arrays[target], split["test"]),
        "args": vars(args),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize(root: Path, rows: List[Dict[str, object]]) -> None:
    flat_rows = []
    for row in rows:
        flat = {"target": row["target"], "seed": row["seed"], "best_epoch": row["best_epoch"], "epochs_run": row["epochs_run"]}
        for split_name in ["all", "val", "test"]:
            for key, value in row[split_name].items():
                flat[f"{split_name}_{key}"] = value
        flat_rows.append(flat)
    with (root / "scalar_baseline_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    summary_rows = []
    for target in sorted({str(r["target"]) for r in rows}):
        group = [r for r in flat_rows if r["target"] == target]
        out = {"target": target, "n_seeds": len(group)}
        for key in group[0].keys():
            if key in ("target", "seed"):
                continue
            vals = np.asarray([float(g[key]) for g in group], dtype=np.float64)
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    with (root / "scalar_baseline_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (root / "scalar_baseline_summary.json").write_text(json.dumps({"per_seed": flat_rows, "summary": summary_rows}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct scalar q/g baselines.")
    parser.add_argument("--h5", type=str, default=r"E:\mhw\1\cup\dataset_all_32.h5")
    parser.add_argument("--split-json", type=str, default=r"E:\mhw\1\cup\runs\final_ipnm1_flownet_flux010_30e\split_info.json")
    parser.add_argument("--data-root", type=str, default=r"E:\mhw\1\cup\data")
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\scalar_q_baseline_100e_3seed_20260527")
    parser.add_argument("--targets", type=str, default="q_true,q_table")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--porosity-kernel", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    for seed in seeds:
        for target in targets:
            rows.append(run_one(args, seed, target, root))
    summarize(root, rows)
    print(root / "scalar_baseline_summary.csv")


if __name__ == "__main__":
    main()
