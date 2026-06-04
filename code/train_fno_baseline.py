#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练一个 2D FNO 基线，用于速度场预测，并按当前工程口径计算通量/导流能力相关指标。

设计原则：
1. 走你现有 HDF5 + split_info.json 的数据接口，不额外改数据管线。
2. 输出格式与现有 baseline 保持一致，方便直接接入对比脚本与论文绘图。
3. 输入使用当前数据中的几何/物性通道，并额外拼接二维坐标通道，符合 FNO 常见做法。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from train_poreflownet_baseline import (
    EvalOutputs,
    MinMax2Stats,
    PoreFlowLoss,
    fit_minmax2_stats,
    get_channel_index,
    inverse_minmax2_torch,
    load_split_indices,
    plot_training_curves_from_csv,
    read_channel_order,
    save_predictions,
    set_seed,
    subset_eval,
    transform_minmax2,
    write_report,
)


def build_fno_arrays(h5_path: str, add_coords: bool = True) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:].astype(np.float32)
        Y = f["Y"][:].astype(np.float32)
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)

    order = read_channel_order(h5_path)
    mask_idx = get_channel_index(order, "mask", default=0)
    if mask_idx is None:
        raise ValueError("mask channel not found in HDF5")

    channels = [X[:, i : i + 1] for i in range(X.shape[1])]
    if add_coords:
        _, _, h, w = X.shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
        coord = np.stack([grid_x, grid_y], axis=0)[None]
        coord = np.broadcast_to(coord, (X.shape[0], 2, h, w)).astype(np.float32)
        channels.append(coord)

    x_raw = np.concatenate(channels, axis=1).astype(np.float32)
    mask = X[:, mask_idx : mask_idx + 1].astype(np.float32)
    return {
        "x_raw": x_raw,
        "y_raw": Y.astype(np.float32),
        "mask": mask,
        "rock_type": rock_type,
        "global_id": global_id,
    }


def fit_input_stats(x_raw: np.ndarray, train_indices: np.ndarray) -> List[MinMax2Stats]:
    train_idx = np.asarray(train_indices, dtype=np.int64)
    stats: List[MinMax2Stats] = []
    for ch in range(x_raw.shape[1]):
        stats.append(fit_minmax2_stats(x_raw[train_idx, ch : ch + 1]))
    return stats


def apply_input_stats(x_raw: np.ndarray, stats: List[MinMax2Stats]) -> np.ndarray:
    out = np.empty_like(x_raw, dtype=np.float32)
    for ch, ch_stats in enumerate(stats):
        out[:, ch : ch + 1] = transform_minmax2(x_raw[:, ch : ch + 1], ch_stats)
    return out


class InMemoryFNODataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_t = torch.from_numpy(arrays["x_t"][self.indices])
        self.y_t = torch.from_numpy(arrays["y_t"][self.indices])
        self.y_raw = torch.from_numpy(arrays["y_raw"][self.indices])
        self.mask = torch.from_numpy(arrays["mask"][self.indices])
        self.rock_type = arrays["rock_type"][self.indices]
        self.global_id = arrays["global_id"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_t": self.x_t[idx],
            "y_t": self.y_t[idx],
            "y_raw": self.y_raw[idx],
            "mask": self.mask[idx],
            "index": int(self.indices[idx]),
            "rock_type": int(self.rock_type[idx]),
            "global_id": int(self.global_id[idx]),
        }


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weight_pos[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weight_neg[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral(x) + self.local(x)
        out = self.norm(out)
        return F.gelu(out)


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, modes1: int = 16, modes2: int = 16, depth: int = 4):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2d(width, modes1, modes2) for _ in range(depth)])
        self.mid = nn.Conv2d(width, width, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.mid(x))
        return self.out(x)


def iterate_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_stats: MinMax2Stats,
    criterion: PoreFlowLoss | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_field = 0.0
    total_flux = 0.0
    n_batches = 0

    for batch in loader:
        x_t = batch["x_t"].to(device)
        y_t = batch["y_t"].to(device)
        y_raw = batch["y_raw"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            pred_t = model(x_t)
            pred_raw = inverse_minmax2_torch(pred_t, y_stats) * mask
            loss, loss_items = criterion(pred_t, y_t, pred_raw, y_raw, mask) if criterion is not None else (None, {})
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        if loss is not None:
            total_loss += float(loss.detach().cpu())
        total_field += loss_items.get("field", 0.0)
        total_flux += loss_items.get("flux", 0.0)
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "field": total_field / denom,
        "flux": total_flux / denom,
    }


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device, y_stats: MinMax2Stats) -> EvalOutputs:
    model.eval()
    q_true_list: List[np.ndarray] = []
    q_pred_list: List[np.ndarray] = []
    rel_err_list: List[np.ndarray] = []
    abs_err_list: List[np.ndarray] = []
    rmse_list: List[np.ndarray] = []
    mae_list: List[np.ndarray] = []
    pore_frac_list: List[np.ndarray] = []
    rock_type_list: List[np.ndarray] = []
    global_id_list: List[np.ndarray] = []
    index_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x_t = batch["x_t"].to(device)
            y_raw = batch["y_raw"].to(device)
            mask = batch["mask"].to(device)

            pred_t = model(x_t)
            pred_raw = inverse_minmax2_torch(pred_t, y_stats) * mask

            mask_np = mask.cpu().numpy().astype(np.float32)
            y_np = y_raw.cpu().numpy().astype(np.float32)
            pred_np = pred_raw.cpu().numpy().astype(np.float32)
            q_true = np.sum(y_np * mask_np, axis=(1, 2, 3))
            q_pred = np.sum(pred_np * mask_np, axis=(1, 2, 3))
            abs_err = np.abs(q_pred - q_true)
            rel_err = abs_err / (np.abs(q_true) + 1.0e-8)

            pore_count = np.sum(mask_np, axis=(1, 2, 3)) + 1.0e-8
            sq_err = ((pred_np - y_np) ** 2) * mask_np
            ab_err = np.abs(pred_np - y_np) * mask_np
            rmse = np.sqrt(np.sum(sq_err, axis=(1, 2, 3)) / pore_count)
            mae = np.sum(ab_err, axis=(1, 2, 3)) / pore_count
            pore_frac = np.mean(mask_np, axis=(1, 2, 3))

            q_true_list.append(q_true)
            q_pred_list.append(q_pred)
            rel_err_list.append(rel_err)
            abs_err_list.append(abs_err)
            rmse_list.append(rmse)
            mae_list.append(mae)
            pore_frac_list.append(pore_frac)
            rock_type_list.append(np.asarray(batch["rock_type"], dtype=np.int16))
            global_id_list.append(np.asarray(batch["global_id"], dtype=np.int32))
            index_list.append(np.asarray(batch["index"], dtype=np.int32))
            pred_list.append(pred_np.astype(np.float16))

    return EvalOutputs(
        q_true=np.concatenate(q_true_list, axis=0),
        q_pred=np.concatenate(q_pred_list, axis=0),
        rel_err=np.concatenate(rel_err_list, axis=0),
        abs_err=np.concatenate(abs_err_list, axis=0),
        rmse_pore=np.concatenate(rmse_list, axis=0),
        mae_pore=np.concatenate(mae_list, axis=0),
        pore_frac=np.concatenate(pore_frac_list, axis=0),
        rock_type=np.concatenate(rock_type_list, axis=0),
        global_id=np.concatenate(global_id_list, axis=0),
        index=np.concatenate(index_list, axis=0),
        pred_ux=np.concatenate(pred_list, axis=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 2D FNO 速度场预测基线。")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-flux", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=28)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes1", type=int, default=16)
    parser.add_argument("--modes2", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-coords", action="store_true", help="不拼接 x/y 坐标通道")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_fno_arrays(args.h5, add_coords=not args.no_coords)
    split = load_split_indices(args.split_json, arrays["rock_type"], arrays["global_id"])
    input_stats = fit_input_stats(arrays["x_raw"], split["train"])
    y_stats = fit_minmax2_stats(arrays["y_raw"][split["train"]])
    arrays["x_t"] = apply_input_stats(arrays["x_raw"], input_stats)
    arrays["y_t"] = transform_minmax2(arrays["y_raw"], y_stats)

    train_ds = InMemoryFNODataset(arrays, split["train"])
    val_ds = InMemoryFNODataset(arrays, split["val"])
    all_ds = InMemoryFNODataset(arrays, np.arange(arrays["y_raw"].shape[0], dtype=np.int64))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = FNO2d(
        in_channels=int(arrays["x_t"].shape[1]),
        width=args.width,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
    ).to(device)
    criterion = PoreFlowLoss(lambda_flux=args.lambda_flux)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history_rows: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_metrics = iterate_batches(model, train_loader, device, y_stats, criterion=criterion, optimizer=optimizer)
        val_metrics = iterate_batches(model, val_loader, device, y_stats, criterion=criterion, optimizer=None)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_field": train_metrics["field"],
            "train_flux": train_metrics["flux"],
            "val_loss": val_metrics["loss"],
            "val_field": val_metrics["field"],
            "val_flux": val_metrics["flux"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.6f} "
            f"(field {train_metrics['field']:.6f}, flux {train_metrics['flux']:.6f}) | "
            f"val {val_metrics['loss']:.6f} "
            f"(field {val_metrics['field']:.6f}, flux {val_metrics['flux']:.6f})"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "best_val_loss": best_val,
                    "input_stats": [asdict(s) for s in input_stats],
                    "y_stats": asdict(y_stats),
                },
                out_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch}")
                break

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    plot_training_curves_from_csv(str(out_dir / "training_history.csv"), str(out_dir))

    with open(out_dir / "transform_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_stats": [asdict(s) for s in input_stats],
                "y_stats": asdict(y_stats),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    pred_all = predict_dataset(model, all_loader, device, y_stats)
    pred_test = subset_eval(pred_all, split["test"])
    save_predictions(str(out_dir / "predictions.npz"), pred_all)
    all_summary = write_report(str(out_dir / "report.csv"), pred_all)
    test_summary = write_report(str(out_dir / "report_test.csv"), pred_test)

    summary = {
        "paper": {
            "name": "2D Fourier Neural Operator baseline",
            "family": "neural operator",
            "reproduction_level": "local-implementation-2d-fno",
            "aligned_points": [
                "Fourier neural operator blocks",
                "global frequency mixing",
                "coordinate augmentation",
                "same train/val/test split",
                "same flux-aware evaluation metrics",
            ],
            "notes": [
                "This baseline predicts the velocity field directly and then derives flux-related quantities.",
                "Input channels follow the current HDF5 setup, with optional x/y coordinate maps.",
                "The output and report format are kept consistent with the existing baselines for fair comparison.",
            ],
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_run": int(len(history_rows)),
        },
        "all_metrics": all_summary,
        "test_metrics": test_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved FNO baseline outputs to: {out_dir}")


if __name__ == "__main__":
    main()
