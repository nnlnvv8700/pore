#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PoreFlow-Net 官方代码风格的 2D 适配版基线。

目的：
1. 尽量贴近官方开源实现的关键设计，而不是只保留一个“名字类似”的 baseline。
2. 在当前 2D yz patch 数据上做可运行的近似复现，并和现有模型做公平对比。

与官方代码对齐的部分：
- 四个单通道输入分支
- 每个分支独立残差编码器
- 各分支在 skip level 上拼接
- BatchNorm + SELU
- 全局 minMax_2 归一化到 [-1, 1]
- 默认主损失使用 MAE

和官方不同的部分：
- 原文与官方仓库是 3D 80^3 子体块，这里只能做 2D 64x64 patch
- 原文使用 e_pore / tof_L / tof_R / MIS_z
- 当前数据没有真正的 TOF，因此这里构造两张 TOF-like 替代图：
  从 patch 左/右边界出发的累计“孔隙阻力”图
- MIS_z 也无法严格恢复，因此这里用局部孔隙率 + 全局几何标量混合图替代
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def stats_basic(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def stats_three(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def read_channel_order(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        raw = None
        if "channel_order" in f.attrs:
            raw = f.attrs["channel_order"]
        elif "X" in f and "channel_order" in f["X"].attrs:
            raw = f["X"].attrs["channel_order"]
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "U"):
        return [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in raw.tolist()
        ]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def get_channel_index(order: List[str], name: str, default: int | None = None) -> int | None:
    for i, item in enumerate(order):
        if item.lower() == name.lower():
            return i
    return default


def load_split_indices(split_json_path: str, rock_type: np.ndarray, global_id: np.ndarray) -> Dict[str, np.ndarray]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    index_by_key = {}
    for idx, (rt, gid) in enumerate(zip(rock_type.tolist(), global_id.tolist())):
        index_by_key[(int(rt), int(gid))] = idx

    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for rt_str, rt_info in split_info["by_rock_type"].items():
        rt = int(rt_str)
        for split_name in out.keys():
            gids = rt_info[f"{split_name}_global_ids"]
            for gid in gids:
                key = (rt, int(gid))
                if key not in index_by_key:
                    raise KeyError(f"Missing sample for split key {key}")
                out[split_name].append(index_by_key[key])

    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


@dataclass
class MinMax2Stats:
    mean: float
    min: float
    range: float
    std: float
    max: float
    maxAbs: float
    minAbs: float
    p95: float
    new_min: float


def fit_minmax2_stats(arr: np.ndarray) -> MinMax2Stats:
    x = arr.astype(np.float32)
    x_mean = float(x.mean())
    x_min = float(x.min())
    x_max = float(x.max())
    x_std = float(x.std())
    x_range = float(max(x_max - x_min, 1.0e-8))
    x_max_abs = float(np.max(np.abs(x)))
    positive = np.abs(x[x > 0])
    x_min_abs = float(np.min(positive)) if positive.size > 0 else 0.0
    nz = x[x != 0]
    x_p95 = float(np.percentile(nz, 95)) if nz.size > 0 else x_max
    x_new_min = float(np.finfo(np.float32).eps * x_max) if x_max != 0 else 0.0
    return MinMax2Stats(
        mean=x_mean,
        min=x_min,
        range=x_range,
        std=x_std,
        max=x_max,
        maxAbs=x_max_abs,
        minAbs=x_min_abs,
        p95=x_p95,
        new_min=x_new_min,
    )


def transform_minmax2(arr: np.ndarray, stats: MinMax2Stats) -> np.ndarray:
    return ((arr - stats.min) * 2.0 / stats.range - 1.0).astype(np.float32)


def inverse_minmax2(arr: np.ndarray, stats: MinMax2Stats) -> np.ndarray:
    return ((arr + 1.0) * stats.range / 2.0 + stats.min).astype(np.float32)


def build_feature_maps(h5_path: str, porosity_kernel: int) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:].astype(np.float32)
        Y = f["Y"][:].astype(np.float32)
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
        eta = f["eta"][:].astype(np.float32) if "eta" in f else np.zeros(X.shape[0], dtype=np.float32)
        ri = f["RI"][:].astype(np.float32) if "RI" in f else np.zeros(X.shape[0], dtype=np.float32)
        r0_ra = f["R0_RA"][:].astype(np.float32) if "R0_RA" in f else np.zeros(X.shape[0], dtype=np.float32)

    order = read_channel_order(h5_path)
    mask_idx = get_channel_index(order, "mask", default=0)
    dist_idx = get_channel_index(order, "dist", default=1 if X.shape[1] > 1 else None)
    eta_idx = get_channel_index(order, "eta_map", default=2 if X.shape[1] > 2 else None)
    if mask_idx is None:
        raise ValueError("mask channel not found in HDF5")

    mask = X[:, mask_idx].astype(np.float32)
    dist = X[:, dist_idx].astype(np.float32) if dist_idx is not None else np.zeros_like(mask)
    if eta_idx is not None:
        eta_map = X[:, eta_idx].astype(np.float32)
    else:
        eta_map = np.broadcast_to(eta[:, None, None], mask.shape).astype(np.float32)

    # 官方第一分支 e_pore 的替代：当前数据已有欧氏距离图
    e_pore_like = dist * mask

    # 当前数据没有真正的 TOF。
    # 这里构造两个方向性的 TOF-like 特征，模拟“从左右边界进入 patch 的累计阻力”。
    resistance = np.where(mask > 0.5, 1.0 / np.maximum(dist, 1.0e-3), 0.0).astype(np.float32)
    tof_left_like = np.cumsum(resistance, axis=2).astype(np.float32)
    tof_right_like = np.flip(np.cumsum(np.flip(resistance, axis=2), axis=2), axis=2).astype(np.float32)
    tof_left_like *= mask
    tof_right_like *= mask

    # 官方第四分支 MIS_z 的替代：
    # 用局部孔隙率与全局几何标量组合，提供 patch 级几何上下文。
    local_porosity = uniform_filter(mask, size=(1, porosity_kernel, porosity_kernel), mode="nearest").astype(np.float32)
    eta_norm = (eta - eta.min()) / max(float(eta.max() - eta.min()), 1.0e-8)
    ri_norm = (ri - ri.min()) / max(float(ri.max() - ri.min()), 1.0e-8)
    r0_norm = (r0_ra - r0_ra.min()) / max(float(r0_ra.max() - r0_ra.min()), 1.0e-8)
    global_geom = (
        0.50 * eta_map
        + 0.20 * np.broadcast_to(eta_norm[:, None, None], mask.shape)
        + 0.15 * np.broadcast_to(ri_norm[:, None, None], mask.shape)
        + 0.15 * np.broadcast_to(r0_norm[:, None, None], mask.shape)
    ).astype(np.float32)
    mis_like = (0.6 * local_porosity + 0.4 * global_geom) * mask

    return {
        "x0": e_pore_like[:, None],
        "x1": tof_left_like[:, None],
        "x2": tof_right_like[:, None],
        "x3": mis_like[:, None],
        "mask": mask[:, None].astype(np.float32),
        "y_raw": Y.astype(np.float32),
        "rock_type": rock_type,
        "global_id": global_id,
    }


def fit_transforms(arrays: Dict[str, np.ndarray], train_indices: np.ndarray) -> Dict[str, MinMax2Stats]:
    train_idx = np.asarray(train_indices, dtype=np.int64)
    return {
        "x0": fit_minmax2_stats(arrays["x0"][train_idx]),
        "x1": fit_minmax2_stats(arrays["x1"][train_idx]),
        "x2": fit_minmax2_stats(arrays["x2"][train_idx]),
        "x3": fit_minmax2_stats(arrays["x3"][train_idx]),
        "y": fit_minmax2_stats(arrays["y_raw"][train_idx]),
    }


def apply_transforms(arrays: Dict[str, np.ndarray], stats: Dict[str, MinMax2Stats]) -> Dict[str, np.ndarray]:
    out = dict(arrays)
    out["x0_t"] = transform_minmax2(arrays["x0"], stats["x0"])
    out["x1_t"] = transform_minmax2(arrays["x1"], stats["x1"])
    out["x2_t"] = transform_minmax2(arrays["x2"], stats["x2"])
    out["x3_t"] = transform_minmax2(arrays["x3"], stats["x3"])
    out["y_t"] = transform_minmax2(arrays["y_raw"], stats["y"])
    return out


class InMemoryBranchDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x0_t = torch.from_numpy(arrays["x0_t"][self.indices])
        self.x1_t = torch.from_numpy(arrays["x1_t"][self.indices])
        self.x2_t = torch.from_numpy(arrays["x2_t"][self.indices])
        self.x3_t = torch.from_numpy(arrays["x3_t"][self.indices])
        self.y_t = torch.from_numpy(arrays["y_t"][self.indices])
        self.y_raw = torch.from_numpy(arrays["y_raw"][self.indices])
        self.mask = torch.from_numpy(arrays["mask"][self.indices])
        self.rock_type = arrays["rock_type"][self.indices]
        self.global_id = arrays["global_id"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x0_t": self.x0_t[idx],
            "x1_t": self.x1_t[idx],
            "x2_t": self.x2_t[idx],
            "x3_t": self.x3_t[idx],
            "y_t": self.y_t[idx],
            "y_raw": self.y_raw[idx],
            "mask": self.mask[idx],
            "index": int(self.indices[idx]),
            "rock_type": int(self.rock_type[idx]),
            "global_id": int(self.global_id[idx]),
        }


class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        return out + residual


class Encoder2D(nn.Module):
    def __init__(self, base_filters: int):
        super().__init__()
        self.first_main = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.SELU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
        )
        self.first_short = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=1, stride=1),
            nn.BatchNorm2d(base_filters),
        )
        self.block2 = ResBlock2D(base_filters, base_filters * 2, stride=2)
        self.block3 = ResBlock2D(base_filters * 2, base_filters * 4, stride=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out0 = self.first_main(x) + self.first_short(x)
        out1 = self.block2(out0)
        out2 = self.block3(out1)
        return [out0, out1, out2]


class Decoder2D(nn.Module):
    def __init__(self, base_filters: int):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.rb1 = ResBlock2D(base_filters * 8 + base_filters * 16, base_filters * 4, stride=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.rb2 = ResBlock2D(base_filters * 4 + base_filters * 8, base_filters * 2, stride=1)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.rb3 = ResBlock2D(base_filters * 2 + base_filters * 4, base_filters, stride=1)

    def forward(self, x: torch.Tensor, skip0: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.rb1(x)
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.rb2(x)
        x = self.up3(x)
        x = torch.cat([x, skip0], dim=1)
        x = self.rb3(x)
        return x


class PoreFlowNetOfficial2D(nn.Module):
    def __init__(self, base_filters: int = 10):
        super().__init__()
        self.enc0 = Encoder2D(base_filters)
        self.enc1 = Encoder2D(base_filters)
        self.enc2 = Encoder2D(base_filters)
        self.enc3 = Encoder2D(base_filters)
        self.bridge = ResBlock2D(base_filters * 16, base_filters * 8, stride=2)
        self.decoder = Decoder2D(base_filters)
        self.out_conv = nn.Conv2d(base_filters, 1, kernel_size=1)
        self.out_act = nn.SELU(inplace=True)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        b0 = self.enc0(x0)
        b1 = self.enc1(x1)
        b2 = self.enc2(x2)
        b3 = self.enc3(x3)

        skip0 = torch.cat([b0[0], b1[0], b2[0], b3[0]], dim=1)
        skip1 = torch.cat([b0[1], b1[1], b2[1], b3[1]], dim=1)
        skip2 = torch.cat([b0[2], b1[2], b2[2], b3[2]], dim=1)
        x = self.bridge(skip2)
        x = self.decoder(x, skip0, skip1, skip2)
        x = self.out_act(self.out_conv(x))
        return x


class PoreFlowLoss(nn.Module):
    def __init__(
        self,
        lambda_flux: float = 0.0,
        lambda_dist: float = 0.0,
        lambda_poisson: float = 0.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        self.lambda_flux = float(lambda_flux)
        self.lambda_dist = float(lambda_dist)
        self.lambda_poisson = float(lambda_poisson)
        self.eps = float(eps)

    def _distribution_loss(self, pred_raw: torch.Tensor, y_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses = []
        for pred_i, true_i, mask_i in zip(pred_raw[:, 0], y_raw[:, 0], mask[:, 0]):
            pore = mask_i > 0.5
            if torch.count_nonzero(pore) < 2:
                continue
            p = pred_i[pore]
            t = true_i[pore]
            scale = torch.mean(torch.abs(t)) + self.eps
            mean_loss = torch.abs(torch.mean(p) - torch.mean(t)) / scale
            std_loss = torch.abs(torch.std(p, unbiased=False) - torch.std(t, unbiased=False)) / scale
            qp = torch.quantile(p, torch.tensor([0.5, 0.9], dtype=p.dtype, device=p.device))
            qt = torch.quantile(t, torch.tensor([0.5, 0.9], dtype=t.dtype, device=t.device))
            quantile_loss = torch.mean(torch.abs(qp - qt) / scale)
            losses.append((mean_loss + std_loss + quantile_loss) / 3.0)
        if not losses:
            return torch.zeros((), dtype=pred_raw.dtype, device=pred_raw.device)
        return torch.stack(losses).mean()

    def _poisson_shape_loss(self, pred_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred_raw.shape[-1] < 3 or pred_raw.shape[-2] < 3:
            return torch.zeros((), dtype=pred_raw.dtype, device=pred_raw.device)
        center = pred_raw[:, :, 1:-1, 1:-1]
        lap = (
            pred_raw[:, :, :-2, 1:-1]
            + pred_raw[:, :, 2:, 1:-1]
            + pred_raw[:, :, 1:-1, :-2]
            + pred_raw[:, :, 1:-1, 2:]
            - 4.0 * center
        )
        interior = (
            (mask[:, :, 1:-1, 1:-1] > 0.5)
            & (mask[:, :, :-2, 1:-1] > 0.5)
            & (mask[:, :, 2:, 1:-1] > 0.5)
            & (mask[:, :, 1:-1, :-2] > 0.5)
            & (mask[:, :, 1:-1, 2:] > 0.5)
        ).to(pred_raw.dtype)
        count = interior.sum(dim=(1, 2, 3), keepdim=True)
        valid = count > 0
        if not bool(valid.any()):
            return torch.zeros((), dtype=pred_raw.dtype, device=pred_raw.device)
        lap_mean = (lap * interior).sum(dim=(1, 2, 3), keepdim=True) / torch.clamp(count, min=1.0)
        residual = (lap - lap_mean) * interior
        u_scale = (torch.abs(center) * interior).sum(dim=(1, 2, 3), keepdim=True) / torch.clamp(count, min=1.0)
        per_sample = residual.abs().sum(dim=(1, 2, 3), keepdim=True) / torch.clamp(count, min=1.0)
        per_sample = per_sample / (u_scale + self.eps)
        return per_sample[valid].mean()

    def forward(
        self,
        pred_t: torch.Tensor,
        y_t: torch.Tensor,
        pred_raw: torch.Tensor,
        y_raw: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        field_loss = torch.mean(torch.abs(pred_t - y_t))
        flux_loss = torch.zeros((), dtype=pred_t.dtype, device=pred_t.device)
        dist_loss = torch.zeros((), dtype=pred_t.dtype, device=pred_t.device)
        poisson_loss = torch.zeros((), dtype=pred_t.dtype, device=pred_t.device)
        if self.lambda_flux > 0:
            q_pred = (pred_raw * mask).sum(dim=(1, 2, 3))
            q_true = (y_raw * mask).sum(dim=(1, 2, 3))
            flux_loss = torch.mean(torch.abs(q_pred - q_true) / (torch.abs(q_true) + self.eps))
        if self.lambda_dist > 0:
            dist_loss = self._distribution_loss(pred_raw, y_raw, mask)
        if self.lambda_poisson > 0:
            poisson_loss = self._poisson_shape_loss(pred_raw, mask)
        total = (
            field_loss
            + self.lambda_flux * flux_loss
            + self.lambda_dist * dist_loss
            + self.lambda_poisson * poisson_loss
        )
        return total, {
            "field": float(field_loss.detach().cpu()),
            "flux": float(flux_loss.detach().cpu()),
            "dist": float(dist_loss.detach().cpu()),
            "poisson": float(poisson_loss.detach().cpu()),
        }


@dataclass
class EvalOutputs:
    q_true: np.ndarray
    q_pred: np.ndarray
    rel_err: np.ndarray
    abs_err: np.ndarray
    rmse_pore: np.ndarray
    mae_pore: np.ndarray
    pore_frac: np.ndarray
    rock_type: np.ndarray
    global_id: np.ndarray
    index: np.ndarray
    pred_ux: np.ndarray


def inverse_minmax2_torch(arr: torch.Tensor, stats: MinMax2Stats) -> torch.Tensor:
    return (arr + 1.0) * stats.range / 2.0 + stats.min


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
    total_dist = 0.0
    total_poisson = 0.0
    n_batches = 0

    for batch in loader:
        x0_t = batch["x0_t"].to(device)
        x1_t = batch["x1_t"].to(device)
        x2_t = batch["x2_t"].to(device)
        x3_t = batch["x3_t"].to(device)
        y_t = batch["y_t"].to(device)
        y_raw = batch["y_raw"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            pred_t = model(x0_t, x1_t, x2_t, x3_t)
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
        total_dist += loss_items.get("dist", 0.0)
        total_poisson += loss_items.get("poisson", 0.0)
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "field": total_field / denom,
        "flux": total_flux / denom,
        "dist": total_dist / denom,
        "poisson": total_poisson / denom,
    }


def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device, y_stats: MinMax2Stats) -> EvalOutputs:
    model.eval()
    q_true_list = []
    q_pred_list = []
    rel_err_list = []
    abs_err_list = []
    rmse_list = []
    mae_list = []
    pore_frac_list = []
    rock_type_list = []
    global_id_list = []
    index_list = []
    pred_list = []

    with torch.no_grad():
        for batch in loader:
            x0_t = batch["x0_t"].to(device)
            x1_t = batch["x1_t"].to(device)
            x2_t = batch["x2_t"].to(device)
            x3_t = batch["x3_t"].to(device)
            y_raw = batch["y_raw"].to(device)
            mask = batch["mask"].to(device)

            pred_t = model(x0_t, x1_t, x2_t, x3_t)
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


def write_report(path: str, pred: EvalOutputs) -> Dict[str, float]:
    rel_stats = stats_basic(pred.rel_err)
    abs_stats = stats_basic(pred.abs_err)
    rmse_stats = stats_three(pred.rmse_pore)
    mae_stats = stats_three(pred.mae_pore)
    q_stats = {
        "min": float(np.min(pred.q_true)),
        "median": float(np.median(pred.q_true)),
        "p10": float(np.percentile(pred.q_true, 10)),
        "p90": float(np.percentile(pred.q_true, 90)),
    }
    r2_val = compute_r2(pred.q_pred, pred.q_true)

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(
            "N,RelFluxErr_mean,RelFluxErr_median,RelFluxErr_p90,RelFluxErr_p95,RelFluxErr_max,"
            "AbsFluxErr_mean,AbsFluxErr_median,AbsFluxErr_p90,AbsFluxErr_p95,AbsFluxErr_max,"
            "RMSE_pore_mean,RMSE_pore_median,RMSE_pore_p90,"
            "MAE_pore_mean,MAE_pore_median,MAE_pore_p90,"
            "Flux_R2,q_true_min,q_true_median,q_true_p10,q_true_p90\n"
        )
        f.write(
            f"{pred.q_true.shape[0]},{rel_stats['mean']:.6f},{rel_stats['median']:.6f},{rel_stats['p90']:.6f},{rel_stats['p95']:.6f},{rel_stats['max']:.6f},"
            f"{abs_stats['mean']:.6f},{abs_stats['median']:.6f},{abs_stats['p90']:.6f},{abs_stats['p95']:.6f},{abs_stats['max']:.6f},"
            f"{rmse_stats['mean']:.6f},{rmse_stats['median']:.6f},{rmse_stats['p90']:.6f},"
            f"{mae_stats['mean']:.6f},{mae_stats['median']:.6f},{mae_stats['p90']:.6f},"
            f"{r2_val:.6f},{q_stats['min']:.6f},{q_stats['median']:.6f},{q_stats['p10']:.6f},{q_stats['p90']:.6f}\n"
        )

    return {
        "N": int(pred.q_true.shape[0]),
        "RelFluxErr_mean": rel_stats["mean"],
        "RelFluxErr_median": rel_stats["median"],
        "RelFluxErr_p90": rel_stats["p90"],
        "AbsFluxErr_mean": abs_stats["mean"],
        "RMSE_pore_mean": rmse_stats["mean"],
        "MAE_pore_mean": mae_stats["mean"],
        "Flux_R2": r2_val,
    }


def subset_eval(pred: EvalOutputs, keep_indices: Iterable[int]) -> EvalOutputs:
    keep = np.asarray(sorted(set(int(x) for x in keep_indices)), dtype=np.int64)
    mask = np.isin(pred.index.astype(np.int64), keep)
    return EvalOutputs(
        q_true=pred.q_true[mask],
        q_pred=pred.q_pred[mask],
        rel_err=pred.rel_err[mask],
        abs_err=pred.abs_err[mask],
        rmse_pore=pred.rmse_pore[mask],
        mae_pore=pred.mae_pore[mask],
        pore_frac=pred.pore_frac[mask],
        rock_type=pred.rock_type[mask],
        global_id=pred.global_id[mask],
        index=pred.index[mask],
        pred_ux=pred.pred_ux[mask],
    )


def save_predictions(path: str, pred: EvalOutputs) -> None:
    np.savez_compressed(
        path,
        q_true=pred.q_true.astype(np.float32),
        q_pred=pred.q_pred.astype(np.float32),
        rel_err=pred.rel_err.astype(np.float32),
        abs_err=pred.abs_err.astype(np.float32),
        rmse_pore=pred.rmse_pore.astype(np.float32),
        mae_pore=pred.mae_pore.astype(np.float32),
        pore_frac=pred.pore_frac.astype(np.float32),
        rock_type=pred.rock_type.astype(np.int16),
        global_id=pred.global_id.astype(np.int32),
        index=pred.index.astype(np.int32),
        pred_ux=pred.pred_ux.astype(np.float16),
    )


def plot_training_curves_from_csv(history_csv: str, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping training curves")
        return

    rows = []
    with open(history_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("training_history.csv is empty, skipping training curves")
        return

    epochs = np.asarray([int(r["epoch"]) for r in rows], dtype=np.int32)
    train_loss = np.asarray([float(r["train_loss"]) for r in rows], dtype=np.float64)
    val_loss = np.asarray([float(r["val_loss"]) for r in rows], dtype=np.float64)
    train_field = np.asarray([float(r["train_field"]) for r in rows], dtype=np.float64)
    val_field = np.asarray([float(r["val_field"]) for r in rows], dtype=np.float64)
    train_flux = np.asarray([float(r["train_flux"]) for r in rows], dtype=np.float64)
    val_flux = np.asarray([float(r["val_flux"]) for r in rows], dtype=np.float64)
    train_dist = np.asarray([float(r.get("train_dist", 0.0)) for r in rows], dtype=np.float64)
    val_dist = np.asarray([float(r.get("val_dist", 0.0)) for r in rows], dtype=np.float64)
    lr = np.asarray([float(r["lr"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_loss, "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.plot(epochs, train_field, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_field, "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Field Loss")
    ax.set_title("Field Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.plot(epochs, train_flux, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_flux, "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flux Loss")
    ax.set_title("Flux Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if np.any((train_flux > 0) | (val_flux > 0)):
        ax.set_yscale("log")

    ax = axes[1, 0]
    ax.plot(epochs, train_dist, "b-", label="Train", linewidth=2)
    ax.plot(epochs, val_dist, "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Distribution Loss")
    ax.set_title("Distribution Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if np.any((train_dist > 0) | (val_dist > 0)):
        ax.set_yscale("log")

    ax = axes[1, 1]
    best_idx = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_idx])
    ax.plot(epochs, val_loss, "r-", linewidth=2, label="Val Loss")
    ax.scatter([best_epoch], [val_loss[best_idx]], color="black", s=40, zorder=3, label=f"Best @ {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Best Validation Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1, 2]
    ax.plot(epochs, lr, color="orange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.suptitle("Training Curves (PoreFlow-Net Baseline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = Path(out_dir) / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"training curves saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个更贴近官方 PoreFlow-Net 的 2D 适配基线。")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-flux", type=float, default=0.0)
    parser.add_argument("--lambda-dist", type=float, default=0.0)
    parser.add_argument("--lambda-poisson", type=float, default=0.0)
    parser.add_argument("--base-filters", type=int, default=10)
    parser.add_argument("--porosity-kernel", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_raw = build_feature_maps(args.h5, porosity_kernel=args.porosity_kernel)
    split = load_split_indices(args.split_json, arrays_raw["rock_type"], arrays_raw["global_id"])
    transform_stats = fit_transforms(arrays_raw, split["train"])
    arrays = apply_transforms(arrays_raw, transform_stats)

    train_ds = InMemoryBranchDataset(arrays, split["train"])
    val_ds = InMemoryBranchDataset(arrays, split["val"])
    all_ds = InMemoryBranchDataset(arrays, np.arange(arrays["y_raw"].shape[0], dtype=np.int64))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = PoreFlowNetOfficial2D(base_filters=args.base_filters).to(device)
    criterion = PoreFlowLoss(
        lambda_flux=args.lambda_flux,
        lambda_dist=args.lambda_dist,
        lambda_poisson=args.lambda_poisson,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history_rows: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_metrics = iterate_batches(model, train_loader, device, transform_stats["y"], criterion=criterion, optimizer=optimizer)
        val_metrics = iterate_batches(model, val_loader, device, transform_stats["y"], criterion=criterion, optimizer=None)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_field": train_metrics["field"],
            "train_flux": train_metrics["flux"],
            "train_dist": train_metrics["dist"],
            "train_poisson": train_metrics["poisson"],
            "val_loss": val_metrics["loss"],
            "val_field": val_metrics["field"],
            "val_flux": val_metrics["flux"],
            "val_dist": val_metrics["dist"],
            "val_poisson": val_metrics["poisson"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.6f} "
            f"(field {train_metrics['field']:.6f}, flux {train_metrics['flux']:.6f}, "
            f"dist {train_metrics['dist']:.6f}, poisson {train_metrics['poisson']:.6f}) | "
            f"val {val_metrics['loss']:.6f} "
            f"(field {val_metrics['field']:.6f}, flux {val_metrics['flux']:.6f}, "
            f"dist {val_metrics['dist']:.6f}, poisson {val_metrics['poisson']:.6f})"
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
                    "transform_stats": {k: asdict(v) for k, v in transform_stats.items()},
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
        json.dump({k: asdict(v) for k, v in transform_stats.items()}, f, indent=2, ensure_ascii=False)

    pred_all = predict_dataset(model, all_loader, device, transform_stats["y"])
    pred_test = subset_eval(pred_all, split["test"])
    save_predictions(str(out_dir / "predictions.npz"), pred_all)
    all_summary = write_report(str(out_dir / "report.csv"), pred_all)
    test_summary = write_report(str(out_dir / "report_test.csv"), pred_test)

    summary = {
        "paper": {
            "name": "Santos et al. 2020 PoreFlow-Net baseline",
            "doi": "10.1016/j.advwatres.2020.103539",
            "reproduction_level": "official-code-inspired-2d-adaptation",
            "aligned_points": [
                "four single-channel branches",
                "residual branch encoders",
                "concatenated skip levels",
                "BatchNorm + SELU",
                "minMax_2 normalization",
                "MAE velocity-field objective",
                "optional flux, distribution, and Poisson-shape physics losses",
            ],
            "notes": [
                "Original code is 3D TensorFlow/Keras; this script is a 2D PyTorch adaptation.",
                "Current dataset has no true TOF, so TOF-like surrogate branches are used.",
                "MIS_z is also replaced by a local-porosity/global-geometry mixed map.",
            ],
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_run": int(len(history_rows)),
            "lambda_flux": float(args.lambda_flux),
            "lambda_dist": float(args.lambda_dist),
            "lambda_poisson": float(args.lambda_poisson),
        },
        "all_metrics": all_summary,
        "test_metrics": test_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved baseline outputs to: {out_dir}")


if __name__ == "__main__":
    main()
