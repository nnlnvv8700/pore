#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified architecture comparison for IPNM1 velocity-to-conductance models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_poreflownet_baseline import (
    MinMax2Stats,
    PoreFlowLoss,
    compute_r2,
    fit_minmax2_stats,
    get_channel_index,
    inverse_minmax2_torch,
    load_split_indices,
    read_channel_order,
    set_seed,
    stats_basic,
    transform_minmax2,
)
from train_fno_baseline import FNO2d


def ensure_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, None]
    raise ValueError(f"Expected 3D/4D array, got {arr.shape}")


def build_arrays(h5_path: str, porosity_kernel: int = 11) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        x_h5 = f["X"][:].astype(np.float32)
        y_raw = ensure_nchw(f["Y"][:].astype(np.float32))
        rock_type = f["rock_type"][:].astype(np.int16)
        global_id = f["global_id"][:].astype(np.int32)
        eta = f["eta"][:].astype(np.float32) if "eta" in f else np.zeros(x_h5.shape[0], dtype=np.float32)
        ri = f["RI"][:].astype(np.float32) if "RI" in f else np.zeros(x_h5.shape[0], dtype=np.float32)
        r0_ra = f["R0_RA"][:].astype(np.float32) if "R0_RA" in f else np.zeros(x_h5.shape[0], dtype=np.float32)

    order = read_channel_order(h5_path)
    mask_idx = get_channel_index(order, "mask", default=0)
    dist_idx = get_channel_index(order, "dist", default=1 if x_h5.shape[1] > 1 else None)
    eta_idx = get_channel_index(order, "eta_map", default=2 if x_h5.shape[1] > 2 else None)
    if mask_idx is None:
        raise ValueError("mask channel not found")

    mask = x_h5[:, mask_idx : mask_idx + 1].astype(np.float32)
    dist = x_h5[:, dist_idx : dist_idx + 1].astype(np.float32) if dist_idx is not None else np.zeros_like(mask)
    eta_map = x_h5[:, eta_idx : eta_idx + 1].astype(np.float32) if eta_idx is not None else np.zeros_like(mask)

    n, _, h, w = mask.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    coords = np.stack([grid_x, grid_y], axis=0)[None]
    coords = np.broadcast_to(coords, (n, 2, h, w)).astype(np.float32)

    mask2 = mask[:, 0]
    dist2 = dist[:, 0]
    resistance = np.where(mask2 > 0.5, 1.0 / np.maximum(dist2, 1.0e-3), 0.0).astype(np.float32)
    tof_left = np.cumsum(resistance, axis=2).astype(np.float32) * mask2
    tof_right = np.flip(np.cumsum(np.flip(resistance, axis=2), axis=2), axis=2).astype(np.float32) * mask2
    local_porosity = uniform_filter(mask2, size=(1, porosity_kernel, porosity_kernel), mode="nearest").astype(np.float32)

    eta_norm = (eta - eta.min()) / max(float(eta.max() - eta.min()), 1.0e-8)
    ri_norm = (ri - ri.min()) / max(float(ri.max() - ri.min()), 1.0e-8)
    r0_norm = (r0_ra - r0_ra.min()) / max(float(r0_ra.max() - r0_ra.min()), 1.0e-8)
    global_geom = (
        0.50 * eta_map[:, 0]
        + 0.20 * np.broadcast_to(eta_norm[:, None, None], mask2.shape)
        + 0.15 * np.broadcast_to(ri_norm[:, None, None], mask2.shape)
        + 0.15 * np.broadcast_to(r0_norm[:, None, None], mask2.shape)
    ).astype(np.float32)
    mis_like = (0.6 * local_porosity + 0.4 * global_geom) * mask2

    geom = np.concatenate(
        [
            mask,
            dist,
            eta_map,
            tof_left[:, None],
            tof_right[:, None],
            local_porosity[:, None],
            mis_like[:, None],
            coords,
        ],
        axis=1,
    ).astype(np.float32)
    branch = np.concatenate(
        [dist * mask, tof_left[:, None], tof_right[:, None], mis_like[:, None]],
        axis=1,
    ).astype(np.float32)
    scalar = np.stack([eta_norm, ri_norm, r0_norm], axis=1).astype(np.float32)

    return {
        "geom_raw": geom,
        "branch_raw": branch,
        "scalar_raw": scalar,
        "y_raw": y_raw.astype(np.float32),
        "mask": mask.astype(np.float32),
        "rock_type": rock_type,
        "global_id": global_id,
    }


def fit_channel_stats(arr: np.ndarray, train_idx: np.ndarray) -> List[MinMax2Stats]:
    return [fit_minmax2_stats(arr[train_idx, ch : ch + 1]) for ch in range(arr.shape[1])]


def apply_channel_stats(arr: np.ndarray, stats: List[MinMax2Stats]) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for ch, st in enumerate(stats):
        out[:, ch : ch + 1] = transform_minmax2(arr[:, ch : ch + 1], st)
    return out


class UnifiedDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.geom = torch.from_numpy(arrays["geom_t"][self.indices])
        self.branch = torch.from_numpy(arrays["branch_t"][self.indices])
        self.scalar = torch.from_numpy(arrays["scalar_t"][self.indices])
        self.y_t = torch.from_numpy(arrays["y_t"][self.indices])
        self.y_raw = torch.from_numpy(arrays["y_raw"][self.indices])
        self.mask = torch.from_numpy(arrays["mask"][self.indices])
        self.rock_type = arrays["rock_type"][self.indices]
        self.global_id = arrays["global_id"][self.indices]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "geom": self.geom[i],
            "branch": self.branch[i],
            "scalar": self.scalar[i],
            "y_t": self.y_t[i],
            "y_raw": self.y_raw[i],
            "mask": self.mask[i],
            "index": int(self.indices[i]),
            "rock_type": int(self.rock_type[i]),
            "global_id": int(self.global_id[i]),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int, base: int = 24, multitask: bool = False):
        super().__init__()
        self.multitask = bool(multitask)
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.q_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 2),
            nn.GELU(),
            nn.Linear(base * 2, 1),
        ) if self.multitask else None

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        x1 = self.e1(geom)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.b(self.pool(x3))
        x = self.d3(torch.cat([self.u3(xb), x3], dim=1))
        x = self.d2(torch.cat([self.u2(x), x2], dim=1))
        x = self.d1(torch.cat([self.u1(x), x1], dim=1))
        field = self.out(x)
        if self.q_head is None:
            return field, None
        return field, self.q_head(xb).view(-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + self.skip(x))


class ResEncoderDecoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 24):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.GELU())
        self.e1 = ResBlock(base, base)
        self.e2 = ResBlock(base, base * 2, stride=2)
        self.e3 = ResBlock(base * 2, base * 4, stride=2)
        self.e4 = ResBlock(base * 4, base * 6, stride=2)
        self.d3 = ResBlock(base * 6 + base * 4, base * 4)
        self.d2 = ResBlock(base * 4 + base * 2, base * 2)
        self.d1 = ResBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        x0 = self.e1(self.stem(geom))
        x1 = self.e2(x0)
        x2 = self.e3(x1)
        x3 = self.e4(x2)
        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.d3(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([x, x0], dim=1))
        return self.out(x), None


class ConvNeXtBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.BatchNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, ch * 4, 1)
        self.pw2 = nn.Conv2d(ch * 4, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + y


class ConvNeXtEncoderDecoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 24):
        super().__init__()
        self.s0 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), ConvNeXtBlock(base))
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 2, stride=2), ConvNeXtBlock(base * 2))
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 2, stride=2), ConvNeXtBlock(base * 4))
        self.down3 = nn.Sequential(nn.Conv2d(base * 4, base * 6, 2, stride=2), ConvNeXtBlock(base * 6))
        self.fuse2 = ConvBlock(base * 6 + base * 4, base * 4)
        self.fuse1 = ConvBlock(base * 4 + base * 2, base * 2)
        self.fuse0 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        x0 = self.s0(geom)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.fuse2(torch.cat([F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False), x2], dim=1))
        x = self.fuse1(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), x1], dim=1))
        x = self.fuse0(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), x0], dim=1))
        return self.out(x), None


class FNOAdapter(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, depth: int = 4, modes: int = 16):
        super().__init__()
        self.net = FNO2d(in_channels=in_ch, width=width, depth=depth, modes1=modes, modes2=modes)

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        return self.net(geom), None


class PoreBranchEncoder(nn.Module):
    def __init__(self, base: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.SELU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
        )
        self.short = nn.Sequential(nn.Conv2d(1, base, 1), nn.BatchNorm2d(base))
        self.b1 = ResBlock(base, base * 2, stride=2)
        self.b2 = ResBlock(base * 2, base * 4, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        y0 = self.stem(x) + self.short(x)
        y1 = self.b1(y0)
        y2 = self.b2(y1)
        return [y0, y1, y2]


class IPNM1FlowNet(nn.Module):
    def __init__(self, base: int = 10):
        super().__init__()
        self.encoders = nn.ModuleList([PoreBranchEncoder(base) for _ in range(4)])
        self.bridge = ResBlock(base * 16, base * 8, stride=2)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec2 = ResBlock(base * 4 + base * 16, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ResBlock(base * 2 + base * 8, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = ResBlock(base + base * 4, base)
        self.out = nn.Sequential(nn.Conv2d(base, 1, 1), nn.SELU(inplace=True))

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        if branch is None:
            raise ValueError("IPNM1FlowNet requires branch input")
        feats = [enc(branch[:, i : i + 1]) for i, enc in enumerate(self.encoders)]
        skip0 = torch.cat([f[0] for f in feats], dim=1)
        skip1 = torch.cat([f[1] for f in feats], dim=1)
        skip2 = torch.cat([f[2] for f in feats], dim=1)
        x = self.bridge(skip2)
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        x = self.dec0(torch.cat([self.up0(x), skip0], dim=1))
        return self.out(x), None


class BranchNet(nn.Module):
    def __init__(self, in_ch: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class DeepONetLite(nn.Module):
    def __init__(self, in_ch: int, width: int = 64, height: int = 64, width_px: int = 64):
        super().__init__()
        self.branch = BranchNet(in_ch, width)
        self.trunk = TrunkNet(width)
        yy = torch.linspace(-1.0, 1.0, height)
        xx = torch.linspace(-1.0, 1.0, width_px)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        self.register_buffer("coords", coords)
        self.bias = nn.Parameter(torch.zeros(1))
        self.height = int(height)
        self.width_px = int(width_px)

    def forward(self, geom: torch.Tensor, branch: torch.Tensor | None = None, scalar: torch.Tensor | None = None):
        coeff = self.branch(geom)
        basis = self.trunk(self.coords)
        out = coeff @ basis.T / math.sqrt(float(coeff.shape[1]))
        out = out.view(geom.shape[0], 1, self.height, self.width_px) + self.bias
        return out, None


def make_model(name: str, in_ch: int, h: int, w: int, base: int, fno_width: int) -> nn.Module:
    if name == "ipnm1_flownet":
        return IPNM1FlowNet(base=max(8, base // 2))
    if name == "unet":
        return UNetSmall(in_ch, base=base, multitask=False)
    if name == "res_ed":
        return ResEncoderDecoder(in_ch, base=base)
    if name == "convnext_ed":
        return ConvNeXtEncoderDecoder(in_ch, base=base)
    if name == "fno":
        return FNOAdapter(in_ch, width=fno_width, depth=4, modes=16)
    if name == "deeponet_lite":
        return DeepONetLite(in_ch, width=max(64, base * 3), height=h, width_px=w)
    if name == "multitask_cnn":
        return UNetSmall(in_ch, base=base, multitask=True)
    raise ValueError(f"Unknown model: {name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_stats: MinMax2Stats,
    criterion: PoreFlowLoss,
    optimizer: torch.optim.Optimizer | None,
    lambda_head: float,
    residual_mode: str = "none",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "field": 0.0, "flux": 0.0, "dist": 0.0, "poisson": 0.0, "head": 0.0}
    residual_indices: List[np.ndarray] = []
    residual_values: List[np.ndarray] = []
    n_batches = 0
    for batch in loader:
        geom = batch["geom"].to(device)
        y_t = batch["y_t"].to(device)
        y_raw = batch["y_raw"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(is_train):
            pred_t, q_head = model(geom, batch["branch"].to(device), batch["scalar"].to(device))
            pred_raw = inverse_minmax2_torch(pred_t, y_stats) * mask
            loss, items = criterion(pred_t, y_t, pred_raw, y_raw, mask)
            head_loss = torch.zeros((), dtype=pred_t.dtype, device=device)
            if q_head is not None and lambda_head > 0:
                q_true = (y_raw * mask).sum(dim=(1, 2, 3))
                head_loss = torch.mean(torch.abs(q_head - q_true) / (torch.abs(q_true) + 1.0e-8))
                loss = loss + lambda_head * head_loss
            if is_train and residual_mode != "none":
                pore_count = torch.clamp(mask.sum(dim=(1, 2, 3)), min=1.0)
                field_res = (torch.abs(pred_raw - y_raw) * mask).sum(dim=(1, 2, 3))
                field_scale = (torch.abs(y_raw) * mask).sum(dim=(1, 2, 3)) + 1.0e-8
                field_res = field_res / field_scale
                q_pred = (pred_raw * mask).sum(dim=(1, 2, 3))
                q_true_res = (y_raw * mask).sum(dim=(1, 2, 3))
                flux_res = torch.abs(q_pred - q_true_res) / (torch.abs(q_true_res) + 1.0e-8)
                if residual_mode == "field":
                    sample_res = field_res
                elif residual_mode == "flux":
                    sample_res = flux_res
                elif residual_mode == "joint":
                    sample_res = 0.5 * field_res + 0.5 * flux_res
                else:
                    sample_res = torch.zeros_like(field_res)
                # The count keeps this residual pore-normalized while avoiding an unused tensor warning.
                sample_res = sample_res + 0.0 * pore_count
                residual_indices.append(np.asarray(batch["index"], dtype=np.int64))
                residual_values.append(sample_res.detach().cpu().numpy().astype(np.float64))
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["field"] += items.get("field", 0.0)
        totals["flux"] += items.get("flux", 0.0)
        totals["dist"] += items.get("dist", 0.0)
        totals["poisson"] += items.get("poisson", 0.0)
        totals["head"] += float(head_loss.detach().cpu())
        n_batches += 1
    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if residual_indices:
        out["residual_index"] = np.concatenate(residual_indices, axis=0)
        out["residual_value"] = np.concatenate(residual_values, axis=0)
    return out


def build_ars_weights(
    train_indices: np.ndarray,
    residual_index: np.ndarray,
    residual_value: np.ndarray,
    previous: np.ndarray,
    eta: float,
    tau: float,
    ema: float,
    clip_p: float,
) -> Tuple[np.ndarray, np.ndarray]:
    index_to_pos = {int(idx): pos for pos, idx in enumerate(train_indices.tolist())}
    scores = np.asarray(previous, dtype=np.float64).copy()
    seen = np.zeros_like(scores, dtype=bool)
    for idx, value in zip(residual_index.tolist(), residual_value.tolist()):
        pos = index_to_pos.get(int(idx))
        if pos is None or not np.isfinite(value):
            continue
        scores[pos] = (1.0 - ema) * float(value) + ema * float(previous[pos])
        seen[pos] = True
    if not np.any(seen):
        probs = np.full_like(scores, 1.0 / max(scores.size, 1), dtype=np.float64)
        return scores, probs
    finite = scores[np.isfinite(scores)]
    hi = np.percentile(finite, float(clip_p)) if finite.size else 1.0
    hi = max(float(hi), 1.0e-12)
    clipped = np.clip(np.nan_to_num(scores, nan=0.0, posinf=hi, neginf=0.0), 0.0, hi)
    adaptive = np.power(clipped + 1.0e-12, float(tau))
    if float(adaptive.sum()) <= 0:
        adaptive = np.ones_like(adaptive)
    adaptive = adaptive / adaptive.sum()
    uniform = np.full_like(adaptive, 1.0 / max(adaptive.size, 1), dtype=np.float64)
    probs = (1.0 - float(eta)) * uniform + float(eta) * adaptive
    return scores, np.maximum(probs, 1.0e-12)


def update_softadapt_lambda(
    history: List[Dict[str, float]],
    current_lambda: float,
    base_lambda: float,
    warmup: int,
    lookback: int,
    tau: float,
    ema: float,
    min_lambda: float,
    max_lambda: float,
) -> float:
    if len(history) <= int(warmup) or len(history) <= int(lookback):
        return float(current_lambda)
    recent = history[-1]
    past = history[-1 - int(lookback)]
    eps = 1.0e-12
    ratios = np.asarray(
        [
            float(recent["train_field"]) / (float(past["train_field"]) + eps),
            float(recent["train_flux"]) / (float(past["train_flux"]) + eps),
        ],
        dtype=np.float64,
    )
    ratios = np.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
    logits = (ratios - np.max(ratios)) / max(float(tau), eps)
    weights = np.exp(logits)
    weights = 2.0 * weights / max(float(weights.sum()), eps)
    target_lambda = float(base_lambda) * float(weights[1] / max(weights[0], eps))
    updated = float(ema) * float(current_lambda) + (1.0 - float(ema)) * target_lambda
    return float(np.clip(updated, float(min_lambda), float(max_lambda)))


def update_softadapt_lambdas(
    history: List[Dict[str, float]],
    current: Dict[str, float],
    base: Dict[str, float],
    warmup: int,
    lookback: int,
    tau: float,
    ema: float,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    if len(history) <= int(warmup) or len(history) <= int(lookback):
        return dict(current)
    keys = ["field"] + [k for k in ["flux", "dist", "poisson"] if float(base.get(k, 0.0)) > 0.0]
    recent = history[-1]
    past = history[-1 - int(lookback)]
    eps = 1.0e-12
    ratios = []
    for key in keys:
        ratios.append(float(recent[f"train_{key}"]) / (float(past[f"train_{key}"]) + eps))
    ratios_arr = np.nan_to_num(np.asarray(ratios, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    logits = (ratios_arr - np.max(ratios_arr)) / max(float(tau), eps)
    weights = np.exp(logits)
    weights = float(len(keys)) * weights / max(float(weights.sum()), eps)
    weight_by_key = {key: float(weight) for key, weight in zip(keys, weights.tolist())}
    field_weight = max(weight_by_key.get("field", 1.0), eps)
    updated = dict(current)
    for key in ["flux", "dist", "poisson"]:
        if float(base.get(key, 0.0)) <= 0.0:
            updated[key] = float(base.get(key, 0.0))
            continue
        target = float(base[key]) * weight_by_key[key] / field_weight
        lo, hi = bounds[key]
        value = float(ema) * float(current.get(key, base[key])) + (1.0 - float(ema)) * target
        updated[key] = float(np.clip(value, float(lo), float(hi)))
    return updated


def choose_validation_score(metrics: Dict[str, float], mode: str, base_lambda_flux: float) -> float:
    if mode == "flux":
        return float(metrics["flux"])
    if mode == "field":
        return float(metrics["field"])
    return float(metrics["field"]) + float(base_lambda_flux) * float(metrics["flux"])


def predict(model: nn.Module, loader: DataLoader, device: torch.device, y_stats: MinMax2Stats) -> Dict[str, np.ndarray]:
    model.eval()
    outs: Dict[str, List[np.ndarray]] = {k: [] for k in ["pred_ux", "q_true", "q_pred", "index", "rock_type", "global_id"]}
    with torch.no_grad():
        for batch in loader:
            geom = batch["geom"].to(device)
            mask = batch["mask"].to(device)
            y_raw = batch["y_raw"].to(device)
            pred_t, _ = model(geom, batch["branch"].to(device), batch["scalar"].to(device))
            pred_raw = inverse_minmax2_torch(pred_t, y_stats) * mask
            q_true = (y_raw * mask).sum(dim=(1, 2, 3))
            q_pred = (pred_raw * mask).sum(dim=(1, 2, 3))
            outs["pred_ux"].append(pred_raw.cpu().numpy().astype(np.float16))
            outs["q_true"].append(q_true.cpu().numpy().astype(np.float32))
            outs["q_pred"].append(q_pred.cpu().numpy().astype(np.float32))
            outs["index"].append(np.asarray(batch["index"], dtype=np.int32))
            outs["rock_type"].append(np.asarray(batch["rock_type"], dtype=np.int16))
            outs["global_id"].append(np.asarray(batch["global_id"], dtype=np.int32))
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def subset_by_indices(pred: Dict[str, np.ndarray], keep_indices: np.ndarray) -> Dict[str, np.ndarray]:
    keep = np.isin(pred["index"].astype(np.int64), np.asarray(keep_indices, dtype=np.int64))
    return {k: v[keep] for k, v in pred.items()}


def summarize_prediction(pred: Dict[str, np.ndarray], y_all: np.ndarray, mask_all: np.ndarray) -> Dict[str, float]:
    idx = pred["index"].astype(np.int64)
    p = pred["pred_ux"].astype(np.float64)
    y = y_all[idx].astype(np.float64)
    m = mask_all[idx].astype(bool)
    pore_true = y[m]
    pore_pred = p[m]
    err = pore_pred - pore_true
    q_true = pred["q_true"].astype(np.float64)
    q_pred = pred["q_pred"].astype(np.float64)
    rel = np.abs(q_pred - q_true) / (np.abs(q_true) + 1.0e-8)
    return {
        "n": int(idx.size),
        "pixel_r2": compute_r2(pore_pred, pore_true),
        "pixel_rmse": float(np.sqrt(np.mean(err ** 2))),
        "pixel_mae": float(np.mean(np.abs(err))),
        "pixel_1_minus_nrmse_range": float(1.0 - np.sqrt(np.mean(err ** 2)) / (pore_true.max() - pore_true.min() + 1.0e-12)),
        "q_mean_re": float(np.mean(rel)),
        "q_median_re": float(np.median(rel)),
        "q_p90_re": float(np.percentile(rel, 90)),
        "q_r2": compute_r2(q_pred, q_true),
    }


def conductance_summary(pred: Dict[str, np.ndarray], q_table_by_index: np.ndarray) -> Dict[str, float]:
    idx = pred["index"].astype(np.int64)
    q_pred = pred["q_pred"].astype(np.float64)
    q_table = q_table_by_index[idx].astype(np.float64)
    rel = np.abs(q_pred - q_table) / (np.abs(q_table) + 1.0e-12)
    return {
        "table_g_mean_re": float(np.mean(rel)),
        "table_g_median_re": float(np.median(rel)),
        "table_g_r2": compute_r2(q_pred * 5000.0, q_table * 5000.0),
    }


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
        vals = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for token in line.replace(",", " ").split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
        if vals:
            # Teacher tables satisfy g/q=5000 for the selected units.
            # The final numeric value in these per-sample files is the table g.
            out[i] = vals[-1] / 5000.0
    if not np.isfinite(out).all():
        raise ValueError("Could not load all table q values from permeability files")
    return out


def write_history(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(compare_dir: Path, run_summaries: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for item in run_summaries:
        hist = list(csv.DictReader((compare_dir / str(item["run"]) / "training_history.csv").open("r", encoding="utf-8")))
        ep = [int(r["epoch"]) for r in hist]
        val = [float(r["val_loss"]) for r in hist]
        axes[0].plot(ep, val, label=str(item["model"]))
        axes[1].plot(ep, [float(r["val_flux"]) for r in hist], label=str(item["model"]))
    axes[0].set_title("Validation total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation flux loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative q loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "loss_curves_comparison.png", dpi=180)
    plt.close(fig)

    labels = [str(x["model"]) for x in run_summaries]
    metrics = ["pixel_r2_all", "q_r2_all", "table_g_r2_all"]
    titles = ["Velocity pixel R2", "q R2", "Table g R2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [float(x[metric]) for x in run_summaries]
        ax.bar(labels, vals, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylim(max(0.0, min(vals) - 0.08), 1.01)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "r2_metrics_comparison.png", dpi=180)
    plt.close(fig)

    metrics = ["q_mean_re_all", "table_g_mean_re_all"]
    titles = ["q mean relative error", "Table g mean relative error"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [100.0 * float(x[metric]) for x in run_summaries]
        ax.bar(labels, vals, color="#f58518")
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(compare_dir / "relative_error_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified architecture comparison.")
    parser.add_argument("--h5", type=str, default=r"E:\mhw\1\cup\dataset_all_32.h5")
    parser.add_argument("--split-json", type=str, default=r"E:\mhw\1\cup\runs\final_ipnm1_flownet_flux010_30e\split_info.json")
    parser.add_argument("--data-root", type=str, default=r"E:\mhw\1\cup\data")
    parser.add_argument("--out-dir", type=str, default=r"E:\mhw\1\cup\runs\arch_compare_20260527")
    parser.add_argument("--models", type=str, default="unet,res_ed,convnext_ed,fno,deeponet_lite,multitask_cnn")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-flux", type=float, default=0.10)
    parser.add_argument("--lambda-dist", type=float, default=0.0)
    parser.add_argument("--lambda-poisson", type=float, default=0.0)
    parser.add_argument("--lambda-head", type=float, default=0.05)
    parser.add_argument("--ars-mode", type=str, default="none", choices=["none", "field", "flux", "joint"])
    parser.add_argument("--ars-warmup", type=int, default=10)
    parser.add_argument("--ars-eta", type=float, default=0.5)
    parser.add_argument("--ars-tau", type=float, default=0.5)
    parser.add_argument("--ars-ema", type=float, default=0.7)
    parser.add_argument("--ars-clip-p", type=float, default=95.0)
    parser.add_argument("--adaptive-lambda", type=str, default="none", choices=["none", "softadapt", "softadapt_all"])
    parser.add_argument("--adaptive-warmup", type=int, default=10)
    parser.add_argument("--adaptive-lookback", type=int, default=1)
    parser.add_argument("--adaptive-tau", type=float, default=0.5)
    parser.add_argument("--adaptive-ema", type=float, default=0.8)
    parser.add_argument("--adaptive-min-lambda", type=float, default=0.01)
    parser.add_argument("--adaptive-max-lambda", type=float, default=1.0)
    parser.add_argument("--adaptive-min-dist", type=float, default=0.001)
    parser.add_argument("--adaptive-max-dist", type=float, default=0.10)
    parser.add_argument("--adaptive-min-poisson", type=float, default=0.0001)
    parser.add_argument("--adaptive-max-poisson", type=float, default=0.01)
    parser.add_argument("--select-metric", type=str, default="field_flux", choices=["field_flux", "flux", "field"])
    parser.add_argument("--base", type=int, default=20)
    parser.add_argument("--fno-width", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_arrays(args.h5)
    split = load_split_indices(args.split_json, arrays["rock_type"], arrays["global_id"])
    geom_stats = fit_channel_stats(arrays["geom_raw"], split["train"])
    branch_stats = fit_channel_stats(arrays["branch_raw"], split["train"])
    scalar_min = arrays["scalar_raw"][split["train"]].min(axis=0, keepdims=True)
    scalar_rng = np.maximum(arrays["scalar_raw"][split["train"]].max(axis=0, keepdims=True) - scalar_min, 1.0e-8)
    y_stats = fit_minmax2_stats(arrays["y_raw"][split["train"]])
    arrays["geom_t"] = apply_channel_stats(arrays["geom_raw"], geom_stats)
    arrays["branch_t"] = apply_channel_stats(arrays["branch_raw"], branch_stats)
    arrays["scalar_t"] = ((arrays["scalar_raw"] - scalar_min) * 2.0 / scalar_rng - 1.0).astype(np.float32)
    arrays["y_t"] = transform_minmax2(arrays["y_raw"], y_stats)

    train_ds = UnifiedDataset(arrays, split["train"])
    val_ds = UnifiedDataset(arrays, split["val"])
    all_ds = UnifiedDataset(arrays, np.arange(arrays["y_raw"].shape[0], dtype=np.int64))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_all = arrays["y_raw"].astype(np.float64)
    mask_all = arrays["mask"].astype(bool)
    q_table = load_table_q(args.h5, args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    criterion = PoreFlowLoss(
        lambda_flux=args.lambda_flux,
        lambda_dist=args.lambda_dist,
        lambda_poisson=args.lambda_poisson,
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_summaries: List[Dict[str, object]] = []

    for model_name in models:
        run_dir = out_dir / model_name
        if run_dir.exists() and args.overwrite:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        _, _, h, w = arrays["geom_t"].shape
        model = make_model(model_name, arrays["geom_t"].shape[1], h, w, args.base, args.fno_width).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        best_val = float("inf")
        best_epoch = -1
        patience_left = args.patience
        history: List[Dict[str, float]] = []
        ars_scores = np.ones(train_ds.indices.shape[0], dtype=np.float64)
        ars_probs = np.full(train_ds.indices.shape[0], 1.0 / max(train_ds.indices.shape[0], 1), dtype=np.float64)
        effective_lambdas = {
            "flux": float(args.lambda_flux),
            "dist": float(args.lambda_dist),
            "poisson": float(args.lambda_poisson),
        }
        start = time.time()

        print(f"\n=== Training {model_name} ===")
        for epoch in range(1, args.epochs + 1):
            criterion.lambda_flux = float(effective_lambdas["flux"])
            criterion.lambda_dist = float(effective_lambdas["dist"])
            criterion.lambda_poisson = float(effective_lambdas["poisson"])
            if args.ars_mode == "none" or epoch <= args.ars_warmup:
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
                ars_weight_max = 1.0 / max(len(train_ds), 1)
                ars_weight_mean = ars_weight_max
            else:
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(ars_probs, dtype=torch.double),
                    num_samples=len(train_ds),
                    replacement=True,
                )
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
                ars_weight_max = float(np.max(ars_probs))
                ars_weight_mean = float(np.mean(ars_probs))
            residual_mode = args.ars_mode if args.ars_mode != "none" else "none"
            train_metrics = train_epoch(
                model, train_loader, device, y_stats, criterion, optimizer, args.lambda_head, residual_mode=residual_mode
            )
            if args.ars_mode != "none" and "residual_index" in train_metrics:
                ars_scores, ars_probs = build_ars_weights(
                    train_ds.indices,
                    train_metrics.pop("residual_index"),
                    train_metrics.pop("residual_value"),
                    ars_scores,
                    eta=args.ars_eta,
                    tau=args.ars_tau,
                    ema=args.ars_ema,
                    clip_p=args.ars_clip_p,
                )
            val_metrics = train_epoch(model, val_loader, device, y_stats, criterion, None, args.lambda_head)
            val_select_loss = choose_validation_score(val_metrics, args.select_metric, args.lambda_flux)
            scheduler.step(val_select_loss)
            row = {
                "epoch": epoch,
                "lambda_flux_eff": effective_lambdas["flux"],
                "lambda_dist_eff": effective_lambdas["dist"],
                "lambda_poisson_eff": effective_lambdas["poisson"],
                "train_loss": train_metrics["loss"],
                "train_field": train_metrics["field"],
                "train_flux": train_metrics["flux"],
                "train_dist": train_metrics["dist"],
                "train_poisson": train_metrics["poisson"],
                "train_head": train_metrics["head"],
                "val_loss": val_metrics["loss"],
                "val_field": val_metrics["field"],
                "val_flux": val_metrics["flux"],
                "val_dist": val_metrics["dist"],
                "val_poisson": val_metrics["poisson"],
                "val_head": val_metrics["head"],
                "val_select_loss": val_select_loss,
                "ars_weight_mean": ars_weight_mean,
                "ars_weight_max": ars_weight_max,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if args.adaptive_lambda == "softadapt":
                effective_lambdas["flux"] = update_softadapt_lambda(
                    history,
                    current_lambda=effective_lambdas["flux"],
                    base_lambda=args.lambda_flux,
                    warmup=args.adaptive_warmup,
                    lookback=args.adaptive_lookback,
                    tau=args.adaptive_tau,
                    ema=args.adaptive_ema,
                    min_lambda=args.adaptive_min_lambda,
                    max_lambda=args.adaptive_max_lambda,
                )
            elif args.adaptive_lambda == "softadapt_all":
                effective_lambdas = update_softadapt_lambdas(
                    history,
                    current=effective_lambdas,
                    base={"flux": args.lambda_flux, "dist": args.lambda_dist, "poisson": args.lambda_poisson},
                    warmup=args.adaptive_warmup,
                    lookback=args.adaptive_lookback,
                    tau=args.adaptive_tau,
                    ema=args.adaptive_ema,
                    bounds={
                        "flux": (args.adaptive_min_lambda, args.adaptive_max_lambda),
                        "dist": (args.adaptive_min_dist, args.adaptive_max_dist),
                        "poisson": (args.adaptive_min_poisson, args.adaptive_max_poisson),
                    },
                )
            print(
                f"{model_name} epoch {epoch:03d} | "
                f"lambda f/q/d/p {1.0:.1f}/{row['lambda_flux_eff']:.4f}/{row['lambda_dist_eff']:.4f}/{row['lambda_poisson_eff']:.5f} | "
                f"train {train_metrics['loss']:.6f} | val {val_metrics['loss']:.6f} "
                f"(field {val_metrics['field']:.6f}, flux {val_metrics['flux']:.6f})"
            )
            if val_select_loss < best_val:
                best_val = val_select_loss
                best_epoch = epoch
                patience_left = args.patience
                torch.save({"model_state": model.state_dict(), "best_epoch": best_epoch, "best_val": best_val}, run_dir / "best.pt")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        elapsed = time.time() - start
        write_history(run_dir / "training_history.csv", history)
        ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        pred_all = predict(model, all_loader, device, y_stats)
        np.savez_compressed(run_dir / "predictions.npz", **pred_all)
        pred_val = subset_by_indices(pred_all, split["val"])
        pred_test = subset_by_indices(pred_all, split["test"])

        all_metrics = summarize_prediction(pred_all, y_all, mask_all)
        val_metrics = summarize_prediction(pred_val, y_all, mask_all)
        test_metrics = summarize_prediction(pred_test, y_all, mask_all)
        g_metrics = conductance_summary(pred_all, q_table)
        g_val = conductance_summary(pred_val, q_table)
        g_test = conductance_summary(pred_test, q_table)
        summary = {
            "model": model_name,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "epochs_run": int(len(history)),
            "elapsed_seconds": float(elapsed),
            "all": all_metrics | g_metrics,
            "val": val_metrics | g_val,
            "test": test_metrics | g_test,
            "args": vars(args),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        flat = {
            "run": model_name,
            "model": model_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "elapsed_seconds": elapsed,
        }
        for split_name, metrics in [("all", summary["all"]), ("val", summary["val"]), ("test", summary["test"])]:
            for key, value in metrics.items():
                flat[f"{key}_{split_name}"] = value
        run_summaries.append(flat)

    csv_path = out_dir / "architecture_metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(run_summaries)
    (out_dir / "architecture_metrics_summary.json").write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
    plot_outputs(out_dir, run_summaries)

    readme = out_dir / "README.md"
    lines = [
        "# Architecture Comparison",
        "",
        "Unified split and loss: `field + 0.10 * integrated_flux`.",
        "",
        "| Model | Pixel R2 all | Pixel R2 val | q mean RE all | q R2 all | Table g mean RE | Table g R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(run_summaries, key=lambda x: float(x["pixel_r2_all"]), reverse=True):
        lines.append(
            f"| {r['model']} | {float(r['pixel_r2_all']):.4f} | {float(r['pixel_r2_val']):.4f} | "
            f"{100.0 * float(r['q_mean_re_all']):.2f}% | {float(r['q_r2_all']):.4f} | "
            f"{100.0 * float(r['table_g_mean_re_all']):.2f}% | {float(r['table_g_r2_all']):.4f} |"
        )
    lines += [
        "",
        "Figures:",
        "",
        "- `loss_curves_comparison.png`",
        "- `r2_metrics_comparison.png`",
        "- `relative_error_comparison.png`",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
