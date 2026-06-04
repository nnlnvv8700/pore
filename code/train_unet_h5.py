
# -*- coding: utf-8 -*-
R"""
Train U-Net for Porous Media Flow Prediction (multi-rock HDF5).

This script trains a lightweight 2D U-Net to predict normalized ux velocity fields
from pore geometry inputs. It supports:
  - stratified splits by (rock_type, global_id)
  - flux-balanced and type+flux-balanced sampling
  - rock_type conditioning via extra input channel and FiLM
  - mixed flux loss (absolute + relative)
  - per rock_type metrics and CSV logging

Example commands (Windows):
  Multi-rock training (recommended)
  python train_unet_h5.py --h5 "E:\mhw\1\pore\dataset_all_32.h5" --out_dir "E:\mhw\1\pore\runs\unet_all32_v2" --epochs 200 --batch_size 96 --type_flux_balanced_sampler --use_rock_type_channel --use_softplus --lambda_q 10 --q_mix --use_film

  Fine-tune rock_3
  python train_unet_h5.py --h5 "E:\mhw\1\pore\dataset_all_32.h5" --out_dir "E:\mhw\1\pore\runs\unet_all32_ft_rock3" --epochs 50 --batch_size 96 --finetune_rock_type 3 --resume "E:\mhw\1\pore\runs\unet_all32_v2\best.pt" --lr 1e-4 --use_rock_type_channel --use_softplus --lambda_q 10 --q_mix --use_film
"""

import os
import json
import argparse
import random
import atexit
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

PERM_LINE_RE = re.compile(
    r"^\s*(?P<t>\d+)\s+"
    r"(?P<rho_Q>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<permeability>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<conductivity>[+-]?\d+(?:\.\d+)?)\s*$"
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# HDF5 helpers
# ============================================================================
def _decode_attr(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ('S', 'U'):
            return ','.join([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in val.tolist()])
    return val


def read_channel_order(h5_path: str) -> Tuple[List[str], Optional[int]]:
    """Read channel_order attr and return (list, mask_idx)."""
    channel_order = None
    with h5py.File(h5_path, 'r') as f:
        if 'X' in f and 'channel_order' in f['X'].attrs:
            channel_order = _decode_attr(f['X'].attrs['channel_order'])
        elif 'channel_order' in f.attrs:
            channel_order = _decode_attr(f.attrs['channel_order'])

    if channel_order is None:
        return [], None

    if isinstance(channel_order, str):
        order_list = [s.strip() for s in channel_order.split(',') if s.strip()]
    elif isinstance(channel_order, (list, tuple)):
        order_list = [str(s).strip() for s in channel_order]
    else:
        order_list = [str(channel_order).strip()]

    mask_idx = None
    for i, name in enumerate(order_list):
        if name.lower() == 'mask':
            mask_idx = i
            break
    return order_list, mask_idx


def get_channel_index(order_list: List[str], name: str) -> Optional[int]:
    """Find channel index by name (case-insensitive)."""
    if not order_list:
        return None
    name = name.lower()
    for i, ch in enumerate(order_list):
        if str(ch).lower() == name:
            return i
    return None


def load_h5_field_for_indices(h5_path: str, field: str, indices: List[int]) -> np.ndarray:
    """Load a 1D field for given indices, preserving order."""
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    out = np.empty(len(indices), dtype=np.int64)
    with h5py.File(h5_path, 'r') as f:
        data = f[field]
        for start in range(0, len(sorted_idx), 4096):
            end = min(start + 4096, len(sorted_idx))
            chunk = sorted_idx[start:end]
            out[start:end] = data[chunk]
    restored = np.empty_like(out)
    restored[order] = out
    return restored


def compute_q_true_for_indices(
    h5_path: str,
    indices: List[int],
    mask_idx: int,
    chunk_size: int = 512
) -> np.ndarray:
    """
    Compute q_true = sum(Y * mask) for specified indices.

    mask uses X's mask channel.
    """
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    q_true_sorted = np.empty(len(indices), dtype=np.float32)

    with h5py.File(h5_path, 'r') as f:
        X = f['X']
        Y = f['Y']
        for start in range(0, len(sorted_idx), chunk_size):
            end = min(start + chunk_size, len(sorted_idx))
            chunk = sorted_idx[start:end]
            x = X[chunk]  # (B, C, P, P)
            y = Y[chunk]  # (B, 1, P, P)
            mask = x[:, mask_idx, :, :]
            q_vals = np.sum(y[:, 0] * mask, axis=(1, 2))
            q_true_sorted[start:end] = q_vals.astype(np.float32)

    q_true = np.empty_like(q_true_sorted)
    q_true[order] = q_true_sorted
    return q_true


def parse_permeability_last_row(path: Path) -> Optional[Dict[str, float]]:
    if not path.is_file():
        return None
    last = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PERM_LINE_RE.match(raw_line.strip())
        if match:
            last = {
                "q_table": float(match.group("rho_Q")),
                "g_table": float(match.group("conductivity")),
                "permeability_table": float(match.group("permeability")),
            }
    return last


def build_table_q_lookup(
    h5_path: str,
    table_root: Optional[str],
    rock_folder_offset: int = 1,
    normalization_exponent: float = 2.0,
    area_scale_exponent: int = 2
) -> Tuple[Dict[int, Tuple[float, float]], int, int]:
    if not table_root:
        return {}, 0, 0

    root = Path(table_root)
    lookup: Dict[int, Tuple[float, float]] = {}
    missing = 0
    with h5py.File(h5_path, "r") as f:
        rock_types = f["rock_type"][:].astype(np.int64)
        local_ids = f["local_id"][:].astype(np.int64)
        scale_s = f["scale_s"][:].astype(np.float64)
        for idx, (rock_type, local_id, s) in enumerate(zip(rock_types, local_ids, scale_s)):
            folder = str(int(rock_type) + int(rock_folder_offset))
            parsed = parse_permeability_last_row(root / folder / f"permeability_{int(local_id)}.dat")
            if parsed is None:
                missing += 1
                continue
            q_table = abs(float(parsed["q_table"]))
            scale_factor = float(s) ** (float(normalization_exponent) + int(area_scale_exponent))
            lookup[int(idx)] = (q_table * scale_factor, abs(float(parsed["g_table"])))
    return lookup, len(lookup), missing

# ============================================================================
# Dataset and split
# ============================================================================
class PorousH5Dataset(Dataset):
    """
    HDF5 Dataset for porous media flow prediction.

    Reads data lazily from HDF5 file without loading everything into memory.
    """

    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        mask_idx: int,
        use_rock_type_channel: bool = False,
        dist_idx: Optional[int] = None,
        num_rock_types: int = 6,
        table_q_lookup: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        self.h5_path = h5_path
        self.indices = indices
        self.mask_idx = mask_idx
        self.dist_idx = dist_idx
        self.use_rock_type_channel = use_rock_type_channel
        self.num_rock_types = max(int(num_rock_types), 1)
        self.rock_norm_denom = max(self.num_rock_types - 1, 1)
        self.table_q_lookup = table_q_lookup or {}
        self._h5_file = None

        with h5py.File(h5_path, 'r') as f:
            self.total_samples = f['X'].shape[0]
            self.base_in_channels = f['X'].shape[1]
            self.in_channels = self.base_in_channels + (1 if use_rock_type_channel else 0)
            self.patch_size = f['X'].shape[2]
            self.has_rock_type = 'rock_type' in f
            self.has_global_id = 'global_id' in f
            self.has_local_id = 'local_id' in f
            self.has_scale_s = 'scale_s' in f

        if not self.has_rock_type:
            raise ValueError("HDF5 missing required dataset: /rock_type")
        if not self.has_global_id:
            raise ValueError("HDF5 missing required dataset: /global_id")

    def _open_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._open_h5()
        real_idx = self.indices[idx]

        x = self._h5_file['X'][real_idx]  # (C, P, P)
        y = self._h5_file['Y'][real_idx]  # (1, P, P)

        if self.mask_idx < 0 or self.mask_idx >= x.shape[0]:
            raise ValueError(f"mask_idx {self.mask_idx} out of bounds for X with {x.shape[0]} channels.")
        mask = x[self.mask_idx:self.mask_idx + 1].copy()
        if self.dist_idx is not None:
            if self.dist_idx < 0 or self.dist_idx >= x.shape[0]:
                raise ValueError(f"dist_idx {self.dist_idx} out of bounds for X with {x.shape[0]} channels.")
            dist = x[self.dist_idx:self.dist_idx + 1].copy()
        else:
            dist = np.zeros_like(mask, dtype=np.float32)

        rock_type = int(self._h5_file['rock_type'][real_idx])
        global_id = int(self._h5_file['global_id'][real_idx])
        local_id = int(self._h5_file['local_id'][real_idx]) if self.has_local_id else -1
        scale_s = float(self._h5_file['scale_s'][real_idx]) if self.has_scale_s else float('nan')
        table_vals = self.table_q_lookup.get(int(real_idx))
        if table_vals is None:
            q_table_norm, g_table, has_table = 0.0, 0.0, 0.0
        else:
            q_table_norm, g_table = table_vals
            has_table = 1.0

        if self.use_rock_type_channel:
            rock_val = rock_type / float(self.rock_norm_denom)
            rock_val = float(np.clip(rock_val, 0.0, 1.0))
            rock_map = np.full((1, x.shape[1], x.shape[2]), rock_val, dtype=np.float32)
            x = np.concatenate([x, rock_map], axis=0)

        return {
            'x': torch.from_numpy(x.astype(np.float32)),
            'y': torch.from_numpy(y.astype(np.float32)),
            'mask': torch.from_numpy(mask.astype(np.float32)),
            'dist': torch.from_numpy(dist.astype(np.float32)),
            'rock_type': rock_type,
            'global_id': global_id,
            'local_id': local_id,
            'scale_s': np.float32(scale_s),
            'q_table_norm': torch.tensor([q_table_norm], dtype=torch.float32),
            'g_table': torch.tensor([g_table], dtype=torch.float32),
            'has_table': torch.tensor([has_table], dtype=torch.float32),
            'idx': real_idx
        }

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


def split_by_global_id_stratified(
    h5_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], Dict]:
    """
    Split dataset by (rock_type, global_id) to prevent leakage.

    Patches from the same global_id are kept in a single split.
    The split is stratified by rock_type.
    """
    with h5py.File(h5_path, 'r') as f:
        if 'rock_type' not in f or 'global_id' not in f:
            raise ValueError("HDF5 must contain rock_type and global_id for stratified split.")
        rock_types = f['rock_type'][:]
        global_ids = f['global_id'][:]

    gid_to_indices = defaultdict(list)
    gid_to_type = {}
    for idx, (gid, rt) in enumerate(zip(global_ids, rock_types)):
        gid_int = int(gid)
        rt_int = int(rt)
        gid_to_indices[gid_int].append(idx)
        if gid_int not in gid_to_type:
            gid_to_type[gid_int] = rt_int
        elif gid_to_type[gid_int] != rt_int:
            print(f"Warning: global_id {gid_int} has mixed rock_type values.")

    type_to_gids = defaultdict(list)
    for gid, rt in gid_to_type.items():
        type_to_gids[int(rt)].append(int(gid))

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    split_info = {
        'seed': seed,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': 1.0 - train_ratio - val_ratio
        },
        'by_rock_type': {}
    }

    for rt in sorted(type_to_gids.keys()):
        gids = type_to_gids[rt]
        rng.shuffle(gids)
        n_ids = len(gids)
        n_train = int(n_ids * train_ratio)
        n_val = int(n_ids * val_ratio)
        n_test = n_ids - n_train - n_val

        if n_ids <= 1:
            n_train, n_val, n_test = n_ids, 0, 0
        elif n_ids == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            if n_val < 1:
                n_val = 1
            if n_test < 1:
                n_test = 1
            n_train = n_ids - n_val - n_test
            while n_train < 1:
                if n_val > 1:
                    n_val -= 1
                    n_train += 1
                elif n_test > 1:
                    n_test -= 1
                    n_train += 1
                else:
                    break
        train_gids = gids[:n_train]
        val_gids = gids[n_train:n_train + n_val]
        test_gids = gids[n_train + n_val:n_train + n_val + n_test]

        train_count = 0
        val_count = 0
        test_count = 0

        for gid in train_gids:
            idxs = gid_to_indices[gid]
            train_indices.extend(idxs)
            train_count += len(idxs)
        for gid in val_gids:
            idxs = gid_to_indices[gid]
            val_indices.extend(idxs)
            val_count += len(idxs)
        for gid in test_gids:
            idxs = gid_to_indices[gid]
            test_indices.extend(idxs)
            test_count += len(idxs)

        split_info['by_rock_type'][str(rt)] = {
            'train_global_ids': train_gids,
            'val_global_ids': val_gids,
            'test_global_ids': test_gids,
            'n_train_ids': len(train_gids),
            'n_val_ids': len(val_gids),
            'n_test_ids': len(test_gids),
            'n_train_samples': train_count,
            'n_val_samples': val_count,
            'n_test_samples': test_count
        }

    split_info['n_train_samples'] = len(train_indices)
    split_info['n_val_samples'] = len(val_indices)
    split_info['n_test_samples'] = len(test_indices)

    return train_indices, val_indices, test_indices, split_info

# ============================================================================
# Samplers
# ============================================================================
class FluxBalancedBatchSampler(Sampler[List[int]]):
    """Balance q_true bins within each batch (no rock_type balancing)."""

    def __init__(self, indices_by_bin: List[List[int]], dataset_len: int, batch_size: int, seed: int = 42):
        self.indices_by_bin = indices_by_bin
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.valid_bins = [i for i, b in enumerate(indices_by_bin) if len(b) > 0]
        if len(self.valid_bins) == 0:
            raise ValueError("No samples available for flux-balanced sampling.")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset_len // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        num_bins = len(self.valid_bins)
        base = self.batch_size // num_bins
        rem = self.batch_size % num_bins

        for _ in range(self.__len__()):
            counts = [base] * num_bins
            if rem > 0:
                extra = rng.permutation(num_bins)[:rem]
                for i in extra:
                    counts[i] += 1

            batch = []
            for i, bin_idx in enumerate(self.valid_bins):
                candidates = self.indices_by_bin[bin_idx]
                for _ in range(counts[i]):
                    choice = candidates[int(rng.randint(len(candidates)))]
                    batch.append(choice)
            rng.shuffle(batch)
            yield batch


class TypeFluxBalancedBatchSampler(Sampler[List[int]]):
    """Balance rock_type and q_true bins inside each rock_type."""

    def __init__(
        self,
        indices_by_type_bin: Dict[int, List[List[int]]],
        dataset_len: int,
        batch_size: int,
        seed: int = 42
    ):
        self.indices_by_type_bin = indices_by_type_bin
        self.present_types = sorted(indices_by_type_bin.keys())
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        if len(self.present_types) == 0:
            raise ValueError("No rock_type data available for balanced sampling.")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dataset_len // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        num_types = len(self.present_types)
        base = self.batch_size // num_types
        rem = self.batch_size % num_types

        valid_bins = {}
        bin_cycles = {}
        bin_ptr = {}
        for rt in self.present_types:
            bins = self.indices_by_type_bin[rt]
            non_empty = [i for i, b in enumerate(bins) if len(b) > 0]
            if len(non_empty) == 0:
                raise ValueError(f"rock_type {rt} has no samples for sampling.")
            valid_bins[rt] = non_empty
            bin_cycles[rt] = rng.permutation(non_empty).tolist()
            bin_ptr[rt] = 0

        def sample_one(rt: int) -> int:
            if bin_ptr[rt] >= len(bin_cycles[rt]):
                bin_cycles[rt] = rng.permutation(valid_bins[rt]).tolist()
                bin_ptr[rt] = 0
            bin_id = bin_cycles[rt][bin_ptr[rt]]
            bin_ptr[rt] += 1
            candidates = self.indices_by_type_bin[rt][bin_id]
            return candidates[int(rng.randint(len(candidates)))]

        for _ in range(self.__len__()):
            counts = [base] * num_types
            if rem > 0:
                extra_types = rng.permutation(num_types)[:rem]
                for i in extra_types:
                    counts[i] += 1

            batch = []
            for i, rt in enumerate(self.present_types):
                for _ in range(counts[i]):
                    batch.append(sample_one(rt))
            rng.shuffle(batch)
            yield batch


def build_flux_bins(q_true: np.ndarray, n_bins: int) -> List[List[int]]:
    """Build indices by q_true quantile bins."""
    q_vals = q_true
    bin_edges = np.quantile(q_vals, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] = bin_edges[-1] + 1e-8
    bin_ids = np.digitize(q_vals, bin_edges[1:-1])
    bins = []
    for b in range(n_bins):
        bins.append(np.where(bin_ids == b)[0].tolist())
    return bins


def build_type_bin_indices(rock_types: np.ndarray, q_true: np.ndarray, n_bins: int) -> Dict[int, List[List[int]]]:
    """Build indices_by_type_bin[type][bin] = list of dataset indices."""
    indices_by_type_bin: Dict[int, List[List[int]]] = {}
    unique_types = sorted(np.unique(rock_types).tolist())
    for rt in unique_types:
        type_mask = (rock_types == rt)
        idxs = np.where(type_mask)[0]
        if idxs.size == 0:
            continue
        q_vals = q_true[idxs]
        bin_edges = np.quantile(q_vals, np.linspace(0, 1, n_bins + 1))
        bin_edges[-1] = bin_edges[-1] + 1e-8
        bin_ids = np.digitize(q_vals, bin_edges[1:-1])
        bins: List[List[int]] = []
        for b in range(n_bins):
            bins.append(idxs[bin_ids == b].tolist())
        indices_by_type_bin[int(rt)] = bins
    return indices_by_type_bin

# ============================================================================
# Model
# ============================================================================
class ConvBlock(nn.Module):
    """Double conv block with GroupNorm + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        num_groups = min(num_groups, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block: Upsample + Concat + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightUNet(nn.Module):
    """Lightweight U-Net for 32x32 or 64x64 inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        use_softplus: bool = False,
        use_scale_head: bool = True,
        use_film: bool = True,
        film_dim: int = 16,
        num_rock_types: int = 6
    ):
        super().__init__()
        C = base_channels
        self.use_softplus = use_softplus
        self.use_scale_head = use_scale_head
        self.use_film = use_film
        self.film_dim = film_dim
        self.num_rock_types = num_rock_types

        self.enc1 = ConvBlock(in_channels, C)
        self.enc2 = DownBlock(C, C * 2)
        self.enc3 = DownBlock(C * 2, C * 4)
        self.bottleneck = DownBlock(C * 4, C * 8)

        self.dec3 = UpBlock(C * 8 + C * 4, C * 4)
        self.dec2 = UpBlock(C * 4 + C * 2, C * 2)
        self.dec1 = UpBlock(C * 2 + C, C)

        self.out_conv = nn.Conv2d(C, out_channels, 1)

        if use_softplus:
            self.softplus = nn.Softplus(beta=1.0)
        if use_scale_head:
            self.scale_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(C * 8, C * 2),
                nn.SiLU(inplace=True),
                nn.Linear(C * 2, 1),
                nn.Softplus(beta=1.0)
            )
        if use_film:
            self.film_embed = nn.Embedding(num_rock_types, film_dim)
            self.film_enc1 = nn.Linear(film_dim, 2 * C)
            self.film_enc2 = nn.Linear(film_dim, 2 * (C * 2))
            self.film_enc3 = nn.Linear(film_dim, 2 * (C * 4))
            self.film_bot = nn.Linear(film_dim, 2 * (C * 8))
            self.film_dec3 = nn.Linear(film_dim, 2 * (C * 4))
            self.film_dec2 = nn.Linear(film_dim, 2 * (C * 2))
            self.film_dec1 = nn.Linear(film_dim, 2 * C)

    def _apply_film(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rock_type: Optional[torch.Tensor] = None
    ):
        if self.use_film:
            if rock_type is None:
                raise ValueError("rock_type is required when use_film=True")
            film_feat = self.film_embed(rock_type.long())
        else:
            film_feat = None

        e1 = self.enc1(x)
        if film_feat is not None:
            e1 = self._apply_film(e1, self.film_enc1(film_feat))
        e2 = self.enc2(e1)
        if film_feat is not None:
            e2 = self._apply_film(e2, self.film_enc2(film_feat))
        e3 = self.enc3(e2)
        if film_feat is not None:
            e3 = self._apply_film(e3, self.film_enc3(film_feat))
        b = self.bottleneck(e3)
        if film_feat is not None:
            b = self._apply_film(b, self.film_bot(film_feat))
        d3 = self.dec3(b, e3)
        if film_feat is not None:
            d3 = self._apply_film(d3, self.film_dec3(film_feat))
        d2 = self.dec2(d3, e2)
        if film_feat is not None:
            d2 = self._apply_film(d2, self.film_dec2(film_feat))
        d1 = self.dec1(d2, e1)
        if film_feat is not None:
            d1 = self._apply_film(d1, self.film_dec1(film_feat))
        out = self.out_conv(d1)

        if self.use_softplus:
            out = self.softplus(out)

        if self.use_scale_head:
            scale = self.scale_head(b).view(-1, 1, 1, 1)
            out = out * scale
        return out

# ============================================================================
# Loss
# ============================================================================
class PorousFlowLoss(nn.Module):
    """Physics-informed loss with mixed flux error and optional negative penalty."""

    def __init__(
        self,
        lambda_field: float = 1.0,
        lambda_solid: float = 0.1,
        lambda_neg: float = 1.0,
        lambda_q: float = 10.0,
        lambda_grad: float = 0.05,
        lambda_peak: float = 0.1,
        lambda_wall: float = 10.0,
        lambda_mono: float = 2.0,
        lambda_dist_vel: float = 2.0,
        dist_vel_margin: float = 0.0,
        lambda_underpred: float = 5.0,
        underpred_alpha: float = 3.0,
        lambda_ratio: float = 5.0,
        lambda_stokes: float = 0.0,
        stokes_dist_thr: float = 0.10,
        stokes_norm_eps: float = 1.0e-6,
        lambda_table_q: float = 0.0,
        lambda_table_g: float = 0.0,
        table_normalization_exponent: float = 2.0,
        table_area_scale_exponent: int = 2,
        table_g_rho: float = 1.0,
        table_g_ax: float = 1.0e-4,
        table_g_segment_length: float = 4.0,
        table_g_mu_lbm: float = 0.5,
        table_g_target_mu: float = 1.0,
        q_mix: bool = True,
        q_thresh: float = 0.02,
        eps_q: float = 1e-3,
        median_q: float = 1.0,
        q_weight_power: float = 1.0,
        q_weight_clip_min: float = 0.5,
        q_weight_clip_max: float = 5.0,
        flux_weighted: bool = False,
        flux_w_alpha: float = 0.7,
        flux_w_min: float = 0.5,
        flux_w_max: float = 3.0,
        wall_dist_thr: float = 0.15,
        mono_bins: int = 6,
        mono_dist_max: float = 1.0,
        peak_percentile: float = 0.9,
        enable_neg_penalty: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()
        self.lambda_field = lambda_field
        self.lambda_solid = lambda_solid
        self.lambda_neg = lambda_neg
        self.lambda_q = lambda_q
        self.lambda_grad = lambda_grad
        self.lambda_peak = lambda_peak
        self.lambda_wall = lambda_wall
        self.lambda_mono = lambda_mono
        self.lambda_dist_vel = lambda_dist_vel
        self.dist_vel_margin = dist_vel_margin
        self.lambda_underpred = lambda_underpred
        self.underpred_alpha = underpred_alpha
        self.lambda_ratio = lambda_ratio
        self.lambda_stokes = lambda_stokes
        self.stokes_dist_thr = stokes_dist_thr
        self.stokes_norm_eps = stokes_norm_eps
        self.lambda_table_q = lambda_table_q
        self.lambda_table_g = lambda_table_g
        self.table_scale_exponent = float(table_normalization_exponent) + int(table_area_scale_exponent)
        driver = abs(float(table_g_rho) * float(table_g_ax) * float(table_g_segment_length) * float(table_g_mu_lbm))
        if driver <= 0:
            raise ValueError("table_g_rho * table_g_ax * table_g_segment_length * table_g_mu_lbm must be non-zero.")
        self.table_g_factor = float(table_g_target_mu) / driver
        self.q_mix = q_mix
        self.q_thresh = q_thresh
        self.eps_q = eps_q
        self.median_q = median_q
        self.q_weight_power = q_weight_power
        self.q_weight_clip_min = q_weight_clip_min
        self.q_weight_clip_max = q_weight_clip_max
        self.flux_weighted = flux_weighted
        self.flux_w_alpha = flux_w_alpha
        self.flux_w_min = flux_w_min
        self.flux_w_max = flux_w_max
        self.wall_dist_thr = wall_dist_thr
        self.mono_bins = mono_bins
        self.mono_dist_max = mono_dist_max
        self.peak_percentile = peak_percentile
        self.enable_neg_penalty = enable_neg_penalty
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        dist: Optional[torch.Tensor] = None,
        q_table_norm: Optional[torch.Tensor] = None,
        g_table: Optional[torch.Tensor] = None,
        scale_s: Optional[torch.Tensor] = None,
        has_table: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        pore_mask = mask
        solid_mask = 1.0 - mask

        field_diff_sq = (pred - target) ** 2 * pore_mask
        L_field = field_diff_sq.sum() / (pore_mask.sum() + self.eps)

        solid_penalty = pred ** 2 * solid_mask
        L_solid = solid_penalty.sum() / (solid_mask.sum() + self.eps)

        if self.enable_neg_penalty:
            neg_penalty = F.relu(-pred) * pore_mask
            L_neg = neg_penalty.sum() / (pore_mask.sum() + self.eps)
        else:
            L_neg = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        q_pred = (pred * pore_mask).sum(dim=(2, 3))
        q_true = (target * pore_mask).sum(dim=(2, 3))
        abs_err = torch.abs(q_pred - q_true)

        # Flux weighting to emphasize high-q samples
        if self.flux_weighted:
            weight = (torch.abs(q_true) / (self.median_q + self.eps)) ** self.flux_w_alpha
            weight = torch.clamp(weight, min=self.flux_w_min, max=self.flux_w_max)
        elif self.q_weight_power > 0:
            weight = (torch.abs(q_true) / (self.median_q + self.eps)) ** self.q_weight_power
            weight = torch.clamp(weight, min=self.q_weight_clip_min, max=self.q_weight_clip_max)
        else:
            weight = torch.ones_like(q_true)

        if self.q_mix:
            abs_q = torch.abs(q_true)
            rel_err = abs_err / (abs_q + self.eps_q)
            use_abs = abs_q < self.q_thresh
            flux_err = torch.where(use_abs, abs_err, rel_err)
        else:
            flux_err = abs_err / (torch.abs(q_true) + self.eps_q)
        L_flux = (weight * flux_err).mean()

        # Gradient loss (pore-aware)
        if self.lambda_grad > 0:
            dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]
            dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]
            mask_dx = pore_mask[:, :, :, 1:] * pore_mask[:, :, :, :-1]
            mask_dy = pore_mask[:, :, 1:, :] * pore_mask[:, :, :-1, :]
            L_grad_x = (torch.abs(dx_pred - dx_true) * mask_dx).sum() / (mask_dx.sum() + self.eps)
            L_grad_y = (torch.abs(dy_pred - dy_true) * mask_dy).sum() / (mask_dy.sum() + self.eps)
            L_grad = 0.5 * (L_grad_x + L_grad_y)
        else:
            L_grad = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Peak loss on high-value regions (relative error for scale-invariance)
        if self.lambda_peak > 0:
            B = target.size(0)
            pore_mask_bin = pore_mask > 0
            flat_mask = pore_mask_bin.view(B, -1)
            flat_target = target.view(B, -1)
            pore_counts = flat_mask.sum(dim=1)
            has_pore = pore_counts > 0
            if has_pore.any():
                k = torch.clamp(
                    (pore_counts.float() * (1.0 - self.peak_percentile)).ceil().to(torch.long),
                    min=1
                )
                max_k = int(k[has_pore].max().item())
                masked_vals = flat_target.masked_fill(~flat_mask, float('-inf'))
                topk_vals = torch.topk(masked_vals, k=max_k, dim=1).values
                gather_idx = (k - 1).clamp(min=0).unsqueeze(1)
                thresh = torch.gather(topk_vals, 1, gather_idx).squeeze(1)
                thresh = torch.where(
                    has_pore,
                    thresh,
                    torch.full_like(thresh, float('inf'))
                )
                peak_mask = (target >= thresh.view(-1, 1, 1, 1)).float() * pore_mask
                # Use relative error: |pred - target| / (target + eps) for scale-invariance
                peak_rel_err = torch.abs(pred - target) / (target + self.eps) * peak_mask
                L_peak = peak_rel_err.sum() / (peak_mask.sum() + self.eps)
            else:
                L_peak = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_peak = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Wall loss (near-boundary no-slip)
        if self.lambda_wall > 0 and dist is not None:
            wall_band = (pore_mask > 0.5) & (dist < self.wall_dist_thr)
            wall_count = wall_band.sum()
            if wall_count > 0:
                L_wall = (pred ** 2)[wall_band].sum() / (wall_count + self.eps)
            else:
                L_wall = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_wall = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Monotonic loss (center faster with larger dist)
        if self.lambda_mono > 0 and dist is not None:
            edges = torch.linspace(0.0, self.mono_dist_max, self.mono_bins + 1, device=pred.device)
            total_mono = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
            count_mono = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
            B = pred.size(0)
            for i in range(B):
                d = dist[i, 0]
                p = pred[i, 0]
                m = (pore_mask[i, 0] > 0.5) & (d <= self.mono_dist_max)
                if m.sum() == 0:
                    continue
                means = []
                valid = []
                for k in range(self.mono_bins):
                    bin_mask = m & (d >= edges[k]) & (d < edges[k + 1])
                    if bin_mask.any():
                        means.append(p[bin_mask].mean())
                        valid.append(True)
                    else:
                        means.append(None)
                        valid.append(False)
                for k in range(self.mono_bins - 1):
                    if valid[k] and valid[k + 1]:
                        total_mono = total_mono + F.relu(means[k] - means[k + 1])
                        count_mono = count_mono + 1.0
            L_mono = total_mono / (count_mono + self.eps)
        else:
            L_mono = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Under-prediction penalty (asymmetric loss: penalize pred < target more)
        if self.lambda_underpred > 0:
            diff_pore = (pred - target) * pore_mask  # negative where under-predicted
            # Asymmetric weight: under-prediction gets alpha times more penalty
            asym_weight = torch.where(
                diff_pore < 0,
                torch.full_like(diff_pore, self.underpred_alpha),  # under-pred
                torch.ones_like(diff_pore)                         # over-pred
            )
            L_underpred = (asym_weight * diff_pore ** 2 * pore_mask).sum() / (pore_mask.sum() + self.eps)
        else:
            L_underpred = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Ratio loss: penalize pred/target != 1 (scale-invariant, works on small values)
        if self.lambda_ratio > 0:
            # Only on pore pixels with significant target values
            t_pore = target * pore_mask
            p_pore = pred * pore_mask
            significant = (t_pore > self.eps).float() * pore_mask
            if significant.sum() > 0:
                ratio = (p_pore + self.eps) / (t_pore + self.eps)
                ratio = torch.clamp(ratio, min=1e-6, max=1e6)
                log_ratio = torch.log(ratio) * significant
                L_ratio = (log_ratio ** 2).sum() / (significant.sum() + self.eps)
            else:
                L_ratio = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_ratio = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        if self.lambda_table_q > 0 and q_table_norm is not None and has_table is not None:
            q_table = torch.abs(q_table_norm.to(device=pred.device, dtype=pred.dtype)).view(-1)
            valid = has_table.to(device=pred.device, dtype=pred.dtype).view(-1) > 0.5
            if valid.any():
                q_pred_abs = torch.abs(q_pred.view(-1))
                log_ratio_table = torch.log((q_pred_abs[valid] + self.eps_q) / (q_table[valid] + self.eps_q))
                L_table_q = (log_ratio_table ** 2).mean()
            else:
                L_table_q = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_table_q = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        if self.lambda_table_g > 0 and g_table is not None and scale_s is not None and has_table is not None:
            g_ref = torch.abs(g_table.to(device=pred.device, dtype=pred.dtype)).view(-1)
            s = torch.abs(scale_s.to(device=pred.device, dtype=pred.dtype)).view(-1)
            valid = (has_table.to(device=pred.device, dtype=pred.dtype).view(-1) > 0.5) & (g_ref > self.eps_q) & (s > self.eps)
            if valid.any():
                q_pred_abs = torch.abs(q_pred.view(-1))
                scale_factor = torch.pow(torch.clamp(s[valid], min=self.eps), self.table_scale_exponent)
                g_pred = q_pred_abs[valid] / (scale_factor + self.eps) * float(self.table_g_factor)
                log_ratio_g = torch.log((g_pred + self.eps_q) / (g_ref[valid] + self.eps_q))
                L_table_g = (log_ratio_g ** 2).mean()
            else:
                L_table_g = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_table_g = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        if self.lambda_stokes > 0:
            center = pred[:, :, 1:-1, 1:-1]
            lap = (
                pred[:, :, 1:-1, 2:] + pred[:, :, 1:-1, :-2] +
                pred[:, :, 2:, 1:-1] + pred[:, :, :-2, 1:-1] -
                4.0 * center
            )
            interior = (
                pore_mask[:, :, 1:-1, 1:-1] *
                pore_mask[:, :, 1:-1, 2:] *
                pore_mask[:, :, 1:-1, :-2] *
                pore_mask[:, :, 2:, 1:-1] *
                pore_mask[:, :, :-2, 1:-1]
            )
            if dist is not None:
                interior = interior * (dist[:, :, 1:-1, 1:-1] >= self.stokes_dist_thr).to(dtype=pred.dtype)
            denom = interior.sum(dim=(1, 2, 3), keepdim=True)
            valid = denom.view(-1) > 0.5
            if valid.any():
                lap_mean = (lap * interior).sum(dim=(1, 2, 3), keepdim=True) / (denom + self.eps)
                centered_lap = (lap - lap_mean) * interior
                per_sample = (centered_lap ** 2).sum(dim=(1, 2, 3)) / (denom.view(-1) + self.eps)
                # Normalize by the local velocity scale so the residual has a useful
                # magnitude across throat sizes and is not numerically silent.
                u_scale = (torch.abs(center) * interior).sum(dim=(1, 2, 3)) / (denom.view(-1) + self.eps)
                per_sample = per_sample / (u_scale ** 2 + self.stokes_norm_eps)
                L_stokes = per_sample[valid].mean()
            else:
                L_stokes = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        else:
            L_stokes = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        # Dist-Vel positive correlation loss (pixel-level)
        # For adjacent pore pixels: if dist increases, velocity should increase
        if self.lambda_dist_vel > 0 and dist is not None:
            d = dist          # (B,1,H,W)
            p = pred          # (B,1,H,W)
            m = pore_mask     # (B,1,H,W)

            # Horizontal neighbors
            dd_x = d[:, :, :, 1:] - d[:, :, :, :-1]  # delta dist
            dp_x = p[:, :, :, 1:] - p[:, :, :, :-1]  # delta pred
            mm_x = m[:, :, :, 1:] * m[:, :, :, :-1]  # both pore

            # Vertical neighbors
            dd_y = d[:, :, 1:, :] - d[:, :, :-1, :]
            dp_y = p[:, :, 1:, :] - p[:, :, :-1, :]
            mm_y = m[:, :, 1:, :] * m[:, :, :-1, :]

            # Violation: dist increases (dd>0) but velocity decreases (dp<0)
            # Penalty = ReLU(-dp - margin) * |dd| where dd > 0
            margin = self.dist_vel_margin
            viol_x_pos = F.relu(-dp_x - margin) * F.relu(dd_x) * mm_x
            viol_x_neg = F.relu(dp_x - margin) * F.relu(-dd_x) * mm_x
            viol_y_pos = F.relu(-dp_y - margin) * F.relu(dd_y) * mm_y
            viol_y_neg = F.relu(dp_y - margin) * F.relu(-dd_y) * mm_y

            total_viol = viol_x_pos.sum() + viol_x_neg.sum() + viol_y_pos.sum() + viol_y_neg.sum()
            total_pairs = mm_x.sum() + mm_y.sum()
            L_dist_vel = total_viol / (total_pairs + self.eps)
        else:
            L_dist_vel = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()

        total = (self.lambda_field * L_field +
                 self.lambda_solid * L_solid +
                 self.lambda_neg * L_neg +
                 self.lambda_q * L_flux +
                 self.lambda_grad * L_grad +
                 self.lambda_peak * L_peak +
                 self.lambda_wall * L_wall +
                 self.lambda_mono * L_mono +
                 self.lambda_dist_vel * L_dist_vel +
                 self.lambda_underpred * L_underpred +
                 self.lambda_ratio * L_ratio +
                 self.lambda_stokes * L_stokes +
                 self.lambda_table_q * L_table_q +
                 self.lambda_table_g * L_table_g)

        return {
            'total': total,
            'field': L_field,
            'solid': L_solid,
            'neg': L_neg,
            'flux': L_flux,
            'grad': L_grad,
            'peak': L_peak,
            'wall': L_wall,
            'mono': L_mono,
            'dist_vel': L_dist_vel,
            'underpred': L_underpred,
            'ratio': L_ratio,
            'stokes': L_stokes,
            'table_q': L_table_q,
            'table_g': L_table_g
        }


# ============================================================================
# Metrics
# ============================================================================
def compute_r2(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    if q_pred.size == 0:
        return float('nan')
    ss_res = np.sum((q_pred - q_true) ** 2)
    ss_tot = np.sum((q_true - np.mean(q_true)) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot


def summarize_rel_err(rel_err: np.ndarray) -> Dict[str, float]:
    if rel_err.size == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'p90': float('nan'), 'p95': float('nan')}
    return {
        'mean': float(np.mean(rel_err)),
        'median': float(np.median(rel_err)),
        'p90': float(np.percentile(rel_err, 90)),
        'p95': float(np.percentile(rel_err, 95))
    }


def summarize_metrics(rmse: np.ndarray, mae: np.ndarray, q_pred: np.ndarray, q_true: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(q_pred - q_true)
    rel_err = abs_err / (np.abs(q_true) + 1e-8)
    rel_stats = summarize_rel_err(rel_err)
    return {
        'RMSE_pore_mean': float(np.mean(rmse)) if rmse.size > 0 else float('nan'),
        'MAE_pore_mean': float(np.mean(mae)) if mae.size > 0 else float('nan'),
        'RelFluxErr_mean': rel_stats['mean'],
        'RelFluxErr_median': rel_stats['median'],
        'RelFluxErr_p90': rel_stats['p90'],
        'RelFluxErr_p95': rel_stats['p95'],
        'AbsFluxErr_mean': float(np.mean(abs_err)) if abs_err.size > 0 else float('nan'),
        'q_R2': compute_r2(q_pred, q_true)
    }

# ============================================================================
# Training and evaluation
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PorousFlowLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    clip_grad_norm: float = 1.0,
    debug_nan: bool = False,
    nan_debug_state: Optional[Dict[str, int]] = None,
    epoch: int = 0
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_field = 0.0
    total_flux = 0.0
    total_wall = 0.0
    total_mono = 0.0
    total_dist_vel = 0.0
    total_underpred = 0.0
    total_ratio = 0.0
    total_stokes = 0.0
    total_table_q = 0.0
    total_table_g = 0.0
    n_batches = 0

    all_rmse = []
    all_mae = []
    all_q_pred = []
    all_q_true = []

    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = batch['mask'].to(device)
        dist = batch['dist'].to(device)
        q_table_norm = batch.get('q_table_norm')
        g_table = batch.get('g_table')
        scale_s = batch.get('scale_s')
        has_table = batch.get('has_table')
        if q_table_norm is not None:
            q_table_norm = q_table_norm.to(device)
        if g_table is not None:
            g_table = g_table.to(device)
        if scale_s is not None:
            scale_s = scale_s.to(device)
        if has_table is not None:
            has_table = has_table.to(device)
        rock_type = None
        if getattr(model, "use_film", False):
            rock_type = batch['rock_type'].to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                pred = model(x, rock_type=rock_type)
                losses = criterion(
                    pred, y, mask, dist=dist,
                    q_table_norm=q_table_norm,
                    g_table=g_table,
                    scale_s=scale_s,
                    has_table=has_table
                )
                loss_total = losses['total']
            if not torch.isfinite(loss_total):
                if debug_nan and nan_debug_state is not None and nan_debug_state['count'] < nan_debug_state['max']:
                    with torch.no_grad():
                        pred_min = pred.min().item()
                        pred_max = pred.max().item()
                        pred_neg = bool((pred <= -criterion.eps).any().item())
                        t_min = y.min().item()
                        t_max = y.max().item()
                        t_pore = y * mask
                        significant = (t_pore > criterion.eps).float() * mask
                        sig_sum = significant.sum().item()
                    print(
                        f"[NaN] epoch {epoch} step {step} loss={loss_total.item()} "
                        f"pred_min={pred_min:.6g} pred_max={pred_max:.6g} pred<=-eps={pred_neg} "
                        f"t_min={t_min:.6g} t_max={t_max:.6g} significant_sum={sig_sum:.6g}"
                    )
                    nan_debug_state['count'] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss_total).backward()
            if clip_grad_norm and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x, rock_type=rock_type)
            losses = criterion(
                pred, y, mask, dist=dist,
                q_table_norm=q_table_norm,
                g_table=g_table,
                scale_s=scale_s,
                has_table=has_table
            )
            loss_total = losses['total']
            if not torch.isfinite(loss_total):
                if debug_nan and nan_debug_state is not None and nan_debug_state['count'] < nan_debug_state['max']:
                    with torch.no_grad():
                        pred_min = pred.min().item()
                        pred_max = pred.max().item()
                        pred_neg = bool((pred <= -criterion.eps).any().item())
                        t_min = y.min().item()
                        t_max = y.max().item()
                        t_pore = y * mask
                        significant = (t_pore > criterion.eps).float() * mask
                        sig_sum = significant.sum().item()
                    print(
                        f"[NaN] epoch {epoch} step {step} loss={loss_total.item()} "
                        f"pred_min={pred_min:.6g} pred_max={pred_max:.6g} pred<=-eps={pred_neg} "
                        f"t_min={t_min:.6g} t_max={t_max:.6g} significant_sum={sig_sum:.6g}"
                    )
                    nan_debug_state['count'] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_total.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        total_loss += losses['total'].item()
        total_field += losses['field'].item()
        total_flux += losses['flux'].item()
        total_wall += losses.get('wall', torch.tensor(0.0, device=device)).item()
        total_mono += losses.get('mono', torch.tensor(0.0, device=device)).item()
        total_dist_vel += losses.get('dist_vel', torch.tensor(0.0, device=device)).item()
        total_underpred += losses.get('underpred', torch.tensor(0.0, device=device)).item()
        total_ratio += losses.get('ratio', torch.tensor(0.0, device=device)).item()
        total_stokes += losses.get('stokes', torch.tensor(0.0, device=device)).item()
        total_table_q += losses.get('table_q', torch.tensor(0.0, device=device)).item()
        total_table_g += losses.get('table_g', torch.tensor(0.0, device=device)).item()
        n_batches += 1

        with torch.no_grad():
            diff = (pred - y) * mask
            pore_sum = mask.sum(dim=(1, 2, 3)) + 1e-8
            mse = (diff ** 2).sum(dim=(1, 2, 3)) / pore_sum
            rmse = torch.sqrt(mse).cpu().numpy()
            mae = (torch.abs(diff).sum(dim=(1, 2, 3)) / pore_sum).cpu().numpy()
            q_pred = (pred * mask).sum(dim=(2, 3)).cpu().numpy().flatten()
            q_true = (y * mask).sum(dim=(2, 3)).cpu().numpy().flatten()

            all_rmse.append(rmse)
            all_mae.append(mae)
            all_q_pred.append(q_pred)
            all_q_true.append(q_true)

    rmse_arr = np.concatenate(all_rmse, axis=0) if all_rmse else np.array([])
    mae_arr = np.concatenate(all_mae, axis=0) if all_mae else np.array([])
    q_pred_arr = np.concatenate(all_q_pred, axis=0) if all_q_pred else np.array([])
    q_true_arr = np.concatenate(all_q_true, axis=0) if all_q_true else np.array([])

    metrics = summarize_metrics(rmse_arr, mae_arr, q_pred_arr, q_true_arr)
    metrics['loss'] = total_loss / max(n_batches, 1)
    metrics['L_field'] = total_field / max(n_batches, 1)
    metrics['L_flux'] = total_flux / max(n_batches, 1)
    metrics['L_wall'] = total_wall / max(n_batches, 1)
    metrics['L_mono'] = total_mono / max(n_batches, 1)
    metrics['L_dist_vel'] = total_dist_vel / max(n_batches, 1)
    metrics['L_underpred'] = total_underpred / max(n_batches, 1)
    metrics['L_ratio'] = total_ratio / max(n_batches, 1)
    metrics['L_stokes'] = total_stokes / max(n_batches, 1)
    metrics['L_table_q'] = total_table_q / max(n_batches, 1)
    metrics['L_table_g'] = total_table_g / max(n_batches, 1)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PorousFlowLoss,
    device: torch.device,
    q_thresh: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    model.eval()

    total_loss = 0.0
    total_field = 0.0
    total_flux = 0.0
    total_wall = 0.0
    total_mono = 0.0
    total_dist_vel = 0.0
    total_underpred = 0.0
    total_ratio = 0.0
    total_stokes = 0.0
    total_table_q = 0.0
    total_table_g = 0.0
    n_batches = 0

    all_rmse = []
    all_mae = []
    all_q_pred = []
    all_q_true = []
    all_types = []

    for batch in tqdm(loader, desc='Val', leave=False):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = batch['mask'].to(device)
        dist = batch['dist'].to(device)
        q_table_norm = batch.get('q_table_norm')
        g_table = batch.get('g_table')
        scale_s = batch.get('scale_s')
        has_table = batch.get('has_table')
        if q_table_norm is not None:
            q_table_norm = q_table_norm.to(device)
        if g_table is not None:
            g_table = g_table.to(device)
        if scale_s is not None:
            scale_s = scale_s.to(device)
        if has_table is not None:
            has_table = has_table.to(device)
        rock_type_cpu = batch['rock_type'].cpu().numpy().astype(np.int64)
        rock_type_dev = None
        if getattr(model, "use_film", False):
            rock_type_dev = batch['rock_type'].to(device, dtype=torch.long)

        pred = model(x, rock_type=rock_type_dev)
        losses = criterion(
            pred, y, mask, dist=dist,
            q_table_norm=q_table_norm,
            g_table=g_table,
            scale_s=scale_s,
            has_table=has_table
        )
        total_loss += losses['total'].item()
        total_field += losses['field'].item()
        total_flux += losses['flux'].item()
        total_wall += losses.get('wall', torch.tensor(0.0, device=device)).item()
        total_mono += losses.get('mono', torch.tensor(0.0, device=device)).item()
        total_dist_vel += losses.get('dist_vel', torch.tensor(0.0, device=device)).item()
        total_underpred += losses.get('underpred', torch.tensor(0.0, device=device)).item()
        total_ratio += losses.get('ratio', torch.tensor(0.0, device=device)).item()
        total_stokes += losses.get('stokes', torch.tensor(0.0, device=device)).item()
        total_table_q += losses.get('table_q', torch.tensor(0.0, device=device)).item()
        total_table_g += losses.get('table_g', torch.tensor(0.0, device=device)).item()
        n_batches += 1

        diff = (pred - y) * mask
        pore_sum = mask.sum(dim=(1, 2, 3)) + 1e-8
        mse = (diff ** 2).sum(dim=(1, 2, 3)) / pore_sum
        rmse = torch.sqrt(mse).cpu().numpy()
        mae = (torch.abs(diff).sum(dim=(1, 2, 3)) / pore_sum).cpu().numpy()
        q_pred = (pred * mask).sum(dim=(2, 3)).cpu().numpy().flatten()
        q_true = (y * mask).sum(dim=(2, 3)).cpu().numpy().flatten()

        all_rmse.append(rmse)
        all_mae.append(mae)
        all_q_pred.append(q_pred)
        all_q_true.append(q_true)
        all_types.append(rock_type_cpu)

    rmse_arr = np.concatenate(all_rmse, axis=0) if all_rmse else np.array([])
    mae_arr = np.concatenate(all_mae, axis=0) if all_mae else np.array([])
    q_pred_arr = np.concatenate(all_q_pred, axis=0) if all_q_pred else np.array([])
    q_true_arr = np.concatenate(all_q_true, axis=0) if all_q_true else np.array([])
    types_arr = np.concatenate(all_types, axis=0) if all_types else np.array([], dtype=np.int64)

    overall = summarize_metrics(rmse_arr, mae_arr, q_pred_arr, q_true_arr)
    overall['loss'] = total_loss / max(n_batches, 1)
    overall['L_field'] = total_field / max(n_batches, 1)
    overall['L_flux'] = total_flux / max(n_batches, 1)
    overall['L_wall'] = total_wall / max(n_batches, 1)
    overall['L_mono'] = total_mono / max(n_batches, 1)
    overall['L_dist_vel'] = total_dist_vel / max(n_batches, 1)
    overall['L_underpred'] = total_underpred / max(n_batches, 1)
    overall['L_ratio'] = total_ratio / max(n_batches, 1)
    overall['L_stokes'] = total_stokes / max(n_batches, 1)
    overall['L_table_q'] = total_table_q / max(n_batches, 1)
    overall['L_table_g'] = total_table_g / max(n_batches, 1)

    # High-q metrics (ignore low |q_true| for best-model selection)
    if q_thresh is not None and q_true_arr.size > 0:
        hi_mask = np.abs(q_true_arr) >= float(q_thresh)
        if hi_mask.any():
            rel_hi = np.abs(q_pred_arr[hi_mask] - q_true_arr[hi_mask]) / (np.abs(q_true_arr[hi_mask]) + 1e-8)
            overall['RelFluxErr_p90_hiQ'] = float(np.percentile(rel_hi, 90))
            overall['RelFluxErr_median_hiQ'] = float(np.median(rel_hi))
        else:
            overall['RelFluxErr_p90_hiQ'] = float('nan')
            overall['RelFluxErr_median_hiQ'] = float('nan')
    else:
        overall['RelFluxErr_p90_hiQ'] = float('nan')
        overall['RelFluxErr_median_hiQ'] = float('nan')

    per_type = {}
    for rt in sorted(np.unique(types_arr).tolist()) if types_arr.size > 0 else []:
        mask_rt = (types_arr == rt)
        per_type[int(rt)] = summarize_metrics(
            rmse_arr[mask_rt],
            mae_arr[mask_rt],
            q_pred_arr[mask_rt],
            q_true_arr[mask_rt]
        )
        per_type[int(rt)]['n_samples'] = int(mask_rt.sum())

    return overall, per_type


def load_checkpoint(model: nn.Module, optimizer, scheduler, ckpt_path: str, device: torch.device, finetune: bool):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    start_epoch = 1
    best_metric = None

    if not finetune:
        if 'optimizer_state_dict' in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1
        if 'best_metric' in ckpt:
            best_metric = ckpt['best_metric']

    q_stats = None
    if isinstance(ckpt, dict) and any(k in ckpt for k in ('q_thresh', 'median_q', 'eps_q')):
        q_stats = {
            'q_thresh': ckpt.get('q_thresh', None),
            'median_q': ckpt.get('median_q', None),
            'eps_q': ckpt.get('eps_q', None)
        }

    return start_epoch, best_metric, q_stats

# ============================================================================
# Main training
# ============================================================================
def train(args):
    set_seed(args.seed)

    # Read channel order / mask index
    channel_order, mask_idx = read_channel_order(args.h5)
    if mask_idx is None:
        print("Warning: channel_order not found or missing 'mask'. Using mask_idx=0.")
        mask_idx = 0
    else:
        print(f"Channel order: {channel_order}")
        print(f"Using mask_idx={mask_idx}")
    dist_idx = get_channel_index(channel_order, 'dist')
    if dist_idx is None:
        print("Warning: channel_order missing 'dist'. dist-based losses will be disabled.")
    else:
        print(f"Using dist_idx={dist_idx}")
    if dist_idx is None and (args.lambda_wall > 0 or args.lambda_mono > 0 or args.lambda_dist_vel > 0):
        raise ValueError("dist channel not found but lambda_wall/lambda_mono/lambda_dist_vel > 0. "
                         "Please include dist in channel_order or set lambda_wall=lambda_mono=lambda_dist_vel=0.")

    # Split ratios
    if not np.isclose(args.split_train + args.split_val + args.split_test, 1.0):
        raise ValueError("split_train + split_val + split_test must sum to 1.0")

    # Finetune handling
    if args.finetune_rock_type is not None and args.lr == args.default_lr:
        args.lr = 1e-4
        print("Finetune detected: setting lr to 1e-4 (override default).")

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # Device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data split
    print("Splitting dataset by (rock_type, global_id) with stratification...")
    train_idx, val_idx, test_idx, split_info = split_by_global_id_stratified(
        args.h5, train_ratio=args.split_train, val_ratio=args.split_val, seed=args.seed
    )

    # Optional finetune: filter train indices by rock_type
    if args.finetune_rock_type is not None:
        train_types = load_h5_field_for_indices(args.h5, 'rock_type', train_idx)
        keep_mask = (train_types.astype(np.int64) == int(args.finetune_rock_type))
        train_idx = list(np.asarray(train_idx)[keep_mask])
        print(f"Finetune rock_type={args.finetune_rock_type}: using {len(train_idx)} train samples.")

    table_q_lookup, table_q_count, table_q_missing = build_table_q_lookup(
        args.h5,
        args.conductance_table_root,
        rock_folder_offset=args.rock_folder_offset,
        normalization_exponent=args.table_normalization_exponent,
        area_scale_exponent=args.table_area_scale_exponent
    )
    if args.conductance_table_root:
        print(f"Conductance table labels: loaded={table_q_count}, missing={table_q_missing}, "
              f"root={args.conductance_table_root}")

    # Save split info
    split_info['use_rock_type_channel'] = bool(args.use_rock_type_channel)
    split_info['use_film'] = bool(args.use_film)
    split_info['film_dim'] = int(args.film_dim)
    split_info['num_rock_types'] = int(args.num_rock_types)
    split_path = os.path.join(args.out_dir, 'split_info.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info saved to: {split_path}")

    # Datasets
    train_ds = PorousH5Dataset(
        args.h5, train_idx, mask_idx,
        use_rock_type_channel=args.use_rock_type_channel,
        dist_idx=dist_idx,
        num_rock_types=args.num_rock_types,
        table_q_lookup=table_q_lookup
    )
    val_ds = PorousH5Dataset(
        args.h5, val_idx, mask_idx,
        use_rock_type_channel=args.use_rock_type_channel,
        dist_idx=dist_idx,
        num_rock_types=args.num_rock_types,
        table_q_lookup=table_q_lookup
    )
    atexit.register(train_ds.close)
    atexit.register(val_ds.close)

    # Precompute q_true for training set
    print("Computing q_true for training set...")
    q_true_train = compute_q_true_for_indices(args.h5, train_idx, mask_idx)
    q_abs_train = np.abs(q_true_train)
    median_q = float(np.median(q_abs_train)) if q_abs_train.size > 0 else 0.0
    eps_q = 0.05 * median_q if median_q > 0 else 1e-4

    if args.q_thresh is None:
        q_thresh = float(np.percentile(q_abs_train, 10)) if q_abs_train.size > 0 else 0.02
        if q_thresh <= 0:
            q_thresh = 0.02
    else:
        q_thresh = float(args.q_thresh)

    print(f"Median |q_true| (training): {median_q:.6f}")
    print(f"eps_q for flux loss: {eps_q:.6f}")
    print(f"q_thresh for mixed flux loss: {q_thresh:.6f}")
    if args.flux_weighted:
        print(f"Flux weighted: True (alpha={args.flux_w_alpha}, wmin={args.flux_w_min}, wmax={args.flux_w_max})")
    else:
        print(f"Flux weighted: False (using q_weight_power={args.q_weight_power}, "
              f"clip=[{args.q_weight_clip_min},{args.q_weight_clip_max}])")
    print(f"Wall/Mono: lambda_wall={args.lambda_wall}, wall_dist_thr={args.wall_dist_thr}, "
          f"lambda_mono={args.lambda_mono}, mono_bins={args.mono_bins}, mono_dist_max={args.mono_dist_max}, "
          f"warmup_epochs={args.physics_warmup_epochs}")

    # Worker init for HDF5 multiprocessing
    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)

    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
        'worker_init_fn': worker_init_fn
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    # Sampler selection
    if args.type_flux_balanced_sampler and args.balanced_sampler:
        print("Both --balanced_sampler and type+flux sampler enabled; using flux-balanced sampler only.")
        args.type_flux_balanced_sampler = False

    if args.type_flux_balanced_sampler:
        print("Creating type + flux balanced batch sampler...")
        train_types = load_h5_field_for_indices(args.h5, 'rock_type', train_idx).astype(np.int64)
        indices_by_type_bin = build_type_bin_indices(train_types, q_true_train, args.bin_count)
        batch_sampler = TypeFluxBalancedBatchSampler(
            indices_by_type_bin=indices_by_type_bin,
            dataset_len=len(train_ds),
            batch_size=args.batch_size,
            seed=args.seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **loader_kwargs
        )
    elif args.balanced_sampler:
        print("Creating flux-balanced batch sampler...")
        bins = build_flux_bins(q_true_train, args.bin_count)
        batch_sampler = FluxBalancedBatchSampler(
            indices_by_bin=bins,
            dataset_len=len(train_ds),
            batch_size=args.batch_size,
            seed=args.seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )

    # Model
    model = LightUNet(
        in_channels=train_ds.in_channels,
        out_channels=1,
        base_channels=args.base_channels,
        use_softplus=args.use_softplus,
        use_scale_head=args.use_scale_head,
        use_film=args.use_film,
        film_dim=args.film_dim,
        num_rock_types=args.num_rock_types
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_film:
        print(f"  FiLM conditioning: enabled (num_rock_types={args.num_rock_types}, film_dim={args.film_dim})")

    # Loss
    criterion = PorousFlowLoss(
        lambda_field=args.lambda_field,
        lambda_solid=args.lambda_solid,
        lambda_neg=args.lambda_neg,
        lambda_q=args.lambda_q,
        lambda_grad=args.lambda_grad,
        lambda_peak=args.lambda_peak,
        lambda_wall=args.lambda_wall,
        lambda_mono=args.lambda_mono,
        lambda_dist_vel=args.lambda_dist_vel,
        lambda_underpred=args.lambda_underpred,
        underpred_alpha=args.underpred_alpha,
        lambda_ratio=args.lambda_ratio,
        lambda_stokes=args.lambda_stokes,
        stokes_dist_thr=args.stokes_dist_thr,
        stokes_norm_eps=args.stokes_norm_eps,
        lambda_table_q=args.lambda_table_q,
        lambda_table_g=args.lambda_table_g,
        table_normalization_exponent=args.table_normalization_exponent,
        table_area_scale_exponent=args.table_area_scale_exponent,
        table_g_rho=args.table_g_rho,
        table_g_ax=args.table_g_ax,
        table_g_segment_length=args.table_g_segment_length,
        table_g_mu_lbm=args.table_g_mu_lbm,
        table_g_target_mu=args.table_g_target_mu,
        q_mix=args.q_mix,
        q_thresh=q_thresh,
        eps_q=eps_q,
        median_q=median_q,
        q_weight_power=args.q_weight_power,
        q_weight_clip_min=args.q_weight_clip_min,
        q_weight_clip_max=args.q_weight_clip_max,
        flux_weighted=args.flux_weighted,
        flux_w_alpha=args.flux_w_alpha,
        flux_w_min=args.flux_w_min,
        flux_w_max=args.flux_w_max,
        wall_dist_thr=args.wall_dist_thr,
        mono_bins=args.mono_bins,
        mono_dist_max=args.mono_dist_max,
        peak_percentile=args.peak_percentile,
        enable_neg_penalty=not args.use_softplus
    )
    base_lambda_wall = args.lambda_wall
    base_lambda_mono = args.lambda_mono
    base_lambda_dist_vel = args.lambda_dist_vel
    base_lambda_grad = args.lambda_grad
    base_lambda_peak = args.lambda_peak
    base_lambda_underpred = args.lambda_underpred
    base_lambda_ratio = args.lambda_ratio
    base_lambda_stokes = args.lambda_stokes
    base_lambda_table_q = args.lambda_table_q
    base_lambda_table_g = args.lambda_table_g

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Mixed precision
    use_amp = (device.type == 'cuda')
    scaler = GradScaler() if use_amp else None

    # Resume
    start_epoch = 1
    best_metric_val = None
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        start_epoch, best_metric_val, ckpt_q_stats = load_checkpoint(
            model, optimizer, scheduler, args.resume, device, finetune=(args.finetune_rock_type is not None)
        )
        if ckpt_q_stats is not None:
            if args.finetune_rock_type is None:
                if ckpt_q_stats.get('median_q') is not None:
                    median_q = float(ckpt_q_stats['median_q'])
                if ckpt_q_stats.get('eps_q') is not None:
                    eps_q = float(ckpt_q_stats['eps_q'])
                if ckpt_q_stats.get('q_thresh') is not None:
                    q_thresh = float(ckpt_q_stats['q_thresh'])
                criterion.median_q = median_q
                criterion.eps_q = eps_q
                criterion.q_thresh = q_thresh
                print(f"Using q stats from checkpoint: median_q={median_q:.6f}, "
                      f"eps_q={eps_q:.6f}, q_thresh={q_thresh:.6f}")
            else:
                print("Checkpoint q stats found, but finetune enabled; keeping current q stats.")
        print(f"Resumed from epoch {start_epoch}")

    # History
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_L_field': [],
        'train_L_flux': [],
        'train_L_wall': [],
        'train_L_mono': [],
        'train_L_dist_vel': [],
        'train_L_underpred': [],
        'train_L_ratio': [],
        'train_L_stokes': [],
        'train_L_table_q': [],
        'train_L_table_g': [],
        'val_L_field': [],
        'val_L_flux': [],
        'val_L_wall': [],
        'val_L_mono': [],
        'val_L_dist_vel': [],
        'val_L_underpred': [],
        'val_L_ratio': [],
        'val_L_stokes': [],
        'val_L_table_q': [],
        'val_L_table_g': [],
        'train_RMSE_pore_mean': [],
        'train_MAE_pore_mean': [],
        'train_RelFluxErr_mean': [],
        'train_RelFluxErr_median': [],
        'train_RelFluxErr_p90': [],
        'train_RelFluxErr_p95': [],
        'train_AbsFluxErr_mean': [],
        'train_q_R2': [],
        'val_RMSE_pore_mean': [],
        'val_MAE_pore_mean': [],
        'val_RelFluxErr_mean': [],
        'val_RelFluxErr_median': [],
        'val_RelFluxErr_p90': [],
        'val_RelFluxErr_p95': [],
        'val_RelFluxErr_median_hiQ': [],
        'val_RelFluxErr_p90_hiQ': [],
        'val_AbsFluxErr_mean': [],
        'val_q_R2': [],
        'lr': []
    }
    history_by_type = []

    if best_metric_val is None:
        best_metric_val = float('inf')

    nan_debug_state = {'count': 0, 'max': args.debug_nan_max}

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Loss weights: field={args.lambda_field}, solid={args.lambda_solid}, neg={args.lambda_neg}, "
          f"flux={args.lambda_q}, wall={args.lambda_wall}, mono={args.lambda_mono}, dist_vel={args.lambda_dist_vel}")
    print(f"  underpred={args.lambda_underpred}(alpha={args.underpred_alpha}), ratio={args.lambda_ratio}, "
          f"stokes={args.lambda_stokes}(dist_thr={args.stokes_dist_thr})")
    print(f"  table_q={args.lambda_table_q}, table_g={args.lambda_table_g}, "
          f"conductance_table_root={args.conductance_table_root}")
    print("-" * 110)

    for epoch in range(start_epoch, args.epochs + 1):
        # Deterministic sampling per epoch
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        # Loss warmup: keep only core losses early, then ramp up auxiliary terms
        if args.physics_warmup_epochs > 0:
            if epoch <= args.physics_warmup_epochs:
                ramp = 0.0
            else:
                ramp = min(1.0, (epoch - args.physics_warmup_epochs) / float(args.physics_warmup_epochs))
        else:
            ramp = 1.0

        criterion.lambda_wall = base_lambda_wall * ramp
        criterion.lambda_mono = base_lambda_mono * ramp
        criterion.lambda_dist_vel = base_lambda_dist_vel * ramp
        criterion.lambda_grad = base_lambda_grad * ramp
        criterion.lambda_peak = base_lambda_peak * ramp
        criterion.lambda_underpred = base_lambda_underpred * ramp
        criterion.lambda_ratio = base_lambda_ratio * ramp
        criterion.lambda_stokes = base_lambda_stokes * ramp
        criterion.lambda_table_q = base_lambda_table_q * ramp
        criterion.lambda_table_g = base_lambda_table_g * ramp

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            clip_grad_norm=args.clip_grad_norm,
            debug_nan=args.debug_nan,
            nan_debug_state=nan_debug_state,
            epoch=epoch
        )

        # Validate
        val_metrics, val_by_type = evaluate_epoch(
            model, val_loader, criterion, device, q_thresh=q_thresh
        )

        # Scheduler (monitor val P90 or median RelFluxErr)
        if args.best_ignore_low_q and not np.isnan(val_metrics.get('RelFluxErr_p90_hiQ', np.nan)):
            metric_for_sched = val_metrics['RelFluxErr_p90_hiQ'] if args.best_metric == 'p90' else val_metrics['RelFluxErr_median_hiQ']
        else:
            metric_for_sched = val_metrics['RelFluxErr_p90'] if args.best_metric == 'p90' else val_metrics['RelFluxErr_median']
        scheduler.step(metric_for_sched)
        current_lr = optimizer.param_groups[0]['lr']

        # Log overall
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"L_field/L_q/L_wall/L_dv: "
            f"{train_metrics['L_field']:.4f}/{train_metrics['L_flux']:.4f}/"
            f"{train_metrics['L_wall']:.4f}/{train_metrics.get('L_dist_vel',0):.4f} | "
            f"L_up/L_rat: {train_metrics.get('L_underpred',0):.4f}/{train_metrics.get('L_ratio',0):.4f} | "
            f"L_stk/L_tg: {train_metrics.get('L_stokes',0):.4f}/{train_metrics.get('L_table_g',0):.4f} | "
            f"RelFluxErr mean/med/p90/p95: {val_metrics['RelFluxErr_mean']:.4f}/"
            f"{val_metrics['RelFluxErr_median']:.4f}/{val_metrics['RelFluxErr_p90']:.4f}/"
            f"{val_metrics['RelFluxErr_p95']:.4f} | "
            f"RMSE mean: {val_metrics['RMSE_pore_mean']:.4f} | "
            f"R2_q: {val_metrics['q_R2']:.4f}"
        )
        if args.best_ignore_low_q:
            print(
                f"  HiQ RelFluxErr med/p90: "
                f"{val_metrics.get('RelFluxErr_median_hiQ', float('nan')):.4f}/"
                f"{val_metrics.get('RelFluxErr_p90_hiQ', float('nan')):.4f}"
            )

        # Log per rock_type (val)
        for rt in sorted(val_by_type.keys()):
            m = val_by_type[rt]
            print(
                f"  rock_{rt}: RelFluxErr mean/med/p90/p95="
                f"{m['RelFluxErr_mean']:.4f}/{m['RelFluxErr_median']:.4f}/"
                f"{m['RelFluxErr_p90']:.4f}/{m['RelFluxErr_p95']:.4f} | "
                f"R2_q: {m['q_R2']:.4f} | N={m.get('n_samples', 0)}"
            )

        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_L_field'].append(train_metrics.get('L_field', float('nan')))
        history['train_L_flux'].append(train_metrics.get('L_flux', float('nan')))
        history['train_L_wall'].append(train_metrics.get('L_wall', float('nan')))
        history['train_L_mono'].append(train_metrics.get('L_mono', float('nan')))
        history['train_L_dist_vel'].append(train_metrics.get('L_dist_vel', float('nan')))
        history['train_L_underpred'].append(train_metrics.get('L_underpred', float('nan')))
        history['train_L_ratio'].append(train_metrics.get('L_ratio', float('nan')))
        history['train_L_stokes'].append(train_metrics.get('L_stokes', float('nan')))
        history['train_L_table_q'].append(train_metrics.get('L_table_q', float('nan')))
        history['train_L_table_g'].append(train_metrics.get('L_table_g', float('nan')))
        history['val_L_field'].append(val_metrics.get('L_field', float('nan')))
        history['val_L_flux'].append(val_metrics.get('L_flux', float('nan')))
        history['val_L_wall'].append(val_metrics.get('L_wall', float('nan')))
        history['val_L_mono'].append(val_metrics.get('L_mono', float('nan')))
        history['val_L_dist_vel'].append(val_metrics.get('L_dist_vel', float('nan')))
        history['val_L_underpred'].append(val_metrics.get('L_underpred', float('nan')))
        history['val_L_ratio'].append(val_metrics.get('L_ratio', float('nan')))
        history['val_L_stokes'].append(val_metrics.get('L_stokes', float('nan')))
        history['val_L_table_q'].append(val_metrics.get('L_table_q', float('nan')))
        history['val_L_table_g'].append(val_metrics.get('L_table_g', float('nan')))
        history['train_RMSE_pore_mean'].append(train_metrics['RMSE_pore_mean'])
        history['train_MAE_pore_mean'].append(train_metrics['MAE_pore_mean'])
        history['train_RelFluxErr_mean'].append(train_metrics['RelFluxErr_mean'])
        history['train_RelFluxErr_median'].append(train_metrics['RelFluxErr_median'])
        history['train_RelFluxErr_p90'].append(train_metrics['RelFluxErr_p90'])
        history['train_RelFluxErr_p95'].append(train_metrics['RelFluxErr_p95'])
        history['train_AbsFluxErr_mean'].append(train_metrics['AbsFluxErr_mean'])
        history['train_q_R2'].append(train_metrics['q_R2'])
        history['val_RMSE_pore_mean'].append(val_metrics['RMSE_pore_mean'])
        history['val_MAE_pore_mean'].append(val_metrics['MAE_pore_mean'])
        history['val_RelFluxErr_mean'].append(val_metrics['RelFluxErr_mean'])
        history['val_RelFluxErr_median'].append(val_metrics['RelFluxErr_median'])
        history['val_RelFluxErr_p90'].append(val_metrics['RelFluxErr_p90'])
        history['val_RelFluxErr_p95'].append(val_metrics['RelFluxErr_p95'])
        history['val_RelFluxErr_median_hiQ'].append(val_metrics.get('RelFluxErr_median_hiQ', float('nan')))
        history['val_RelFluxErr_p90_hiQ'].append(val_metrics.get('RelFluxErr_p90_hiQ', float('nan')))
        history['val_AbsFluxErr_mean'].append(val_metrics['AbsFluxErr_mean'])
        history['val_q_R2'].append(val_metrics['q_R2'])
        history['lr'].append(current_lr)

        for rt, m in val_by_type.items():
            history_by_type.append({
                'epoch': epoch,
                'rock_type': int(rt),
                'RelFluxErr_mean': m['RelFluxErr_mean'],
                'RelFluxErr_median': m['RelFluxErr_median'],
                'RelFluxErr_p90': m['RelFluxErr_p90'],
                'RelFluxErr_p95': m['RelFluxErr_p95'],
                'q_R2': m['q_R2'],
                'n_samples': m.get('n_samples', 0)
            })

        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'args': vars(args),
            'q_thresh': q_thresh,
            'eps_q': eps_q,
            'median_q': median_q,
            'best_metric': best_metric_val
        }
        torch.save(checkpoint, os.path.join(args.out_dir, 'last.pt'))

        # Best checkpoint by metric (optionally ignore low-q samples)
        if args.best_ignore_low_q and not np.isnan(val_metrics.get('RelFluxErr_p90_hiQ', np.nan)):
            metric_for_best = val_metrics['RelFluxErr_p90_hiQ'] if args.best_metric == 'p90' else val_metrics['RelFluxErr_median_hiQ']
        else:
            metric_for_best = val_metrics['RelFluxErr_p90'] if args.best_metric == 'p90' else val_metrics['RelFluxErr_median']
        if metric_for_best < best_metric_val:
            best_metric_val = metric_for_best
            checkpoint['best_metric'] = best_metric_val
            torch.save(checkpoint, os.path.join(args.out_dir, 'best.pt'))
            print(f"  -> New best model saved ({args.best_metric}: {best_metric_val:.4f})")

    # Save history CSV
    import csv
    history_path = os.path.join(args.out_dir, 'training_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        for i in range(len(history['epoch'])):
            row = {k: history[k][i] for k in history.keys()}
            writer.writerow(row)
    print(f"\nTraining history saved to: {history_path}")

    history_by_type_path = os.path.join(args.out_dir, 'training_history_by_rock_type.csv')
    if history_by_type:
        fieldnames = list(history_by_type[0].keys())
        with open(history_by_type_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history_by_type:
                writer.writerow(row)
        print(f"Per rock_type history saved to: {history_by_type_path}")

    # Plot training curves
    plot_training_curves(history, history_by_type, args.out_dir)

    train_ds.close()
    val_ds.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation {args.best_metric}: {best_metric_val:.4f}")
    print(f"Checkpoints saved to: {args.out_dir}")
    print("=" * 80)


# ============================================================================
# Visualization
# ============================================================================
def plot_training_curves(history: Dict, history_by_type: List[Dict], out_dir: str):
    """Plot and save comprehensive training curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    epochs = history['epoch']
    
    # =========================================================================
    # Figure 1: Main training curves (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. RelFluxErr (mean, median, p90, p95)
    ax = axes[0, 1]
    ax.plot(epochs, history['val_RelFluxErr_mean'], 'g-', label='Mean', linewidth=2)
    ax.plot(epochs, history['val_RelFluxErr_median'], 'b-', label='Median', linewidth=1.5)
    ax.plot(epochs, history['val_RelFluxErr_p90'], 'orange', label='P90', linewidth=1.5)
    ax.plot(epochs, history['val_RelFluxErr_p95'], 'r-', label='P95', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RelFluxErr')
    ax.set_title('Validation Relative Flux Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 3. R² (Flux)
    ax = axes[0, 2]
    ax.plot(epochs, history['train_q_R2'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_q_R2'], 'r-', label='Val', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² (Flux)')
    ax.set_title('Flux R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.05])

    # 4. RMSE (Pore)
    ax = axes[1, 0]
    ax.plot(epochs, history['train_RMSE_pore_mean'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_RMSE_pore_mean'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE (Pore Region)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. MAE (Pore)
    ax = axes[1, 1]
    ax.plot(epochs, history['train_MAE_pore_mean'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_MAE_pore_mean'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('MAE (Pore Region)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Learning Rate
    ax = axes[1, 2]
    ax.plot(epochs, history['lr'], 'orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('Training Curves (Overall)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {fig_path}")

    # =========================================================================
    # Figure 2: Per rock_type RelFluxErr curves
    # =========================================================================
    if not history_by_type:
        return

    # Collect data per rock_type
    rock_types = sorted(set(r['rock_type'] for r in history_by_type))
    if len(rock_types) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, max(6, len(rock_types))))

    # Plot per-type RelFluxErr P90
    ax = axes[0]
    for i, rt in enumerate(rock_types):
        rt_data = [r for r in history_by_type if r['rock_type'] == rt]
        rt_epochs = [r['epoch'] for r in rt_data]
        rt_p90 = [r['RelFluxErr_p90'] for r in rt_data]
        ax.plot(rt_epochs, rt_p90, color=colors[i], label=f'rock_{rt}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RelFluxErr P90')
    ax.set_title('Per Rock Type: RelFluxErr P90')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot per-type RelFluxErr median
    ax = axes[1]
    for i, rt in enumerate(rock_types):
        rt_data = [r for r in history_by_type if r['rock_type'] == rt]
        rt_epochs = [r['epoch'] for r in rt_data]
        rt_median = [r['RelFluxErr_median'] for r in rt_data]
        ax.plot(rt_epochs, rt_median, color=colors[i], label=f'rock_{rt}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RelFluxErr Median')
    ax.set_title('Per Rock Type: RelFluxErr Median')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot per-type RelFluxErr mean
    ax = axes[2]
    for i, rt in enumerate(rock_types):
        rt_data = [r for r in history_by_type if r['rock_type'] == rt]
        rt_epochs = [r['epoch'] for r in rt_data]
        rt_mean = [r['RelFluxErr_mean'] for r in rt_data]
        ax.plot(rt_epochs, rt_mean, color=colors[i], label=f'rock_{rt}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RelFluxErr Mean')
    ax.set_title('Per Rock Type: RelFluxErr Mean')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot per-type R²
    ax = axes[3]
    for i, rt in enumerate(rock_types):
        rt_data = [r for r in history_by_type if r['rock_type'] == rt]
        rt_epochs = [r['epoch'] for r in rt_data]
        rt_r2 = [r['q_R2'] for r in rt_data]
        ax.plot(rt_epochs, rt_r2, color=colors[i], label=f'rock_{rt}', linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² (Flux)')
    ax.set_title('Per Rock Type: Flux R²')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.05])

    # Final epoch bar chart: RelFluxErr P90 by rock_type
    ax = axes[4]
    final_epoch = max(r['epoch'] for r in history_by_type)
    final_data = [r for r in history_by_type if r['epoch'] == final_epoch]
    final_data = sorted(final_data, key=lambda x: x['rock_type'])
    x_pos = np.arange(len(final_data))
    p90_vals = [r['RelFluxErr_p90'] for r in final_data]
    bars = ax.bar(x_pos, p90_vals, color=[colors[r['rock_type']] for r in final_data])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"rock_{r['rock_type']}" for r in final_data])
    ax.set_ylabel('RelFluxErr P90')
    ax.set_title(f'Final Epoch ({final_epoch}): RelFluxErr P90 by Rock Type')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, p90_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Final epoch bar chart: R² by rock_type
    ax = axes[5]
    r2_vals = [r['q_R2'] for r in final_data]
    bars = ax.bar(x_pos, r2_vals, color=[colors[r['rock_type']] for r in final_data])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"rock_{r['rock_type']}" for r in final_data])
    ax.set_ylabel('R² (Flux)')
    ax.set_title(f'Final Epoch ({final_epoch}): Flux R² by Rock Type')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Training Curves (Per Rock Type)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(out_dir, 'training_curves_by_rock_type.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per rock_type curves saved to: {fig_path}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train U-Net for porous media flow prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--h5', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for checkpoints and logs')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--physics_warmup_epochs', type=int, default=30,
                        help='Core-only epochs before ramping auxiliary losses (wall/mono/grad/peak/etc.)')

    # Split ratios
    parser.add_argument('--split_train', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--split_val', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--split_test', type=float, default=0.1, help='Test split ratio')

    # Loss weights
    parser.add_argument('--lambda_field', type=float, default=1.0, help='Weight for field loss')
    parser.add_argument('--lambda_solid', type=float, default=0.1, help='Weight for solid penalty')
    parser.add_argument('--lambda_q', type=float, default=10.0, help='Weight for flux loss (increased for better R²)')
    parser.add_argument('--lambda_neg', type=float, default=1.0, help='Weight for negative velocity penalty')
    parser.add_argument('--lambda_wall', type=float, default=10.0, help='Weight for wall (near-boundary) loss')
    parser.add_argument('--lambda_mono', type=float, default=2.0, help='Weight for monotonic (center-faster) loss')
    parser.add_argument('--lambda_dist_vel', type=float, default=2.0, help='Weight for dist-velocity positive correlation loss')
    parser.add_argument('--lambda_underpred', type=float, default=5.0, help='Weight for asymmetric under-prediction penalty')
    parser.add_argument('--underpred_alpha', type=float, default=3.0, help='Asymmetric ratio: under-prediction gets alpha times more penalty')
    parser.add_argument('--lambda_ratio', type=float, default=5.0, help='Weight for log-ratio loss (scale-invariant)')
    parser.add_argument('--lambda_stokes', type=float, default=0.0,
                        help='Weight for interior Stokes/Poiseuille residual loss on predicted velocity fields')
    parser.add_argument('--stokes_dist_thr', type=float, default=0.10,
                        help='Use only pore pixels with dist_norm >= this threshold for Stokes residual')
    parser.add_argument('--stokes_norm_eps', type=float, default=1.0e-6,
                        help='Velocity-scale epsilon for normalized Stokes residual')
    parser.add_argument('--lambda_table_q', type=float, default=0.0,
                        help='Weight for table-q/log-conductance loss from permeability_<local_id>.dat')
    parser.add_argument('--lambda_table_g', type=float, default=0.0,
                        help='Weight for direct table conductance log-ratio loss')
    parser.add_argument('--conductance_table_root', type=str, default=None,
                        help='Root containing rock folders with permeability_<local_id>.dat files')
    parser.add_argument('--rock_folder_offset', type=int, default=1,
                        help='Folder offset mapping HDF5 rock_type to table folders; current data uses rock_type+1')
    parser.add_argument('--table_normalization_exponent', type=float, default=2.0,
                        help='Velocity normalization exponent used to convert table q to training q units')
    parser.add_argument('--table_area_scale_exponent', type=int, default=2,
                        help='Area scale exponent used to convert table q to training q units')
    parser.add_argument('--table_g_rho', type=float, default=1.0,
                        help='LBM density used when converting predicted q to conductance labels')
    parser.add_argument('--table_g_ax', type=float, default=1.0e-4,
                        help='LBM body acceleration used when converting predicted q to conductance labels')
    parser.add_argument('--table_g_segment_length', type=float, default=4.0,
                        help='Local segment length used when converting predicted q to conductance labels')
    parser.add_argument('--table_g_mu_lbm', type=float, default=0.5,
                        help='LBM viscosity used when converting predicted q to conductance labels')
    parser.add_argument('--table_g_target_mu', type=float, default=1.0,
                        help='Target viscosity used when converting predicted q to conductance labels')

    # Samplers
    parser.add_argument('--balanced_sampler', action='store_true', help='Use flux-balanced sampler')
    parser.add_argument('--type_flux_balanced_sampler', dest='type_flux_balanced_sampler', action='store_true',
                        help='Use rock_type + flux balanced sampler (default)')
    parser.add_argument('--no_type_flux_balanced_sampler', dest='type_flux_balanced_sampler', action='store_false',
                        help='Disable type+flux balanced sampler (use shuffle if no sampler specified)')
    parser.set_defaults(type_flux_balanced_sampler=True)
    parser.add_argument('--bin_count', type=int, default=8, help='Number of q_true bins')

    # Model options
    parser.add_argument('--use_softplus', action='store_true', help='Use softplus activation for non-negative output')
    parser.add_argument('--use_rock_type_channel', action='store_true', help='Append rock_type map as extra input channel')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channel count for U-Net')
    parser.add_argument('--use_scale_head', dest='use_scale_head', action='store_true',
                        help='Use flux scale head to calibrate magnitude (default)')
    parser.add_argument('--no_scale_head', dest='use_scale_head', action='store_false',
                        help='Disable flux scale head')
    parser.set_defaults(use_scale_head=True)

    # FiLM conditioning (rock_type)
    parser.add_argument('--use_film', dest='use_film', action='store_true',
                        help='Enable FiLM conditioning by rock_type (default)')
    parser.add_argument('--no_film', dest='use_film', action='store_false',
                        help='Disable FiLM conditioning')
    parser.set_defaults(use_film=True)
    parser.add_argument('--film_dim', type=int, default=16, help='FiLM embedding dimension')
    parser.add_argument('--num_rock_types', type=int, default=6, help='Number of rock_type categories')

    # Flux loss options
    parser.add_argument('--q_mix', dest='q_mix', action='store_true', help='Enable mixed absolute/relative flux loss')
    parser.add_argument('--no_q_mix', dest='q_mix', action='store_false', help='Disable mixed flux loss (relative only)')
    parser.set_defaults(q_mix=True)
    parser.add_argument('--q_thresh', type=float, default=None, help='Threshold for mixed flux loss (default: P10 of |q_true|)')
    parser.add_argument('--q_weight_power', type=float, default=1.0, help='Power for high-q weighting in flux loss')
    parser.add_argument('--q_weight_clip_min', type=float, default=0.5, help='Min clamp for flux weight')
    parser.add_argument('--q_weight_clip_max', type=float, default=5.0, help='Max clamp for flux weight')
    parser.add_argument('--flux_weighted', action='store_true', help='Enable q_true-weighted flux loss (alpha/wmin/wmax)')
    parser.add_argument('--flux_w_alpha', type=float, default=0.7, help='Alpha for flux weighting')
    parser.add_argument('--flux_w_min', type=float, default=0.5, help='Min clamp for flux weight (weighted mode)')
    parser.add_argument('--flux_w_max', type=float, default=3.0, help='Max clamp for flux weight (weighted mode)')
    parser.add_argument('--wall_dist_thr', type=float, default=0.15, help='Distance threshold for wall loss')
    parser.add_argument('--mono_bins', type=int, default=6, help='Number of distance bins for monotonic loss')
    parser.add_argument('--mono_dist_max', type=float, default=1.0, help='Max dist_norm for monotonic loss')
    parser.add_argument('--lambda_grad', type=float, default=0.05, help='Weight for gradient loss')
    parser.add_argument('--lambda_peak', type=float, default=0.1, help='Weight for peak-region loss')
    parser.add_argument('--peak_percentile', type=float, default=0.9, help='Percentile for peak-region mask (0-1)')

    # Resume / finetune
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--finetune_rock_type', type=int, default=None,
                        help='Finetune only on a specific rock_type (0..num_rock_types-1)')

    # Best checkpoint metric
    parser.add_argument('--best_metric', type=str, default='p90', choices=['p90', 'median'], help='Metric for best checkpoint')
    parser.add_argument('--best_ignore_low_q', dest='best_ignore_low_q', action='store_true',
                        help='Ignore low |q_true| when selecting best checkpoint (default)')
    parser.add_argument('--best_include_low_q', dest='best_ignore_low_q', action='store_false',
                        help='Include low |q_true| when selecting best checkpoint')
    parser.set_defaults(best_ignore_low_q=True)

    # System
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers (0 for HDF5 compatibility)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--debug_nan', action='store_true', help='Print debug info when non-finite loss occurs')
    parser.add_argument('--debug_nan_max', type=int, default=3, help='Max number of NaN debug prints')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Clip grad norm (<=0 to disable)')

    args = parser.parse_args()
    args.default_lr = parser.get_default('lr')

    train(args)


if __name__ == '__main__':
    main()
