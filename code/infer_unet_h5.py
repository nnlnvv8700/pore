
# -*- coding: utf-8 -*-
R"""
Infer U-Net for porous media flow prediction (multi-rock HDF5).

This script loads a trained checkpoint and performs inference + evaluation.
Outputs:
  - report.csv (overall metrics)
  - report_by_rock_type.csv
  - worst_cases.csv (Top-K by rel_err)
  - predictions.npz or predictions.h5 (metrics + optional pred_ux)
  - inference_results.png (scatter/boxplots)
  - sample_predictions.png (best/median/worst per rock_type)

Example commands (Windows):
  Stats only (no full field save)
  python infer_unet_h5.py --h5 "E:\mhw\1\pore\dataset_all_32.h5" --ckpt "E:\mhw\1\pore\runs\unet_all32_v2\best.pt" --out_dir "E:\mhw\1\pore\runs\unet_all32_v2\infer" --save_field 0 --use_softplus --use_rock_type_channel --use_film

  Save full fields (npz, float16)
  python infer_unet_h5.py --h5 "E:\mhw\1\pore\dataset_all_32.h5" --ckpt "E:\mhw\1\pore\runs\unet_all32_v2\best.pt" --out_dir "E:\mhw\1\pore\runs\unet_all32_v2\infer" --save_field 1 --pred_format npz --field_dtype float16 --use_softplus --use_rock_type_channel --use_film
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


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


def discover_meta_fields(h5_path: str, n_samples: int) -> List[str]:
    """Discover optional 1D meta fields (excluding X/Y/rock_type/global_id)."""
    exclude = {'X', 'Y', 'rock_type', 'global_id'}
    fields = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if key in exclude:
                continue
            ds = f[key]
            if not hasattr(ds, 'shape'):
                continue
            if len(ds.shape) == 1 and ds.shape[0] == n_samples:
                if ds.dtype.kind in ('i', 'u', 'f'):
                    fields.append(key)
    return fields


def load_h5_field_for_indices(h5_path: str, field: str, indices: np.ndarray) -> np.ndarray:
    """Load a 1D field for given indices, preserving order."""
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    out = np.empty(len(indices), dtype=np.float64)
    with h5py.File(h5_path, 'r') as f:
        data = f[field]
        for start in range(0, len(sorted_idx), 4096):
            end = min(start + 4096, len(sorted_idx))
            chunk = sorted_idx[start:end]
            out[start:end] = data[chunk]
    restored = np.empty_like(out)
    restored[order] = out
    return restored


# ============================================================================
# Dataset
# ============================================================================
class PorousH5InferenceDataset(Dataset):
    """Lazy HDF5 dataset for inference."""

    def __init__(self, h5_path: str, mask_idx: int, use_rock_type_channel: bool = False):
        self.h5_path = h5_path
        self.mask_idx = mask_idx
        self.use_rock_type_channel = use_rock_type_channel
        self._h5_file = None

        with h5py.File(h5_path, 'r') as f:
            if 'X' not in f or 'Y' not in f:
                raise ValueError("HDF5 must contain /X and /Y datasets.")
            if 'rock_type' not in f or 'global_id' not in f:
                raise ValueError("HDF5 must contain /rock_type and /global_id datasets.")
            self.n_samples = f['X'].shape[0]
            self.base_in_channels = f['X'].shape[1]
            self.in_channels = self.base_in_channels + (1 if use_rock_type_channel else 0)
            self.patch_size = f['X'].shape[2]

    def _open_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._open_h5()
        x = self._h5_file['X'][idx]
        y = self._h5_file['Y'][idx]

        if self.mask_idx < 0 or self.mask_idx >= x.shape[0]:
            raise ValueError(f"mask_idx {self.mask_idx} out of bounds for X with {x.shape[0]} channels.")
        mask = x[self.mask_idx:self.mask_idx + 1].copy()

        rock_type = int(self._h5_file['rock_type'][idx])
        global_id = int(self._h5_file['global_id'][idx])

        if self.use_rock_type_channel:
            rock_val = rock_type / 5.0
            rock_map = np.full((1, x.shape[1], x.shape[2]), rock_val, dtype=np.float32)
            x = np.concatenate([x, rock_map], axis=0)

        return {
            'x': torch.from_numpy(x.astype(np.float32)),
            'y': torch.from_numpy(y.astype(np.float32)),
            'mask': torch.from_numpy(mask.astype(np.float32)),
            'rock_type': rock_type,
            'global_id': global_id,
            'idx': idx
        }

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()

# ============================================================================
# Model (same as training)
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

        if self.use_softplus and mask is not None:
            out = out * mask
        return out


def load_checkpoint_model(
    ckpt_path: str,
    in_channels: int,
    use_softplus: bool,
    base_channels: int,
    device: torch.device,
    use_rock_type_channel: bool,
    use_scale_head: bool,
    use_film: Optional[bool],
    film_dim: Optional[int],
    num_rock_types: Optional[int]
) -> Tuple[nn.Module, Dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt_args, dict) and base_channels is None:
        try:
            base_channels = int(ckpt_args.get('base_channels', 32))
        except Exception:
            base_channels = 32
    if base_channels is None:
        base_channels = 32

    ckpt_softplus = use_softplus
    ckpt_rock_ch = use_rock_type_channel
    ckpt_scale = use_scale_head
    ckpt_use_film = False
    ckpt_film_dim = film_dim if film_dim is not None else 16
    ckpt_num_rocks = num_rock_types if num_rock_types is not None else 6
    if isinstance(ckpt_args, dict):
        ckpt_softplus = bool(ckpt_args.get('use_softplus', use_softplus))
        ckpt_rock_ch = bool(ckpt_args.get('use_rock_type_channel', use_rock_type_channel))
        ckpt_scale = bool(ckpt_args.get('use_scale_head', use_scale_head))
        ckpt_use_film = bool(ckpt_args.get('use_film', False))
        if film_dim is None:
            try:
                ckpt_film_dim = int(ckpt_args.get('film_dim', ckpt_film_dim))
            except Exception:
                pass
        if num_rock_types is None:
            try:
                ckpt_num_rocks = int(ckpt_args.get('num_rock_types', ckpt_num_rocks))
            except Exception:
                pass
        if use_film is None:
            use_film = ckpt_use_film
        if film_dim is None:
            film_dim = ckpt_film_dim
        if num_rock_types is None:
            num_rock_types = ckpt_num_rocks

    if use_film is None:
        use_film = ckpt_use_film

    if ckpt_softplus != use_softplus:
        print(f"Warning: checkpoint use_softplus={ckpt_softplus} but inference uses {use_softplus}.")
    if ckpt_rock_ch != use_rock_type_channel:
        print(f"Warning: checkpoint use_rock_type_channel={ckpt_rock_ch} but inference uses {use_rock_type_channel}.")
    if ckpt_scale != use_scale_head:
        print(f"Warning: checkpoint use_scale_head={ckpt_scale} but inference uses {use_scale_head}.")
    if ckpt_use_film != bool(use_film):
        print(f"Warning: checkpoint use_film={ckpt_use_film} but inference uses {bool(use_film)}.")

    model = LightUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
        use_softplus=use_softplus,
        use_scale_head=use_scale_head,
        use_film=bool(use_film),
        film_dim=int(film_dim) if film_dim is not None else ckpt_film_dim,
        num_rock_types=int(num_rock_types) if num_rock_types is not None else ckpt_num_rocks
    )
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {len(unexpected)}")
    return model, ckpt_args

# ============================================================================
# Metrics helpers
# ============================================================================
def compute_r2(q_pred: np.ndarray, q_true: np.ndarray) -> float:
    if q_pred.size == 0:
        return float('nan')
    ss_res = np.sum((q_pred - q_true) ** 2)
    ss_tot = np.sum((q_true - np.mean(q_true)) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot


def stats_basic(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'p90': float('nan'), 'p95': float('nan'), 'max': float('nan')}
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr))
    }


def stats_three(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'p90': float('nan')}
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p90': float(np.percentile(arr, 90))
    }


def compute_divergence(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Compute divergence of 2D vector field (ux, uy)."""
    dux_dy, dux_dx = np.gradient(ux)
    duy_dy, duy_dx = np.gradient(uy)
    return dux_dx + duy_dy


def compute_vector_metrics(
    ux_t: np.ndarray,
    uy_t: np.ndarray,
    ux_p: np.ndarray,
    uy_p: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8
) -> Dict[str, float]:
    pore = mask > 0.5
    solid = ~pore

    if np.any(pore):
        ux_err = ux_p - ux_t
        uy_err = uy_p - uy_t
        speed_t = np.sqrt(ux_t ** 2 + uy_t ** 2)
        speed_p = np.sqrt(ux_p ** 2 + uy_p ** 2)
        speed_err = speed_p - speed_t

        mae_ux = float(np.mean(np.abs(ux_err[pore])))
        rmse_ux = float(np.sqrt(np.mean((ux_err[pore]) ** 2)))
        mae_uy = float(np.mean(np.abs(uy_err[pore])))
        rmse_uy = float(np.sqrt(np.mean((uy_err[pore]) ** 2)))
        mae_speed = float(np.mean(np.abs(speed_err[pore])))
        rmse_speed = float(np.sqrt(np.mean((speed_err[pore]) ** 2)))

        epe = float(np.mean(np.sqrt(ux_err[pore] ** 2 + uy_err[pore] ** 2)))
        dot = (ux_p * ux_t + uy_p * uy_t)
        norm_p = np.sqrt(ux_p ** 2 + uy_p ** 2)
        norm_t = np.sqrt(ux_t ** 2 + uy_t ** 2)
        cos = float(np.mean(dot[pore] / (norm_p[pore] * norm_t[pore] + eps)))

        div = compute_divergence(ux_p, uy_p)
        div_mean = float(np.mean(np.abs(div[pore])))
    else:
        mae_ux = rmse_ux = mae_uy = rmse_uy = mae_speed = rmse_speed = float('nan')
        epe = cos = div_mean = float('nan')

    if np.any(solid):
        speed_p = np.sqrt(ux_p ** 2 + uy_p ** 2)
        solid_speed_mean = float(np.mean(speed_p[solid]))
    else:
        solid_speed_mean = float('nan')

    return {
        'mae_ux': mae_ux,
        'rmse_ux': rmse_ux,
        'mae_uy': mae_uy,
        'rmse_uy': rmse_uy,
        'mae_speed': mae_speed,
        'rmse_speed': rmse_speed,
        'epe': epe,
        'cosine': cos,
        'div_mean': div_mean,
        'solid_speed_mean': solid_speed_mean
    }


def save_visualization(
    out_path: str,
    ux_t: np.ndarray,
    uy_t: np.ndarray,
    ux_p: np.ndarray,
    uy_p: np.ndarray,
    mask: np.ndarray
) -> None:
    """Save combined visualization figure for a single sample."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    pore = mask > 0.5
    ux_t_vis = np.where(pore, ux_t, np.nan)
    uy_t_vis = np.where(pore, uy_t, np.nan)
    ux_p_vis = np.where(pore, ux_p, np.nan)
    uy_p_vis = np.where(pore, uy_p, np.nan)

    speed_t = np.sqrt(ux_t ** 2 + uy_t ** 2)
    speed_p = np.sqrt(ux_p ** 2 + uy_p ** 2)
    speed_t_vis = np.where(pore, speed_t, np.nan)
    speed_p_vis = np.where(pore, speed_p, np.nan)

    speed_err = np.abs(speed_p - speed_t)
    ux_err = np.abs(ux_p - ux_t)
    uy_err = np.abs(uy_p - uy_t)
    speed_err_vis = np.where(pore, speed_err, np.nan)
    ux_err_vis = np.where(pore, ux_err, np.nan)
    uy_err_vis = np.where(pore, uy_err, np.nan)

    fig, axes = plt.subplots(5, 3, figsize=(12, 16))

    # Row 0: |u|
    vmax = np.nanmax(np.stack([speed_t_vis, speed_p_vis])) if np.any(pore) else 1.0
    axes[0, 0].imshow(speed_t_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[0, 0].set_title('|u| GT')
    axes[0, 1].imshow(speed_p_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[0, 1].set_title('|u| Pred')
    axes[0, 2].imshow(speed_err_vis, cmap='magma', interpolation='nearest')
    axes[0, 2].set_title('|u| AbsErr')

    # Row 1: ux
    vmax = np.nanmax(np.stack([ux_t_vis, ux_p_vis])) if np.any(pore) else 1.0
    axes[1, 0].imshow(ux_t_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[1, 0].set_title('ux GT')
    axes[1, 1].imshow(ux_p_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[1, 1].set_title('ux Pred')
    axes[1, 2].imshow(ux_err_vis, cmap='magma', interpolation='nearest')
    axes[1, 2].set_title('ux AbsErr')

    # Row 2: uy
    vmax = np.nanmax(np.stack([uy_t_vis, uy_p_vis])) if np.any(pore) else 1.0
    axes[2, 0].imshow(uy_t_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[2, 0].set_title('uy GT')
    axes[2, 1].imshow(uy_p_vis, cmap='turbo', vmin=0, vmax=vmax, interpolation='nearest')
    axes[2, 1].set_title('uy Pred')
    axes[2, 2].imshow(uy_err_vis, cmap='magma', interpolation='nearest')
    axes[2, 2].set_title('uy AbsErr')

    # Row 3: quiver GT vs Pred
    H, W = ux_t.shape
    step = max(1, H // 16)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    axes[3, 0].quiver(xx, yy, ux_t[::step, ::step], uy_t[::step, ::step], color='k', scale=None)
    axes[3, 0].set_title('Quiver GT')
    axes[3, 1].quiver(xx, yy, ux_p[::step, ::step], uy_p[::step, ::step], color='k', scale=None)
    axes[3, 1].set_title('Quiver Pred')
    axes[3, 2].axis('off')

    # Row 4: streamplot GT vs Pred
    axes[4, 0].streamplot(np.arange(W), np.arange(H), ux_t, uy_t, density=1.0, color='k')
    axes[4, 0].set_title('Stream GT')
    axes[4, 1].streamplot(np.arange(W), np.arange(H), ux_p, uy_p, density=1.0, color='k')
    axes[4, 1].set_title('Stream Pred')
    axes[4, 2].axis('off')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def pick_best_median_worst(indices: np.ndarray, rel_err: np.ndarray) -> Dict[str, int]:
    if indices.size == 0:
        return {}
    order = np.argsort(rel_err)
    best_idx = indices[order[0]]
    worst_idx = indices[order[-1]]
    median_val = np.median(rel_err)
    median_idx = indices[np.argmin(np.abs(rel_err - median_val))]
    return {'best': int(best_idx), 'median': int(median_idx), 'worst': int(worst_idx)}


def safe_device(device_str: Optional[str]) -> torch.device:
    if device_str is None or device_str.lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)

# ============================================================================
# Visualization
# ============================================================================
def plot_inference_results(
    out_dir: str,
    q_true: np.ndarray,
    q_pred: np.ndarray,
    rel_err: np.ndarray,
    abs_err: np.ndarray,
    rmse_pore: np.ndarray,
    pore_frac: np.ndarray,
    rock_type: np.ndarray,
    r2_val: float
):
    """Generate comprehensive inference result plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    # Color palette for rock types
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rock_types_sorted = sorted(np.unique(rock_type).tolist())

    # Downsample for scatter plots
    max_points = 15000
    if q_true.size > max_points:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(q_true.size, max_points, replace=False)
    else:
        sample_idx = np.arange(q_true.size)

    # =========================================================================
    # Figure 1: Main inference results (3x3)
    # =========================================================================
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # (1,1) q_pred vs q_true scatter with identity line
    ax = fig.add_subplot(gs[0, 0])
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type[sample_idx] == rt
        ax.scatter(q_true[sample_idx][mask_rt], q_pred[sample_idx][mask_rt],
                   s=8, alpha=0.5, c=[colors[rt]], label=f'rock_{rt}')
    min_v = min(np.min(q_true), np.min(q_pred))
    max_v = max(np.max(q_true), np.max(q_pred))
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label='y=x')
    ax.set_xlabel('q_true', fontsize=11)
    ax.set_ylabel('q_pred', fontsize=11)
    ax.set_title(f'Flux Prediction (R² = {r2_val:.4f})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Residual (q_pred - q_true) vs q_true
    ax = fig.add_subplot(gs[0, 1])
    residual = q_pred - q_true
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type[sample_idx] == rt
        ax.scatter(q_true[sample_idx][mask_rt], residual[sample_idx][mask_rt],
                   s=8, alpha=0.5, c=[colors[rt]], label=f'rock_{rt}')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('q_true', fontsize=11)
    ax.set_ylabel('q_pred - q_true', fontsize=11)
    ax.set_title('Residual vs q_true', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (1,3) rel_err vs q_true (log scale y)
    ax = fig.add_subplot(gs[0, 2])
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type[sample_idx] == rt
        ax.scatter(q_true[sample_idx][mask_rt], rel_err[sample_idx][mask_rt],
                   s=8, alpha=0.5, c=[colors[rt]], label=f'rock_{rt}')
    ax.set_xlabel('q_true', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Relative Error vs q_true', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (2,1) Boxplot: rel_err by rock_type
    ax = fig.add_subplot(gs[1, 0])
    data = [rel_err[rock_type == rt] for rt in rock_types_sorted]
    bp = ax.boxplot(data, labels=[f'rock_{rt}' for rt in rock_types_sorted],
                    showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[rock_types_sorted[i]])
        patch.set_alpha(0.7)
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('RelErr Distribution by Rock Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (2,2) Violin plot: rel_err by rock_type (capped)
    ax = fig.add_subplot(gs[1, 1])
    rel_err_capped = np.clip(rel_err, 0, np.percentile(rel_err, 99))
    data_capped = [rel_err_capped[rock_type == rt] for rt in rock_types_sorted]
    parts = ax.violinplot(data_capped, positions=range(len(rock_types_sorted)), showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[rock_types_sorted[i]])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(rock_types_sorted)))
    ax.set_xticklabels([f'rock_{rt}' for rt in rock_types_sorted])
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('Relative Error (capped at P99)', fontsize=11)
    ax.set_title('RelErr Violin by Rock Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (2,3) Bar chart: metrics by rock_type
    ax = fig.add_subplot(gs[1, 2])
    x_pos = np.arange(len(rock_types_sorted))
    width = 0.35
    means = [np.mean(rel_err[rock_type == rt]) for rt in rock_types_sorted]
    medians = [np.median(rel_err[rock_type == rt]) for rt in rock_types_sorted]
    p90s = [np.percentile(rel_err[rock_type == rt], 90) for rt in rock_types_sorted]
    ax.bar(x_pos - width/2, medians, width, label='Median', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, p90s, width, label='P90', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'rock_{rt}' for rt in rock_types_sorted])
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('RelErr Median & P90 by Rock Type', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (3,1) R² by rock_type
    ax = fig.add_subplot(gs[2, 0])
    r2_by_type = []
    for rt in rock_types_sorted:
        mask_rt = rock_type == rt
        r2_rt = compute_r2(q_pred[mask_rt], q_true[mask_rt])
        r2_by_type.append(r2_rt)
    bars = ax.bar(x_pos, r2_by_type, color=[colors[rt] for rt in rock_types_sorted], alpha=0.8)
    ax.axhline(r2_val, color='k', linestyle='--', linewidth=1.5, label=f'Overall R²={r2_val:.4f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'rock_{rt}' for rt in rock_types_sorted])
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Flux R² by Rock Type', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2_by_type):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # (3,2) Histogram of rel_err (log scale x)
    ax = fig.add_subplot(gs[2, 1])
    for i, rt in enumerate(rock_types_sorted):
        data_rt = rel_err[rock_type == rt]
        ax.hist(data_rt, bins=50, alpha=0.5, label=f'rock_{rt}', color=colors[rt],
                range=(0, np.percentile(rel_err, 98)))
    ax.set_xlabel('Relative Error', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('RelErr Histogram by Rock Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3,3) rel_err vs pore_frac
    ax = fig.add_subplot(gs[2, 2])
    for i, rt in enumerate(rock_types_sorted):
        mask_rt = rock_type[sample_idx] == rt
        ax.scatter(pore_frac[sample_idx][mask_rt], rel_err[sample_idx][mask_rt],
                   s=8, alpha=0.5, c=[colors[rt]], label=f'rock_{rt}')
    ax.set_xlabel('Pore Fraction', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('RelErr vs Pore Fraction', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Inference Results Summary', fontsize=16, fontweight='bold', y=0.98)
    fig_path = os.path.join(out_dir, 'inference_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {fig_path}")

    # =========================================================================
    # Figure 2: Additional analysis (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (1,1) CDF of rel_err by rock_type
    ax = axes[0, 0]
    for i, rt in enumerate(rock_types_sorted):
        data_rt = np.sort(rel_err[rock_type == rt])
        cdf = np.arange(1, len(data_rt) + 1) / len(data_rt)
        ax.plot(data_rt, cdf, color=colors[rt], linewidth=2, label=f'rock_{rt}')
    ax.axvline(np.median(rel_err), color='k', linestyle='--', alpha=0.7, label=f'Overall Median={np.median(rel_err):.4f}')
    ax.set_xlabel('Relative Error', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Cumulative Distribution of RelErr', fontsize=12, fontweight='bold')
    ax.set_xlim([0, np.percentile(rel_err, 98)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Sample counts by rock_type
    ax = axes[0, 1]
    counts = [np.sum(rock_type == rt) for rt in rock_types_sorted]
    bars = ax.bar(x_pos, counts, color=[colors[rt] for rt in rock_types_sorted], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'rock_{rt}' for rt in rock_types_sorted])
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_title('Sample Distribution by Rock Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    # (2,1) RMSE_pore boxplot by rock_type
    ax = axes[1, 0]
    data_rmse = [rmse_pore[rock_type == rt] for rt in rock_types_sorted]
    bp = ax.boxplot(data_rmse, labels=[f'rock_{rt}' for rt in rock_types_sorted],
                    showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[rock_types_sorted[i]])
        patch.set_alpha(0.7)
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('RMSE (Pore Region)', fontsize=11)
    ax.set_title('RMSE Distribution by Rock Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (2,2) q_true distribution by rock_type
    ax = axes[1, 1]
    bp = ax.boxplot([q_true[rock_type == rt] for rt in rock_types_sorted],
                    labels=[f'rock_{rt}' for rt in rock_types_sorted],
                    showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[rock_types_sorted[i]])
        patch.set_alpha(0.7)
    ax.set_xlabel('Rock Type', fontsize=11)
    ax.set_ylabel('q_true', fontsize=11)
    ax.set_title('q_true Distribution by Rock Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Additional Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'inference_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {fig_path}")


def plot_sample_grid(out_path: str, samples: List[Dict]):
    """Plot sample predictions: True | Predicted | Error (Pred - True), each with colorbar."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        print("Warning: matplotlib not available, skipping sample grid")
        return

    if not samples:
        print("No samples for visualization.")
        return

    n_rows = len(samples)
    n_cols = 3  # True, Predicted, Error

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(samples):
        y = s['y']
        pred = s['pred']
        err = pred - y

        # Use same vmax for True and Predicted for fair comparison
        vmax_field = max(float(np.max(y)), float(np.max(pred)), 1e-12)
        emax = max(float(np.max(np.abs(err))), 1e-12)

        # True (turbo, shared scale)
        ax = axes[i, 0]
        im = ax.imshow(y, cmap='turbo', vmin=0.0, vmax=vmax_field, interpolation='nearest')
        ax.set_title('True', fontsize=12, fontweight='bold')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, format='%.4f')
        cb.ax.tick_params(labelsize=8)

        # Predicted (turbo, shared scale)
        ax = axes[i, 1]
        im = ax.imshow(pred, cmap='turbo', vmin=0.0, vmax=vmax_field, interpolation='nearest')
        ax.set_title('Predicted', fontsize=12, fontweight='bold')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, format='%.4f')
        cb.ax.tick_params(labelsize=8)

        # Error (Pred - True) - use coolwarm with symmetric scale
        ax = axes[i, 2]
        im = ax.imshow(err, cmap='coolwarm', vmin=-emax, vmax=emax, interpolation='nearest')
        ax.set_title('Error (Pred - True)', fontsize=12, fontweight='bold')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, format='%.4f')
        cb.ax.tick_params(labelsize=8)

        # Row label on the left
        label = (
            f"rock_{s['rock_type']} [{s['kind']}]\n"
            f"gid={s['global_id']} idx={s['idx']}\n"
            f"qT={s['q_true']:.4f}\n"
            f"qP={s['q_pred']:.4f}\n"
            f"rel={s['rel_err']:.4f}"
        )
        axes[i, 0].text(-0.25, 0.5, label, transform=axes[i, 0].transAxes,
                        fontsize=9, va='center', ha='right',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, wspace=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved sample grid: {out_path}")


# ============================================================================
# Main inference
# ============================================================================
def run_inference(args):
    os.makedirs(args.out_dir, exist_ok=True)
    save_root = args.save_dir if args.save_dir else args.out_dir
    os.makedirs(save_root, exist_ok=True)
    vis_dir = os.path.join(save_root, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    channel_order, mask_idx = read_channel_order(args.h5)
    if mask_idx is None:
        print("Warning: channel_order not found or missing 'mask'. Using mask_idx=0.")
        mask_idx = 0
    else:
        print(f"Channel order: {channel_order}")
        print(f"Using mask_idx={mask_idx}")

    # Device
    device = safe_device(args.device)
    print(f"Using device: {device}")

    # Dataset
    dataset = PorousH5InferenceDataset(args.h5, mask_idx, use_rock_type_channel=args.use_rock_type_channel)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model, _ = load_checkpoint_model(
        ckpt_path=args.ckpt,
        in_channels=dataset.in_channels,
        use_softplus=args.use_softplus,
        base_channels=args.base_channels,
        device=device,
        use_rock_type_channel=args.use_rock_type_channel,
        use_scale_head=args.use_scale_head,
        use_film=args.use_film,
        film_dim=args.film_dim,
        num_rock_types=args.num_rock_types
    )
    model = model.to(device)
    model.eval()

    # Output arrays
    n = len(dataset)
    q_true = np.zeros(n, dtype=np.float32)
    q_pred = np.zeros(n, dtype=np.float32)
    rel_err = np.zeros(n, dtype=np.float32)
    abs_err = np.zeros(n, dtype=np.float32)
    rmse_pore = np.zeros(n, dtype=np.float32)
    mae_pore = np.zeros(n, dtype=np.float32)
    pore_frac = np.zeros(n, dtype=np.float32)
    rock_type = np.zeros(n, dtype=np.int16)
    global_id = np.zeros(n, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)

    # Prediction storage
    pred_field = None
    pred_h5 = None
    pred_h5_path = None
    if args.save_field == 1:
        if args.pred_format == 'npz':
            pred_field = np.zeros((n, 1, dataset.patch_size, dataset.patch_size), dtype=np.float16 if args.field_dtype == 'float16' else np.float32)
        else:
            pred_h5_path = os.path.join(args.out_dir, 'predictions.h5')
            pred_h5 = h5py.File(pred_h5_path, 'w')
            pred_h5.create_dataset(
                'pred_ux',
                shape=(n, 1, dataset.patch_size, dataset.patch_size),
                dtype=np.float16 if args.field_dtype == 'float16' else np.float32,
                chunks=(1, 1, dataset.patch_size, dataset.patch_size),
                compression='gzip'
            )

    # Metrics CSV (per-sample)
    metrics_path = os.path.join(save_root, 'metrics.csv')
    metrics_fields = [
        'index', 'rock_type', 'global_id',
        'mae_ux', 'rmse_ux', 'mae_uy', 'rmse_uy',
        'mae_speed', 'rmse_speed', 'epe', 'cosine', 'div_mean',
        'solid_speed_mean'
    ]
    metrics_sum = {k: 0.0 for k in metrics_fields if k not in ('index', 'rock_type', 'global_id')}
    metrics_count = {k: 0 for k in metrics_sum.keys()}
    metrics_f = open(metrics_path, 'w', newline='')
    metrics_f.write(','.join(metrics_fields) + '\n')
    metrics_f.flush()

    # Inference loop
    for batch_idx, batch in enumerate(tqdm(loader, desc='Infer', leave=False)):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        mask = batch['mask'].to(device)
        idx = batch['idx'].cpu().numpy().astype(np.int64)
        rock_type_dev = None
        if getattr(model, "use_film", False):
            rock_type_dev = batch['rock_type'].to(device, dtype=torch.long)

        with torch.no_grad():
            pred = model(x, mask=mask if args.use_softplus else None, rock_type=rock_type_dev)
            if not args.use_softplus:
                pred = pred * mask

        q_p = (pred * mask).sum(dim=(2, 3)).cpu().numpy().flatten()
        q_t = (y * mask).sum(dim=(2, 3)).cpu().numpy().flatten()
        abs_e = np.abs(q_p - q_t)
        rel_e = abs_e / (np.abs(q_t) + args.eps_q)

        diff = (pred - y) * mask
        pore_sum = mask.sum(dim=(1, 2, 3)) + 1e-8
        mse = (diff ** 2).sum(dim=(1, 2, 3)) / pore_sum
        rmse = torch.sqrt(mse).cpu().numpy().flatten()
        mae = (torch.abs(diff).sum(dim=(1, 2, 3)) / pore_sum).cpu().numpy().flatten()
        pore = (mask.mean(dim=(1, 2, 3))).cpu().numpy().flatten()

        q_true[idx] = q_t
        q_pred[idx] = q_p
        abs_err[idx] = abs_e
        rel_err[idx] = rel_e
        rmse_pore[idx] = rmse
        mae_pore[idx] = mae
        pore_frac[idx] = pore
        rock_type[idx] = batch['rock_type'].cpu().numpy().astype(np.int16)
        global_id[idx] = batch['global_id'].cpu().numpy().astype(np.int32)

        # Per-sample metrics (vector field)
        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()[:, 0]
        c_true = y_np.shape[1]
        c_pred = pred_np.shape[1]
        if c_true >= 2:
            if c_pred < 2:
                raise ValueError("Model output has <2 channels but Y has 2 channels (ux, uy).")
            ux_t = y_np[:, 0]
            uy_t = y_np[:, 1]
            ux_p = pred_np[:, 0]
            uy_p = pred_np[:, 1]
        else:
            ux_t = y_np[:, 0]
            uy_t = np.zeros_like(ux_t)
            ux_p = pred_np[:, 0]
            uy_p = np.zeros_like(ux_p)

        for j, sample_idx in enumerate(idx):
            m = compute_vector_metrics(
                ux_t[j], uy_t[j],
                ux_p[j], uy_p[j],
                mask_np[j]
            )
            metrics_f.write(
                f"{int(sample_idx)},{int(rock_type[sample_idx])},{int(global_id[sample_idx])},"
                f"{m['mae_ux']:.6e},{m['rmse_ux']:.6e},{m['mae_uy']:.6e},{m['rmse_uy']:.6e},"
                f"{m['mae_speed']:.6e},{m['rmse_speed']:.6e},{m['epe']:.6e},{m['cosine']:.6e},{m['div_mean']:.6e},"
                f"{m['solid_speed_mean']:.6e}\n"
            )
            for k in metrics_sum.keys():
                v = m.get(k, float('nan'))
                if np.isfinite(v):
                    metrics_sum[k] += float(v)
                    metrics_count[k] += 1

        # Save visualizations
        if args.max_vis > 0 and (batch_idx % args.vis_every == 0):
            vis_n = min(args.max_vis, pred_np.shape[0])
            for j in range(vis_n):
                vis_path = os.path.join(
                    vis_dir,
                    f"batch{batch_idx:05d}_idx{int(idx[j]):06d}.png"
                )
                save_visualization(
                    vis_path,
                    ux_t[j], uy_t[j],
                    ux_p[j], uy_p[j],
                    mask_np[j]
                )

        if args.save_field == 1:
            pred_np_save = pred.cpu().numpy().astype(np.float16 if args.field_dtype == 'float16' else np.float32)
            if args.pred_format == 'npz':
                pred_field[idx] = pred_np_save
            else:
                pred_h5['pred_ux'][idx] = pred_np_save

    if pred_h5 is not None:
        pred_h5.flush()

    metrics_f.close()

    # Metrics summary
    summary = {}
    for k in metrics_sum.keys():
        if metrics_count[k] > 0:
            summary[k] = metrics_sum[k] / metrics_count[k]
        else:
            summary[k] = float('nan')
    summary['n_samples'] = int(n)
    summary_path = os.path.join(save_root, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved metrics summary: {summary_path}")

    # Overall stats
    rel_stats = stats_basic(rel_err)
    abs_stats = stats_basic(abs_err)
    rmse_stats = stats_three(rmse_pore)
    mae_stats = stats_three(mae_pore)
    r2_val = compute_r2(q_pred, q_true)

    q_true_stats = {
        'min': float(np.min(q_true)) if q_true.size > 0 else float('nan'),
        'median': float(np.median(q_true)) if q_true.size > 0 else float('nan'),
        'p10': float(np.percentile(q_true, 10)) if q_true.size > 0 else float('nan'),
        'p90': float(np.percentile(q_true, 90)) if q_true.size > 0 else float('nan')
    }

    # report.csv
    report_path = os.path.join(args.out_dir, 'report.csv')
    with open(report_path, 'w', newline='') as f:
        f.write(
            'N,RelFluxErr_mean,RelFluxErr_median,RelFluxErr_p90,RelFluxErr_p95,RelFluxErr_max,'
            'AbsFluxErr_mean,AbsFluxErr_median,AbsFluxErr_p90,AbsFluxErr_p95,AbsFluxErr_max,'
            'RMSE_pore_mean,RMSE_pore_median,RMSE_pore_p90,'
            'MAE_pore_mean,MAE_pore_median,MAE_pore_p90,'
            'Flux_R2,q_true_min,q_true_median,q_true_p10,q_true_p90\n'
        )
        f.write(
            f"{n},{rel_stats['mean']:.6f},{rel_stats['median']:.6f},{rel_stats['p90']:.6f},{rel_stats['p95']:.6f},{rel_stats['max']:.6f},"
            f"{abs_stats['mean']:.6f},{abs_stats['median']:.6f},{abs_stats['p90']:.6f},{abs_stats['p95']:.6f},{abs_stats['max']:.6f},"
            f"{rmse_stats['mean']:.6f},{rmse_stats['median']:.6f},{rmse_stats['p90']:.6f},"
            f"{mae_stats['mean']:.6f},{mae_stats['median']:.6f},{mae_stats['p90']:.6f},"
            f"{r2_val:.6f},{q_true_stats['min']:.6f},{q_true_stats['median']:.6f},{q_true_stats['p10']:.6f},{q_true_stats['p90']:.6f}\n"
        )
    print(f"Saved report: {report_path}")

    # report_by_rock_type.csv
    report_type_path = os.path.join(args.out_dir, 'report_by_rock_type.csv')
    with open(report_type_path, 'w', newline='') as f:
        f.write(
            'rock_type,N,RelFluxErr_mean,RelFluxErr_median,RelFluxErr_p90,RelFluxErr_p95,'
            'AbsFluxErr_mean,AbsFluxErr_median,AbsFluxErr_p90,AbsFluxErr_p95,'
            'RMSE_pore_mean,MAE_pore_mean,Flux_R2,q_true_median,q_true_p10,q_true_p90\n'
        )
        for rt in sorted(np.unique(rock_type).tolist()):
            mask_rt = rock_type == rt
            rel_s = stats_basic(rel_err[mask_rt])
            abs_s = stats_basic(abs_err[mask_rt])
            q_true_rt = q_true[mask_rt]
            r2_rt = compute_r2(q_pred[mask_rt], q_true_rt)
            f.write(
                f"{rt},{int(mask_rt.sum())},{rel_s['mean']:.6f},{rel_s['median']:.6f},{rel_s['p90']:.6f},{rel_s['p95']:.6f},"
                f"{abs_s['mean']:.6f},{abs_s['median']:.6f},{abs_s['p90']:.6f},{abs_s['p95']:.6f},"
                f"{float(np.mean(rmse_pore[mask_rt])):.6f},{float(np.mean(mae_pore[mask_rt])):.6f},"
                f"{r2_rt:.6f},{float(np.median(q_true_rt)):.6f},{float(np.percentile(q_true_rt, 10)):.6f},{float(np.percentile(q_true_rt, 90)):.6f}\n"
            )
    print(f"Saved report by rock_type: {report_type_path}")

    # worst_cases.csv
    topk = min(args.topk, n)
    order = np.argsort(rel_err)[::-1][:topk]
    meta_fields = discover_meta_fields(args.h5, n)
    meta_data = {}
    for field in meta_fields:
        meta_data[field] = load_h5_field_for_indices(args.h5, field, order)

    worst_path = os.path.join(args.out_dir, 'worst_cases.csv')
    with open(worst_path, 'w', newline='') as f:
        header = [
            'index', 'rock_type', 'global_id', 'q_true', 'q_pred', 'rel_err', 'abs_err',
            'rmse_pore', 'mae_pore', 'pore_frac'
        ] + meta_fields
        f.write(','.join(header) + '\n')
        for i, idx in enumerate(order):
            row = [
                int(idx), int(rock_type[idx]), int(global_id[idx]),
                q_true[idx], q_pred[idx], rel_err[idx], abs_err[idx],
                rmse_pore[idx], mae_pore[idx], pore_frac[idx]
            ]
            row += [meta_data[field][i] for field in meta_fields]
            f.write(','.join([str(v) for v in row]) + '\n')
    print(f"Saved worst cases: {worst_path}")

    # Save predictions (metrics + optional pred_ux)
    pred_path = os.path.join(args.out_dir, f"predictions.{args.pred_format}")
    if args.pred_format == 'npz':
        save_dict = {
            'q_true': q_true,
            'q_pred': q_pred,
            'rel_err': rel_err,
            'abs_err': abs_err,
            'rmse_pore': rmse_pore,
            'mae_pore': mae_pore,
            'pore_frac': pore_frac,
            'rock_type': rock_type,
            'global_id': global_id,
            'index': indices
        }
        if args.save_field == 1:
            save_dict['pred_ux'] = pred_field
        np.savez_compressed(pred_path, **save_dict)
    else:
        if pred_h5 is None:
            pred_h5 = h5py.File(pred_path, 'w')
        pred_h5.create_dataset('q_true', data=q_true)
        pred_h5.create_dataset('q_pred', data=q_pred)
        pred_h5.create_dataset('rel_err', data=rel_err)
        pred_h5.create_dataset('abs_err', data=abs_err)
        pred_h5.create_dataset('rmse_pore', data=rmse_pore)
        pred_h5.create_dataset('mae_pore', data=mae_pore)
        pred_h5.create_dataset('pore_frac', data=pore_frac)
        pred_h5.create_dataset('rock_type', data=rock_type)
        pred_h5.create_dataset('global_id', data=global_id)
        pred_h5.create_dataset('index', data=indices)
        pred_h5.close()
    print(f"Saved predictions: {pred_path}")

    # Plots - comprehensive inference results
    plot_inference_results(
        out_dir=args.out_dir,
        q_true=q_true,
        q_pred=q_pred,
        rel_err=rel_err,
        abs_err=abs_err,
        rmse_pore=rmse_pore,
        pore_frac=pore_frac,
        rock_type=rock_type,
        r2_val=r2_val
    )

    # Sample grid: random 10 samples
    sample_indices = []
    sample_n = 10
    if n > 0:
        rng = np.random.default_rng()
        pick_count = min(sample_n, n)
        picked = rng.choice(n, size=pick_count, replace=False)
        for idx in picked.tolist():
            sample_indices.append(("random", int(rock_type[idx]), int(idx)))

    samples = []
    if sample_indices:
        # Second pass to fetch sample predictions
        model.eval()
        with h5py.File(args.h5, 'r') as f:
            X = f['X']
            Y = f['Y']
            for kind, rt, idx in sample_indices:
                x = X[idx]
                y = Y[idx]
                mask = x[mask_idx]
                if args.use_rock_type_channel:
                    rock_val = rt / 5.0
                    rock_map = np.full((1, x.shape[1], x.shape[2]), rock_val, dtype=np.float32)
                    x_in = np.concatenate([x, rock_map], axis=0)
                else:
                    x_in = x
                x_t = torch.from_numpy(x_in.astype(np.float32)).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                rock_t = None
                if getattr(model, "use_film", False):
                    rock_t = torch.tensor([rt], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(x_t, mask=mask_t if args.use_softplus else None, rock_type=rock_t)
                    if not args.use_softplus:
                        pred = pred * mask_t
                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
                samples.append({
                    'kind': kind,
                    'rock_type': rt,
                    'global_id': int(global_id[idx]),
                    'idx': int(idx),
                    'q_true': float(q_true[idx]),
                    'q_pred': float(q_pred[idx]),
                    'rel_err': float(rel_err[idx]),
                    'rmse_pore': float(rmse_pore[idx]),
                    'pore_frac': float(pore_frac[idx]),
                    'mask': mask,
                    'y': y[0],
                    'pred': pred_np
                })

    plot_sample_grid(os.path.join(args.out_dir, 'sample_predictions.png'), samples)


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Infer U-Net for porous media flow prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--h5', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pt)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory (legacy)')
    parser.add_argument('--save_dir', type=str, default='', help='Directory for new metrics/visualizations (default: out_dir)')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto|cuda|cpu')
    parser.add_argument('--max_vis', type=int, default=0, help='Max visualizations per batch (0 to disable)')
    parser.add_argument('--vis_every', type=int, default=1, help='Visualize every N batches')

    parser.add_argument('--use_softplus', action='store_true', help='Use softplus output (pred = softplus(raw) * mask)')
    parser.add_argument('--use_rock_type_channel', action='store_true', help='Append rock_type map as extra input channel')
    parser.add_argument('--use_scale_head', dest='use_scale_head', action='store_true',
                        help='Use flux scale head to calibrate magnitude (default)')
    parser.add_argument('--no_scale_head', dest='use_scale_head', action='store_false',
                        help='Disable flux scale head')
    parser.set_defaults(use_scale_head=True)
    parser.add_argument('--use_film', dest='use_film', action='store_true',
                        help='Enable FiLM conditioning by rock_type (default: follow checkpoint)')
    parser.add_argument('--no_film', dest='use_film', action='store_false',
                        help='Disable FiLM conditioning')
    parser.set_defaults(use_film=None)
    parser.add_argument('--film_dim', type=int, default=None, help='FiLM embedding dimension (use checkpoint if None)')
    parser.add_argument('--num_rock_types', type=int, default=None, help='Number of rock_type categories (use checkpoint if None)')

    parser.add_argument('--save_field', type=int, default=0, choices=[0, 1], help='Save full pred_ux field')
    parser.add_argument('--pred_format', type=str, default='npz', choices=['npz', 'h5'], help='Prediction output format')
    parser.add_argument('--field_dtype', type=str, default='float16', choices=['float16', 'float32'], help='Field dtype when saving pred_ux')

    parser.add_argument('--topk', type=int, default=200, help='Top-K worst cases by rel_err')
    parser.add_argument('--eps_q', type=float, default=1e-8, help='Epsilon for rel_err')

    parser.add_argument('--base_channels', type=int, default=None, help='U-Net base channels (use checkpoint if None)')

    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()
