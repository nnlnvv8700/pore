# -*- coding: utf-8 -*-
"""
Build HDF5 Dataset for Porous Media U-Net Training (Optimized Version)

优化策略：
1. 多尺度滑窗：使用多个 raw_patch 尺寸，捕获不同大小的孔隙
2. 自适应孔隙检测：先检测连通孔隙区域，以孔隙质心为中心提取patch
3. 放宽触边限制：允许少量触边（max_border_pore_frac > 0）
4. 数据增强：旋转90°增强（可选）
5. 多层提取：可选提取多个x_layer切片
6. 智能步长：根据孔隙分布自适应调整步长

Example usage:
    python build_hdf5_dataset_v2.py --root "E:\mhw\1\pore\data" --rocks 1 2 3 4 5 6 \
        --out_h5 "E:\mhw\1\pore\dataset_all_32_v2.h5" \
        --raw_patches 32 48 64 --patch 32 --stride_raw 4 --R1 8 \
        --max_border_pore_frac 0.05 --use_pore_detection --augment_rot90

Author: Data Engineering Pipeline (Optimized)
Date: 2026-02-04
"""

import os
import re
import csv
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict

import numpy as np
import h5py
from scipy.ndimage import zoom, label, distance_transform_edt, center_of_mass

# 版本信息
__version__ = "4.0.0"  # Optimized version for more valid data


# ===========================================================================
# 1) 文件匹配 (与原版相同)
# ===========================================================================
OUT_RE = re.compile(r"^OUT_(\d+)_(\d+)\.dat$", re.IGNORECASE)
IN_RE = re.compile(r"^poretype_sub(\d+)\.in$", re.IGNORECASE)


def scan_pairs(folder: str) -> Dict[int, Dict]:
    """Scan a folder for OUT_<id>_<step>.dat files."""
    pairs: Dict[int, Dict] = {}
    if not os.path.isdir(folder):
        return pairs
    
    for fn in os.listdir(folder):
        m = OUT_RE.match(fn)
        if m:
            id_ = int(m.group(1))
            step = int(m.group(2))
            pairs.setdefault(id_, {"in": None, "outs": []})
            pairs[id_]["outs"].append((step, os.path.join(folder, fn)))
            continue
        m = IN_RE.match(fn)
        if m:
            id_ = int(m.group(1))
            pairs.setdefault(id_, {"in": None, "outs": []})
            pairs[id_]["in"] = os.path.join(folder, fn)
    return pairs


def choose_latest_out(outs: List[Tuple[int, str]]) -> Tuple[int, str]:
    """Select the OUT file with maximum step."""
    outs_sorted = sorted(outs, key=lambda x: x[0])
    return outs_sorted[-1]


# ===========================================================================
# 2) 解析 Tecplot OUT (与原版相同)
# ===========================================================================
def _find_header_and_dims(out_path: str) -> Tuple[int, int, int, int]:
    """Parse header to find I, J, K dimensions and data start line."""
    I = J = K = None
    skip_rows = 0
    zone_re = re.compile(r"\bI\s*=\s*(\d+)\s+J\s*=\s*(\d+)\s+K\s*=\s*(\d+)", re.IGNORECASE)

    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            skip_rows += 1
            m = zone_re.search(line)
            if m:
                I, J, K = int(m.group(1)), int(m.group(2)), int(m.group(3))

            s = line.strip()
            if s and (s[0].isdigit() or (s[0] == '-' and len(s) > 1 and s[1].isdigit())):
                skip_rows -= 1
                break

    if I is None or J is None or K is None:
        raise ValueError(f"Cannot parse I,J,K from ZONE line in: {out_path}")
    return skip_rows, I, J, K


def load_out_as_2d(
    out_path: str,
    x_layer: int = 1,
    solid_ux_tol: float = 1e-6,
    debug_walls: bool = False
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    """Load OUT file and extract 2D slice."""
    skip_rows, I, J, K = _find_header_and_dims(out_path)
    data = np.loadtxt(out_path, skiprows=skip_rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"Unexpected numeric table shape: {data.shape} in {out_path}")

    X = data[:, 0].astype(np.int32) - 1
    Y = data[:, 1].astype(np.int32) - 1
    Z = data[:, 2].astype(np.int32) - 1
    ux = data[:, 4].astype(np.float32)
    walls = data[:, 7].astype(np.int16)

    ux3 = np.zeros((I, J, K), dtype=np.float32)
    w3 = np.zeros((I, J, K), dtype=np.int16)
    ux3[X, Y, Z] = ux
    w3[X, Y, Z] = walls

    xi = int(x_layer) - 1
    if xi < 0 or xi >= I:
        raise ValueError(f"x_layer must be in [1,{I}], got {x_layer}")

    walls2d = w3[xi, :, :]
    ux2d = ux3[xi, :, :].copy()

    # walls 语义：0/2 空隙，1/-1 固体
    mask2d = np.isin(walls2d, [0, 2]).astype(np.uint8)
    ux2d *= mask2d.astype(np.float32)

    if debug_walls:
        uniq = np.unique(walls2d)
        print(f"[DEBUG] walls unique values in {os.path.basename(out_path)}: {uniq}")

    solid_ux_max = float(np.max(np.abs(ux2d[mask2d == 0]))) if np.any(mask2d == 0) else 0.0
    if solid_ux_max > solid_ux_tol:
        print(f"[WARN] solid_ux_max={solid_ux_max:.3e} > tol={solid_ux_tol:.1e} in {out_path}")

    return mask2d, ux2d, (I, J, K), solid_ux_max


# ===========================================================================
# 3) Patch 工具函数 (增强版)
# ===========================================================================
def center_crop_or_pad(img: np.ndarray, out_hw: Tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    """Center crop or pad image to target size."""
    H, W = img.shape
    outH, outW = out_hw

    y0 = max((H - outH) // 2, 0)
    x0 = max((W - outW) // 2, 0)
    cropped = img[y0:y0 + min(outH, H), x0:x0 + min(outW, W)]

    ch, cw = cropped.shape
    pad_top = max((outH - ch) // 2, 0)
    pad_bottom = max(outH - ch - pad_top, 0)
    pad_left = max((outW - cw) // 2, 0)
    pad_right = max(outW - cw - pad_left, 0)

    if pad_top or pad_bottom or pad_left or pad_right:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value
        )
    return cropped.astype(img.dtype, copy=False)


def compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute centroid of pore pixels."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        H, W = mask.shape
        return H / 2.0, W / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def shift_with_zero_pad(img: np.ndarray, dy: int, dx: int, pad_value: float = 0.0) -> np.ndarray:
    """Shift image with zero padding."""
    H, W = img.shape
    out = np.full((H, W), pad_value, dtype=img.dtype)
    
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    
    return out


def touches_border(mask: np.ndarray, max_border_pore_frac: float = 0.0) -> bool:
    """Check whether pore pixels touch patch border."""
    if mask.size == 0:
        return True
    border = np.concatenate([
        mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
    ]).astype(np.float32)
    border_pore_frac = float(border.mean())
    return border_pore_frac > max_border_pore_frac


def recenter_patch(mask: np.ndarray, ux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Align pore centroid to patch center."""
    H, W = mask.shape
    
    if mask.sum() == 0:
        return mask.copy(), ux.copy(), 0, 0, False
    
    cy, cx = compute_centroid(mask)
    center_y, center_x = H / 2.0, W / 2.0
    dy = int(round(center_y - cy))
    dx = int(round(center_x - cx))
    
    mask_centered = shift_with_zero_pad(mask.astype(np.float32), dy, dx, pad_value=0.0)
    mask_centered = (mask_centered > 0.5).astype(np.uint8)
    
    ux_centered = shift_with_zero_pad(ux, dy, dx, pad_value=0.0)
    ux_centered = ux_centered * mask_centered.astype(np.float32)
    
    return mask_centered, ux_centered, dy, dx, True


def zoom_to_target_radius(
    mask: np.ndarray, 
    ux: np.ndarray, 
    R0: float, 
    R1: float, 
    patch_size: int,
    scale_clamp: Tuple[float, float] = (0.1, 10.0)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Physics-informed spatial normalization with Hagen-Poiseuille scaling."""
    if R0 <= 1e-8:
        return (
            np.zeros((patch_size, patch_size), dtype=np.uint8), 
            np.zeros((patch_size, patch_size), dtype=np.float32), 
            1.0
        )
    
    s = float(R1 / R0)
    s = max(min(s, scale_clamp[1]), scale_clamp[0])
    
    mask_f = mask.astype(np.float32)
    mask_z = zoom(mask_f, zoom=(s, s), order=0, mode='constant', cval=0.0)
    ux_z = zoom(ux, zoom=(s, s), order=1, mode='constant', cval=0.0)
    
    mask_n = center_crop_or_pad(mask_z, (patch_size, patch_size), pad_value=0.0)
    mask_n = (mask_n > 0.5).astype(np.uint8)
    
    ux_n = center_crop_or_pad(ux_z, (patch_size, patch_size), pad_value=0.0).astype(np.float32)
    ux_n *= mask_n.astype(np.float32)
    ux_n *= (s ** 2)

    return mask_n, ux_n, s


def rotate_90(mask: np.ndarray, ux: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate mask and ux by 90*k degrees."""
    mask_r = np.rot90(mask, k=k)
    ux_r = np.rot90(ux, k=k)
    return mask_r.copy(), ux_r.copy()


# ===========================================================================
# 4) 孔隙检测与智能提取
# ===========================================================================
def detect_pore_regions(mask2d: np.ndarray, min_area: int = 50) -> List[Dict]:
    """
    检测所有连通孔隙区域，返回其位置和大小信息。
    
    Returns:
        List of {centroid: (cy, cx), area: int, bbox: (y0, y1, x0, x1), RA: float}
    """
    labeled, num_features = label(mask2d)
    regions = []
    
    for i in range(1, num_features + 1):
        region_mask = (labeled == i)
        area = int(region_mask.sum())
        
        if area < min_area:
            continue
        
        ys, xs = np.where(region_mask)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        RA = math.sqrt(area / math.pi)
        
        regions.append({
            'centroid': (cy, cx),
            'area': area,
            'bbox': (y0, y1, x0, x1),
            'RA': RA
        })
    
    return regions


def extract_patch_at_center(
    mask2d: np.ndarray, 
    ux2d: np.ndarray, 
    cy: float, 
    cx: float, 
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    以给定中心点提取patch，如果超出边界则padding。
    """
    H, W = mask2d.shape
    half = patch_size // 2
    
    # 计算提取区域
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    
    # 创建输出数组
    mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
    ux_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    # 计算有效区域
    src_y0 = max(0, y0)
    src_y1 = min(H, y1)
    src_x0 = max(0, x0)
    src_x1 = min(W, x1)
    
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    
    if src_y1 > src_y0 and src_x1 > src_x0:
        mask_patch[dst_y0:dst_y1, dst_x0:dst_x1] = mask2d[src_y0:src_y1, src_x0:src_x1]
        ux_patch[dst_y0:dst_y1, dst_x0:dst_x1] = ux2d[src_y0:src_y1, src_x0:src_x1]
        return mask_patch, ux_patch, True
    
    return mask_patch, ux_patch, False


# ===========================================================================
# 5) Patch 配置和迭代器 (增强版)
# ===========================================================================
@dataclass
class PatchConfigV2:
    """Enhanced configuration for patch extraction."""
    raw_patches: List[int] = field(default_factory=lambda: [32, 48, 64])  # 多尺度
    patch_size: int = 32
    stride_raw: int = 4
    min_pore_frac: float = 0.05
    max_pore_frac: float = 0.95
    max_border_pore_frac: float = 0.05  # 放宽触边限制
    target_R1: float = 10.0
    compute_dist: bool = True
    scale_clamp: Tuple[float, float] = (0.1, 10.0)
    recenter: bool = True
    use_pore_detection: bool = True  # 使用孔隙检测
    min_pore_area: int = 30  # 最小孔隙面积
    augment_rot90: bool = False  # 旋转增强
    center_check_ratio: float = 0.25  # 中心检查区域比例（放宽）
    center_min_frac_ratio: float = 0.3  # 中心区域最小孔隙率比例


def process_single_patch(
    pm_raw: np.ndarray, 
    pu_raw: np.ndarray, 
    cfg: PatchConfigV2,
    stats: Optional[Dict[str, int]] = None,
    aug_k: int = 0  # 旋转增强参数
) -> Optional[Dict]:
    """
    处理单个patch，返回标准化结果或None。
    """
    P_out = cfg.patch_size
    
    # 旋转增强
    if aug_k > 0:
        pm_raw, pu_raw = rotate_90(pm_raw, pu_raw, aug_k)
    
    pore_frac = float(pm_raw.mean())
    if pore_frac < cfg.min_pore_frac:
        if stats is not None:
            stats['drop_pore_frac_low'] = stats.get('drop_pore_frac_low', 0) + 1
        return None
    if pore_frac > cfg.max_pore_frac:
        if stats is not None:
            stats['drop_pore_frac_high'] = stats.get('drop_pore_frac_high', 0) + 1
        return None

    # 触边检查（放宽）
    if touches_border(pm_raw, cfg.max_border_pore_frac):
        if stats is not None:
            stats['drop_border_pre'] = stats.get('drop_border_pre', 0) + 1
        return None

    # Recenter
    if cfg.recenter:
        pm, pu, shift_dy, shift_dx, valid = recenter_patch(pm_raw, pu_raw)
        if not valid:
            if stats is not None:
                stats['drop_recenter_invalid'] = stats.get('drop_recenter_invalid', 0) + 1
            return None
    else:
        pm, pu = pm_raw.copy(), pu_raw.copy()

    # 后置触边检查（可选，使用更宽松阈值）
    post_border_thresh = min(cfg.max_border_pore_frac * 2, 0.15)
    if touches_border(pm, post_border_thresh):
        if stats is not None:
            stats['drop_border_post'] = stats.get('drop_border_post', 0) + 1
        return None

    # 计算 RA, RI, eta
    A = float(pm.sum())
    if A < cfg.min_pore_area:
        if stats is not None:
            stats['drop_area_low'] = stats.get('drop_area_low', 0) + 1
        return None
    
    RA = math.sqrt(A / math.pi)
    dist_in = distance_transform_edt(pm.astype(bool))
    RI = float(dist_in.max())
    eta = float(RI / (RA + 1e-8))

    q0 = float((pu * pm.astype(np.float32)).sum())

    # Zoom to target radius
    pm_n, pu_n, s = zoom_to_target_radius(
        pm, pu, RA, cfg.target_R1, P_out, 
        scale_clamp=cfg.scale_clamp
    )
    
    # 归一化后孔隙率检查（放宽）
    pore_frac_norm = float(pm_n.mean())
    if pore_frac_norm < cfg.min_pore_frac * 0.8:  # 放宽20%
        if stats is not None:
            stats['drop_pore_frac_norm'] = stats.get('drop_pore_frac_norm', 0) + 1
        return None
    
    # 中心区域检查（放宽）
    P = P_out
    margin = int(P * cfg.center_check_ratio)
    if margin > 0 and margin < P // 2:
        center_region = pm_n[margin:P-margin, margin:P-margin]
        center_pore_frac = float(center_region.mean())
        
        if center_pore_frac < cfg.min_pore_frac * cfg.center_min_frac_ratio:
            if stats is not None:
                stats['drop_center_region'] = stats.get('drop_center_region', 0) + 1
            return None
    
    qn = float((pu_n * pm_n.astype(np.float32)).sum())

    # Distance transform
    if cfg.compute_dist:
        dist_n = distance_transform_edt(pm_n.astype(bool)).astype(np.float32)
        dist_n = dist_n / max(cfg.target_R1, 1e-6)
        dist_n = np.clip(dist_n, 0.0, 2.0)
    else:
        dist_n = None

    return {
        "mask_norm": pm_n,
        "ux_norm": pu_n,
        "dist_norm": dist_n,
        "R0_RA": RA,
        "RI": RI,
        "eta": eta,
        "scale_s": float(s),
        "q0": q0,
        "q_norm": qn,
        "aug_k": aug_k,
    }


def iter_patches_v2(
    mask2d: np.ndarray, 
    ux2d: np.ndarray, 
    cfg: PatchConfigV2, 
    stats: Optional[Dict[str, int]] = None
):
    """
    增强版patch迭代器：多尺度 + 孔隙检测 + 旋转增强
    """
    H, W = mask2d.shape
    P_out = cfg.patch_size
    S = cfg.stride_raw
    
    # 记录已提取的位置，避免过多重复
    extracted_centers: Set[Tuple[int, int]] = set()
    min_center_dist = P_out // 4  # 最小中心距离
    
    def is_duplicate(cy: float, cx: float) -> bool:
        """检查是否与已提取位置过近"""
        cy_i, cx_i = int(cy // min_center_dist), int(cx // min_center_dist)
        return (cy_i, cx_i) in extracted_centers
    
    def mark_extracted(cy: float, cx: float):
        cy_i, cx_i = int(cy // min_center_dist), int(cx // min_center_dist)
        extracted_centers.add((cy_i, cx_i))
    
    # =========================================================
    # 策略1: 基于孔隙检测的提取（优先）
    # =========================================================
    if cfg.use_pore_detection:
        regions = detect_pore_regions(mask2d, min_area=cfg.min_pore_area)
        
        for region in regions:
            cy, cx = region['centroid']
            RA = region['RA']
            
            # 根据孔隙大小选择合适的 raw_patch
            # 目标是让 RA 缩放后接近 target_R1
            best_raw_patch = cfg.raw_patches[0]
            for rp in cfg.raw_patches:
                # 估算在该 raw_patch 下的缩放比例
                # 如果 RA 在 [0.3*rp, 0.7*rp] 范围内比较合适
                if 0.2 * rp < RA * 2 < 0.8 * rp:
                    best_raw_patch = rp
                    break
            
            if is_duplicate(cy, cx):
                continue
            
            # 提取patch
            pm_raw, pu_raw, valid = extract_patch_at_center(
                mask2d, ux2d, cy, cx, best_raw_patch
            )
            
            if not valid:
                continue
            
            if stats is not None:
                stats['total_pore_detect'] = stats.get('total_pore_detect', 0) + 1
            
            # 处理patch
            result = process_single_patch(pm_raw, pu_raw, cfg, stats, aug_k=0)
            if result is not None:
                result['yx0_raw'] = (int(cy - best_raw_patch // 2), int(cx - best_raw_patch // 2))
                result['extract_method'] = 'pore_detect'
                result['raw_patch_size'] = best_raw_patch
                mark_extracted(cy, cx)
                yield result
                
                # 旋转增强
                if cfg.augment_rot90:
                    for k in [1, 2, 3]:
                        result_aug = process_single_patch(pm_raw, pu_raw, cfg, stats, aug_k=k)
                        if result_aug is not None:
                            result_aug['yx0_raw'] = (int(cy - best_raw_patch // 2), int(cx - best_raw_patch // 2))
                            result_aug['extract_method'] = f'pore_detect_rot{k*90}'
                            result_aug['raw_patch_size'] = best_raw_patch
                            yield result_aug
    
    # =========================================================
    # 策略2: 多尺度滑窗提取
    # =========================================================
    for P_raw in cfg.raw_patches:
        # Pad if needed
        mask_padded = mask2d
        ux_padded = ux2d
        H_p, W_p = H, W
        
        if H < P_raw or W < P_raw:
            mask_padded = center_crop_or_pad(mask2d, (max(H, P_raw), max(W, P_raw)), pad_value=0).astype(np.uint8)
            ux_padded = center_crop_or_pad(ux2d, (max(H, P_raw), max(W, P_raw)), pad_value=0.0).astype(np.float32)
            H_p, W_p = mask_padded.shape

        for y0 in range(0, H_p - P_raw + 1, S):
            for x0 in range(0, W_p - P_raw + 1, S):
                if stats is not None:
                    stats['total_sliding'] = stats.get('total_sliding', 0) + 1
                
                # 检查是否已提取
                cy, cx = y0 + P_raw // 2, x0 + P_raw // 2
                if is_duplicate(cy, cx):
                    if stats is not None:
                        stats['skip_duplicate'] = stats.get('skip_duplicate', 0) + 1
                    continue
                
                pm_raw = mask_padded[y0:y0 + P_raw, x0:x0 + P_raw].copy()
                pu_raw = ux_padded[y0:y0 + P_raw, x0:x0 + P_raw].copy()
                
                result = process_single_patch(pm_raw, pu_raw, cfg, stats, aug_k=0)
                if result is not None:
                    result['yx0_raw'] = (y0, x0)
                    result['extract_method'] = 'sliding'
                    result['raw_patch_size'] = P_raw
                    mark_extracted(cy, cx)
                    yield result
                    
                    # 旋转增强（仅对滑窗的部分样本）
                    if cfg.augment_rot90 and np.random.random() < 0.3:  # 30%概率
                        k = np.random.randint(1, 4)
                        result_aug = process_single_patch(pm_raw, pu_raw, cfg, stats, aug_k=k)
                        if result_aug is not None:
                            result_aug['yx0_raw'] = (y0, x0)
                            result_aug['extract_method'] = f'sliding_rot{k*90}'
                            result_aug['raw_patch_size'] = P_raw
                            yield result_aug


# ===========================================================================
# 6) Buffered HDF5 写入 (与原版相同)
# ===========================================================================
class BufferedH5Writer:
    """Buffered HDF5 writer for efficient batch writing."""
    
    def __init__(self, h5_path: str, patch_size: int, n_channels: int, 
                 buffer_size: int = 1024, compression: str = "gzip"):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.compression = compression if compression != "none" else None
        
        self.h5 = None
        self.dsets = {}
        self.buffer = {
            'X': [], 'Y': [],
            'rock_type': [], 'local_id': [], 'global_id': [],
            'step': [], 'R0_RA': [], 'RI': [], 'eta': [],
            'scale_s': [], 'q0': [], 'q_norm': [],
            'yx0_raw': [], 'dims_3d': [], 'raw_patch_size': []
        }
        self.total_written = 0
    
    def open(self, attrs: Dict):
        """Open HDF5 file and create datasets."""
        out_dir = os.path.dirname(self.h5_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.h5 = h5py.File(self.h5_path, "w")
        
        P = self.patch_size
        C = self.n_channels
        comp = self.compression
        
        self.dsets['X'] = self.h5.create_dataset(
            'X', shape=(0, C, P, P), maxshape=(None, C, P, P),
            dtype=np.float32, chunks=(min(64, self.buffer_size), C, P, P), compression=comp
        )
        self.dsets['Y'] = self.h5.create_dataset(
            'Y', shape=(0, 1, P, P), maxshape=(None, 1, P, P),
            dtype=np.float32, chunks=(min(64, self.buffer_size), 1, P, P), compression=comp
        )
        
        self.dsets['rock_type'] = self.h5.create_dataset(
            'rock_type', shape=(0,), maxshape=(None,), dtype=np.int8, 
            chunks=(1024,), compression=comp
        )
        self.dsets['local_id'] = self.h5.create_dataset(
            'local_id', shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1024,), compression=comp
        )
        self.dsets['global_id'] = self.h5.create_dataset(
            'global_id', shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1024,), compression=comp
        )
        self.dsets['step'] = self.h5.create_dataset(
            'step', shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1024,), compression=comp
        )
        self.dsets['R0_RA'] = self.h5.create_dataset(
            'R0_RA', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['RI'] = self.h5.create_dataset(
            'RI', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['eta'] = self.h5.create_dataset(
            'eta', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['scale_s'] = self.h5.create_dataset(
            'scale_s', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['q0'] = self.h5.create_dataset(
            'q0', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['q_norm'] = self.h5.create_dataset(
            'q_norm', shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1024,), compression=comp
        )
        self.dsets['yx0_raw'] = self.h5.create_dataset(
            'yx0_raw', shape=(0, 2), maxshape=(None, 2), dtype=np.int32,
            chunks=(1024, 2), compression=comp
        )
        self.dsets['dims_3d'] = self.h5.create_dataset(
            'dims_3d', shape=(0, 3), maxshape=(None, 3), dtype=np.int32,
            chunks=(1024, 3), compression=comp
        )
        self.dsets['raw_patch_size'] = self.h5.create_dataset(
            'raw_patch_size', shape=(0,), maxshape=(None,), dtype=np.int16,
            chunks=(1024,), compression=comp
        )
        
        for k, v in attrs.items():
            self.h5.attrs[k] = v
    
    def add_sample(self, X: np.ndarray, Y: np.ndarray, 
                   rock_type: int, local_id: int, global_id: int,
                   step: int, R0_RA: float, RI: float, eta: float,
                   scale_s: float, q0: float, q_norm: float,
                   yx0_raw: Tuple[int, int], dims_3d: Tuple[int, int, int],
                   raw_patch_size: int = 32):
        """Add a sample to the buffer."""
        self.buffer['X'].append(X)
        self.buffer['Y'].append(Y)
        self.buffer['rock_type'].append(rock_type)
        self.buffer['local_id'].append(local_id)
        self.buffer['global_id'].append(global_id)
        self.buffer['step'].append(step)
        self.buffer['R0_RA'].append(R0_RA)
        self.buffer['RI'].append(RI)
        self.buffer['eta'].append(eta)
        self.buffer['scale_s'].append(scale_s)
        self.buffer['q0'].append(q0)
        self.buffer['q_norm'].append(q_norm)
        self.buffer['yx0_raw'].append(yx0_raw)
        self.buffer['dims_3d'].append(dims_3d)
        self.buffer['raw_patch_size'].append(raw_patch_size)
        
        if len(self.buffer['X']) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to HDF5 and clear."""
        n = len(self.buffer['X'])
        if n == 0:
            return
        
        old_size = self.total_written
        new_size = old_size + n
        
        for name, dset in self.dsets.items():
            new_shape = (new_size,) + dset.shape[1:]
            dset.resize(new_shape)
        
        self.dsets['X'][old_size:new_size] = np.array(self.buffer['X'], dtype=np.float32)
        self.dsets['Y'][old_size:new_size] = np.array(self.buffer['Y'], dtype=np.float32)
        self.dsets['rock_type'][old_size:new_size] = np.array(self.buffer['rock_type'], dtype=np.int8)
        self.dsets['local_id'][old_size:new_size] = np.array(self.buffer['local_id'], dtype=np.int32)
        self.dsets['global_id'][old_size:new_size] = np.array(self.buffer['global_id'], dtype=np.int32)
        self.dsets['step'][old_size:new_size] = np.array(self.buffer['step'], dtype=np.int32)
        self.dsets['R0_RA'][old_size:new_size] = np.array(self.buffer['R0_RA'], dtype=np.float32)
        self.dsets['RI'][old_size:new_size] = np.array(self.buffer['RI'], dtype=np.float32)
        self.dsets['eta'][old_size:new_size] = np.array(self.buffer['eta'], dtype=np.float32)
        self.dsets['scale_s'][old_size:new_size] = np.array(self.buffer['scale_s'], dtype=np.float32)
        self.dsets['q0'][old_size:new_size] = np.array(self.buffer['q0'], dtype=np.float32)
        self.dsets['q_norm'][old_size:new_size] = np.array(self.buffer['q_norm'], dtype=np.float32)
        self.dsets['yx0_raw'][old_size:new_size] = np.array(self.buffer['yx0_raw'], dtype=np.int32)
        self.dsets['dims_3d'][old_size:new_size] = np.array(self.buffer['dims_3d'], dtype=np.int32)
        self.dsets['raw_patch_size'][old_size:new_size] = np.array(self.buffer['raw_patch_size'], dtype=np.int16)
        
        self.total_written = new_size
        
        for k in self.buffer:
            self.buffer[k] = []
    
    def close(self):
        """Flush remaining and close file."""
        self.flush()
        if self.h5:
            self.h5.close()
            self.h5 = None


# ===========================================================================
# 7) 可视化
# ===========================================================================
def plot_sample_patches(samples: List[Dict], out_path: str, show_dist: bool = True):
    """Plot a small grid of sampled patches."""
    if not samples:
        print("No samples to visualize.")
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    n = len(samples)
    n_cols = 4 if show_dist else 3
    n_rows = min(n, 20)  # 最多显示20个
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_rows):
        s = samples[i]
        mask = s['mask']
        ux = s['ux']
        
        ax = axes[i, 0]
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"mask (method={s.get('method', 'N/A')[:10]})" if i == 0 else '', fontsize=9)
        ax.axis('off')

        ax = axes[i, 1]
        vmax_u = max(float(np.max(ux)), 1e-12)
        ax.imshow(ux, cmap='viridis', vmin=0.0, vmax=vmax_u, interpolation='nearest')
        ax.set_title('ux_norm' if i == 0 else '', fontsize=9)
        ax.axis('off')

        if show_dist and s.get('dist') is not None:
            ax = axes[i, 2]
            ax.imshow(s['dist'], cmap='viridis', interpolation='nearest')
            ax.set_title('dist_norm' if i == 0 else '', fontsize=9)
            ax.axis('off')
            
            ax = axes[i, 3]
            ax.imshow(s['eta'], cmap='plasma', interpolation='nearest')
            ax.set_title('eta_map' if i == 0 else '', fontsize=9)
            ax.axis('off')
        else:
            ax = axes[i, 2]
            ax.imshow(s['eta'], cmap='plasma', interpolation='nearest')
            ax.set_title('eta_map' if i == 0 else '', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {out_path}")


# ===========================================================================
# 8) 主程序
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Build HDF5 dataset (optimized version) for porous media U-Net training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径
    ap.add_argument("--root", type=str, required=True,
                    help="数据根目录（包含 1/, 2/, ... 子目录）")
    ap.add_argument("--rocks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="要处理的岩石类型")
    ap.add_argument("--out_h5", type=str, required=True,
                    help="输出 HDF5 文件路径")
    
    # 可选
    ap.add_argument("--manifest", type=str, default="",
                    help="输出 manifest.csv 路径")
    ap.add_argument("--x_layer", type=int, default=1,
                    help="提取的 X 层索引（1-based）")
    ap.add_argument("--x_layers", type=int, nargs="*", default=[],
                    help="多层提取（覆盖 --x_layer）")
    
    # Patch 参数 (优化版)
    ap.add_argument("--raw_patches", type=int, nargs="+", default=[32, 48, 64],
                    help="多尺度原始patch大小列表")
    ap.add_argument("--patch", type=int, default=32,
                    help="输出 patch 大小")
    ap.add_argument("--stride_raw", type=int, default=4,
                    help="滑窗步长")
    ap.add_argument("--R1", type=float, default=8.0,
                    help="目标等效半径")
    
    # 过滤参数 (放宽)
    ap.add_argument("--min_pore_frac", type=float, default=0.05,
                    help="最小孔隙率阈值")
    ap.add_argument("--max_pore_frac", type=float, default=0.95,
                    help="最大孔隙率阈值")
    ap.add_argument("--max_border_pore_frac", type=float, default=0.05,
                    help="触边孔隙比例阈值（放宽到0.05）")
    ap.add_argument("--min_pore_area", type=int, default=30,
                    help="最小孔隙面积")
    
    # 优化选项
    ap.add_argument("--use_pore_detection", action="store_true", default=True,
                    help="使用孔隙检测智能提取")
    ap.add_argument("--no_pore_detection", action="store_true",
                    help="禁用孔隙检测")
    ap.add_argument("--augment_rot90", action="store_true",
                    help="启用90度旋转增强")
    ap.add_argument("--center_check_ratio", type=float, default=0.25,
                    help="中心检查区域比例")
    ap.add_argument("--center_min_frac_ratio", type=float, default=0.3,
                    help="中心区域最小孔隙率比例")
    
    # 特征选项
    ap.add_argument("--no_dist", action="store_true",
                    help="不计算距离变换特征")
    ap.add_argument("--no_recenter", action="store_true",
                    help="禁用孔隙质心对齐")
    ap.add_argument("--solid_ux_tol", type=float, default=1e-6,
                    help="固体区速度容忍阈值")
    ap.add_argument("--skip_bad_solid_ux", action="store_true",
                    help="若固体区速度超阈值则跳过")
    
    # 存储选项
    ap.add_argument("--compression", type=str, default="gzip",
                    choices=["gzip", "lzf", "none"],
                    help="HDF5 压缩方式")
    ap.add_argument("--buffer", type=int, default=1024,
                    help="写入缓冲区大小")
    ap.add_argument("--viz_samples", type=int, default=0,
                    help="抽样可视化数量")
    ap.add_argument("--viz_out", type=str, default="",
                    help="可视化输出路径")
    
    # 缩放限制
    ap.add_argument("--scale_min", type=float, default=0.1,
                    help="最小缩放因子")
    ap.add_argument("--scale_max", type=float, default=10.0,
                    help="最大缩放因子")
    
    args = ap.parse_args()
    
    # 处理 x_layers
    if args.x_layers:
        x_layers = args.x_layers
    else:
        x_layers = [args.x_layer]
    
    use_pore_detection = args.use_pore_detection and not args.no_pore_detection
    
    print(f"\n{'='*70}")
    print(f"Build HDF5 Dataset v{__version__} (Optimized)")
    print(f"{'='*70}")
    print(f"Root directory:     {args.root}")
    print(f"Rock types:         {args.rocks}")
    print(f"Output HDF5:        {args.out_h5}")
    print(f"Raw patches:        {args.raw_patches}")
    print(f"Output patch:       {args.patch}x{args.patch}")
    print(f"Stride:             {args.stride_raw}")
    print(f"Target R1:          {args.R1}")
    print(f"X layers:           {x_layers}")
    print(f"Pore detection:     {use_pore_detection}")
    print(f"Augment rot90:      {args.augment_rot90}")
    print(f"Max border frac:    {args.max_border_pore_frac}")
    print(f"Center check ratio: {args.center_check_ratio}")
    print(f"{'='*70}\n")
    
    # Config
    cfg = PatchConfigV2(
        raw_patches=args.raw_patches,
        patch_size=args.patch,
        stride_raw=args.stride_raw,
        min_pore_frac=args.min_pore_frac,
        max_pore_frac=args.max_pore_frac,
        max_border_pore_frac=args.max_border_pore_frac,
        target_R1=args.R1,
        compute_dist=(not args.no_dist),
        scale_clamp=(args.scale_min, args.scale_max),
        recenter=(not args.no_recenter),
        use_pore_detection=use_pore_detection,
        min_pore_area=args.min_pore_area,
        augment_rot90=args.augment_rot90,
        center_check_ratio=args.center_check_ratio,
        center_min_frac_ratio=args.center_min_frac_ratio,
    )
    
    # Channels
    C = 3 if not args.no_dist else 2
    channel_order = "mask,dist,eta_map" if not args.no_dist else "mask,eta_map"
    
    # Attributes
    attrs = {
        "patch_size": args.patch,
        "raw_patches": args.raw_patches,
        "stride_raw": args.stride_raw,
        "target_R1": float(args.R1),
        "x_layers": x_layers,
        "channels": C,
        "channel_order": channel_order,
        "rock_types": args.rocks,
        "version": __version__,
        "use_pore_detection": use_pore_detection,
        "augment_rot90": args.augment_rot90,
    }
    
    # Writer
    writer = BufferedH5Writer(
        args.out_h5, args.patch, C,
        buffer_size=args.buffer,
        compression=args.compression
    )
    writer.open(attrs)

    # Visualization sampling
    viz_samples: List[Dict] = []
    viz_seen = 0
    drop_stats: Dict[str, int] = {}
    
    # Manifest
    manifest_writer = None
    manifest_file = None
    if args.manifest:
        manifest_dir = os.path.dirname(args.manifest)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        manifest_file = open(args.manifest, "w", newline="", encoding="utf-8")
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow([
            "rock_type", "local_id", "global_id", "step", "x_layer", "out_file", 
            "n_patches", "dims_3d", "note"
        ])
    
    # Process
    total_patches = 0
    rock_stats = defaultdict(int)
    method_stats = defaultdict(int)
    
    for rock_k in args.rocks:
        rock_type = rock_k - 1
        rock_folder = os.path.join(args.root, str(rock_k))
        
        if not os.path.isdir(rock_folder):
            print(f"[WARN] Rock folder not found: {rock_folder}")
            continue
        
        print(f"\n--- Processing Rock Type {rock_k} (rock_type={rock_type}) ---")
        
        pairs = scan_pairs(rock_folder)
        local_ids = sorted(pairs.keys())
        
        for local_id in local_ids:
            outs = pairs[local_id]["outs"]
            if not outs:
                continue
            
            step, out_path = choose_latest_out(outs)
            global_id = rock_type * 1_000_000 + local_id
            
            for x_layer in x_layers:
                note = ""
                try:
                    mask2d, ux2d, dims3d, solid_ux_max = load_out_as_2d(
                        out_path,
                        x_layer=x_layer,
                        solid_ux_tol=args.solid_ux_tol,
                    )
                    if solid_ux_max > args.solid_ux_tol:
                        note = f"solid_ux_max={solid_ux_max:.3e}"
                        if args.skip_bad_solid_ux:
                            continue
                    
                    n_patch_this = 0
                    
                    for p in iter_patches_v2(mask2d, ux2d, cfg, stats=drop_stats):
                        mask_n = p["mask_norm"].astype(np.float32)
                        ux_n = p["ux_norm"].astype(np.float32)
                        eta_val = np.float32(p["eta"])
                        eta_map = np.full((args.patch, args.patch), eta_val, dtype=np.float32)
                        
                        if args.no_dist:
                            X = np.stack([mask_n, eta_map], axis=0)
                        else:
                            dist_n = p["dist_norm"].astype(np.float32)
                            X = np.stack([mask_n, dist_n, eta_map], axis=0)
                        
                        Y = ux_n[None, :, :]
                        
                        writer.add_sample(
                            X=X, Y=Y,
                            rock_type=rock_type,
                            local_id=local_id,
                            global_id=global_id,
                            step=step,
                            R0_RA=p["R0_RA"],
                            RI=p["RI"],
                            eta=p["eta"],
                            scale_s=p["scale_s"],
                            q0=p["q0"],
                            q_norm=p["q_norm"],
                            yx0_raw=p["yx0_raw"],
                            dims_3d=dims3d,
                            raw_patch_size=p.get("raw_patch_size", args.raw_patches[0]),
                        )

                        # Track method
                        method = p.get('extract_method', 'unknown')
                        method_stats[method] += 1

                        # Reservoir sampling
                        if args.viz_samples > 0:
                            viz_seen += 1
                            sample = {
                                'mask': mask_n,
                                'ux': ux_n,
                                'dist': p.get("dist_norm"),
                                'eta': eta_map,
                                'method': method,
                            }
                            if len(viz_samples) < args.viz_samples:
                                viz_samples.append(sample)
                            else:
                                j = np.random.randint(0, viz_seen)
                                if j < args.viz_samples:
                                    viz_samples[j] = sample
                        
                        n_patch_this += 1
                    
                    if manifest_writer:
                        manifest_writer.writerow([
                            rock_type, local_id, global_id, step, x_layer,
                            os.path.basename(out_path), n_patch_this, str(dims3d), note
                        ])
                    
                    rock_stats[rock_k] += n_patch_this
                    total_patches += n_patch_this
                    
                    if n_patch_this > 0:
                        print(f"  [OK] local_id={local_id} x_layer={x_layer} patches={n_patch_this}")
                
                except Exception as e:
                    note = f"ERROR: {e}"
                    if manifest_writer:
                        manifest_writer.writerow([
                            rock_type, local_id, global_id, step, x_layer,
                            os.path.basename(out_path), 0, "", note
                        ])
                    print(f"  [ERR] local_id={local_id} x_layer={x_layer}: {e}")
    
    # Close
    writer.close()
    if manifest_file:
        manifest_file.close()

    # Visualization
    if args.viz_samples > 0 and viz_samples:
        if args.viz_out:
            viz_path = args.viz_out
        else:
            base, _ = os.path.splitext(args.out_h5)
            viz_path = base + "_samples.png"
        plot_sample_patches(viz_samples, viz_path, show_dist=(not args.no_dist))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DONE!")
    print(f"{'='*70}")
    print(f"Output HDF5:    {args.out_h5}")
    print(f"Total patches:  {total_patches}")
    print(f"Patch size:     {args.patch}x{args.patch}")
    print(f"Channels (X):   {C} ({channel_order})")
    
    if drop_stats:
        print("\nDrop/Processing stats:")
        for k, v in sorted(drop_stats.items()):
            print(f"  {k}: {v}")
    
    print(f"\nExtraction method stats:")
    for method, count in sorted(method_stats.items()):
        print(f"  {method}: {count}")
    
    print(f"\nPatches per rock type:")
    for rock_k in sorted(rock_stats.keys()):
        print(f"  Rock {rock_k}: {rock_stats[rock_k]}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
