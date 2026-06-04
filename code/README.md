# 多孔介质流动场预测项目

## 项目概述

本项目使用深度学习方法（U-Net 卷积神经网络）预测多孔介质中的流动速度场。通过输入孔隙几何结构，模型可以快速预测流体在孔隙中的速度分布，替代耗时的 CFD（计算流体力学）仿真。

### 应用场景
- 岩石孔隙结构中的流体流动模拟
- 石油/天然气渗流预测
- 地下水流动建模

---

## 1. 数据处理流程

### 1.1 原始数据格式

原始数据来自 CFD 仿真软件（Tecplot格式），包含：
- **输入文件**: `poretype_sub{id}.in` - 孔隙几何信息
- **输出文件**: `OUT_{id}_{step}.dat` - 仿真速度场结果

数据目录结构：
```
data/
├── 1/          # 岩石类型1 (rock_type=0)
│   ├── OUT_1_1000.dat
│   ├── OUT_2_1000.dat
│   └── ...
├── 2/          # 岩石类型2 (rock_type=1)
├── 3/          # 岩石类型3 (rock_type=2)
├── 4/          # 岩石类型4 (rock_type=3)
├── 5/          # 岩石类型5 (rock_type=4)
└── 6/          # 岩石类型6 (rock_type=5)
```

### 1.2 数据提取流程 (`build_hdf5_dataset.py`)

1. **解析 Tecplot 文件**
   - 读取 3D 网格数据 (I × J × K)
   - 提取指定 x_layer 的 2D 切片
   - 获取孔隙掩码 (mask) 和 x 方向速度 (ux)

2. **滑窗提取 Patch**
   - 以固定步长 (stride) 在 2D 切面上滑动
   - 提取 `raw_patch × raw_patch` 大小的区域
   
3. **有效性过滤**
   - 孔隙率检查: 孔隙占比在合理范围内
   - 触边检查: 孔隙不能触及 patch 边界（保证完整孔隙）
   - 中心区域检查: patch 中心区域必须有足够孔隙
   - 连通性检查: 只保留主连通分量

4. **孔隙中心对齐 (Recenter)**
   - 计算孔隙区域的质心
   - 将质心平移到 patch 中心
   - 保证孔隙位置标准化

5. **空间归一化 (Zoom)**
   - 计算等效半径 `R0 = sqrt(A/π)`，其中 `A` 为孔隙面积
   - 按 `s = R1/R0` 缩放几何结构
   - 统一到目标等效半径 `R1`

---

## 2. 归一化方法

### 2.1 几何归一化

**目标**: 不同大小的孔隙结构归一化到相同尺度

```
s = R1 / R0
```

其中:
- `R0`: 原始孔隙的等效半径
- `R1`: 目标等效半径（超参数，如 8 或 10）
- `s`: 缩放因子

### 2.2 速度归一化（Hagen-Poiseuille 定律）

根据圆管流动的 Hagen-Poiseuille 定律，速度与半径平方成正比：

```
u ∝ R²
```

因此，速度归一化公式：

```
u_norm = u_raw × s² = u_raw × (R1/R0)²
```

**物理意义**: 
- 如果孔隙缩小 (s < 1)，速度按 s² 减小
- 如果孔隙放大 (s > 1)，速度按 s² 增大

### 2.3 输入通道

HDF5 数据集中 X 的通道顺序 (channel_order):
| 通道 | 名称 | 说明 |
|------|------|------|
| 0 | mask | 孔隙掩码 (1=孔隙, 0=固体) |
| 1 | dist | 距离场 (到最近壁面的距离, 归一化) |
| 2 | eta_map | 局部孔隙率 (可选) |

### 2.4 输出目标

- **Y**: 归一化后的 x 方向速度场 `ux_norm`
- 形状: `(N, 1, H, W)`
- 固体区域速度为 0

---

## 3. CNN 模型架构

### 3.1 LightUNet

采用轻量级 U-Net 架构，专为 32×32 或 64×64 小尺寸输入设计：

```
输入 (B, C_in, H, W)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Encoder                                     │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│ │ enc1    │→ │ enc2    │→ │ enc3    │      │
│ │ Conv×2  │  │ Pool+   │  │ Pool+   │      │
│ │ C=32    │  │ Conv×2  │  │ Conv×2  │      │
│ │         │  │ C=64    │  │ C=128   │      │
│ └────┬────┘  └────┬────┘  └────┬────┘      │
│      │ skip1      │ skip2      │ skip3     │
│      │            │            │           │
│      │            │            ▼           │
│      │            │     ┌──────────────┐   │
│      │            │     │ Bottleneck   │   │
│      │            │     │ Pool+Conv×2  │   │
│      │            │     │ C=256        │   │
│      │            │     └──────┬───────┘   │
│      │            │            │           │
└──────┼────────────┼────────────┼───────────┘
       │            │            │
┌──────┼────────────┼────────────┼───────────┐
│ Decoder           │            │           │
│      │            │            ▼           │
│      │            │     ┌──────────────┐   │
│      │            └────→│ dec3         │   │
│      │                  │ Up+Cat+Conv  │   │
│      │                  │ C=128        │   │
│      │                  └──────┬───────┘   │
│      │                         │           │
│      │                  ┌──────┴───────┐   │
│      └─────────────────→│ dec2         │   │
│                         │ Up+Cat+Conv  │   │
│                         │ C=64         │   │
│                         └──────┬───────┘   │
│                                │           │
│                         ┌──────┴───────┐   │
│                         │ dec1         │   │
│ skip1 ─────────────────→│ Up+Cat+Conv  │   │
│                         │ C=32         │   │
│                         └──────┬───────┘   │
└────────────────────────────────┼───────────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ 1×1 Conv     │
                          │ → Softplus   │
                          │ × mask       │
                          └──────────────┘
                                 │
                                 ▼
                          输出 (B, 1, H, W)
```

### 3.2 模型组件

| 组件 | 说明 |
|------|------|
| **ConvBlock** | 两层 3×3 卷积 + GroupNorm + SiLU 激活 |
| **DownBlock** | 2×2 MaxPool + ConvBlock |
| **UpBlock** | 2× 双线性上采样 + Skip连接 + ConvBlock |
| **Softplus** | 保证输出非负 |
| **Mask乘法** | 确保固体区域输出为0 |

### 3.3 FiLM 条件化 (Feature-wise Linear Modulation)

针对不同岩石类型 (rock_type) 的条件化：

```
FiLM(x) = x × (1 + γ) + β
```

其中 `γ, β` 由 rock_type 嵌入计算得到，应用于每个编码/解码层。

### 3.4 可选配置

| 参数 | 说明 |
|------|------|
| `--use_softplus` | 使用 Softplus 激活保证输出非负 |
| `--use_rock_type_channel` | 添加 rock_type 作为额外输入通道 |
| `--use_film` | 使用 FiLM 进行 rock_type 条件化 |
| `--base_channels` | 基础通道数 (默认 32) |

---

## 4. 损失函数

采用物理信息增强的多项损失函数：

### 4.1 基础损失

| 损失项 | 公式 | 说明 |
|--------|------|------|
| **L_field** | `Σ(pred-target)²·mask / Σmask` | 孔隙区域 MSE |
| **L_solid** | `Σpred²·(1-mask) / Σ(1-mask)` | 固体区域惩罚 |
| **L_neg** | `Σ ReLU(-pred)·mask / Σmask` | 负值惩罚 |

### 4.2 通量损失 (Flux Loss)

```
L_flux = mean(w_i × FluxErr_i)
```

**Mixed Flux Error** (混合绝对/相对误差):
```
FluxErr = |q_pred - q_true|           if |q_true| < τ
        = |q_pred - q_true| / |q_true|  otherwise
```

其中 `q = Σ(u × mask)` 为总通量。

### 4.3 物理约束损失

| 损失项 | 说明 |
|--------|------|
| **L_grad** | 梯度一致性（平滑性约束）|
| **L_peak** | 峰值区域精度（高速区域权重更大）|
| **L_wall** | 壁面无滑移约束（近壁速度应接近0）|
| **L_mono** | 单调性约束（中心速度应大于边缘）|

### 4.4 总损失

```
L_total = λ_field × L_field + λ_solid × L_solid + λ_neg × L_neg 
        + λ_q × L_flux + λ_grad × L_grad + λ_peak × L_peak 
        + λ_wall × L_wall + λ_mono × L_mono
```

---

## 5. 训练流程

### 5.1 数据划分

**分层划分策略**:
- 按 `(rock_type, global_id)` 分组
- 每组按比例划分 train/val
- 保证每种岩石类型在训练集和验证集中都有代表

### 5.2 采样策略

**TypeFluxBalancedBatchSampler**:
1. 每个 batch 包含所有 rock_type 的样本（类型均衡）
2. 每个 rock_type 内按通量 (q) 分 bin，均匀采样（通量均衡）
3. 避免高通量样本被忽视

### 5.3 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 96 | 批大小 |
| `--lr` | 1e-3 | 初始学习率 |
| `--weight_decay` | 1e-4 | L2 正则化 |
| `--patience` | 30 | 早停耐心值 |

### 5.4 学习率调度

使用 `ReduceLROnPlateau`:
- 监控验证集损失
- 10 epoch 无改善则 lr × 0.5
- 最小学习率 1e-6

### 5.5 训练命令示例

```bash
python train_unet_h5.py \
  --h5 "dataset_all_32.h5" \
  --out_dir "runs/unet_v1" \
  --epochs 200 \
  --batch_size 96 \
  --type_flux_balanced_sampler \
  --use_rock_type_channel \
  --use_softplus \
  --use_film \
  --lambda_q 10 \
  --q_mix
```

### 5.6 输出文件

```
runs/unet_v1_YYYYMMDD_HHMMSS/
├── best.pt              # 最佳模型权重
├── last.pt              # 最后一轮权重
├── config.json          # 训练配置
├── history.json         # 训练历史
├── training_log.csv     # 每轮指标
└── training_curves.png  # 训练曲线图
```

---

## 6. 推理/测试流程

### 6.1 推理命令示例

```bash
python infer_unet_h5.py \
  --h5 "dataset_all_32.h5" \
  --ckpt "runs/unet_v1/best.pt" \
  --out_dir "runs/unet_v1/infer" \
  --use_softplus \
  --use_rock_type_channel \
  --use_film
```

### 6.2 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **RMSE_pore** | `sqrt(Σ(pred-true)²·mask / Σmask)` | 孔隙区域均方根误差 |
| **MAE_pore** | `Σ|pred-true|·mask / Σmask` | 孔隙区域平均绝对误差 |
| **RelFluxErr** | `|q_pred - q_true| / |q_true|` | 相对通量误差 |
| **R²** | `1 - Σ(q_pred-q_true)² / Σ(q_true-q̄)²` | 通量拟合优度 |

### 6.3 输出文件

```
runs/unet_v1/infer/
├── report.csv               # 总体指标
├── report_by_rock_type.csv  # 按岩石类型的指标
├── worst_cases.csv          # 误差最大的样本
├── predictions.npz          # 预测结果
├── inference_results.png    # 散点图/箱线图
└── sample_predictions.png   # 样本可视化
```

---

## 7. 文件清单

| 脚本 | 功能 |
|------|------|
| `build_hdf5_dataset.py` | 从 CFD 数据构建 HDF5 数据集 |
| `train_unet_h5.py` | 训练 U-Net 模型 |
| `train_unet_h5_v2.py` | 训练 V2（增强中心速度预测）|
| `infer_unet_h5.py` | 推理和评估 |
| `visualize_metrics.py` | 可视化评估指标 |

---

## 8. 快速开始

### Step 1: 构建数据集
```bash
python build_hdf5_dataset.py \
  --root "E:\mhw\1\pore\data" \
  --rocks 1 2 3 4 5 6 \
  --out_h5 "dataset_all_32.h5" \
  --patch 32 --R1 8
```

### Step 2: 训练模型
```bash
python train_unet_h5.py \
  --h5 "dataset_all_32.h5" \
  --out_dir "runs/unet_v1" \
  --epochs 200 --batch_size 96 \
  --type_flux_balanced_sampler \
  --use_rock_type_channel --use_softplus --use_film \
  --lambda_q 10 --q_mix
```

### Step 3: 推理测试
```bash
python infer_unet_h5.py \
  --h5 "dataset_all_32.h5" \
  --ckpt "runs/unet_v1/best.pt" \
  --out_dir "runs/unet_v1/infer" \
  --use_softplus --use_rock_type_channel --use_film
```

---

## 9. 依赖环境

```
Python >= 3.8
PyTorch >= 1.10
numpy
h5py
scipy
matplotlib
tqdm
```

---

## 作者

mhw 
Date: 2026-02
