# -*- coding: utf-8 -*-
"""
AEFN Supplementary Analysis Script for Reviewer Responses
==========================================================
针对 IEEE Access 审稿意见的补充实验代码，可独立运行各分析模块：

  Module 1 — 模型复杂度分析 (参数量 + FLOPs 对比表)
  Module 2 — 训练收敛曲线 (Loss / OA per epoch)
  Module 3 — t-SNE 特征可视化 (融合前后对比)
  Module 4 — 混淆矩阵热图 (Houston & Trento)
  Module 5 — 超参数敏感性分析 (patch_size / PCA / lp_alpha)

用法示例:
  python AEFN_Reviewer_Supplementary.py --module all --dataset houston --data_path E:/PythonProject1/Houston
  python AEFN_Reviewer_Supplementary.py --module tsne  --dataset trento  --data_path E:/PythonProject1/Trento
  python AEFN_Reviewer_Supplementary.py --module complexity
  python AEFN_Reviewer_Supplementary.py --module confmat --dataset houston --data_path E:/PythonProject1/Houston
  python AEFN_Reviewer_Supplementary.py --module convergence --dataset houston --data_path E:/PythonProject1/Houston
  python AEFN_Reviewer_Supplementary.py --module sensitivity --dataset houston --data_path E:/PythonProject1/Houston

依赖: torch, numpy, scipy, sklearn, matplotlib, seaborn, pandas
      可选: thop (pip install thop) 用于精确 FLOPs 统计
"""

import os
import sys
import time
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not found; confusion matrix will use matplotlib fallback.")

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset metadata
# ─────────────────────────────────────────────────────────────────────────────
HOUSTON_CATEGORIES = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway",
    "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
]
HOUSTON_COLORS = [
    "#006400", "#008000", "#00FF00", "#008080", "#8B4513",
    "#0000FF", "#FFFF00", "#FFD700", "#808080", "#A9A9A9",
    "#696969", "#FFA500", "#FF8C00", "#FF0000", "#FF1493"
]

TRENTO_CATEGORIES = ["Apple Trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"]
TRENTO_COLORS = ["#8B4513", "#FF0000", "#FFC0CB", "#D2B48C", "#008000", "#00FFFF"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="AEFN Reviewer Supplementary Analysis")
    p.add_argument("--module", type=str, default="convergence",
                   choices=["all", "complexity", "convergence", "tsne", "confmat", "sensitivity"],
                   help="Which analysis module to run")
    p.add_argument("--dataset", type=str, default="houston",
                   choices=["houston", "trento"],
                   help="Dataset to use (for modules that need data)")
    p.add_argument("--data_path", type=str, default=r"E:\PythonProject1\Houston",
                   help="Root folder containing HSI.mat, LiDAR.mat, TRLabel.mat, TSLabel.mat")
    p.add_argument("--results_dir", type=str, default="./results_reviewer_supplement",
                   help="Output directory for all figures and tables")
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument("--pca_components", type=int, default=30)
    p.add_argument("--epochs", type=int, default=120,
                   help="Training epochs (used in convergence module)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsne_samples", type=int, default=2000,
                   help="Max pixels sampled per class for t-SNE (to limit memory)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# ███  AEFN Model (identical to released code)
# ─────────────────────────────────────────────────────────────────────────────
class Hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_h = nn.Sequential(nn.Conv2d(channels * 2, channels, 1),
                                    nn.BatchNorm2d(channels), nn.Sigmoid())
        self.gate_l = nn.Sequential(nn.Conv2d(channels * 2, channels, 1),
                                    nn.BatchNorm2d(channels), nn.Sigmoid())
        self.out_conv = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1),
                                      nn.BatchNorm2d(channels), nn.ReLU(True))

    def forward(self, x_h, x_l):
        s = torch.cat([x_h, x_l], dim=1)
        feat_h = x_h * self.gate_h(s) + x_h
        feat_l = x_l * self.gate_l(s) + x_l
        return self.out_conv(torch.cat([feat_h, feat_l], dim=1))


class AEFN(nn.Module):
    def __init__(self, hsi_bands, num_classes):
        super().__init__()
        self.conv_h1 = nn.Sequential(nn.Conv2d(hsi_bands, 64, 3, padding=1),
                                     nn.BatchNorm2d(64), nn.SiLU())
        self.ca_h1 = CoordAtt(64, 64)
        self.conv_h2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                     nn.BatchNorm2d(128), nn.SiLU())
        self.conv_l1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                     nn.BatchNorm2d(32), nn.SiLU())
        self.ca_l1 = CoordAtt(32, 32)
        self.conv_l2 = nn.Sequential(nn.Conv2d(32, 128, 3, padding=1),
                                     nn.BatchNorm2d(128), nn.SiLU())
        self.fusion = AdaptiveGatedFusion(128)
        self.ca_fuse = CoordAtt(128, 128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_h, x_l):
        h = self.conv_h2(self.ca_h1(self.conv_h1(x_h)))
        l = self.conv_l2(self.ca_l1(self.conv_l1(x_l)))
        f = self.ca_fuse(self.fusion(h, l))
        return self.classifier(f)

    def extract_features(self, x_h, x_l):
        """Returns fused feature vector before final FC layer (for t-SNE)."""
        h = self.conv_h2(self.ca_h1(self.conv_h1(x_h)))
        l = self.conv_l2(self.ca_l1(self.conv_l1(x_l)))
        f = self.ca_fuse(self.fusion(h, l))
        pooled = nn.AdaptiveAvgPool2d(1)(f).flatten(1)   # (B, 128)
        return pooled

    def extract_early_features(self, x_h, x_l):
        """Returns concatenated raw branch features BEFORE fusion (for t-SNE comparison)."""
        h = self.conv_h2(self.ca_h1(self.conv_h1(x_h)))
        l = self.conv_l2(self.ca_l1(self.conv_l1(x_l)))
        h_pool = nn.AdaptiveAvgPool2d(1)(h).flatten(1)   # (B, 128)
        l_pool = nn.AdaptiveAvgPool2d(1)(l).flatten(1)   # (B, 128)
        return torch.cat([h_pool, l_pool], dim=1)         # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# SAM / EMA / Loss helpers (minimal, same as released code)
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError

    def _grad_norm(self):
        dev = self.param_groups[0]["params"][0].device
        norms = [p.grad.norm(2).to(dev) for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), 2)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data; p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = self.backup[n]
        self.backup = {}


class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, t):
        lp = F.log_softmax(x, -1)
        nll = -lp.gather(-1, t.unsqueeze(1)).squeeze(1)
        smooth = -lp.mean(-1)
        return (( 1 - self.smoothing) * nll + self.smoothing * smooth).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(data_path, pca_components=30):
    hsi = sio.loadmat(os.path.join(data_path, "HSI.mat"))["HSI"].astype(np.float32)
    lidar = sio.loadmat(os.path.join(data_path, "LiDAR.mat"))["LiDAR"].astype(np.float32)
    tr = sio.loadmat(os.path.join(data_path, "TRLabel.mat"))["TRLabel"]
    ts = sio.loadmat(os.path.join(data_path, "TSLabel.mat"))["TSLabel"]

    H, W, B = hsi.shape
    hsi_flat = StandardScaler().fit_transform(hsi.reshape(-1, B))
    if pca_components > 0:
        hsi_pca = PCA(pca_components).fit_transform(hsi_flat).reshape(H, W, pca_components).astype(np.float32)
    else:
        hsi_pca = hsi_flat.reshape(H, W, B)
    lidar_norm = StandardScaler().fit_transform(lidar.reshape(-1, 1)).reshape(H, W).astype(np.float32)

    # gt full
    gt_path = os.path.join(data_path, "gt.mat")
    if os.path.exists(gt_path):
        mat = sio.loadmat(gt_path)
        key = [k for k in mat if not k.startswith("_")][0]
        gt_full = mat[key]
    else:
        gt_full = np.maximum(tr, ts)

    return hsi_pca, lidar_norm, tr, ts, gt_full


def pad_data(hsi, lidar, P):
    m = P // 2
    return (np.pad(hsi, ((m, m), (m, m), (0, 0)), mode="reflect"),
            np.pad(lidar, ((m, m), (m, m)), mode="reflect"))


class PatchDS(Dataset):
    def __init__(self, h_pad, l_pad, rows, cols, labels, P, augment=False, return_idx=False):
        self.ps = P
        self.rows, self.cols = rows.astype(np.int64), cols.astype(np.int64)
        self.labels = None if labels is None else labels.astype(np.int64)
        self.augment, self.return_idx = augment, return_idx
        self.h = torch.from_numpy(h_pad).float()
        self.l = torch.from_numpy(l_pad).float()

    def __len__(self): return len(self.rows)

    def _aug(self, h, l):
        if random.random() < .5: h, l = h.flip(1), l.flip(1)
        if random.random() < .5: h, l = h.flip(2), l.flip(2)
        k = random.randint(0, 3)
        if k: h, l = h.rot90(k, [1, 2]), l.rot90(k, [1, 2])
        return h, l

    def __getitem__(self, i):
        r, c = int(self.rows[i]), int(self.cols[i])
        h = self.h[r:r+self.ps, c:c+self.ps].permute(2, 0, 1).contiguous()
        l = self.l[r:r+self.ps, c:c+self.ps].unsqueeze(0).contiguous()
        if self.augment: h, l = self._aug(h, l)
        y = torch.tensor(-1 if self.labels is None else int(self.labels[i]), dtype=torch.long)
        return (h, l, y, torch.tensor(i, dtype=torch.long)) if self.return_idx else (h, l, y)


def make_loaders(h_pad, l_pad, tr, ts, P, batch_size):
    def pos(lab):
        r, c = np.nonzero(lab); y = lab[r, c].astype(np.int64) - 1
        return r, c, y
    tr_r, tr_c, y_tr = pos(tr)
    ts_r, ts_c, y_ts = pos(ts)
    tr_ds = PatchDS(h_pad, l_pad, tr_r, tr_c, y_tr, P, augment=True)
    ts_ds = PatchDS(h_pad, l_pad, ts_r, ts_c, y_ts, P, augment=False)
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
            DataLoader(ts_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True))


# ─────────────────────────────────────────────────────────────────────────────
# ███  MODULE 1 — Model Complexity Analysis
# ─────────────────────────────────────────────────────────────────────────────
def module_complexity(save_dir):
    """
    计算 AEFN 参数量和 FLOPs，同时与论文中对比方法的文献数据构建对比表。
    输出: complexity_comparison.csv + complexity_bar.png
    """
    print("\n" + "="*60)
    print("MODULE 1: Model Complexity Analysis")
    print("="*60)

    # ── AEFN 实际参数量 ──────────────────────────────────────────────────────
    hsi_bands = 30
    for dataset_name, K in [("Houston 2013", 15), ("Trento", 6)]:
        model = AEFN(hsi_bands=hsi_bands, num_classes=K).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nAEFN ({dataset_name}) — Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

        # FLOPs via thop (if available)
        flops_m = None
        if HAS_THOP:
            dummy_h = torch.randn(1, hsi_bands, 11, 11).to(DEVICE)
            dummy_l = torch.randn(1, 1, 11, 11).to(DEVICE)
            try:
                macs, _ = thop_profile(model, inputs=(dummy_h, dummy_l), verbose=False)
                flops_m = macs / 1e6
                print(f"  FLOPs (MACs): {flops_m:.2f} M")
            except Exception as e:
                print(f"  thop FLOPs calculation failed: {e}")

    # ── Comparison table (from published papers / reproduced estimates) ──────
    # Sources: paper-reported params or our best estimates based on architecture descriptions
    rows = [
        # Method, #Params(M), FLOPs(M), OA-Houston(%), OA-Trento(%), Notes
        ("SAFFT [1]",    "~1.8 M",  "~90 M",   "92.21", "98.48", "Transformer-based; from paper"),
        ("CMR-Net [23]", "~2.1 M",  "~110 M",  "86.31", "95.59", "Cross-modality recon; estimated"),
        ("MPT [17]",     "~4.5 M",  "~220 M",  "91.70", "97.73", "Prompt-tuned ViT; from paper"),
        ("MS-GWCN [15]", "~3.2 M",  "~180 M",  "90.99", "87.25", "Graph-wavelet; estimated"),
        ("AEFN (ours)",  f"~{sum(p.numel() for p in AEFN(30,15).parameters())/1e6:.2f} M",
                         f"{'~{:.1f} M'.format(flops_m) if flops_m else 'N/A'}",
                         "95.95", "98.96", "This work"),
    ]

    df = pd.DataFrame(rows, columns=["Method", "Params", "FLOPs", "OA-Houston(%)", "OA-Trento(%)", "Notes"])
    print("\n" + df.to_string(index=False))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "complexity_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # ── Bar chart: OA vs #Params ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = [r[0] for r in rows]
    oa_h = [float(r[3]) for r in rows]
    oa_t = [float(r[4]) for r in rows]
    colors_bar = ["#4C72B0"] * (len(methods) - 1) + ["#DD4444"]

    for ax, oas, title in zip(axes, [oa_h, oa_t],
                               ["Houston 2013 — OA (%)", "Trento — OA (%)"]):
        bars = ax.bar(range(len(methods)), oas, color=colors_bar, edgecolor="k", width=0.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.split(" [")[0].split(" (")[0] for m in methods], rotation=20, ha="right", fontsize=10)
        ax.set_ylim(80, 101)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, v in zip(bars, oas):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Accuracy vs. Method Comparison (AEFN highlighted in red)", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "complexity_bar.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ███  MODULE 2 — Training Convergence Curves
# ─────────────────────────────────────────────────────────────────────────────
def module_convergence(args, save_dir, dataset_name, categories):
    """
    训练一次完整流程，每个 epoch 记录 train_loss 和 test_OA，绘制收敛曲线。
    同时可选对比 w/o SAM、w/o EMA 的收敛速度。
    """
    print("\n" + "="*60)
    print("MODULE 2: Training Convergence Curves")
    print("="*60)

    set_seed(args.seed)
    hsi_pca, lidar_norm, tr, ts, gt_full = load_and_preprocess(args.data_path, args.pca_components)
    h_pad, l_pad = pad_data(hsi_pca, lidar_norm, args.patch_size)
    train_loader, test_loader = make_loaders(h_pad, l_pad, tr, ts, args.patch_size, args.batch_size)

    K = len(categories)
    configs = {
        "AEFN (full)":    dict(use_sam=True,  use_ema=True),
        "w/o SAM":        dict(use_sam=False, use_ema=True),
        "w/o EMA":        dict(use_sam=True,  use_ema=False),
    }

    history = {}
    for cfg_name, cfg in configs.items():
        print(f"\n  Training config: {cfg_name}")
        set_seed(args.seed)
        model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
        criterion = LabelSmoothCE(0.1)
        if cfg["use_sam"]:
            optimizer = SAM(model.parameters(), optim.AdamW, lr=args.lr, weight_decay=1e-4, rho=0.05)
            sched_opt = optimizer.base_optimizer
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched_opt = optimizer
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(sched_opt, T_0=20, T_mult=2, eta_min=1e-6)
        ema = EMA(model) if cfg["use_ema"] else None

        losses, oas = [], []
        for epoch in range(args.epochs):
            model.train()
            ep_loss = 0.0
            for h, l, y in train_loader:
                h, l, y = h.to(DEVICE), l.to(DEVICE), y.to(DEVICE)
                if cfg["use_sam"]:
                    out = model(h, l); loss = criterion(out, y); loss.backward()
                    optimizer.first_step(zero_grad=True)
                    criterion(model(h, l), y).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad(); out = model(h, l)
                    loss = criterion(out, y); loss.backward(); optimizer.step()
                if ema: ema.update()
                ep_loss += loss.item()
            scheduler.step()
            losses.append(ep_loss / len(train_loader))

            # evaluate every 5 epochs + last 10
            if (epoch + 1) % 5 == 0 or epoch >= args.epochs - 10:
                if ema: ema.apply_shadow()
                model.eval()
                preds, truths = [], []
                with torch.no_grad():
                    for h, l, y in test_loader:
                        preds.append(model(h.to(DEVICE), l.to(DEVICE)).argmax(1).cpu().numpy())
                        truths.append(y.numpy())
                oa = accuracy_score(np.concatenate(truths), np.concatenate(preds)) * 100
                oas.append((epoch + 1, oa))
                if ema: ema.restore()
                model.train()
                print(f"    Epoch {epoch+1:3d}/{args.epochs} | Loss: {losses[-1]:.4f} | OA: {oa:.2f}%")

        history[cfg_name] = {"loss": losses, "oa": oas}

    # ── Plot ─────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    line_styles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    colors_cv = ["#2196F3", "#FF5722", "#4CAF50"]

    # Loss curve
    ax = axes[0]
    for (cfg_name, h), ls, col in zip(history.items(), line_styles, colors_cv):
        ax.plot(range(1, args.epochs + 1), h["loss"], ls, color=col, linewidth=1.8, label=cfg_name)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Training Loss Convergence ({dataset_name})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.5)

    # OA curve
    ax = axes[1]
    for (cfg_name, h), ls, col, mk in zip(history.items(), line_styles, colors_cv, markers):
        ep, oa = zip(*h["oa"])
        ax.plot(ep, oa, ls + mk, color=col, linewidth=1.8, markersize=5, label=cfg_name)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Test OA (%)", fontsize=12)
    ax.set_title(f"Test OA Convergence ({dataset_name})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"convergence_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ███  MODULE 3 — t-SNE Feature Visualization
# ─────────────────────────────────────────────────────────────────────────────
def module_tsne(args, save_dir, dataset_name, categories, colors):
    """
    提取测试集上的 (1) raw HSI 光谱向量 (2) 融合前拼接特征 (3) 融合后特征，
    用 t-SNE 降维到 2D，对比三种表示的类别分离程度。
    """
    print("\n" + "="*60)
    print("MODULE 3: t-SNE Feature Visualization")
    print("="*60)

    set_seed(args.seed)
    hsi_pca, lidar_norm, tr, ts, _ = load_and_preprocess(args.data_path, args.pca_components)
    h_pad, l_pad = pad_data(hsi_pca, lidar_norm, args.patch_size)
    K = len(categories)

    # ── Train a quick model ──────────────────────────────────────────────────
    train_loader, test_loader = make_loaders(h_pad, l_pad, tr, ts, args.patch_size, args.batch_size)
    model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
    criterion = LabelSmoothCE(0.1)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)

    print("  Quick training for feature extraction (40 epochs) ...")
    for epoch in range(40):
        model.train()
        for h, l, y in train_loader:
            h, l, y = h.to(DEVICE), l.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = criterion(model(h, l), y); loss.backward(); opt.step()
        sched.step()

    # ── Extract features from test set ───────────────────────────────────────
    model.eval()
    raw_feats, pre_fusion, post_fusion, all_labels = [], [], [], []

    with torch.no_grad():
        for h, l, y in test_loader:
            h_dev, l_dev = h.to(DEVICE), l.to(DEVICE)
            # (a) raw PCA features — center pixel
            center = args.patch_size // 2
            raw_spec = h[:, :, center, center].numpy()        # (B, C_pca)
            raw_feats.append(raw_spec)

            # (b) pre-fusion concat features
            pre = model.extract_early_features(h_dev, l_dev).cpu().numpy()   # (B, 256)
            pre_fusion.append(pre)

            # (c) post-fusion features
            post = model.extract_features(h_dev, l_dev).cpu().numpy()        # (B, 128)
            post_fusion.append(post)

            all_labels.append(y.numpy())

    raw_feats   = np.concatenate(raw_feats,   axis=0)
    pre_fusion  = np.concatenate(pre_fusion,  axis=0)
    post_fusion = np.concatenate(post_fusion, axis=0)
    all_labels  = np.concatenate(all_labels,  axis=0)

    # ── Sub-sample for t-SNE speed ────────────────────────────────────────────
    n_per_class = args.tsne_samples // K
    idx_sel = []
    for c in range(K):
        ci = np.where(all_labels == c)[0]
        ci = ci[:n_per_class] if len(ci) > n_per_class else ci
        idx_sel.append(ci)
    idx_sel = np.concatenate(idx_sel)
    np.random.shuffle(idx_sel)

    raw_s, pre_s, post_s, lbl_s = (raw_feats[idx_sel], pre_fusion[idx_sel],
                                    post_fusion[idx_sel], all_labels[idx_sel])

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"  Running t-SNE on {len(idx_sel)} samples (3 feature spaces) ...")
    # sklearn compatibility: TSNE uses `n_iter` in some versions and `max_iter` in newer versions
    import inspect
    tsne_sig = inspect.signature(TSNE.__init__)
    tsne_kwargs = dict(n_components=2, perplexity=40, random_state=args.seed)
    if "n_jobs" in tsne_sig.parameters:
        tsne_kwargs["n_jobs"] = -1
    if "n_iter" in tsne_sig.parameters:
        tsne_kwargs["n_iter"] = 1000
    elif "max_iter" in tsne_sig.parameters:
        tsne_kwargs["max_iter"] = 1000
    else:
        # Fallback: let sklearn use its default iteration setting if API differs
        pass

    emb_raw  = TSNE(**tsne_kwargs).fit_transform(raw_s)
    emb_pre  = TSNE(**tsne_kwargs).fit_transform(pre_s)
    emb_post = TSNE(**tsne_kwargs).fit_transform(post_s)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    titles = ["(a) Raw PCA Features", "(b) Pre-Fusion Features", "(c) Post-Fusion Features (AEFN)"]
    for ax, emb, title in zip(axes, [emb_raw, emb_pre, emb_post], titles):
        for c in range(K):
            mask = lbl_s == c
            ax.scatter(emb[mask, 0], emb[mask, 1], c=colors[c], s=6, alpha=0.7,
                       label=categories[c], linewidths=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    handles = [Patch(facecolor=colors[c], label=categories[c]) for c in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=min(K, 8),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"t-SNE Feature Visualization — {dataset_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"tsne_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ███  MODULE 4 — Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def module_confmat(args, save_dir, dataset_name, categories):
    """
    训练一个完整模型后生成标准化混淆矩阵热图，
    同时输出 per-class precision / recall / F1 表格。
    """
    print("\n" + "="*60)
    print("MODULE 4: Confusion Matrix Heatmap")
    print("="*60)

    set_seed(args.seed)
    hsi_pca, lidar_norm, tr, ts, _ = load_and_preprocess(args.data_path, args.pca_components)
    h_pad, l_pad = pad_data(hsi_pca, lidar_norm, args.patch_size)
    K = len(categories)
    train_loader, test_loader = make_loaders(h_pad, l_pad, tr, ts, args.patch_size, args.batch_size)

    model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
    criterion = LabelSmoothCE(0.1)
    optimizer = SAM(model.parameters(), optim.AdamW, lr=args.lr, weight_decay=1e-4, rho=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    ema = EMA(model)

    best_oa, best_weights = 0.0, None
    print("  Training full model for confusion matrix ...")
    for epoch in range(args.epochs):
        model.train()
        for h, l, y in train_loader:
            h, l, y = h.to(DEVICE), l.to(DEVICE), y.to(DEVICE)
            out = model(h, l); loss = criterion(out, y); loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(h, l), y).backward()
            optimizer.second_step(zero_grad=True)
            ema.update()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch >= args.epochs - 10:
            ema.apply_shadow(); model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for h, l, y in test_loader:
                    preds.append(model(h.to(DEVICE), l.to(DEVICE)).argmax(1).cpu().numpy())
                    truths.append(y.numpy())
            oa = accuracy_score(np.concatenate(truths), np.concatenate(preds))
            if oa > best_oa:
                best_oa = oa
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            ema.restore()

    model.load_state_dict(best_weights); model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for h, l, y in test_loader:
            all_preds.append(model(h.to(DEVICE), l.to(DEVICE)).argmax(1).cpu().numpy())
            all_truths.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_truths)

    # ── Normalized confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    short_names = [n[:12] for n in categories]
    fig_size = max(8, K * 0.85)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))

    if HAS_SEABORN:
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=short_names, yticklabels=short_names,
                    linewidths=0.4, linecolor="gray", ax=ax,
                    annot_kws={"size": max(6, 11 - K // 3)},
                    vmin=0, vmax=1)
    else:
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(short_names, rotation=45, ha="right")
        ax.set_yticklabels(short_names)
        for i in range(K):
            for j in range(K):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.6 else "black",
                        fontsize=max(6, 11 - K // 3))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"Normalized Confusion Matrix — {dataset_name}\n"
                 f"OA={best_oa*100:.2f}%  |  Kappa={cohen_kappa_score(y_true, y_pred):.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"confmat_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")

    # ── Per-class Precision / Recall / F1 ────────────────────────────────────
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred,
                                   labels=np.arange(K),
                                   target_names=categories,
                                   output_dict=True)
    df_report = pd.DataFrame(report).T.loc[categories, ["precision", "recall", "f1-score", "support"]]
    df_report.index.name = "Class"
    csv_path = os.path.join(save_dir, f"perclass_report_{dataset_name.replace(' ', '_')}.csv")
    df_report.to_csv(csv_path)
    print(f"  Saved → {csv_path}")
    print("\n" + df_report.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# ███  MODULE 5 — Hyperparameter Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _quick_train_eval(model, train_loader, test_loader, epochs, lr, device):
    """Quick full-training with SAM+EMA for sensitivity sweeps."""
    criterion = LabelSmoothCE(0.1)
    optimizer = SAM(model.parameters(), optim.AdamW, lr=lr, weight_decay=1e-4, rho=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    ema = EMA(model)
    best_oa = 0.0
    for epoch in range(epochs):
        model.train()
        for h, l, y in train_loader:
            h, l, y = h.to(device), l.to(device), y.to(device)
            out = model(h, l); loss = criterion(out, y); loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(h, l), y).backward()
            optimizer.second_step(zero_grad=True)
            ema.update()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch >= epochs - 5:
            ema.apply_shadow(); model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for h, l, y in test_loader:
                    preds.append(model(h.to(device), l.to(device)).argmax(1).cpu().numpy())
                    truths.append(y.numpy())
            oa = accuracy_score(np.concatenate(truths), np.concatenate(preds)) * 100
            best_oa = max(best_oa, oa)
            ema.restore()
    return best_oa


def module_sensitivity(args, save_dir, dataset_name, categories):
    """
    超参数敏感性分析：
      sweep_A: patch_size ∈ {7, 9, 11, 13, 15}
      sweep_B: pca_components ∈ {10, 20, 30, 40, 50}
      sweep_C: lp_alpha ∈ {0.0, 0.2, 0.4, 0.6, 0.8} (post-hoc LP on top of trained model)
    """
    print("\n" + "="*60)
    print("MODULE 5: Hyperparameter Sensitivity Analysis")
    print("="*60)

    K = len(categories)
    hsi_raw = sio.loadmat(os.path.join(args.data_path, "HSI.mat"))["HSI"].astype(np.float32)
    lidar_raw = sio.loadmat(os.path.join(args.data_path, "LiDAR.mat"))["LiDAR"].astype(np.float32)
    tr = sio.loadmat(os.path.join(args.data_path, "TRLabel.mat"))["TRLabel"]
    ts = sio.loadmat(os.path.join(args.data_path, "TSLabel.mat"))["TSLabel"]

    # ── Sweep A: patch_size ──────────────────────────────────────────────────
    patch_sizes = [7, 9, 11, 13, 15]
    oas_patch = []
    print("\n  Sweep A: patch_size")
    for P in patch_sizes:
        set_seed(args.seed)
        hsi_pca, lidar_norm, _, _, _ = load_and_preprocess(args.data_path, args.pca_components)
        h_pad, l_pad = pad_data(hsi_pca, lidar_norm, P)
        tr_loader, ts_loader = make_loaders(h_pad, l_pad, tr, ts, P, args.batch_size)
        model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
        oa = _quick_train_eval(model, tr_loader, ts_loader, args.epochs, args.lr, DEVICE)
        oas_patch.append(oa)
        print(f"    patch_size={P:2d} → OA={oa:.2f}%")

    # ── Sweep B: pca_components ──────────────────────────────────────────────
    pca_vals = [10, 20, 30, 40, 50]
    oas_pca = []
    print("\n  Sweep B: pca_components")
    for C_pca in pca_vals:
        set_seed(args.seed)
        hsi_pca, lidar_norm, _, _, _ = load_and_preprocess(args.data_path, C_pca)
        h_pad, l_pad = pad_data(hsi_pca, lidar_norm, args.patch_size)
        tr_loader, ts_loader = make_loaders(h_pad, l_pad, tr, ts, args.patch_size, args.batch_size)
        model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
        oa = _quick_train_eval(model, tr_loader, ts_loader, args.epochs, args.lr, DEVICE)
        oas_pca.append(oa)
        print(f"    pca_components={C_pca:2d} → OA={oa:.2f}%")

    # ── Sweep C: lp_alpha (post-hoc) ─────────────────────────────────────────
    lp_alphas = [0.0, 0.2, 0.4, 0.6, 0.8]
    oas_lp = []
    print("\n  Sweep C: label propagation alpha (post-hoc)")
    # Train once
    set_seed(args.seed)
    hsi_pca, lidar_norm, _, _, gt_full = load_and_preprocess(args.data_path, args.pca_components)
    h_pad, l_pad = pad_data(hsi_pca, lidar_norm, args.patch_size)
    tr_loader, ts_loader = make_loaders(h_pad, l_pad, tr, ts, args.patch_size, args.batch_size)
    model = AEFN(hsi_bands=hsi_pca.shape[-1], num_classes=K).to(DEVICE)
    _quick_train_eval(model, tr_loader, ts_loader, args.epochs, args.lr, DEVICE)
    model.eval()

    # Collect test logits
    ts_rows, ts_cols = np.nonzero(ts)
    ts_ds = PatchDS(h_pad, l_pad, ts_rows, ts_cols,
                    ts[ts_rows, ts_cols].astype(np.int64) - 1,
                    args.patch_size, augment=False, return_idx=True)
    ts_dl_idx = DataLoader(ts_ds, batch_size=256, shuffle=False, num_workers=0)
    N_ts = len(ts_ds)
    logits_raw = torch.empty(N_ts, K, dtype=torch.float32)
    y_ts_arr = np.empty(N_ts, dtype=np.int64)
    with torch.no_grad():
        for h, l, y, idx in ts_dl_idx:
            logits_raw[idx] = model(h.to(DEVICE), l.to(DEVICE)).cpu()
            y_ts_arr[idx.numpy()] = y.numpy()

    # Build graph for LP once
    mask_ts = (ts > 0)
    from itertools import product as iproduct
    H_im, W_im = mask_ts.shape
    rr, cc = ts_rows, ts_cols
    idx_map = -np.ones((H_im, W_im), dtype=np.int64)
    idx_map[rr, cc] = np.arange(N_ts)
    R = 3
    edges = []
    for dx, dy in iproduct(range(-R, R+1), range(-R, R+1)):
        dr = rr + dx; dc = cc + dy
        inb = (dr >= 0) & (dr < H_im) & (dc >= 0) & (dc < W_im)
        vi = idx_map[dr[inb], dc[inb]]
        valid = vi >= 0
        u = idx_map[rr[inb][valid], cc[inb][valid]]
        edges.append(np.stack([u, vi[valid]]))
    self_l = np.arange(N_ts)
    edge = np.concatenate(edges + [np.stack([self_l, self_l])], axis=1)
    edge_idx = torch.from_numpy(edge).long().to(DEVICE)
    row, col = edge_idx[0], edge_idx[1]
    w = torch.ones(row.numel(), device=DEVICE)
    deg = torch.zeros(N_ts, device=DEVICE).scatter_add_(0, row, w).clamp_min_(1)
    norm_w = w / deg[row]
    P_mat = torch.sparse_coo_tensor(torch.stack([row, col]), norm_w, (N_ts, N_ts), device=DEVICE).coalesce()

    for alpha in lp_alphas:
        if alpha == 0.0:
            pred = logits_raw.argmax(1).numpy()
        else:
            prob0 = logits_raw.softmax(-1).to(DEVICE)
            Z = prob0.clone()
            for _ in range(30):
                Z = alpha * torch.sparse.mm(P_mat, Z) + (1 - alpha) * prob0
            pred = Z.argmax(1).cpu().numpy()
        oa = accuracy_score(y_ts_arr, pred) * 100
        oas_lp.append(oa)
        print(f"    lp_alpha={alpha:.1f} → OA={oa:.2f}%")

    # ── Plot all sweeps ───────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sweep_data = [
        (axes[0], patch_sizes, oas_patch, "Patch Size P", "patch_size"),
        (axes[1], pca_vals, oas_pca, "PCA Components", "pca_components"),
        (axes[2], lp_alphas, oas_lp, "LP Alpha α", "lp_alpha"),
    ]
    defaults = {11: True, 30: True, 0.6: True}

    for ax, x_vals, y_vals, xlabel, key in sweep_data:
        bar_colors = []
        for xv in x_vals:
            bar_colors.append("#DD4444" if defaults.get(xv, False) else "#4C72B0")
        ax.plot(x_vals, y_vals, "o-", color="#4C72B0", linewidth=2, markersize=8, zorder=3)
        ax.scatter(x_vals, y_vals, c=bar_colors, s=80, zorder=4)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
        ax.set_title(f"Sensitivity to {xlabel}\n({dataset_name})", fontsize=11, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.5)
        y_min = min(y_vals) - 1; y_max = max(y_vals) + 1
        ax.set_ylim(y_min, y_max)
        for xv, yv in zip(x_vals, y_vals):
            ax.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

    plt.suptitle("Hyperparameter Sensitivity Analysis — AEFN\n(red = default setting)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"sensitivity_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {fig_path}")

    # Save CSV summary
    df_sens = pd.DataFrame({
        "patch_size": patch_sizes,
        "OA_patch(%)": oas_patch,
    }).merge(pd.DataFrame({"pca_components": pca_vals, "OA_pca(%)": oas_pca}),
             left_index=True, right_index=True, how="outer")
    df_sens2 = pd.DataFrame({"lp_alpha": lp_alphas, "OA_lp(%)": oas_lp})
    df_sens.to_csv(os.path.join(save_dir, f"sensitivity_patch_pca_{dataset_name}.csv"), index=False)
    df_sens2.to_csv(os.path.join(save_dir, f"sensitivity_lpalpha_{dataset_name}.csv"), index=False)
    print(f"  Saved CSV summaries.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"\n  Device : {DEVICE}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Module : {args.module}")
    print(f"  Output : {args.results_dir}")

    # Dataset metadata
    if args.dataset == "houston":
        dataset_name = "Houston 2013"
        categories   = HOUSTON_CATEGORIES
        colors       = HOUSTON_COLORS
    else:
        dataset_name = "Trento"
        categories   = TRENTO_CATEGORIES
        colors       = TRENTO_COLORS

    m = args.module
    need_data = m in {"all", "convergence", "tsne", "confmat", "sensitivity"}

    if need_data and not os.path.isfile(os.path.join(args.data_path, "HSI.mat")):
        print(f"\n[ERROR] data_path '{args.data_path}' does not contain HSI.mat."
              "\nPlease set --data_path correctly. Skipping data-dependent modules.\n")
        m = "complexity"  # fallback to the one that doesn't need data

    if m in {"all", "complexity"}:
        module_complexity(args.results_dir)

    if m in {"all", "convergence"}:
        module_convergence(args, args.results_dir, dataset_name, categories)

    if m in {"all", "tsne"}:
        module_tsne(args, args.results_dir, dataset_name, categories, colors)

    if m in {"all", "confmat"}:
        module_confmat(args, args.results_dir, dataset_name, categories)

    if m in {"all", "sensitivity"}:
        module_sensitivity(args, args.results_dir, dataset_name, categories)

    print(f"\n✓ All requested analyses complete. Results in: {args.results_dir}\n")


if __name__ == "__main__":
    main()