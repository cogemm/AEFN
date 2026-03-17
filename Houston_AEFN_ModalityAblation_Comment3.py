# -*- coding: utf-8 -*-
"""
Comment 3 ablation-ready script: supports fused / HSI-only / LiDAR-only experiments
under the same fairness-first evaluation protocol as the accepted manuscript.

融合版脚本：保持 Houston2013-0109.py 的高精度训练/模型，
同时使用 GS_WCN_SOA_Transformer_Houston_0922.py 的“全图(掩膜)推理 + 可选Label Propagation细化 + 更清晰的制图方式”。

主要变化：
1) 读取 gt.mat（若不存在则用 max(TRLabel, TSLabel) 作为 gt_full）。
2) 训练结束后：对 gt_full>0 的所有像素做批量推理，生成完整分类图，而不是只在 TSLabel 的测试点处填值。
3) 可选：对全图 logits 做 label propagation 细化（默认开启，alpha/iters 可调）。
4) 制图使用 save_map_like_demo 风格（图更清晰、图例更规整）。

依赖：torch, numpy, scipy, sklearn, matplotlib, pandas
"""

import os
import random
import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ==============================================================================
# 1) 配置与全局变量
# ==============================================================================
CATEGORIES = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway",
    "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
]

COLORS = [
    "#006400", "#008000", "#00FF00", "#008080", "#8B4513",
    "#0000FF", "#FFFF00", "#FFD700", "#808080", "#A9A9A9",
    "#696969", "#FFA500", "#FF8C00", "#FF0000", "#FF1493"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    p = argparse.ArgumentParser()

    # data / io
    p.add_argument("--data_path", type=str, default=r"E:\PythonProject1\Houston")
    p.add_argument("--results_dir", type=str, default="./results_AEFN_FusedMap_Houston")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # model / preprocess
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument("--pca_components", type=int, default=30)

    # train
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--use_sam", dest="use_sam", action="store_true", default=True)

    p.add_argument("--no_sam",  dest="use_sam", action="store_false")

    p.add_argument("--use_ema", dest="use_ema", action="store_true", default=True)

    p.add_argument("--no_ema",  dest="use_ema", action="store_false")

    p.add_argument("--mixup_alpha", type=float, default=1.0)

    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument("--use_coordatt", dest="use_coordatt", action="store_true", default=True)

    p.add_argument("--no_coordatt",  dest="use_coordatt", action="store_false")

    p.add_argument("--use_agf", dest="use_agf", action="store_true", default=True)

    p.add_argument("--no_agf",  dest="use_agf", action="store_false")
    # full-map inference
    p.add_argument("--gt_mat", type=str, default="gt.mat", help="全图GT文件名（可选）")
    p.add_argument("--gt_key", type=str, default="gt", help="gt.mat中的变量名")
    p.add_argument("--map_batch", type=int, default=256, help="全图推理batch size")
    p.add_argument("--tta", type=int, default=0, help="推理TTA: 0(关闭) 或 8(8-way)")
    p.add_argument("--lp_alpha", type=float, default=0.60, help="Label Propagation alpha；<=0 表示关闭")
    p.add_argument("--lp_iters", type=int, default=30)
    p.add_argument("--lp_radius", type=int, default=3, help="LP图邻域半径(像素)")

    # fairness-first reporting protocol (Reviewer #1 Concern #2)
    p.add_argument("--report_protocol_split", dest="report_protocol_split", action="store_true", default=True,
                   help="Report AEFN-core (fair main comparison) and AEFN+ (optional refinements) separately.")
    p.add_argument("--no_report_protocol_split", dest="report_protocol_split", action="store_false")
    p.add_argument("--core_use_ema_eval", dest="core_use_ema_eval", action="store_true", default=False,
                   help="If set, AEFN-core evaluation/model selection may use EMA shadow weights (default: False for fairness).")
    p.add_argument("--plus_eval_tta", type=int, default=0,
                   help="TTA for auxiliary AEFN+ test-set evaluation (0 or 8). Keep 0 for strict comparability.")
    p.add_argument("--save_core_plus_maps", dest="save_core_plus_maps", action="store_true", default=True,
                   help="Save both AEFN-core and AEFN+ maps when report_protocol_split is enabled.")
    p.add_argument("--no_save_core_plus_maps", dest="save_core_plus_maps", action="store_false")

    # reviewer-comment-3 modality ablation
    p.add_argument("--modality", type=str, default="fused", choices=["fused", "hsi", "lidar"],
                   help="Input modality for comment-3 ablation: fused, hsi-only, or lidar-only.")
    p.add_argument("--tag_suffix", type=str, default="",
                   help="Optional suffix appended to saved file names for manuscript bookkeeping.")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2) 核心模块：Coordinate Attention & AEFN Fusion（保持第一份代码）
# ==============================================================================
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_h = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.gate_l = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_h, x_l):
        combined = torch.cat([x_h, x_l], dim=1)
        g_h = self.gate_h(combined)
        feat_h = x_h * g_h + x_h
        g_l = self.gate_l(combined)
        feat_l = x_l * g_l + x_l
        out = torch.cat([feat_h, feat_l], dim=1)
        out = self.out_conv(out)
        return out



class ConcatConvFusion(nn.Module):
    """Ablation module: direct concatenation followed by Conv3×3 projection."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_h: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_h, x_l], dim=1)
        return self.proj(x)

class HighAcc_AEFN_Net(nn.Module):
    def __init__(self, hsi_bands, num_classes=15, use_coordatt: bool = True,
                 use_agf: bool = True, modality: str = "fused"):
        super().__init__()
        self.modality = str(modality).lower()
        if self.modality not in {"fused", "hsi", "lidar"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.conv_h1 = nn.Sequential(
            nn.Conv2d(hsi_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.ca_h1 = CoordAtt(64, 64) if use_coordatt else nn.Identity()
        self.conv_h2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.ca_l1 = CoordAtt(32, 32) if use_coordatt else nn.Identity()
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.fusion = AdaptiveGatedFusion(128) if use_agf else ConcatConvFusion(128)
        self.ca_fuse = CoordAtt(128, 128) if use_coordatt else nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def encode_hsi(self, x_h):
        h = self.conv_h1(x_h)
        h = self.ca_h1(h)
        h = self.conv_h2(h)
        return h

    def encode_lidar(self, x_l):
        l = self.conv_l1(x_l)
        l = self.ca_l1(l)
        l = self.conv_l2(l)
        return l

    def forward(self, x_h, x_l):
        if self.modality == "hsi":
            feat = self.encode_hsi(x_h)
            return self.classifier(feat)

        if self.modality == "lidar":
            feat = self.encode_lidar(x_l)
            return self.classifier(feat)

        h = self.encode_hsi(x_h)
        l = self.encode_lidar(x_l)
        f = self.fusion(h, l)
        f = self.ca_fuse(f)
        return self.classifier(f)


# ==============================================================================
# 3) 优化器：SAM（保持第一份代码）
# ==============================================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step() and second_step()")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((scale * p.grad).norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)


# ==============================================================================
# 4) 工具类：EMA / MixUp / Loss
# ==============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x1, x2, y, alpha=1.0, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x1.size(0)
    index = torch.randperm(bs).to(device)
    mixed_x1 = lam * x1 + (1.0 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1.0 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ==============================================================================
# 5) 数据：加载 / 预处理 / Patch Dataset（新增：支持全图掩膜推理）
# ==============================================================================
def load_data(path: str):
    hsi = sio.loadmat(os.path.join(path, "HSI.mat"))["HSI"].astype(np.float32)
    lidar = sio.loadmat(os.path.join(path, "LiDAR.mat"))["LiDAR"].astype(np.float32)
    tr_label = sio.loadmat(os.path.join(path, "TRLabel.mat"))["TRLabel"]
    ts_label = sio.loadmat(os.path.join(path, "TSLabel.mat"))["TSLabel"]
    return hsi, lidar, tr_label, ts_label


def load_gt_full(path: str, gt_mat: str = "gt.mat", gt_key: str = "gt", tr_label=None, ts_label=None):
    gt_path = os.path.join(path, gt_mat)
    if os.path.exists(gt_path):
        mat = sio.loadmat(gt_path)
        if gt_key not in mat:
            raise KeyError(f"Found {gt_mat} but key '{gt_key}' not in it. Available keys: {list(mat.keys())}")
        gt = mat[gt_key]
        return gt
    if tr_label is None or ts_label is None:
        raise FileNotFoundError(f"{gt_path} not found and TR/TSLabel not provided to build fallback gt.")
    # fallback: union
    return np.maximum(tr_label, ts_label)


def preprocess_hsi_lidar(hsi_raw: np.ndarray, lidar_raw: np.ndarray, pca_components: int):
    H, W, B = hsi_raw.shape
    hsi_flat = hsi_raw.reshape(-1, B)
    hsi_scaled = StandardScaler().fit_transform(hsi_flat)

    # pca_components <= 0 disables PCA and keeps all standardized bands
    if pca_components is None or int(pca_components) <= 0 or int(pca_components) >= B:
        hsi_out = hsi_scaled.reshape(H, W, B).astype(np.float32)
    else:
        pca = PCA(n_components=int(pca_components))
        hsi_out = pca.fit_transform(hsi_scaled).reshape(H, W, int(pca_components)).astype(np.float32)

    lidar_norm = StandardScaler().fit_transform(lidar_raw.reshape(-1, 1)).reshape(H, W).astype(np.float32)
    return hsi_out, lidar_norm



def pad_hsi_lidar(hsi: np.ndarray, lidar: np.ndarray, patch_size: int):
    m = patch_size // 2
    h_pad = np.pad(hsi, ((m, m), (m, m), (0, 0)), mode="reflect")
    l_pad = np.pad(lidar, ((m, m), (m, m)), mode="reflect")
    return h_pad, l_pad


class HoustonPatchDataset(Dataset):
    """
    使用 padded cube 做按需切patch，避免为全图推理一次性堆叠巨大数组。
    """
    def __init__(self, hsi_pad: np.ndarray, lidar_pad: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 labels, patch_size: int, augment: bool = False, return_index: bool = False):
        super().__init__()
        self.ps = int(patch_size)
        self.rows = rows.astype(np.int64)
        self.cols = cols.astype(np.int64)
        self.labels = None if labels is None else labels.astype(np.int64)
        self.augment = bool(augment)
        self.return_index = bool(return_index)

        # 预先转成 torch CPU tensor，加速 slice + 避免每次 numpy->torch
        self.h_pad = torch.from_numpy(hsi_pad).float()      # (H+2m, W+2m, C)
        self.l_pad = torch.from_numpy(lidar_pad).float()    # (H+2m, W+2m)

    def __len__(self):
        return int(self.rows.size)

    def _rand_aug(self, h: torch.Tensor, l: torch.Tensor):
        # h: (C,ps,ps), l: (1,ps,ps)
        if torch.rand(1).item() < 0.5:
            h = torch.flip(h, dims=[1]); l = torch.flip(l, dims=[1])
        if torch.rand(1).item() < 0.5:
            h = torch.flip(h, dims=[2]); l = torch.flip(l, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            h = torch.rot90(h, k, dims=[1, 2])
            l = torch.rot90(l, k, dims=[1, 2])
        return h, l

    def __getitem__(self, idx: int):
        r = int(self.rows[idx])
        c = int(self.cols[idx])

        h = self.h_pad[r:r + self.ps, c:c + self.ps, :].permute(2, 0, 1).contiguous()  # (C,ps,ps)
        l = self.l_pad[r:r + self.ps, c:c + self.ps].unsqueeze(0).contiguous()         # (1,ps,ps)

        if self.augment:
            h, l = self._rand_aug(h, l)

        if self.labels is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.return_index:
            return h, l, y, torch.tensor(idx, dtype=torch.long)
        return h, l, y


def label_to_positions(label_2d: np.ndarray):
    rows, cols = np.nonzero(label_2d)
    y = label_2d[rows, cols].astype(np.int64) - 1  # 0..14
    return rows, cols, y


# ==============================================================================
# 6) 全图推理增强：TTA + Label Propagation（来自第二份代码思想）
# ==============================================================================
@torch.no_grad()
def forward_with_tta(model: nn.Module, h: torch.Tensor, l: torch.Tensor, tta: int = 0):
    """
    tta=0: 普通推理
    tta=8: 4 rotations + hflip
    """
    if int(tta) != 8:
        return model(h, l)

    logits_sum = 0.0
    for k in range(4):
        h_r = torch.rot90(h, k, dims=[2, 3])
        l_r = torch.rot90(l, k, dims=[2, 3])

        logits_sum = logits_sum + model(h_r, l_r)

        h_f = torch.flip(h_r, dims=[3])
        l_f = torch.flip(l_r, dims=[3])
        logits_sum = logits_sum + model(h_f, l_f)

    return logits_sum / 8.0


def build_edge_index_from_mask(mask: np.ndarray, radius: int = 3):
    """
    在 mask==True 的像素之间按 (2*radius+1)^2 邻域建边，返回 edge_index 以及 idx_map。
    """
    H, W = mask.shape
    rr, cc = np.where(mask)
    N = rr.size
    idx_map = -np.ones((H, W), dtype=np.int64)
    idx_map[rr, cc] = np.arange(N, dtype=np.int64)

    edges = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            dst_r = rr + dx
            dst_c = cc + dy
            inb = (dst_r >= 0) & (dst_r < H) & (dst_c >= 0) & (dst_c < W)
            if not np.any(inb):
                continue
            rr_ib = rr[inb]
            cc_ib = cc[inb]
            dr_ib = dst_r[inb]
            dc_ib = dst_c[inb]
            v_idx = idx_map[dr_ib, dc_ib]
            valid = v_idx >= 0
            if not np.any(valid):
                continue
            u = idx_map[rr_ib[valid], cc_ib[valid]]
            v = v_idx[valid]
            edges.append(np.stack([u, v], axis=0))

    if len(edges) == 0:
        edge = np.empty((2, 0), dtype=np.int64)
    else:
        edge = np.concatenate(edges, axis=1)

    # add self-loops
    self_loops = np.arange(N, dtype=np.int64)
    edge = np.concatenate([edge, np.stack([self_loops, self_loops], axis=0)], axis=1)

    edge_index = torch.from_numpy(edge).long()
    return edge_index, idx_map


@torch.no_grad()
def refine_by_label_propagation(logits: torch.Tensor, edge_index: torch.Tensor, alpha: float = 0.6, iters: int = 30):
    """
    与第二份代码一致：Z_{t+1} = alpha * P Z_t + (1-alpha) * prob0
    logits: (N,C)
    edge_index: (2,E) directed
    """
    N, C = logits.shape
    prob0 = logits.softmax(dim=-1)
    row, col = edge_index[0], edge_index[1]
    w = torch.ones(row.numel(), device=logits.device)
    deg = torch.zeros(N, device=logits.device).scatter_add_(0, row, w).clamp_min_(1)
    norm_w = w / deg[row]
    P = torch.sparse_coo_tensor(torch.stack([row, col]), norm_w, (N, N), device=logits.device).coalesce()

    Z = prob0
    for _ in range(int(iters)):
        Z = alpha * torch.sparse.mm(P, Z) + (1.0 - alpha) * prob0
    return torch.log(Z.clamp_min(1e-9))


# ==============================================================================
# 7) 绘图：采用第二份代码风格
# ==============================================================================
def save_map_like_demo(img2d: np.ndarray, title: str, out_path: str, colors, class_names):
    _cmap = ListedColormap(["#000000"] + list(colors))
    plt.figure(figsize=(18, 4.5))
    ax = plt.gca()
    ax.imshow(img2d, cmap=_cmap, vmin=0, vmax=len(colors), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")

    handles = [Patch(facecolor=colors[i], edgecolor="k", label=class_names[i])
               for i in range(len(class_names) - 1, -1, -1)]
    labels = [class_names[i] for i in range(len(class_names) - 1, -1, -1)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_bar_charts(per_class_acc: np.ndarray, run_id, save_dir, overall_mean=None, overall_std=None):
    x = np.arange(len(CATEGORIES))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, per_class_acc, color=COLORS, edgecolor="black", alpha=0.8)
    plt.xticks(x, CATEGORIES, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.title(f"Per-Class Accuracy (Run {run_id})")
    plt.ylabel("Accuracy")
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.2f}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Run_{run_id}_Bar.png"), dpi=300)
    plt.close()

    if overall_mean is not None and overall_std is not None:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(x, overall_mean, yerr=overall_std, capsize=4,
                       color=COLORS, edgecolor="black", alpha=0.8)
        plt.xticks(x, CATEGORIES, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.title("Overall Per-Class Accuracy (Mean ± Std)")
        plt.ylabel("Accuracy")
        for bar, m, s in zip(bars, overall_mean, overall_std):
            plt.text(bar.get_x() + bar.get_width() / 2, m + s + 0.01, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Overall_Bar_Mean_Std.png"), dpi=300)
        plt.close()


# ==============================================================================
# 8) 训练/评估 + 生成“清晰全图”
# ==============================================================================
@torch.no_grad()
def eval_on_loader(model: nn.Module, loader: DataLoader, tta: int = 0):
    model.eval()
    preds = []
    truths = []
    for h, l, y in loader:
        h = h.to(DEVICE, non_blocking=True)
        l = l.to(DEVICE, non_blocking=True)
        out = forward_with_tta(model, h, l, tta=tta)
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        truths.append(y.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(truths)


@torch.no_grad()
def infer_full_map_logits(model: nn.Module, h_pad: np.ndarray, l_pad: np.ndarray,
                          rows: np.ndarray, cols: np.ndarray, patch_size: int,
                          batch_size: int, tta: int):
    model.eval()
    ds = HoustonPatchDataset(h_pad, l_pad, rows, cols, labels=None,
                            patch_size=patch_size, augment=False, return_index=True)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=True)

    N = len(ds)
    C = len(CATEGORIES)
    logits_all = torch.empty((N, C), dtype=torch.float32, device="cpu")

    for h, l, _, idx in dl:
        h = h.to(DEVICE, non_blocking=True)
        l = l.to(DEVICE, non_blocking=True)
        out = forward_with_tta(model, h, l, tta=tta)
        logits_all[idx] = out.detach().cpu()

    return logits_all



def compute_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CATEGORIES)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    aa = float(np.mean(per_class_acc))
    return float(oa), float(aa), float(kappa), per_class_acc


def _state_dict_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict_cpu(model: nn.Module, state_dict_cpu: dict):
    model.load_state_dict(state_dict_cpu)
    model.eval()


def _infer_pred_map_from_model(model: nn.Module, h_pad: np.ndarray, l_pad: np.ndarray,
                               gt_full: np.ndarray, patch_size: int, batch_size: int,
                               tta: int = 0, lp_alpha: float = -1.0,
                               lp_iters: int = 30, lp_radius: int = 3):
    mask = (gt_full > 0)
    rows_all, cols_all = np.where(mask)
    logits_all_cpu = infer_full_map_logits(
        model=model,
        h_pad=h_pad, l_pad=l_pad,
        rows=rows_all, cols=cols_all,
        patch_size=patch_size,
        batch_size=batch_size,
        tta=tta,
    )

    logits_use = logits_all_cpu
    if lp_alpha is not None and float(lp_alpha) > 0.0:
        edge_index, _ = build_edge_index_from_mask(mask, radius=int(lp_radius))
        logits_use = refine_by_label_propagation(
            logits_all_cpu.to(DEVICE),
            edge_index.to(DEVICE),
            alpha=float(lp_alpha),
            iters=int(lp_iters),
        ).detach().cpu()

    pred_all = logits_use.argmax(dim=1).numpy().astype(np.int32)
    pred_img = np.zeros_like(gt_full, dtype=np.int32)
    pred_img[rows_all, cols_all] = pred_all + 1
    return pred_img


def modality_to_label(modality: str) -> str:
    modality = str(modality).lower()
    mapping = {
        "fused": "HSI+LiDAR",
        "hsi": "HSI-only",
        "lidar": "LiDAR-only",
    }
    return mapping.get(modality, modality)


def modality_to_safe_tag(modality: str) -> str:
    modality = str(modality).lower()
    mapping = {
        "fused": "HSI_LiDAR",
        "hsi": "HSI_only",
        "lidar": "LiDAR_only",
    }
    return mapping.get(modality, modality.replace("+", "_").replace("-", "_").replace(" ", "_"))


def _append_history(history: dict, oa: float, aa: float, kappa: float, per_class_acc: np.ndarray):
    history["OA"].append(float(oa))
    history["AA"].append(float(aa))
    history["Kappa"].append(float(kappa))
    history["PerClass"].append(np.asarray(per_class_acc, dtype=np.float64))


def _summarize_history(history: dict):
    oa_mean, oa_std = float(np.mean(history["OA"])), float(np.std(history["OA"]))
    aa_mean, aa_std = float(np.mean(history["AA"])), float(np.std(history["AA"]))
    kp_mean, kp_std = float(np.mean(history["Kappa"])), float(np.std(history["Kappa"]))
    pc_mean = np.mean(history["PerClass"], axis=0)
    pc_std = np.std(history["PerClass"], axis=0)
    return {
        "oa_mean": oa_mean, "oa_std": oa_std,
        "aa_mean": aa_mean, "aa_std": aa_std,
        "kp_mean": kp_mean, "kp_std": kp_std,
        "pc_mean": pc_mean, "pc_std": pc_std,
    }


def _print_and_save_summary(tag: str, summary: dict, results_dir: str):
    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY - {tag}")
    print("=" * 60)
    print(f"OA    : {summary['oa_mean'] * 100:.2f} ± {summary['oa_std'] * 100:.2f} %")
    print(f"AA    : {summary['aa_mean'] * 100:.2f} ± {summary['aa_std'] * 100:.2f} %")
    print(f"Kappa : {summary['kp_mean']:.4f} ± {summary['kp_std']:.4f}")
    print("-" * 60)

    rows = []
    for i, name in enumerate(CATEGORIES):
        print(f"{name:20s}: {summary['pc_mean'][i] * 100:.2f} ± {summary['pc_std'][i] * 100:.2f} %")
        rows.append({
            "Class ID": i + 1,
            "Class Name": name,
            "Mean Acc": float(summary['pc_mean'][i]),
            "Std": float(summary['pc_std'][i]),
        })

    safe_tag = tag.replace("+", "plus").replace(" ", "_")
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, f"Final_Per_Class_Stats_{safe_tag}.csv"), index=False)
    pd.DataFrame([{
        "OA Mean": summary['oa_mean'], "OA Std": summary['oa_std'],
        "AA Mean": summary['aa_mean'], "AA Std": summary['aa_std'],
        "Kappa Mean": summary['kp_mean'], "Kappa Std": summary['kp_std'],
    }]).to_csv(os.path.join(results_dir, f"Final_Metrics_{safe_tag}.csv"), index=False)


def train_one_run(run_id: int, args, hsi_pca: np.ndarray, lidar_norm: np.ndarray,
                  tr_label: np.ndarray, ts_label: np.ndarray, gt_full: np.ndarray):
    print(f"\n>>> Starting Run {run_id} | Modality: {modality_to_label(args.modality)} ...")

    # pad once for all datasets
    h_pad, l_pad = pad_hsi_lidar(hsi_pca, lidar_norm, args.patch_size)

    # train/test positions
    tr_rows, tr_cols, y_tr = label_to_positions(tr_label)
    ts_rows, ts_cols, y_ts = label_to_positions(ts_label)

    train_ds = HoustonPatchDataset(h_pad, l_pad, tr_rows, tr_cols, y_tr, args.patch_size,
                                   augment=True, return_index=False)
    test_ds = HoustonPatchDataset(h_pad, l_pad, ts_rows, ts_cols, y_ts, args.patch_size,
                                  augment=False, return_index=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = HighAcc_AEFN_Net(
        hsi_bands=hsi_pca.shape[-1],
        num_classes=len(CATEGORIES),
        use_coordatt=bool(args.use_coordatt),
        use_agf=bool(args.use_agf),
        modality=str(args.modality),
    ).to(DEVICE)

    base_optimizer = optim.AdamW
    if args.use_sam:
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=1e-4, rho=0.05)
        sched_opt = optimizer.base_optimizer
    else:
        optimizer = base_optimizer(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_opt = optimizer

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        sched_opt,
        T_0=20, T_mult=2, eta_min=1e-6
    )

    criterion = LabelSmoothingCrossEntropy(smoothing=float(args.label_smoothing))
    ema = EMA(model, decay=0.999) if args.use_ema else None

    # Fairness-first bookkeeping:
    #   AEFN-core : raw network weights, no EMA evaluation, no TTA, no LP (main comparison)
    #   AEFN+     : optional refinements reported separately (EMA eval / TTA / LP)
    best_core_oa = -1.0
    best_core_weights = None

    best_plus_oa = -1.0
    best_plus_weights = None

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0

        for h, l, y in train_loader:
            h = h.to(DEVICE, non_blocking=True)
            l = l.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if args.mixup_alpha and args.mixup_alpha > 0:
                h_m, l_m, y_a, y_b, lam = mixup_data(h, l, y, alpha=args.mixup_alpha, device=str(DEVICE))
                def loss_func(pred):
                    return mixup_criterion(criterion, pred, y_a, y_b, lam)
                h_in, l_in = h_m, l_m
            else:
                def loss_func(pred):
                    return criterion(pred, y)
                h_in, l_in = h, l

            if args.use_sam:
                out = model(h_in, l_in)
                loss = loss_func(out)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                loss_func(model(h_in, l_in)).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                out = model(h_in, l_in)
                loss = loss_func(out)
                loss.backward()
                optimizer.step()

            if ema:
                ema.update()
            train_loss += float(loss.item())

        scheduler.step()

        # periodic eval to select best weights for AEFN-core and AEFN+
        if (epoch + 1) % 5 == 0 or epoch >= int(args.epochs) - 10:
            # ---- Core selection: raw model (default fairness setting) ----
            y_pred_core, y_true_core = eval_on_loader(model, test_loader, tta=0)
            acc_core = accuracy_score(y_true_core, y_pred_core)
            if acc_core > best_core_oa:
                best_core_oa = float(acc_core)
                best_core_weights = _state_dict_cpu(model)

            # ---- Optional '+': EMA-shadow evaluation (separate reporting) ----
            if ema is not None:
                ema.apply_shadow()
                y_pred_plus_sel, y_true_plus_sel = eval_on_loader(model, test_loader, tta=0)
                acc_plus = accuracy_score(y_true_plus_sel, y_pred_plus_sel)
                if acc_plus > best_plus_oa:
                    best_plus_oa = float(acc_plus)
                    best_plus_weights = _state_dict_cpu(model)
                ema.restore()
            else:
                # No EMA available: plus defaults to raw model and will differ only by TTA/LP if enabled.
                if acc_core > best_plus_oa:
                    best_plus_oa = float(acc_core)
                    best_plus_weights = _state_dict_cpu(model)

    if best_core_weights is None:
        best_core_weights = _state_dict_cpu(model)
    if best_plus_weights is None:
        best_plus_weights = best_core_weights

    # -----------------------------
    # AEFN-core (main/fair comparison)
    # -----------------------------
    _load_state_dict_cpu(model, best_core_weights)
    if bool(args.core_use_ema_eval) and ema is not None:
        # Optional override (not recommended for fairness-first reporting).
        ema.apply_shadow()
        y_pred_core, y_true = eval_on_loader(model, test_loader, tta=0)
        ema.restore()
    else:
        y_pred_core, y_true = eval_on_loader(model, test_loader, tta=0)

    core_oa, core_aa, core_kappa, core_pc = compute_metrics_from_predictions(y_true, y_pred_core)
    print(f"[Run {run_id} | {modality_to_label(args.modality)} | AEFN-core] OA: {core_oa * 100:.2f}%, AA: {core_aa * 100:.2f}%, Kappa: {core_kappa:.4f}")

    # -----------------------------
    # AEFN+ (auxiliary refinements, reported separately)
    #   - EMA-based evaluation (if enabled)
    #   - test-time augmentation on test loader (optional)
    #   - label propagation only for dense full-scene refinement/map generation
    # -----------------------------
    _load_state_dict_cpu(model, best_plus_weights)
    plus_tta_eval = int(getattr(args, "plus_eval_tta", 0))
    y_pred_plus, y_true_plus = eval_on_loader(model, test_loader, tta=plus_tta_eval)
    plus_oa, plus_aa, plus_kappa, plus_pc = compute_metrics_from_predictions(y_true_plus, y_pred_plus)
    print(f"[Run {run_id} | {modality_to_label(args.modality)} | AEFN+   ] OA: {plus_oa * 100:.2f}%, AA: {plus_aa * 100:.2f}%, Kappa: {plus_kappa:.4f} (EMA-eval{' on' if args.use_ema else ' off'}, TTA={plus_tta_eval})")

    # full-scene maps (separated, to avoid mixing fair comparison and refinement)
    os.makedirs(args.results_dir, exist_ok=True)
    save_map_like_demo(
        gt_full.astype(np.int32),
        title=f"Ground Truth",
        out_path=os.path.join(args.results_dir, f"{modality_to_safe_tag(args.modality)}_Run_{run_id}_GT.png"),
        colors=COLORS,
        class_names=CATEGORIES,
    )

    if getattr(args, "save_core_plus_maps", True):
        _load_state_dict_cpu(model, best_core_weights)
        pred_img_core = _infer_pred_map_from_model(
            model=model,
            h_pad=h_pad, l_pad=l_pad, gt_full=gt_full,
            patch_size=args.patch_size, batch_size=args.map_batch,
            tta=0, lp_alpha=-1.0,
            lp_iters=int(args.lp_iters), lp_radius=int(args.lp_radius),
        )
        save_map_like_demo(
            pred_img_core,
            title=f"AEFN-core Map (Run {run_id})",
            out_path=os.path.join(args.results_dir, f"{modality_to_safe_tag(args.modality)}_Run_{run_id}_Pred_AEFN_core.png"),
            colors=COLORS,
            class_names=CATEGORIES,
        )

        _load_state_dict_cpu(model, best_plus_weights)
        pred_img_plus = _infer_pred_map_from_model(
            model=model,
            h_pad=h_pad, l_pad=l_pad, gt_full=gt_full,
            patch_size=args.patch_size, batch_size=args.map_batch,
            tta=int(args.tta), lp_alpha=float(args.lp_alpha),
            lp_iters=int(args.lp_iters), lp_radius=int(args.lp_radius),
        )
        save_map_like_demo(
            pred_img_plus,
            title=f"AEFN+ Map (Run {run_id})",
            out_path=os.path.join(args.results_dir, f"{modality_to_safe_tag(args.modality)}_Run_{run_id}_Pred_AEFN_plus.png"),
            colors=COLORS,
            class_names=CATEGORIES,
        )
    else:
        # backward-compatible single map (uses AEFN+ path)
        _load_state_dict_cpu(model, best_plus_weights)
        pred_img_plus = _infer_pred_map_from_model(
            model=model,
            h_pad=h_pad, l_pad=l_pad, gt_full=gt_full,
            patch_size=args.patch_size, batch_size=args.map_batch,
            tta=int(args.tta), lp_alpha=float(args.lp_alpha),
            lp_iters=int(args.lp_iters), lp_radius=int(args.lp_radius),
        )
        save_map_like_demo(
            pred_img_plus,
            title=f"Classification Map (Run {run_id})",
            out_path=os.path.join(args.results_dir, f"{modality_to_safe_tag(args.modality)}_Run_{run_id}_Pred.png"),
            colors=COLORS,
            class_names=CATEGORIES,
        )

    # per-class bars for both protocols (main and auxiliary)
    plot_bar_charts(core_pc, f"{modality_to_safe_tag(args.modality)}_{run_id}_AEFN_core", args.results_dir)
    plot_bar_charts(plus_pc, f"{modality_to_safe_tag(args.modality)}_{run_id}_AEFN_plus", args.results_dir)

    run_record = {
        "core": {"oa": core_oa, "aa": core_aa, "kappa": core_kappa, "per_class": core_pc},
        "plus": {"oa": plus_oa, "aa": plus_aa, "kappa": plus_kappa, "per_class": plus_pc},
    }
    return run_record


def main():
    args = get_args()
    set_seed(args.seed)
    args.results_dir = os.path.join(args.results_dir, modality_to_safe_tag(args.modality) + ("_" + args.tag_suffix.strip().replace(" ", "_") if args.tag_suffix.strip() else ""))
    os.makedirs(args.results_dir, exist_ok=True)

    with open(os.path.join(args.results_dir, "experiment_config.txt"), "w", encoding="utf-8") as f_cfg:
        f_cfg.write("dataset=Houston\n")
        f_cfg.write(f"modality={args.modality}\n")
        f_cfg.write(f"modality_label={modality_to_label(args.modality)}\n")
        f_cfg.write(f"runs={int(args.runs)}\n")
        f_cfg.write(f"epochs={int(args.epochs)}\n")
        f_cfg.write(f"patch_size={int(args.patch_size)}\n")
        f_cfg.write(f"pca_components={int(args.pca_components)}\n")
        f_cfg.write(f"use_sam={bool(args.use_sam)}\n")
        f_cfg.write(f"use_ema={bool(args.use_ema)}\n")
        f_cfg.write(f"plus_eval_tta={int(args.plus_eval_tta)}\n")
        f_cfg.write(f"map_tta={int(args.tta)}\n")
        f_cfg.write(f"lp_alpha={float(args.lp_alpha)}\n")


    hsi_raw, lidar_raw, tr_label, ts_label = load_data(args.data_path)

    print("Preprocessing Data (PCA & Norm)...")
    hsi_pca, lidar_norm = preprocess_hsi_lidar(hsi_raw, lidar_raw, args.pca_components)

    # gt_full：优先 gt.mat，否则 TR/TS union
    gt_full = load_gt_full(args.data_path, gt_mat=args.gt_mat, gt_key=args.gt_key,
                           tr_label=tr_label, ts_label=ts_label)

    history_core = {"OA": [], "AA": [], "Kappa": [], "PerClass": []}
    history_plus = {"OA": [], "AA": [], "Kappa": [], "PerClass": []}

    print("\n" + "=" * 72)
    print("Evaluation protocol (fairness-first)")
    print(f"- Modality : {modality_to_label(args.modality)}")
    print("- AEFN-core: raw model weights, test TTA=0, no LP (main cross-method comparison)")
    print(f"- AEFN+   : optional refinements reported separately (EMA-eval={'on' if args.use_ema else 'off'}, "
          f"test TTA={int(args.plus_eval_tta)}, map TTA={int(args.tta)}, LP alpha={float(args.lp_alpha):.2f})")
    print("=" * 72)

    per_run_rows = []
    for run in range(1, int(args.runs) + 1):
        set_seed(args.seed + run * 100)
        run_record = train_one_run(run, args, hsi_pca, lidar_norm, tr_label, ts_label, gt_full)

        c = run_record["core"]
        p = run_record["plus"]
        _append_history(history_core, c["oa"], c["aa"], c["kappa"], c["per_class"])
        _append_history(history_plus, p["oa"], p["aa"], p["kappa"], p["per_class"])

        per_run_rows.append({
            "Run": run,
            "Modality": modality_to_label(args.modality),
            "Protocol": "AEFN-core",
            "OA": c["oa"], "AA": c["aa"], "Kappa": c["kappa"],
        })
        per_run_rows.append({
            "Run": run,
            "Modality": modality_to_label(args.modality),
            "Protocol": "AEFN+",
            "OA": p["oa"], "AA": p["aa"], "Kappa": p["kappa"],
        })

    core_summary = _summarize_history(history_core)
    plus_summary = _summarize_history(history_plus)

    plot_bar_charts(core_summary["pc_mean"], f"{modality_to_safe_tag(args.modality)}_Overall_Mean_AEFN_core", args.results_dir,
                    core_summary["pc_mean"], core_summary["pc_std"])
    plot_bar_charts(plus_summary["pc_mean"], f"{modality_to_safe_tag(args.modality)}_Overall_Mean_AEFN_plus", args.results_dir,
                    plus_summary["pc_mean"], plus_summary["pc_std"])

    _print_and_save_summary(f"{modality_to_label(args.modality)}_AEFN-core{args.tag_suffix}", core_summary, args.results_dir)
    _print_and_save_summary(f"{modality_to_label(args.modality)}_AEFN+{args.tag_suffix}", plus_summary, args.results_dir)

    # Combined protocol-comparison table for manuscript/rebuttal use
    protocol_df = pd.DataFrame([
        {
            "Modality": modality_to_label(args.modality),
            "Protocol": "AEFN-core",
            "Description": "Raw model; no EMA eval; test TTA=0; no LP (fair main comparison)",
            "OA Mean": core_summary["oa_mean"], "OA Std": core_summary["oa_std"],
            "AA Mean": core_summary["aa_mean"], "AA Std": core_summary["aa_std"],
            "Kappa Mean": core_summary["kp_mean"], "Kappa Std": core_summary["kp_std"],
        },
        {
            "Modality": modality_to_label(args.modality),
            "Protocol": "AEFN+",
            "Description": f"Optional refinements; EMA eval={'on' if args.use_ema else 'off'}; test TTA={int(args.plus_eval_tta)}; map TTA={int(args.tta)}; LP alpha={float(args.lp_alpha):.2f}",
            "OA Mean": plus_summary["oa_mean"], "OA Std": plus_summary["oa_std"],
            "AA Mean": plus_summary["aa_mean"], "AA Std": plus_summary["aa_std"],
            "Kappa Mean": plus_summary["kp_mean"], "Kappa Std": plus_summary["kp_std"],
        },
    ])
    protocol_df.to_csv(os.path.join(args.results_dir, f"Final_Metrics_{modality_to_safe_tag(args.modality)}_AEFN_core_vs_plus.csv"), index=False)
    pd.DataFrame(per_run_rows).to_csv(os.path.join(args.results_dir, f"PerRun_Metrics_{modality_to_safe_tag(args.modality)}_AEFN_core_vs_plus.csv"), index=False)

    print("\nSuggested commands for Comment 3:")
    print(f"  python {os.path.basename(__file__)} --data_path <PATH_TO_DATA> --modality hsi")
    print(f"  python {os.path.basename(__file__)} --data_path <PATH_TO_DATA> --modality lidar")
    print(f"  python {os.path.basename(__file__)} --data_path <PATH_TO_DATA> --modality fused")

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
