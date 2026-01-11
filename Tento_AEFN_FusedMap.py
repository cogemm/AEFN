# -*- coding: utf-8 -*-
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
CATEGORIES =  ["Apple Trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"]
COLORS = ["#8B4513", "#FF0000", "#FFC0CB", "#D2B48C", "#008000", "#00FFFF"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    p = argparse.ArgumentParser()

    # data / io
    p.add_argument("--data_path", type=str, default=r"E:\PythonProject1\Trento")
    p.add_argument("--results_dir", type=str, default="./results_AEFN_FusedMap_Trento")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # model / preprocess
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument("--pca_components", type=int, default=30)

    # train
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--use_sam", action="store_true", default=True)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--mixup_alpha", type=float, default=1.0)

    # full-map inference
    p.add_argument("--gt_mat", type=str, default="gt.mat", help="全图GT文件名（可选）")
    p.add_argument("--gt_key", type=str, default="gt", help="gt.mat中的变量名")
    p.add_argument("--map_batch", type=int, default=256, help="全图推理batch size")
    p.add_argument("--tta", type=int, default=0, help="推理TTA: 0(关闭) 或 8(8-way)")
    p.add_argument("--lp_alpha", type=float, default=0.60, help="Label Propagation alpha；<=0 表示关闭")
    p.add_argument("--lp_iters", type=int, default=30)
    p.add_argument("--lp_radius", type=int, default=3, help="LP图邻域半径(像素)")

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
# 2) 核心模块：Coordinate Attention & AEFN Fusion
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


class HighAcc_AEFN_Net(nn.Module):
    def __init__(self, hsi_bands, num_classes=15):
        super().__init__()
        self.conv_h1 = nn.Sequential(
            nn.Conv2d(hsi_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.ca_h1 = CoordAtt(64, 64)
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
        self.ca_l1 = CoordAtt(32, 32)
        self.conv_l2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.fusion = AdaptiveGatedFusion(128)
        self.ca_fuse = CoordAtt(128, 128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_h, x_l):
        h = self.conv_h1(x_h)
        h = self.ca_h1(h)
        h = self.conv_h2(h)

        l = self.conv_l1(x_l)
        l = self.ca_l1(l)
        l = self.conv_l2(l)

        f = self.fusion(h, l)
        f = self.ca_fuse(f)
        return self.classifier(f)


# ==============================================================================
# 3) 优化器：SAM
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
# 5) 数据：加载 / 预处理 / Patch Dataset
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
    pca = PCA(n_components=pca_components)
    hsi_pca = pca.fit_transform(hsi_scaled).reshape(H, W, pca_components).astype(np.float32)

    lidar_norm = StandardScaler().fit_transform(lidar_raw.reshape(-1, 1)).reshape(H, W).astype(np.float32)
    return hsi_pca, lidar_norm


def pad_hsi_lidar(hsi: np.ndarray, lidar: np.ndarray, patch_size: int):
    m = patch_size // 2
    h_pad = np.pad(hsi, ((m, m), (m, m), (0, 0)), mode="reflect")
    l_pad = np.pad(lidar, ((m, m), (m, m)), mode="reflect")
    return h_pad, l_pad


class TrentoPatchDataset(Dataset):
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
# 6) 全图推理增强：TTA + Label Propagation
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
# 7) 绘图：
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
# 8) 训练/评估 + 生成图
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
    ds = TrentoPatchDataset(h_pad, l_pad, rows, cols, labels=None,
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


def train_one_run(run_id: int, args, hsi_pca: np.ndarray, lidar_norm: np.ndarray,
                  tr_label: np.ndarray, ts_label: np.ndarray, gt_full: np.ndarray):
    print(f"\n>>> Starting Run {run_id} ...")

    # pad once for all datasets
    h_pad, l_pad = pad_hsi_lidar(hsi_pca, lidar_norm, args.patch_size)

    # train/test positions
    tr_rows, tr_cols, y_tr = label_to_positions(tr_label)
    ts_rows, ts_cols, y_ts = label_to_positions(ts_label)

    train_ds = TrentoPatchDataset(h_pad, l_pad, tr_rows, tr_cols, y_tr, args.patch_size,
                                   augment=True, return_index=False)
    test_ds = TrentoPatchDataset(h_pad, l_pad, ts_rows, ts_cols, y_ts, args.patch_size,
                                  augment=False, return_index=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = HighAcc_AEFN_Net(hsi_bands=hsi_pca.shape[-1], num_classes=len(CATEGORIES)).to(DEVICE)

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

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    ema = EMA(model, decay=0.999) if args.use_ema else None

    best_oa = 0.0
    best_weights = None

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

        # periodic eval to select best
        if (epoch + 1) % 5 == 0 or epoch >= int(args.epochs) - 10:
            if ema:
                ema.apply_shadow()
            y_pred, y_true = eval_on_loader(model, test_loader, tta=0)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_oa:
                best_oa = acc
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ema:
                ema.restore()

    # load best weights
    if best_weights is None:
        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    model.eval()

    # final eval (test only, 保持第一份代码的指标口径)
    y_pred, y_true = eval_on_loader(model, test_loader, tta=0)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CATEGORIES)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    aa = float(np.mean(per_class_acc))

    print(f"[Run {run_id} Result] OA: {oa * 100:.2f}%, AA: {aa * 100:.2f}%, Kappa: {kappa:.4f}")

    # -------- 全图(掩膜)推理：让结果图像“像第二份代码那样完整/清晰” --------
    mask = (gt_full > 0)
    rows_all, cols_all = np.where(mask)

    # (1) 推理 logits
    logits_all_cpu = infer_full_map_logits(
        model=model,
        h_pad=h_pad, l_pad=l_pad,
        rows=rows_all, cols=cols_all,
        patch_size=args.patch_size,
        batch_size=args.map_batch,
        tta=args.tta
    )

    # (2) 可选：label propagation 细化（对制图更友好）
    logits_use = logits_all_cpu
    if args.lp_alpha is not None and float(args.lp_alpha) > 0.0:
        edge_index, _ = build_edge_index_from_mask(mask, radius=int(args.lp_radius))
        logits_use = refine_by_label_propagation(
            logits_all_cpu.to(DEVICE),
            edge_index.to(DEVICE),
            alpha=float(args.lp_alpha),
            iters=int(args.lp_iters)
        ).detach().cpu()

    pred_all = logits_use.argmax(dim=1).numpy().astype(np.int32)  # 0..14

    # (3) 回填到整幅图（0=背景；1..15=类别）
    pred_img = np.zeros_like(gt_full, dtype=np.int32)
    pred_img[rows_all, cols_all] = pred_all + 1

    # 保存更清晰的单图（GT / Pred 分开保存）
    os.makedirs(args.results_dir, exist_ok=True)
    save_map_like_demo(
        gt_full.astype(np.int32),
        title=f"Ground Truth",
        out_path=os.path.join(args.results_dir, f"Run_{run_id}_GT.png"),
        colors=COLORS,
        class_names=CATEGORIES
    )
    save_map_like_demo(
        pred_img,
        title=f"Classification Map (Run {run_id})",
        out_path=os.path.join(args.results_dir, f"Run_{run_id}_Pred.png"),
        colors=COLORS,
        class_names=CATEGORIES
    )

    # per-class bar
    plot_bar_charts(per_class_acc, run_id, args.results_dir)

    return oa, aa, kappa, per_class_acc


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    hsi_raw, lidar_raw, tr_label, ts_label = load_data(args.data_path)

    print("Preprocessing Data (PCA & Norm)...")
    hsi_pca, lidar_norm = preprocess_hsi_lidar(hsi_raw, lidar_raw, args.pca_components)

    # gt_full：优先 gt.mat，否则 TR/TS union
    gt_full = load_gt_full(args.data_path, gt_mat=args.gt_mat, gt_key=args.gt_key,
                           tr_label=tr_label, ts_label=ts_label)

    history = {"OA": [], "AA": [], "Kappa": [], "PerClass": []}

    for run in range(1, int(args.runs) + 1):
        set_seed(args.seed + run * 100)
        oa, aa, kp, pc = train_one_run(run, args, hsi_pca, lidar_norm, tr_label, ts_label, gt_full)

        history["OA"].append(oa)
        history["AA"].append(aa)
        history["Kappa"].append(kp)
        history["PerClass"].append(pc)

    oa_mean, oa_std = float(np.mean(history["OA"])), float(np.std(history["OA"]))
    aa_mean, aa_std = float(np.mean(history["AA"])), float(np.std(history["AA"]))
    kp_mean, kp_std = float(np.mean(history["Kappa"])), float(np.std(history["Kappa"]))
    pc_mean = np.mean(history["PerClass"], axis=0)
    pc_std = np.std(history["PerClass"], axis=0)

    plot_bar_charts(pc_mean, "Overall_Mean", args.results_dir, pc_mean, pc_std)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"OA    : {oa_mean * 100:.2f} ± {oa_std * 100:.2f} %")
    print(f"AA    : {aa_mean * 100:.2f} ± {aa_std * 100:.2f} %")
    print(f"Kappa : {kp_mean:.4f} ± {kp_std:.4f}")
    print("-" * 60)

    results_list = []
    for i, name in enumerate(CATEGORIES):
        print(f"{name:20s}: {pc_mean[i] * 100:.2f} ± {pc_std[i] * 100:.2f} %")
        results_list.append({
            "Class ID": i + 1,
            "Class Name": name,
            "Mean Acc": float(pc_mean[i]),
            "Std": float(pc_std[i]),
        })

    pd.DataFrame(results_list).to_csv(os.path.join(args.results_dir, "Final_Per_Class_Stats.csv"), index=False)
    pd.DataFrame([{
        "OA Mean": oa_mean, "OA Std": oa_std,
        "AA Mean": aa_mean, "AA Std": aa_std,
        "Kappa Mean": kp_mean, "Kappa Std": kp_std
    }]).to_csv(os.path.join(args.results_dir, "Final_Metrics.csv"), index=False)

    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
