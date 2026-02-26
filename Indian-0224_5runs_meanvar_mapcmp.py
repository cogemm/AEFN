# -*- coding: utf-8 -*-
"""
Indian Pines processing + HSI-only training script (for IEEE Access revision).
Updated (5-run statistics):
- Runs 5 rounds with different random seeds
- Reports OA / AA / Kappa and per-class accuracies as mean ± variance
- Generates a per-class accuracy comparison figure (line plot, NOT bar chart / NOT confusion matrix)
- Generates a GT-vs-prediction classification map comparison figure (side-by-side) with color chips below
"""

import os
import random
import numpy as np
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
from matplotlib.patches import Patch, Rectangle


# -----------------------------
# 0) Paths (as requested)
# -----------------------------
base_dir = r"E:/PythonProject/HSI-Datas"
rel_dir  = "."

# -----------------------------
# 1) Palette (0 background + 16 classes), as requested
# -----------------------------
colors = [
    "#000000",  # 0: Background
    "#FF5733",  "#FF9F33", "#FFC733", "#80FF33",
    "#33FF57",  "#33FF80", "#33FFB8", "#33B8FF",
    "#33D8FF",  "#3377FF", "#6A33FF", "#9A33FF",
    "#D133FF",  "#FF33D8", "#FF3380", "#FF336A"
]

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_loadmat(candidate_paths):
    last_err = None
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return sio.loadmat(p)
            except Exception as e:
                last_err = e
    raise FileNotFoundError(f"Could not load .mat from any candidate path: {candidate_paths}. Last error: {last_err}")


def _pick_first_data_key(mat_dict):
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise KeyError("No usable variable key found in .mat file.")
    preferred = ["indian_pines_corrected", "indian_pines", "indian_pines_gt", "gt"]
    for p in preferred:
        if p in keys:
            return p
    return keys[0]


def load_indian_pines():
    data_dict = _safe_loadmat([
        os.path.join(base_dir, "Indian_pines.mat"),
        os.path.join(rel_dir,  "Indian_pines.mat"),
    ])
    corrected_data = _safe_loadmat([
        os.path.join(base_dir, "Indian_pines_corrected.mat"),
        os.path.join(rel_dir,  "Indian_pines_corrected.mat"),
    ])
    gt_data = _safe_loadmat([
        os.path.join(base_dir, "Indian_pines_gt.mat"),
        os.path.join(rel_dir,  "Indian_pines_gt.mat"),
    ])

    # Prefer corrected data if present
    key_hsi = _pick_first_data_key(corrected_data)
    hsi = corrected_data[key_hsi].astype(np.float32)  # (H, W, B)

    key_gt = _pick_first_data_key(gt_data)
    gt = gt_data[key_gt].astype(np.int32)  # (H, W), labels {0..16}

    if hsi.ndim != 3:
        raise ValueError(f"HSI must be H×W×B, got shape: {hsi.shape}")
    if gt.ndim != 2:
        raise ValueError(f"GT must be H×W, got shape: {gt.shape}")
    if gt.shape[0] != hsi.shape[0] or gt.shape[1] != hsi.shape[1]:
        raise ValueError(f"HSI/GT spatial mismatch: HSI={hsi.shape}, GT={gt.shape}")

    return hsi, gt


def standardize_and_pca(hsi: np.ndarray, n_components: int = 30):
    H, W, B = hsi.shape
    X = hsi.reshape(-1, B)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    n_components = min(int(n_components), B)
    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    Xp = pca.fit_transform(Xs)

    hsi_pca = Xp.reshape(H, W, n_components).astype(np.float32)
    return hsi_pca, scaler, pca


def reflect_pad_hsi(hsi_pca: np.ndarray, patch_size: int):
    pad = patch_size // 2
    return np.pad(hsi_pca, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def stratified_split(gt: np.ndarray, train_ratio: float = 0.1, seed: int = 42):
    """
    Returns train_mask, test_mask with the same shape as gt.
    Train/test are sampled among gt>0 pixels with stratification per class.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(gt)
    classes = classes[classes > 0]

    train_mask = np.zeros_like(gt, dtype=bool)
    test_mask  = np.zeros_like(gt, dtype=bool)

    for c in classes:
        coords = np.argwhere(gt == c)
        n = len(coords)
        if n == 0:
            continue
        rng.shuffle(coords)
        n_tr = max(1, int(round(n * train_ratio)))
        # Ensure at least 1 test sample if possible
        if n > 1 and n_tr >= n:
            n_tr = n - 1
        tr = coords[:n_tr]
        te = coords[n_tr:]
        train_mask[tr[:, 0], tr[:, 1]] = True
        test_mask[te[:, 0], te[:, 1]] = True

    return train_mask, test_mask


class IndianPinesPatchDataset(Dataset):
    def __init__(self, hsi_pad, gt, mask, patch_size: int = 11, augment: bool = True):
        self.hsi_pad = torch.from_numpy(hsi_pad)  # (H+2p, W+2p, C)
        self.gt = gt
        self.ps = patch_size
        self.augment = augment

        self.rows, self.cols = np.where(mask)
        self.labels = gt[self.rows, self.cols] - 1  # {1..16} -> {0..15}

    def __len__(self):
        return len(self.rows)

    def _augment(self, x):
        if not self.augment:
            return x
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        return x

    def __getitem__(self, idx):
        r = int(self.rows[idx])
        c = int(self.cols[idx])
        patch = self.hsi_pad[r:r + self.ps, c:c + self.ps, :].permute(2, 0, 1).contiguous()  # (C,ps,ps)
        patch = self._augment(patch)
        y = int(self.labels[idx])
        return patch, y


# -----------------------------
# 2) Coordinate Attention (optional, lightweight)
# -----------------------------
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
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

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


class AEFN_HSI_Net(nn.Module):
    def __init__(self, in_bands: int, num_classes: int = 16, use_coordatt: bool = True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.ca1 = CoordAtt(64, 64) if use_coordatt else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.ca2 = CoordAtt(128, 128) if use_coordatt else nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.ca1(self.conv1(x))
        x = self.ca2(self.conv2(x))
        return self.classifier(x)


# -----------------------------
# 3) Visualization helpers
# -----------------------------
def save_map(img2d: np.ndarray, title: str, out_path: str):
    cmap = ListedColormap(colors)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(img2d, cmap=cmap, vmin=0, vmax=len(colors) - 1, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    handles = [Patch(facecolor=colors[i + 1], edgecolor="k", label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES) - 1, -1, -1)]
    labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES) - 1, -1, -1)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_gt_pred_map_comparison_with_colorchips(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: str = "IndianPines_GT_vs_hsi_cnn_Map.png",
    pred_title: str = "hsi_cnn",
    class_names=None,
    class_colors=None,
):
    """
    Create a side-by-side classification map comparison figure like:
    [GT]   [hsi_cnn]
    and place color chips directly below the figure.

    gt_map / pred_map contain labels in {0..16}; 0 is background/unlabeled.
    """
    if gt_map.shape != pred_map.shape:
        raise ValueError(f"gt_map and pred_map must share the same shape, got {gt_map.shape} vs {pred_map.shape}")

    if class_names is None:
        class_names = CLASS_NAMES
    if class_colors is None:
        class_colors = colors[1:17]  # exclude background

    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(16, 9), facecolor="#E6E6E6")
    gs = fig.add_gridspec(2, 2, height_ratios=[12.0, 2.2], hspace=0.02, wspace=0.10)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, :])

    for ax, img, ttl in [(ax_gt, gt_map, "GT"), (ax_pr, pred_map, pred_title)]:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=len(colors) - 1, interpolation="nearest")
        ax.set_title(ttl, fontsize=22, pad=12)
        ax.axis("off")

    # Bottom color chips panel (2 rows x 8 columns), tight under maps
    ax_leg.set_facecolor("#E6E6E6")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    n_cls = len(class_names)
    n_cols = 8
    n_rows = int(np.ceil(n_cls / n_cols))

    left_margin = 0.015
    right_margin = 0.01
    top_y = 0.80
    row_gap = 0.40 if n_rows > 1 else 0.0
    usable_w = 1.0 - left_margin - right_margin
    col_w = usable_w / n_cols

    for i in range(n_cls):
        r = i // n_cols
        c = i % n_cols
        x0 = left_margin + c * col_w
        y0 = top_y - r * row_gap

        # small color rectangle
        rect_w = col_w * 0.12
        rect_h = 0.16
        rect = Rectangle((x0, y0 - rect_h / 2), rect_w, rect_h,
                         facecolor=class_colors[i], edgecolor='k', linewidth=0.7,
                         transform=ax_leg.transAxes, clip_on=False)
        ax_leg.add_patch(rect)

        label = f"{i+1}. {class_names[i]}"
        ax_leg.text(x0 + rect_w + col_w * 0.03, y0, label,
                    transform=ax_leg.transAxes, ha='left', va='center', fontsize=9)

    # Manual layout (avoid tight_layout warning with mixed axes)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.94, bottom=0.05)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def save_per_class_accuracy_comparison_plot(
    per_class_runs: np.ndarray,
    out_path: str = "IndianPines_PerClass_Accuracy_Comparison.png",
    seeds=None,
    class_names=None,
    class_colors=None
):
    """
    per_class_runs: shape [N_runs, 16], values in [0,1]
    Produces a non-bar, non-confusion-matrix comparison figure:
    - top: line plot of each run + mean curve + shaded mean±variance
    - bottom: color rectangles for each land-cover class
    """
    per_class_runs = np.asarray(per_class_runs, dtype=np.float64)
    if per_class_runs.ndim != 2:
        raise ValueError(f"per_class_runs must be 2D, got shape={per_class_runs.shape}")

    n_runs, n_cls = per_class_runs.shape
    x = np.arange(1, n_cls + 1)

    mean_acc = per_class_runs.mean(axis=0)
    var_acc  = per_class_runs.var(axis=0)   # population variance
    std_acc  = per_class_runs.std(axis=0)

    if class_names is None:
        class_names = [f"Class {i}" for i in x]
    if class_colors is None:
        # Skip background color 0
        class_colors = colors[1:1+n_cls]

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 1.25], hspace=0.05)

    # ----- Top: line comparison -----
    ax = fig.add_subplot(gs[0, 0])

    # Each run line (light)
    for i in range(n_runs):
        label = f"Run {i+1}" if seeds is None else f"Run {i+1} (seed={seeds[i]})"
        ax.plot(x, per_class_runs[i] * 100.0, marker='o', linewidth=1.5, alpha=0.55, label=label)

    # Mean line
    ax.plot(x, mean_acc * 100.0, marker='o', linewidth=2.8, label="Mean per-class accuracy")

    # Shaded band: mean ± variance (as explicitly requested)
    lower = np.clip((mean_acc - var_acc) * 100.0, 0.0, 100.0)
    upper = np.clip((mean_acc + var_acc) * 100.0, 0.0, 100.0)
    ax.fill_between(x, lower, upper, alpha=0.20, label="Mean ± Variance")

    # (Optional faint dashed mean ± std for reference; no legend to avoid clutter)
    ax.plot(x, np.clip((mean_acc - std_acc) * 100.0, 0.0, 100.0), linestyle='--', linewidth=1.0, alpha=0.4)
    ax.plot(x, np.clip((mean_acc + std_acc) * 100.0, 0.0, 100.0), linestyle='--', linewidth=1.0, alpha=0.4)

    ax.set_xlim(0.5, n_cls + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])  # names shown in bottom panel
    ax.set_ylabel("Per-class Accuracy (%)")
    ax.set_title("Indian Pines Per-Class Classification Accuracy Comparison (5 Runs)")
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', ncol=2, fontsize=9, frameon=False)

    # ----- Bottom: color rectangles + class names -----
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.set_xlim(0.5, n_cls + 0.5)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # Draw rectangles tightly below the plot
    for i in range(n_cls):
        xi = i + 1
        rect = Rectangle((xi - 0.45, 0.42), 0.90, 0.28,
                         facecolor=class_colors[i], edgecolor='k', linewidth=0.6)
        ax2.add_patch(rect)
        ax2.text(xi, 0.78, str(i + 1), ha='center', va='bottom', fontsize=9)
        ax2.text(xi, 0.35, class_names[i], ha='center', va='top', fontsize=8, rotation=25)

    ax2.text(0.55, 0.98, "Class color chips (Indian Pines land-cover classes)", fontsize=9,
             ha='left', va='top')

    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.07, hspace=0.06)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# 4) Training + evaluation
# -----------------------------
def _evaluate_model(model, test_loader, num_classes=16):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            ps.append(pred)
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    oa = float(accuracy_score(y_true, y_pred))
    aa = float(np.mean(per_class))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    return oa, aa, kappa, per_class, cm


def _infer_label_map(model, hsi_pad: np.ndarray, gt: np.ndarray, patch_size: int, batch_size: int = 512, labeled_only: bool = True):
    """Infer a scene-level label map using the trained model.
    If labeled_only=True, only gt>0 pixels are predicted (others stay 0).
    Returned labels are in {0..16}.
    """
    model.eval()
    if labeled_only:
        rows, cols = np.where(gt > 0)
    else:
        rr, cc = np.indices(gt.shape)
        rows, cols = rr.reshape(-1), cc.reshape(-1)

    pred_map = np.zeros_like(gt, dtype=np.int32)
    ps = int(patch_size)

    with torch.no_grad():
        n = len(rows)
        for st in range(0, n, batch_size):
            ed = min(st + batch_size, n)
            rr = rows[st:ed]
            cc = cols[st:ed]
            # build batch patches
            batch_np = np.empty((len(rr), hsi_pad.shape[2], ps, ps), dtype=np.float32)
            for j, (r, c) in enumerate(zip(rr, cc)):
                patch = hsi_pad[r:r + ps, c:c + ps, :]
                batch_np[j] = np.transpose(patch, (2, 0, 1))
            xb = torch.from_numpy(batch_np).to(DEVICE)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy().astype(np.int32) + 1  # {1..16}
            pred_map[rr, cc] = pred

    return pred_map


def run_one(seed=42, patch_size=11, pca_components=30, train_ratio=0.1,
            epochs=120, batch_size=64, lr=1e-4, use_coordatt=True):
    set_seed(seed)
    hsi, gt = load_indian_pines()
    hsi_pca, _, _ = standardize_and_pca(hsi, n_components=pca_components)
    hsi_pad = reflect_pad_hsi(hsi_pca, patch_size)

    train_mask, test_mask = stratified_split(gt, train_ratio=train_ratio, seed=seed)

    train_ds = IndianPinesPatchDataset(hsi_pad, gt, train_mask, patch_size=patch_size, augment=True)
    test_ds  = IndianPinesPatchDataset(hsi_pad, gt, test_mask,  patch_size=patch_size, augment=False)

    # Avoid BatchNorm1d failure when the last training mini-batch has only 1 sample
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"Empty train/test split: len(train_ds)={len(train_ds)}, len(test_ds)={len(test_ds)}")

    model = AEFN_HSI_Net(in_bands=hsi_pca.shape[-1], num_classes=16, use_coordatt=use_coordatt).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_oa = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            # Extra safety for BatchNorm1d in classifier
            if x.size(0) < 2:
                continue
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validation on test split (kept for reproducibility of the existing script style)
        oa, aa, kappa, per_class, cm = _evaluate_model(model, test_loader, num_classes=16)
        if oa > best_oa:
            best_oa = oa
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    oa, aa, kappa, per_class, cm = _evaluate_model(model, test_loader, num_classes=16)

    # Scene-level prediction map for visualization (predict on labeled pixels only; keep unlabeled as 0)
    pred_map = _infer_label_map(model, hsi_pad, gt, patch_size=patch_size, batch_size=512, labeled_only=True)

    return {
        "seed": int(seed),
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "per_class": per_class.astype(np.float64),  # shape [16]
        "cm": cm.astype(np.int64),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "pred_map": pred_map.astype(np.int32),
    }


# -----------------------------
# 5) Multi-run statistics
# -----------------------------
def _mean_var(arr, axis=0):
    arr = np.asarray(arr, dtype=np.float64)
    return np.mean(arr, axis=axis), np.var(arr, axis=axis)  # variance (not std), as requested


def _fmt_mean_var(x_mean, x_var, is_percent=False):
    if is_percent:
        return f"{x_mean*100:.2f}% ± {x_var*100:.2f}%"
    return f"{x_mean:.4f} ± {x_var:.4f}"


def run_five_rounds(
    seeds=(42, 43, 44, 45, 46),
    patch_size=11,
    pca_components=30,
    train_ratio=0.10,
    epochs=120,
    batch_size=64,
    lr=1e-4,
    use_coordatt=True,
    out_dir="."
):
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    print("=" * 80)
    print("Running 5 rounds on Indian Pines (HSI-only)")
    print(f"DEVICE={DEVICE}, seeds={list(seeds)}")
    print("=" * 80)

    for i, sd in enumerate(seeds, 1):
        result = run_one(
            seed=sd,
            patch_size=patch_size,
            pca_components=pca_components,
            train_ratio=train_ratio,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            use_coordatt=use_coordatt
        )
        all_results.append(result)
        print(f"[Run {i}/5 | seed={sd}] OA={result['oa']*100:.2f}%, "
              f"AA={result['aa']*100:.2f}%, Kappa={result['kappa']:.4f}, "
              f"Train={result['n_train']}, Test={result['n_test']}")

    # Aggregate
    oa_arr = np.array([r["oa"] for r in all_results], dtype=np.float64)
    aa_arr = np.array([r["aa"] for r in all_results], dtype=np.float64)
    kp_arr = np.array([r["kappa"] for r in all_results], dtype=np.float64)
    pc_arr = np.stack([r["per_class"] for r in all_results], axis=0)  # [5, 16]

    oa_mean, oa_var = _mean_var(oa_arr)
    aa_mean, aa_var = _mean_var(aa_arr)
    kp_mean, kp_var = _mean_var(kp_arr)
    pc_mean, pc_var = _mean_var(pc_arr, axis=0)

    print("\n" + "=" * 80)
    print("Final 5-run summary (mean ± variance)")
    print("=" * 80)
    print(f"OA    : {_fmt_mean_var(oa_mean, oa_var, is_percent=True)}")
    print(f"AA    : {_fmt_mean_var(aa_mean, aa_var, is_percent=True)}")
    print(f"Kappa : {_fmt_mean_var(kp_mean, kp_var, is_percent=False)}")
    print("-" * 80)
    print("Per-class Accuracy (mean ± variance)")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i+1:2d}. {name:<36s}: {_fmt_mean_var(pc_mean[i], pc_var[i], is_percent=True)}")
    print("=" * 80)

    # Save summary text
    txt_path = os.path.join(out_dir, "IndianPines_5runs_summary_mean_var.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Indian Pines (HSI-only) - 5 runs summary (mean ± variance)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Seeds: {list(seeds)}\n")
        f.write(f"OA    : {_fmt_mean_var(oa_mean, oa_var, is_percent=True)}\n")
        f.write(f"AA    : {_fmt_mean_var(aa_mean, aa_var, is_percent=True)}\n")
        f.write(f"Kappa : {_fmt_mean_var(kp_mean, kp_var, is_percent=False)}\n\n")
        f.write("Per-class Accuracy (mean ± variance)\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i+1:2d}. {name:<36s}: {_fmt_mean_var(pc_mean[i], pc_var[i], is_percent=True)}\n")

    # Save per-class stats csv
    csv_path = os.path.join(out_dir, "IndianPines_5runs_per_class_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,mean_accuracy,var_accuracy,std_accuracy\n")
        pc_std = np.std(pc_arr, axis=0)
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i+1},\"{name}\",{pc_mean[i]:.8f},{pc_var[i]:.8f},{pc_std[i]:.8f}\n")

    # Save metric summary csv
    metrics_csv = os.path.join(out_dir, "IndianPines_5runs_metrics.csv")
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("run_id,seed,oa,aa,kappa\n")
        for i, r in enumerate(all_results, 1):
            f.write(f"{i},{r['seed']},{r['oa']:.8f},{r['aa']:.8f},{r['kappa']:.8f}\n")
        f.write(f"mean,,{oa_mean:.8f},{aa_mean:.8f},{kp_mean:.8f}\n")
        f.write(f"var,,{oa_var:.8f},{aa_var:.8f},{kp_var:.8f}\n")

    # Save required comparison figure (not bar chart / not confusion matrix)
    fig_path = os.path.join(out_dir, "IndianPines_PerClass_Accuracy_Comparison_Line.png")
    save_per_class_accuracy_comparison_plot(
        per_class_runs=pc_arr,
        out_path=fig_path,
        seeds=list(seeds),
        class_names=CLASS_NAMES,
        class_colors=colors[1:17],
    )
    # Save GT-vs-prediction classification map comparison (style like GT | hsi_cnn)
    _, gt_full = load_indian_pines()
    best_idx = int(np.argmax(oa_arr))
    best_result = all_results[best_idx]
    map_cmp_path = os.path.join(out_dir, "IndianPines_GT_vs_hsi_cnn_Map.png")
    save_gt_pred_map_comparison_with_colorchips(
        gt_map=gt_full.astype(np.int32),
        pred_map=best_result["pred_map"],
        out_path=map_cmp_path,
        pred_title="hsi_cnn",
        class_names=CLASS_NAMES,
        class_colors=colors[1:17],
    )

    print(f"\nSaved summary text: {txt_path}")
    print(f"Saved per-class CSV: {csv_path}")
    print(f"Saved metrics CSV   : {metrics_csv}")
    print(f"Saved comparison fig: {fig_path}")
    print(f"Saved GT-vs-pred fig: {map_cmp_path}")

    return {
        "all_results": all_results,
        "oa_mean": oa_mean, "oa_var": oa_var,
        "aa_mean": aa_mean, "aa_var": aa_var,
        "kappa_mean": kp_mean, "kappa_var": kp_var,
        "per_class_mean": pc_mean, "per_class_var": pc_var,
        "per_class_runs": pc_arr,
        "summary_txt": txt_path,
        "per_class_csv": csv_path,
        "metrics_csv": metrics_csv,
        "comparison_fig": fig_path,
        "map_comparison_fig": map_cmp_path,
    }


if __name__ == "__main__":
    # Default: run 5 rounds and report mean ± variance
    # You can lower epochs (e.g., 80) for faster debugging and restore to 120 for final reporting.
    run_five_rounds(
        seeds=(42, 43, 44, 45, 46),
        patch_size=11,
        pca_components=30,
        train_ratio=0.10,
        epochs=120,
        batch_size=64,
        lr=1e-4,
        use_coordatt=True,
        out_dir="."
    )
