#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gc
import re
import sys
import glob
import zlib
import time
import json
import math
import shutil
import random
import hashlib
import warnings
import tempfile
import subprocess
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.ndimage import rotate as ndimage_rotate

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
except ImportError:
    roc_auc_score = None
    RobustScaler = None
    LogisticRegression = None
    RidgeClassifier = None
    HistGradientBoostingClassifier = None
    ExtraTreesClassifier = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = device
    CUDA_CC_MAJOR = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    AMP_DTYPE = torch.bfloat16 if ((torch.cuda.is_available() and CUDA_CC_MAJOR >= 8 and torch.cuda.is_bf16_supported()) or (not torch.cuda.is_available())) else torch.float16
except ImportError:
    class _NNStub:
        Module = object
    class _TorchStub:
        Tensor = np.ndarray
        nn = _NNStub
        float32 = np.float32
        float16 = np.float16
        bfloat16 = np.float32
        @staticmethod
        def device(name="cpu"):
            return name
        @staticmethod
        def no_grad():
            def _dec(fn):
                return fn
            return _dec
    torch = _TorchStub()
    nn = _NNStub()
    F = None
    TORCH_AVAILABLE = False
    device = "cpu"
    DEVICE = "cpu"
    AMP_DTYPE = np.float32

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

warnings.filterwarnings("ignore")

SEED = 42
CODE_VERSION = "V10.6-COLAB-PRO-9FAM1195D-STAGE0-COORDCONV-RECENTERED"
V107_AUDIT_TAG = "V10.7-FORENSIC-9POINT-AUDITED-STAGE0-AFFINE-INVARIANT"
V105_COMPAT_VERSION = "V10.5-COLAB-PRO-8FAM1123D-EMPIRICAL-MHD-AUDITED"
SCAN_CONTEXT_ANCHOR_K_REF = 24
OFFICIAL_MAX_CAD_MARKS_PER_SCAN = -1
EMPIRICAL_MHD_MAX_SCANS = 25
VAL_RATIO = 0.15
TEST_RATIO = 0.15
FROC_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PY3_EVAL_DIR = os.path.join(tempfile.gettempdir(), "evaluationScript_py3_v107")

"""
Complete Production Implementation of Stage II Stream B Neural Backbones (Section 2.3, Eq. 8):
  - MultiWindowCoordConv3DResNeXtWithDiameterHead (6-Ch 24^3 Multi-Scale CoordConv 3D ResNeXt-SE)
  - FoldIsolatedTriPlanarMIPConvNet (9-Ch 2.5D Axial/Coronal/Sagittal Slab MIP ConvNet)
  - _batch_hard_negative_focal_ranking_loss & 8-View Cube Equivariance Augmentation
"""

def apply_consistent_3d_physical_augmentation(
    v3_b: torch.Tensor,
    vm_b: torch.Tensor,
    aug_code: int,
):
    c_code = int(aug_code) % 8
    if c_code == 0:
        return v3_b, vm_b

    v3_out = v3_b.clone()
    ax_out = vm_b[:, 0:3].clone()
    co_out = vm_b[:, 3:6].clone()
    sa_out = vm_b[:, 6:9].clone()

    if c_code & 1:
        v3_out = torch.flip(v3_out, dims=[-1])
        ax_out = torch.flip(ax_out, dims=[-1])
        co_out = torch.flip(co_out, dims=[-1])
    if c_code & 2:
        v3_out = torch.flip(v3_out, dims=[-2])
        ax_out = torch.flip(ax_out, dims=[-2])
        sa_out = torch.flip(sa_out, dims=[-1])
    if c_code & 4:
        v3_out = torch.flip(v3_out, dims=[-3])
        co_out = torch.flip(co_out, dims=[-2])
        sa_out = torch.flip(sa_out, dims=[-2])

    vm_out = torch.cat([ax_out, co_out, sa_out], dim=1)
    return v3_out, vm_out


def verify_3d_mip_augmentation_geometric_consistency():
    rng = np.random.RandomState(SEED)
    V_test48 = torch.from_numpy(rng.rand(2, 48, 48, 48).astype(np.float32))

    def _build_4ch_and_9mip_from_48(Vt48: torch.Tensor):
        B = Vt48.shape[0]
        Vt5 = Vt48.unsqueeze(1)
        c0 = F.interpolate(Vt5[:, :, 18:30, 18:30, 18:30], size=(24, 24, 24), mode='trilinear', align_corners=True)[:, 0]
        c1 = Vt48[:, 12:36, 12:36, 12:36]
        c2 = Vt48.view(B, 24, 2, 24, 2, 24, 2).mean(dim=(2, 4, 6))
        s1 = F.avg_pool3d(Vt5, kernel_size=3, stride=1, padding=1)[:, 0]
        s3 = F.avg_pool3d(Vt5, kernel_size=7, stride=1, padding=3)[:, 0]
        c3 = (s1 - s3)[:, 12:36, 12:36, 12:36]
        v3 = torch.stack([c0, c1, c2, c3], dim=1)

        Vc32 = Vt48[:, 8:40, 8:40, 8:40]
        ax = torch.stack([
            0.5 * (Vc32[:, 15, :, :] + Vc32[:, 16, :, :]),
            torch.amax(Vc32[:, 13:19, :, :], dim=1),
            torch.amax(Vc32[:, 9:23, :, :], dim=1),
        ], dim=1)
        co = torch.stack([
            0.5 * (Vc32[:, :, 15, :] + Vc32[:, :, 16, :]),
            torch.amax(Vc32[:, :, 13:19, :], dim=2),
            torch.amax(Vc32[:, :, 9:23, :], dim=2),
        ], dim=1)
        sa = torch.stack([
            0.5 * (Vc32[:, :, :, 15] + Vc32[:, :, :, 16]),
            torch.amax(Vc32[:, :, :, 13:19], dim=3),
            torch.amax(Vc32[:, :, :, 9:23], dim=3),
        ], dim=1)
        vm = torch.cat([ax, co, sa], dim=1)
        return v3, vm

    v3_orig, vm_orig = _build_4ch_and_9mip_from_48(V_test48)
    max_3d_err = 0.0
    max_mip_err = 0.0
    for code in range(8):
        Vt_r = V_test48.clone()
        if code & 1:
            Vt_r = torch.flip(Vt_r, dims=[-1])
        if code & 2:
            Vt_r = torch.flip(Vt_r, dims=[-2])
        if code & 4:
            Vt_r = torch.flip(Vt_r, dims=[-3])
        v3_from_3d, vm_from_3d = _build_4ch_and_9mip_from_48(Vt_r)
        v3_aug, vm_aug = apply_consistent_3d_physical_augmentation(v3_orig, vm_orig, code)
        err_3d = float(torch.max(torch.abs(v3_from_3d - v3_aug)).item())
        err_mip = float(torch.max(torch.abs(vm_from_3d - vm_aug)).item())
        max_3d_err = max(max_3d_err, err_3d)
        max_mip_err = max(max_mip_err, err_mip)
    assert max_3d_err < 1e-6, f"8-view 3D pyramid consistency failed: {max_3d_err}"
    assert max_mip_err < 1e-6, f"8-view MIP consistency failed: {max_mip_err}"
    return max_3d_err, max_mip_err


_3D_AUG_MAX_ERR, _MIP_AUG_MAX_ERR = verify_3d_mip_augmentation_geometric_consistency() if TORCH_AVAILABLE else (0.0, 0.0)


class MultiWindowCoordConv3DResNeXtWithDiameterHead(nn.Module):
    """
    V10.6 High-Capacity 6-Channel Multi-Window 3D CoordConv ResNeXt-Attention
    Network with Multi-Task Auxiliary Log-Diameter Regression Head:
      - Dynamically expands the 4-channel `24x24x24` tensor on the GPU with:
        * Ch 4: Subsolid / Ground-Glass Nodule (GGN) Soft-Attenuation Window
                `sqrt(clamp(v_native, 0.05, 0.55) - 0.05)`
        * Ch 5: 3D Radial Gaussian CoordConv Distance Map `exp(-r^2 / (2 * 4.5^2))`
          (reflection-symmetric under all 8 cube reflections!)
      - Outputs `(cls_logit, log_diam_pred)` so training on `diameter_mm` forces
        the 3D backbone to learn physical nodule scale without needing test
        diameter metadata!
    """
    def __init__(self, aux_dim: int = 51):
        super().__init__()
        g24 = torch.arange(24, dtype=torch.float32) - 11.5
        zz24, yy24, xx24 = torch.meshgrid(g24, g24, g24, indexing='ij')
        r24 = torch.sqrt(zz24 ** 2 + yy24 ** 2 + xx24 ** 2)
        coord_gauss = torch.exp(-(r24 ** 2) / (2.0 * (4.5 ** 2))).view(1, 1, 24, 24, 24)
        self.register_buffer('coord_gauss24', coord_gauss)

        self.stem = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(48),
            nn.SiLU(inplace=True),
            nn.Conv3d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(48),
        )
        self.skip1 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(48),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(48, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(80),
            nn.SiLU(inplace=True),
            nn.Conv3d(80, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(80),
        )
        self.skip2 = nn.Sequential(
            nn.Conv3d(48, 80, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(80),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(80, 20),
            nn.SiLU(inplace=True),
            nn.Linear(20, 80),
            nn.Sigmoid(),
        )

        g6 = torch.arange(6, dtype=torch.float32) - 2.5
        zz6, yy6, xx6 = torch.meshgrid(g6, g6, g6, indexing='ij')
        r6 = torch.sqrt(zz6 ** 2 + yy6 ** 2 + xx6 ** 2)
        w_core = (r6 <= 1.6).float()
        w_core = (w_core / w_core.sum()).view(1, 1, 6, 6, 6)
        w_outer = (r6 > 2.2).float()
        w_outer = (w_outer / w_outer.sum()).view(1, 1, 6, 6, 6)
        self.register_buffer('w_core6', w_core)
        self.register_buffer('w_outer6', w_outer)

        self.shared_fc = nn.Sequential(
            nn.Linear(80 * 4 + aux_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.22),
        )
        self.cls_head = nn.Linear(96, 1)
        self.diam_head = nn.Linear(96, 1)

    def forward(self, v3d_4ch: torch.Tensor, x_aux: torch.Tensor):
        B = v3d_4ch.shape[0]
        v_nat = v3d_4ch[:, 1:2]
        v_ggn = torch.sqrt(torch.clamp(v_nat, 0.05, 0.55) - 0.05 + 1e-4)
        v_coord = self.coord_gauss24.expand(B, -1, -1, -1, -1)
        x6 = torch.cat([v3d_4ch, v_ggn, v_coord], dim=1)

        h = self.stem(x6)
        h = F.silu(self.block1(h) + self.skip1(h))
        h = F.silu(self.block2(h) + self.skip2(h))
        w_se = self.se(h).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h * w_se

        core_pool = (h * self.w_core6).sum(dim=(2, 3, 4))
        outer_pool = (h * self.w_outer6).sum(dim=(2, 3, 4))
        diff_pool = core_pool - outer_pool
        gmp = h.amax(dim=(2, 3, 4))
        feat = torch.cat([core_pool, diff_pool, outer_pool, gmp, x_aux], dim=1)
        emb = self.shared_fc(feat)
        return self.cls_head(emb).squeeze(-1), self.diam_head(emb).squeeze(-1)


class FoldIsolatedTriPlanarMIPConvNet(nn.Module):
    def __init__(self, aux_dim: int = 51):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.s1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(48, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.SiLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(48, 80, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(80),
        )

        g8 = torch.arange(8, dtype=torch.float32) - 3.5
        yy8, xx8 = torch.meshgrid(g8, g8, indexing='ij')
        r8 = torch.sqrt(yy8 ** 2 + xx8 ** 2)
        w_c8 = (r8 <= 1.8).float()
        w_c8 = (w_c8 / w_c8.sum()).view(1, 1, 8, 8)
        w_o8 = (r8 > 2.5).float()
        w_o8 = (w_o8 / w_o8.sum()).view(1, 1, 8, 8)
        self.register_buffer('w_core8', w_c8)
        self.register_buffer('w_outer8', w_o8)

        self.head = nn.Sequential(
            nn.Linear(80 * 3 + aux_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.22),
            nn.Linear(64, 1),
        )

    def forward(self, v25d, x_aux):
        h = self.stem(v25d)
        h = F.silu(self.b1(h) + self.s1(h))
        h = F.silu(self.b2(h) + self.s2(h))
        c_pool = (h * self.w_core8).sum(dim=(2, 3))
        o_pool = (h * self.w_outer8).sum(dim=(2, 3))
        gmp = h.amax(dim=(2, 3))
        feat = torch.cat([c_pool, c_pool - o_pool, gmp, x_aux], dim=1)
        return self.head(feat).squeeze(-1)


def _batch_hard_negative_focal_ranking_loss(logits: torch.Tensor, yb: torch.Tensor, pos_w_t: torch.Tensor):
    bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w_t, reduction='none')
    pt = torch.exp(-F.binary_cross_entropy_with_logits(logits, yb, reduction='none'))
    focal_bce = ((1.0 - pt) ** 1.5 * bce).mean()

    pos_mask = (yb > 0.5)
    neg_mask = (yb < 0.5)
    if torch.any(pos_mask) and torch.sum(neg_mask) >= 8:
        pos_s = logits[pos_mask]
        neg_s = logits[neg_mask]
        neg_sorted, _ = torch.sort(neg_s, descending=True)
        k_top1 = max(1, min(len(neg_sorted), max(2, len(neg_sorted) // 40)))
        k_top2 = max(1, min(len(neg_sorted), max(4, len(neg_sorted) // 15)))
        k_top3 = max(1, min(len(neg_sorted), max(8, len(neg_sorted) // 6)))
        tau1 = torch.mean(neg_sorted[:k_top1])
        tau2 = torch.mean(neg_sorted[:k_top2])
        tau3 = torch.mean(neg_sorted[:k_top3])
        rank_loss = (
            0.45 * F.softplus(tau1 + 0.70 - pos_s, beta=2.5).mean()
            + 0.35 * F.softplus(tau2 + 0.55 - pos_s, beta=2.5).mean()
            + 0.20 * F.softplus(tau3 + 0.40 - pos_s, beta=2.5).mean()
        )
        return 0.68 * focal_bce + 0.32 * rank_loss
    return focal_bce
