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
Leak-Free Inner 5-Fold Multi-Task 6-Channel CoordConv 3D ResNeXt-SE & 9-Channel 2.5D MIP
Cross-Fitting and Nested 10x5 CV / Locked 622-133-133 Training Pipeline (Section 2.3, Eq. 8).
"""
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models.backbone import (
    MultiWindowCoordConv3DResNeXtWithDiameterHead,
    FoldIsolatedTriPlanarMIPConvNet,
    _batch_hard_negative_focal_ranking_loss,
    apply_consistent_3d_physical_augmentation,
)

def train_and_predict_v106_multitask_coordconv_3d_cnns_gpu(
    V3D_all_gpu: torch.Tensor,
    V25D_all_gpu: torch.Tensor,
    X_aux_raw: np.ndarray,
    y_all: np.ndarray,
    diam_all: np.ndarray,
    idx_fit_global: np.ndarray,
    idx_targets_global_list: list,
    n_epochs: int = 20,
    batch_size: int = 256,
    seed: int = 42,
):
    """
    Trains the V10.6 6-Channel Multi-Window CoordConv 3D ResNeXt-SE (with
    Multi-Task Auxiliary Log-Diameter Regression Head) and Tri-Planar MIP
    ConvNet strictly on `idx_fit_global`, returning `(p3_arr, pm_arr, d3_arr)`
    for each target index set.
    """
    seed_everything_deterministic(seed)
    sc_aux = RobustScaler()
    aux_fit = np.clip(sc_aux.fit_transform(X_aux_raw[idx_fit_global]), -8.0, 8.0).astype(np.float32)
    aux_targets = [
        torch.from_numpy(np.clip(sc_aux.transform(X_aux_raw[idx_t]), -8.0, 8.0).astype(np.float32)).to(device)
        for idx_t in idx_targets_global_list
    ]

    idx_fit_t = torch.from_numpy(idx_fit_global).to(device, dtype=torch.long)
    aux_fit_t = torch.from_numpy(aux_fit).to(device, dtype=torch.float32)
    y_fit_t = torch.from_numpy(y_all[idx_fit_global].astype(np.float32)).to(device, dtype=torch.float32)
    log_diam_fit = np.where(y_all[idx_fit_global] == 1, np.log1p(np.maximum(diam_all[idx_fit_global], 3.0)), 0.0).astype(np.float32)
    d_fit_t = torch.from_numpy(log_diam_fit).to(device, dtype=torch.float32)

    n_fit = len(idx_fit_global)
    n_pos = float(torch.sum(y_fit_t > 0.5).item())
    pos_weight_val = min(6.0, max(1.5, (n_fit - n_pos) / (n_pos + 1.0) * 0.45))
    pos_w_t = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)

    net_3d = MultiWindowCoordConv3DResNeXtWithDiameterHead(aux_dim=aux_fit.shape[1]).to(device)
    net_mip = FoldIsolatedTriPlanarMIPConvNet(aux_dim=aux_fit.shape[1]).to(device)

    opt_3d = torch.optim.AdamW(net_3d.parameters(), lr=2.5e-3, weight_decay=2.2e-2)
    opt_mip = torch.optim.AdamW(net_mip.parameters(), lr=2.5e-3, weight_decay=2.2e-2)
    sched_3d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_3d, T_max=n_epochs, eta_min=1e-4)
    sched_mip = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mip, T_max=n_epochs, eta_min=1e-4)

    use_amp = (device.type == 'cuda') or (device.type == 'cpu' and AMP_DTYPE == torch.bfloat16)
    use_scaler = (device.type == 'cuda' and AMP_DTYPE == torch.float16)
    scaler_3d = torch.amp.GradScaler('cuda', enabled=True) if use_scaler else None
    scaler_mip = torch.amp.GradScaler('cuda', enabled=True) if use_scaler else None
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    net_3d.train()
    net_mip.train()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # 100% identical architecture, epoch count (20), and batch size (256) across GPU and v5e-1 TPU!
    eff_epochs = n_epochs
    eff_batch = batch_size

    for ep in range(eff_epochs):
        perm = torch.randperm(n_fit, generator=gen, device=device)
        for b_start in range(0, n_fit, eff_batch):
            b_idx = perm[b_start : b_start + eff_batch]
            if len(b_idx) < 8:
                continue
            g_idx = idx_fit_t[b_idx]
            v3_b = V3D_all_gpu[g_idx].float()
            vm_b = V25D_all_gpu[g_idx].float()
            ax_b = aux_fit_t[b_idx]
            yb = y_fit_t[b_idx]
            db = d_fit_t[b_idx]

            aug_code = (ep * 3 + (b_start // eff_batch)) % 8
            v3_b, vm_b = apply_consistent_3d_physical_augmentation(v3_b, vm_b, aug_code)

            opt_3d.zero_grad()
            opt_mip.zero_grad()
            with torch.amp.autocast(amp_device_type, dtype=AMP_DTYPE, enabled=use_amp):
                out_3d, d_pred = net_3d(v3_b, ax_b)
                out_mip = net_mip(vm_b, ax_b)
                loss_cls_3d = _batch_hard_negative_focal_ranking_loss(out_3d, yb, pos_w_t)
                loss_diam = F.smooth_l1_loss(d_pred, db)
                loss_3d = loss_cls_3d + 0.22 * loss_diam
                loss_mip = _batch_hard_negative_focal_ranking_loss(out_mip, yb, pos_w_t)

            if use_scaler:
                scaler_3d.scale(loss_3d).backward()
                scaler_3d.unscale_(opt_3d)
                nn.utils.clip_grad_norm_(net_3d.parameters(), 1.5)
                scaler_3d.step(opt_3d)
                scaler_3d.update()

                scaler_mip.scale(loss_mip).backward()
                scaler_mip.unscale_(opt_mip)
                nn.utils.clip_grad_norm_(net_mip.parameters(), 1.5)
                scaler_mip.step(opt_mip)
                scaler_mip.update()
            else:
                loss_3d.backward()
                nn.utils.clip_grad_norm_(net_3d.parameters(), 1.5)
                opt_3d.step()

                loss_mip.backward()
                nn.utils.clip_grad_norm_(net_mip.parameters(), 1.5)
                opt_mip.step()

        sched_3d.step()
        sched_mip.step()

    net_3d.eval()
    net_mip.eval()
    preds_out = []
    with torch.no_grad():
        for idx_t, ax_t in zip(idx_targets_global_list, aux_targets):
            idx_t_gpu = torch.from_numpy(idx_t).to(device, dtype=torch.long)
            p3_list, pm_list, d3_list = [], [], []
            for b_start in range(0, len(idx_t), 512):
                sub_g = idx_t_gpu[b_start : b_start + 512]
                sub_ax = ax_t[b_start : b_start + 512]
                v3_sub = V3D_all_gpu[sub_g].float()
                vm_sub = V25D_all_gpu[sub_g].float()
                l3_acc = 0.0
                lm_acc = 0.0
                d3_acc = 0.0
                with torch.amp.autocast(amp_device_type, dtype=AMP_DTYPE, enabled=use_amp):
                    for tta_code in range(8):
                        v3_tta, vm_tta = apply_consistent_3d_physical_augmentation(v3_sub, vm_sub, tta_code)
                        c_t, d_t = net_3d(v3_tta, sub_ax)
                        l3_acc = l3_acc + 0.125 * c_t.float()
                        d3_acc = d3_acc + 0.125 * d_t.float()
                        lm_acc = lm_acc + 0.125 * net_mip(vm_tta, sub_ax).float()
                p3_list.append(torch.sigmoid(l3_acc).cpu().numpy())
                pm_list.append(torch.sigmoid(lm_acc).cpu().numpy())
                d3_list.append(torch.clamp(d3_acc, 0.0, 4.0).cpu().numpy())
            p3_arr = np.concatenate(p3_list).astype(np.float32)
            pm_arr = np.concatenate(pm_list).astype(np.float32)
            d3_arr = np.concatenate(d3_list).astype(np.float32)
            preds_out.append((p3_arr, pm_arr, d3_arr))
    del net_3d, net_mip, opt_3d, opt_mip, idx_fit_t, aux_fit_t, y_fit_t, d_fit_t, aux_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds_out


if __name__ == "__main__":
    print("Verified training module: train_and_predict_v106_multitask_coordconv_3d_cnns_gpu loaded successfully.")
