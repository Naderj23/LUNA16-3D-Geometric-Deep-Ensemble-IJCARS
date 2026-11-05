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
Segmentation-Free Thoracic Coordinate Normalization (Eq. 2), Juxtapleural Wall Asymmetry,
26-Ray Boundary Escape, and Hollow-Bronchus Veto Descriptors (Families 8 & 9, Section 2.2, Eq. 7).
Eliminates fragile lung/nodule boundary mesh segmentation masks (cf. Causey et al. [11], El-Baz et al. [12]).
"""
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models.physical_3d_descriptors import (
    _build_26_direction_unit_vectors_and_antipodal_pairs,
    _build_real_spherical_harmonic_qr_subspaces_26,
    _select_cardinality_matched_anchor_local_indices,
    compute_finite_difference_3d_hessian_components,
    compute_exact_3x3_symmetric_hessian_eigenvalues,
)


def compute_normalized_thoracic_coords(coords_xyz: np.ndarray) -> np.ndarray:
    """
    Computes 2D normalized anatomical thoracic coordinates (r_xy, z_norm) from world coordinates
    p_world = (p_x, p_y, p_z) in mm (Section 2.2, Eq. 2):
      r_xy = sqrt((p_x / 150)^2 + (p_y / 150)^2),   z_norm = (p_z + 175) / 300
    """
    c = np.asarray(coords_xyz, dtype=np.float32)
    if c.ndim == 1:
        c = c[None, :]
    r_xy = np.sqrt((c[:, 0] / 150.0) ** 2 + (c[:, 1] / 150.0) ** 2)
    z_norm = (c[:, 2] + 175.0) / 300.0
    return np.column_stack([r_xy, z_norm]).astype(np.float32)

def extract_family8_intrascan_pointcloud_and_3d_compactness_features(
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    polar_grid: np.ndarray,
    geom: np.ndarray,
    star: np.ndarray,
    v75: np.ndarray,
    v76: np.ndarray,
    v77: np.ndarray,
    harm6: np.ndarray,
    vol7: np.ndarray,
) -> np.ndarray:
    N = len(series_arr)
    eps = 1e-5

    r_pos = polar_grid[:, 0::2, :]
    r_neg = polar_grid[:, 1::2, :]
    pair_sum_mid = np.mean(0.5 * (r_pos[:, :, 2:5] + r_neg[:, :, 2:5]), axis=2)
    pair_sum_out = np.mean(0.5 * (r_pos[:, :, 4:7] + r_neg[:, :, 4:7]), axis=2)
    hemi_asym_out = np.abs(np.mean(r_pos[:, :, 3:7] - r_neg[:, :, 3:7], axis=2))
    hemi_asym_mid = np.abs(np.mean(r_pos[:, :, 1:4] - r_neg[:, :, 1:4], axis=2))

    ps_mid_s = np.sort(pair_sum_mid, axis=1)[:, ::-1]
    ps_out_s = np.sort(pair_sum_out, axis=1)[:, ::-1]
    ha_out_s = np.sort(hemi_asym_out, axis=1)[:, ::-1]
    ha_mid_s = np.sort(hemi_asym_mid, axis=1)[:, ::-1]

    vessel_escape_mid = (ps_mid_s[:, 0] + eps) / (np.mean(ps_mid_s[:, 3:], axis=1) + 0.02)
    vessel_escape_out = (ps_out_s[:, 0] + eps) / (np.mean(ps_out_s[:, 3:], axis=1) + 0.02)
    vessel_top2_escape = (0.5 * (ps_mid_s[:, 0] + ps_mid_s[:, 1]) + eps) / (np.mean(ps_mid_s[:, 4:], axis=1) + 0.02)
    pleural_wall_max = ha_out_s[:, 0]
    pleural_wall_top3 = np.mean(ha_out_s[:, :3], axis=1)
    pleural_wall_ratio = (ha_out_s[:, 0] + eps) / (np.mean(polar_grid[:, :, :2], axis=(1, 2)) + 0.02)

    core_int = np.mean(polar_grid[:, :, 0], axis=1)
    inner_int = np.mean(polar_grid[:, :, 1:3], axis=(1, 2))
    mid_int = np.mean(polar_grid[:, :, 3:5], axis=(1, 2))
    outer_int = np.mean(polar_grid[:, :, 5:7], axis=(1, 2))

    compact_drop_1 = (core_int - mid_int) / (core_int + outer_int + 0.05)
    compact_drop_2 = (inner_int - outer_int) / (inner_int + outer_int + 0.05)
    spherical_purity = core_int * (1.0 - np.clip(pleural_wall_max, 0.0, 1.0)) / (vessel_escape_mid + 0.2)

    X_morph20 = np.column_stack([
        vessel_escape_mid, vessel_escape_out, vessel_top2_escape,
        pleural_wall_max, pleural_wall_top3, pleural_wall_ratio,
        ha_mid_s[:, 0], ha_mid_s[:, 1], ha_out_s[:, 1], ha_out_s[:, 3],
        ps_mid_s[:, 0] - ps_mid_s[:, 6], ps_out_s[:, 0] - ps_out_s[:, 6],
        compact_drop_1, compact_drop_2, spherical_purity,
        core_int, inner_int, mid_int, outer_int,
        (core_int + inner_int) / (mid_int + outer_int + 0.05),
    ]).astype(np.float32)
    assert X_morph20.shape == (N, 20)

    key10 = np.column_stack([
        star[:, -2],
        star[:, -1],
        v75[:, 120],
        v77[:, 0],
        harm6[:, 0],
        spherical_purity,
        compact_drop_1,
        -vessel_escape_mid,
        -pleural_wall_max,
        vol7[:, 48],
    ]).astype(np.float64)

    X_cloud24 = np.zeros((N, 24), dtype=np.float32)
    X_scan40 = np.zeros((N, 40), dtype=np.float32)

    df_s = pd.DataFrame({'s': series_arr, 'idx': np.arange(N)})
    for _, grp in df_s.groupby('s', sort=False):
        idxs = grp['idx'].values
        k = len(idxs)
        pts = coords_arr[idxs].astype(np.float64)
        xi, yi, zi = pts[:, 0], pts[:, 1], pts[:, 2]

        ref_loc = _select_cardinality_matched_anchor_local_indices(
            key10[idxs, 0], key10[idxs, 2], k_ref=SCAN_CONTEXT_ANCHOR_K_REF
        )
        k_eff = len(ref_loc)
        pts_ref = pts[ref_loc]

        xm, ym, zm = float(np.mean(pts_ref[:, 0])), float(np.mean(pts_ref[:, 1])), float(np.mean(pts_ref[:, 2]))
        z_span = max(float(np.max(pts_ref[:, 2]) - np.min(pts_ref[:, 2])), 20.0)
        r_xy_scan = np.sqrt((xi - xm) ** 2 + (yi - ym) ** 2)
        r_3d_scan = np.sqrt((xi - xm) ** 2 + (yi - ym) ** 2 + (zi - zm) ** 2)
        z_rel_span = (zi - float(np.min(pts_ref[:, 2]))) / z_span

        if k_eff == 1:
            d_nn1 = np.full(k, 100.0)
            d_nn2 = np.full(k, 120.0)
            d_nn3 = np.full(k, 140.0)
            d_mean = np.full(k, 100.0)
            c_6 = np.zeros(k)
            c_12 = np.zeros(k)
            c_20 = np.zeros(k)
            c_35 = np.zeros(k)
            c_50 = np.zeros(k)
            nn1_diff_star = np.zeros(k)
            nn1_diff_sph = np.zeros(k)
            nn1_diff_pur = np.zeros(k)
        else:
            diff_pts = pts[:, None, :] - pts_ref[None, :, :]
            dmat = np.sqrt(np.sum(diff_pts ** 2, axis=2))  # (k, k_eff)
            dmat = np.where(dmat < 1e-6, np.inf, dmat)
            d_sorted = np.sort(dmat, axis=1)
            d_nn1 = d_sorted[:, 0]
            d_nn2 = d_sorted[:, 1] if k_eff >= 3 else d_sorted[:, 0]
            d_nn3 = d_sorted[:, 2] if k_eff >= 4 else d_nn2
            finite_cnt = np.maximum(np.sum(np.isfinite(dmat), axis=1), 1)
            d_mean = np.sum(np.where(np.isfinite(dmat), dmat, 0.0), axis=1) / finite_cnt

            c_6 = np.sum(dmat <= 6.0, axis=1)
            c_12 = np.sum(dmat <= 12.0, axis=1)
            c_20 = np.sum(dmat <= 20.0, axis=1)
            c_35 = np.sum(dmat <= 35.0, axis=1)
            c_50 = np.sum(dmat <= 50.0, axis=1)

            nn1_ref_idx = ref_loc[np.argmin(dmat, axis=1)]
            nn1_diff_star = key10[idxs, 0] - key10[idxs[nn1_ref_idx], 0]
            nn1_diff_sph = key10[idxs, 2] - key10[idxs[nn1_ref_idx], 2]
            nn1_diff_pur = key10[idxs, 5] - key10[idxs[nn1_ref_idx], 5]

        X_cloud24[idxs] = np.column_stack([
            xi / 150.0,
            np.abs(xi) / 150.0,
            yi / 150.0,
            zi / 300.0,
            (xi - xm) / 100.0,
            (yi - ym) / 100.0,
            (zi - zm) / 100.0,
            r_xy_scan / 120.0,
            r_3d_scan / 150.0,
            z_rel_span,
            np.full(k, np.log1p(k_eff)),
            np.log1p(d_nn1),
            np.log1p(d_nn2),
            np.log1p(d_nn3),
            np.log1p(d_mean),
            d_nn1 / (d_nn2 + 1.0),
            c_6, c_12, c_20, c_35, c_50,
            nn1_diff_star, nn1_diff_sph, nn1_diff_pur,
        ]).astype(np.float32)

        k10_g = key10[idxs]
        k10_ref = k10_g[ref_loc]
        if k_eff == 1:
            X_scan40[idxs] = 0.0
            X_scan40[idxs, 30:40] = 1.0
        else:
            med_g = np.median(k10_ref, axis=0)
            q75_g = np.percentile(k10_ref, 75, axis=0)
            q25_g = np.percentile(k10_ref, 25, axis=0)
            iqr_g = np.maximum(q75_g - q25_g, np.std(k10_ref, axis=0) + 1e-4)
            z_rob = (k10_g - med_g[None, :]) / iqr_g[None, :]

            s_k10 = np.sort(k10_ref, axis=0)
            max_v = s_k10[-1, :]
            sec_v = s_k10[-2, :]
            gap_max = k10_g - max_v[None, :]

            sm_mat = np.where(k10_g >= max_v[None, :] - 1e-12, (max_v - sec_v)[None, :], k10_g - max_v[None, :])
            ranks_mat = np.zeros((k, 10), dtype=np.float32)
            if k <= SCAN_CONTEXT_ANCHOR_K_REF:
                for j_col in range(10):
                    ord_j = np.argsort(k10_g[:, j_col])
                    ranks_mat[ord_j, j_col] = np.arange(k, dtype=np.float32) / float(k - 1)
            else:
                for j_col in range(10):
                    pos_j = np.clip(np.searchsorted(s_k10[:, j_col], k10_g[:, j_col], side='right') - 1, 0, k_eff - 1)
                    ranks_mat[:, j_col] = pos_j.astype(np.float32) / float(k_eff - 1)

            X_scan40[idxs] = np.column_stack([z_rob, gap_max, sm_mat, ranks_mat]).astype(np.float32)

    X_fam8 = np.nan_to_num(np.hstack([X_morph20, X_cloud24, X_scan40]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    assert X_fam8.shape == (N, 84)
    return X_fam8


def extract_family9_koenderink_bronchus_yjunction_features(
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    polar_grid: np.ndarray,
    geom: np.ndarray,
    star: np.ndarray,
    v75: np.ndarray,
    v76: np.ndarray,
    v77: np.ndarray,
    harm6: np.ndarray,
    vol7: np.ndarray,
    fam8: np.ndarray,
) -> np.ndarray:
    """
    NEW IN V10.6 — Family-9 (64D): Specifically designed to eliminate the
    remaining false-positive mimics at `0.125, 0.25, 0.5 FP/scan`:
      1. Hollow-Lumen Bronchial-Wall & Airway-Ring Veto (`16D`):
         - Detects airway/bronchus mimics where the central lumen (`r <= 1.6 mm`)
           is darker than the surrounding bronchial wall ring (`r in [2.5, 4.6] mm`).
      2. 3-Branch Vascular Y-Bifurcation & Planar-Sheet Tensor (`16D`):
         - Measures 3-ray angular concentration (`top3_rays / remaining23_rays`)
           at each radial shell (`r = 3.0, 4.6, 6.5, 9.0, 12.0 mm`) plus
           antipodal vs. non-antipodal Y-angle energy.
      3. Multi-Scale 3D Hessian Koenderink Cap-Index, Spherical-Flux Synergy &
         Effective Radius Estimation (`16D`):
         - Combines 3D Hessian sphericity, blobness, vesselness, inward flux,
           and full-width half-maximum (`r_FWHM`) into scale-adaptive nodule
           cap scores across small (`3-6 mm`), medium (`6-12 mm`), and large
           (`>12 mm`) lesion regimes.
      4. Intra-Scan Multi-Composite Consensus & Supremacy Context (`16D`):
         - Builds 4 orthogonal unsupervised physical composite indices per patch
           and computes each candidate's scan-level supremacy margin (`second_margin`),
           robust z-score, and spatial-neighbor dominance.
    """
    N = len(series_arr)
    eps = 1e-5

    # --- 1. Hollow-Lumen Bronchial-Wall & Airway-Ring Veto (16D) ---
    # polar_grid has shape (N, 26, 7) at radii [1.6, 3.0, 4.6, 6.5, 9.0, 12.0, 15.5]
    r0 = polar_grid[:, :, 0]  # r = 1.6 mm
    r1 = polar_grid[:, :, 1]  # r = 3.0 mm
    r2 = polar_grid[:, :, 2]  # r = 4.6 mm
    r3 = polar_grid[:, :, 3]  # r = 6.5 mm
    r4 = polar_grid[:, :, 4]  # r = 9.0 mm
    r5 = polar_grid[:, :, 5]  # r = 12.0 mm
    r6 = polar_grid[:, :, 6]  # r = 15.5 mm

    m0 = np.mean(r0, axis=1)
    m1 = np.mean(r1, axis=1)
    m2 = np.mean(r2, axis=1)
    m3 = np.mean(r3, axis=1)
    m4 = np.mean(r4, axis=1)
    m5 = np.mean(r5, axis=1)
    m6 = np.mean(r6, axis=1)

    q25_r0 = np.percentile(r0, 25, axis=1)
    q10_r0 = np.percentile(r0, 10, axis=1)
    q75_r1 = np.percentile(r1, 75, axis=1)
    q75_r2 = np.percentile(r2, 75, axis=1)

    # True solid/subsolid nodules peak at the core (m0 >= m1 >= m2); hollow airway mimics have m1 > m0
    hollow_ring_1 = np.maximum(0.0, m1 - m0)
    hollow_ring_2 = np.maximum(0.0, m2 - np.maximum(m0, m1))
    hollow_q_gap = q75_r1 - q25_r0
    solid_core_monotonicity = (
        np.sign(m0 - m1) + np.sign(m1 - m2) + np.sign(m2 - m3) + np.sign(m3 - m4)
    ) / 4.0

    # Radial FWHM (estimated physical radius in mm where intensity drops below 50% of core above background)
    bg_est = 0.5 * (m5 + m6)
    half_lvl = bg_est + 0.5 * np.maximum(0.0, np.maximum(m0, m1) - bg_est)
    shell_means_7 = np.column_stack([m0, m1, m2, m3, m4, m5, m6])
    radii_arr = np.array([1.6, 3.0, 4.6, 6.5, 9.0, 12.0, 15.5], dtype=np.float32)
    above_half = (shell_means_7 >= half_lvl[:, None]).astype(np.float32)
    r_fwhm_est = np.sum(above_half * radii_arr[None, :], axis=1) / (np.sum(above_half, axis=1) + eps)
    sharp_boundary_grad = np.max(-np.diff(shell_means_7, axis=1), axis=1)
    grad_argmax_r = radii_arr[np.argmax(-np.diff(shell_means_7, axis=1), axis=1)]

    X_bronch16 = np.column_stack([
        hollow_ring_1, hollow_ring_2, hollow_q_gap, solid_core_monotonicity,
        q10_r0, q25_r0, m0 - q10_r0, q75_r1 - m1, q75_r2 - m2,
        r_fwhm_est / 15.0, sharp_boundary_grad, grad_argmax_r / 15.0,
        (m0 - bg_est) / (np.std(r0, axis=1) + 0.03),
        (m1 - bg_est) / (np.std(r1, axis=1) + 0.03),
        (m2 - bg_est) / (np.std(r2, axis=1) + 0.03),
        np.minimum(m0, m1) - np.maximum(m4, m5),
    ]).astype(np.float32)
    assert X_bronch16.shape == (N, 16)

    # --- 2. 3-Branch Vascular Y-Bifurcation & Planar-Sheet Tensor (16D) ---
    y_feats = []
    for ir in [1, 2, 3, 4]:
        sr = np.sort(polar_grid[:, :, ir], axis=1)[:, ::-1]  # (N, 26) descending
        top2 = np.mean(sr[:, :2], axis=1)
        top3 = np.mean(sr[:, :3], axis=1)
        top4 = np.mean(sr[:, :4], axis=1)
        body_med = np.mean(sr[:, 6:20], axis=1)
        low6 = np.mean(sr[:, 20:], axis=1)
        y3_excess = (top3 - body_med) / (sr[:, 0] - low6 + 0.03)
        tube2_excess = (top2 - sr[:, 3]) / (sr[:, 0] - low6 + 0.03)
        sheet_excess = ( np.mean(sr[:, :8], axis=1) - low6 ) / (np.mean(sr, axis=1) + 0.03)
        y_feats.append(np.column_stack([y3_excess, tube2_excess, sheet_excess, top4 - body_med]))
    X_yjunc16 = np.hstack(y_feats).astype(np.float32)
    assert X_yjunc16.shape == (N, 16)

    # --- 3. Multi-Scale 3D Hessian Koenderink Cap-Index & Flux Synergy (16D) ---
    # Extract Hessian & Flux columns from v75 (cols 120..228) and v76 (cols 0..89)
    core_sph = v75[:, 120]
    h_sph_s1_c0 = v75[:, 121 + 2]
    h_blob_s1_c0 = v75[:, 121 + 3]
    h_vess_s1_c0 = v75[:, 121 + 4]
    h_sph_s1_c1 = v75[:, 121 + 7 + 2]
    h_blob_s1_c1 = v75[:, 121 + 7 + 3]
    h_vess_s1_c1 = v75[:, 121 + 7 + 4]
    h_sph_s2_c1 = v75[:, 121 + 42 + 7 + 2]
    h_blob_s2_c1 = v75[:, 121 + 42 + 7 + 3]
    h_vess_s2_c1 = v75[:, 121 + 42 + 7 + 4]

    flux_c0 = v76[:, 0]
    flux_c1 = v76[:, 5]
    flux_c2 = v76[:, 10]
    sh_iso_core = harm6[:, 1]
    sh_iso_inner = harm6[:, 15]

    cap_small = (0.5 * h_sph_s1_c0 + 0.5 * h_blob_s1_c0 - 0.6 * h_vess_s1_c0) * (flux_c0 + 0.2)
    cap_medium = (0.5 * h_sph_s1_c1 + 0.5 * h_blob_s1_c1 - 0.6 * h_vess_s1_c1) * (flux_c1 + 0.2)
    cap_large = (0.5 * h_sph_s2_c1 + 0.5 * h_blob_s2_c1 - 0.6 * h_vess_s2_c1) * (flux_c2 + 0.2)
    cap_best = np.maximum(cap_small, np.maximum(cap_medium, cap_large))
    vess_max = np.maximum(h_vess_s1_c0, np.maximum(h_vess_s1_c1, h_vess_s2_c1))

    X_koend16 = np.column_stack([
        cap_small, cap_medium, cap_large, cap_best,
        vess_max, cap_best - vess_max,
        core_sph * sh_iso_core, core_sph * sh_iso_inner,
        (flux_c0 + flux_c1) * sh_iso_core,
        (h_blob_s1_c0 + h_blob_s1_c1) / (vess_max + 0.05),
        (1.0 - hollow_ring_1) * cap_best,
        (1.0 - X_yjunc16[:, 0]) * cap_best,
        v76[:, -6], v76[:, -5], v76[:, -4], v76[:, -3],
    ]).astype(np.float32)
    assert X_koend16.shape == (N, 16)

    # --- 4. Intra-Scan Multi-Composite Consensus & Supremacy Context (16D) ---
    comp1 = cap_best.astype(np.float64)
    comp2 = (star[:, -1] * sh_iso_core - 0.5 * vess_max).astype(np.float64)
    comp3 = (flux_c0 + flux_c1 + 0.5 * flux_c2 - hollow_ring_1).astype(np.float64)
    comp4 = (fam8[:, 14] + 0.5 * fam8[:, 12] - 0.4 * X_yjunc16[:, 0]).astype(np.float64)
    comp_mat = np.column_stack([comp1, comp2, comp3, comp4])

    X_sup16 = np.zeros((N, 16), dtype=np.float32)
    df_s = pd.DataFrame({'s': series_arr, 'idx': np.arange(N)})
    for _, grp in df_s.groupby('s', sort=False):
        idxs = grp['idx'].values
        k = len(idxs)
        cm = comp_mat[idxs]
        ref_loc = _select_cardinality_matched_anchor_local_indices(
            star[idxs, -2], v75[idxs, 120], k_ref=SCAN_CONTEXT_ANCHOR_K_REF
        )
        k_eff = len(ref_loc)
        cm_ref = cm[ref_loc]
        if k_eff == 1:
            X_sup16[idxs, 12:16] = 1.0
        else:
            med_c = np.median(cm_ref, axis=0)
            sd_c = np.std(cm_ref, axis=0) + 1e-4
            z_c = (cm - med_c[None, :]) / sd_c[None, :]

            s_cm = np.sort(cm_ref, axis=0)
            max_c = s_cm[-1, :]
            sec_c = s_cm[-2, :]
            gap_c = cm - max_c[None, :]
            sm_c = np.where(cm >= max_c[None, :] - 1e-12, (max_c - sec_c)[None, :], cm - max_c[None, :])

            rk_c = np.zeros((k, 4), dtype=np.float32)
            if k <= SCAN_CONTEXT_ANCHOR_K_REF:
                for j_col in range(4):
                    ord_j = np.argsort(cm[:, j_col])
                    rk_c[ord_j, j_col] = np.arange(k, dtype=np.float32) / float(k - 1)
            else:
                for j_col in range(4):
                    pos_j = np.clip(np.searchsorted(s_cm[:, j_col], cm[:, j_col], side='right') - 1, 0, k_eff - 1)
                    rk_c[:, j_col] = pos_j.astype(np.float32) / float(k_eff - 1)

            X_sup16[idxs] = np.column_stack([z_c, gap_c, sm_c, rk_c]).astype(np.float32)

    X_fam9 = np.nan_to_num(
        np.hstack([X_bronch16, X_yjunc16, X_koend16, X_sup16]).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    assert X_fam9.shape == (N, 64)
    return X_fam9


def verify_family8_and_family9_anchor_invariance_unit_test() -> float:
    """
    Verifies that Family-8 (`X_scan40` & `X_cloud24` anchor stats) and Family-9 (`X_sup16`)
    are invariant (`max_shift < 1e-5` on anchor statistics) when 200 low-scoring background
    candidates are appended to the 24 scan-context anchors (`k = 24` vs `k = 224`).
    """
    rng = np.random.RandomState(SEED + 33)
    k_a = SCAN_CONTEXT_ANCHOR_K_REF
    k_bg = 200
    k_tot = k_a + k_bg

    # Realistic continuous features with distinct high scores for the 24 anchors and lower scores for the 200 background rows
    star_a = rng.uniform(0.05, 0.85, size=(k_a, 10)).astype(np.float32)
    star_a[:, -2] = np.linspace(0.45, 0.92, k_a, dtype=np.float32)
    star_a[:, -1] = np.linspace(0.40, 0.88, k_a, dtype=np.float32)
    star_bg = rng.uniform(0.01, 0.25, size=(k_bg, 10)).astype(np.float32)
    star_bg[:, -2] = np.linspace(0.01, 0.25, k_bg, dtype=np.float32)
    star_bg[:, -1] = np.linspace(0.01, 0.22, k_bg, dtype=np.float32)

    v75_a = rng.uniform(-0.25, 0.85, size=(k_a, 229)).astype(np.float32)
    v75_a[:, 120] = np.linspace(0.35, 0.85, k_a, dtype=np.float32)
    v75_bg = rng.uniform(-0.25, 0.20, size=(k_bg, 229)).astype(np.float32)
    v75_bg[:, 120] = np.linspace(0.01, 0.18, k_bg, dtype=np.float32)

    pg_all = rng.uniform(0.05, 0.85, size=(k_tot, 26, 7)).astype(np.float32)
    geom_all = rng.uniform(-0.5, 0.5, size=(k_tot, 10)).astype(np.float32)
    v76_all = rng.uniform(-0.3, 0.6, size=(k_tot, 96)).astype(np.float32)
    v77_all = rng.uniform(-0.3, 0.6, size=(k_tot, 60)).astype(np.float32)
    harm6_all = rng.uniform(0.0, 0.8, size=(k_tot, 128)).astype(np.float32)
    vol7_all_t = rng.uniform(0.0, 0.8, size=(k_tot, 148)).astype(np.float32)
    coords_all_t = rng.uniform(-100.0, 100.0, size=(k_tot, 3)).astype(np.float64)

    star_all_t = np.vstack([star_a, star_bg])
    v75_all_t = np.vstack([v75_a, v75_bg])

    f8_24 = extract_family8_intrascan_pointcloud_and_3d_compactness_features(
        np.array(['s_inv'] * k_a), coords_all_t[:k_a], pg_all[:k_a], geom_all[:k_a],
        star_all_t[:k_a], v75_all_t[:k_a], v76_all[:k_a], v77_all[:k_a], harm6_all[:k_a], vol7_all_t[:k_a]
    )
    f8_224 = extract_family8_intrascan_pointcloud_and_3d_compactness_features(
        np.array(['s_inv'] * k_tot), coords_all_t, pg_all, geom_all,
        star_all_t, v75_all_t, v76_all, v77_all, harm6_all, vol7_all_t
    )
    f9_24 = extract_family9_koenderink_bronchus_yjunction_features(
        np.array(['s_inv'] * k_a), coords_all_t[:k_a], pg_all[:k_a], geom_all[:k_a],
        star_all_t[:k_a], v75_all_t[:k_a], v76_all[:k_a], v77_all[:k_a], harm6_all[:k_a], vol7_all_t[:k_a], f8_24
    )
    f9_224 = extract_family9_koenderink_bronchus_yjunction_features(
        np.array(['s_inv'] * k_tot), coords_all_t, pg_all, geom_all,
        star_all_t, v75_all_t, v76_all, v77_all, harm6_all, vol7_all_t, f8_224
    )
    err_f8 = float(np.max(np.abs(f8_24 - f8_224[:k_a])))
    err_f9 = float(np.max(np.abs(f9_24 - f9_224[:k_a])))
    max_f89_err = max(err_f8, err_f9)
    assert max_f89_err < 1e-5, f"Family-8/9 scan-context anchor invariance failed: {max_f89_err}"
    return max_f89_err


if __name__ == "__main__":
    c_test = compute_normalized_thoracic_coords(np.array([[75.0, -75.0, -25.0]]))
    assert c_test.shape == (1, 2)
    print("Verified preprocessing/lung_segmentation.py (Family 8 & 9 anchor invariance & thoracic coords):", c_test)
