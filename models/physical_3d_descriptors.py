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
Complete Production Implementation of the 9 Physical, Differential-Geometric, and
Spherical-Harmonic 3D Descriptor Families (D_pure = 1,166D, Zero CNNs; Section 2.2, Eqs. 2–7).
"""

def compute_finite_difference_3d_hessian_components(Vc: np.ndarray):
    center = Vc[:, 1:-1, 1:-1, 1:-1]
    hzz = Vc[:, 2:, 1:-1, 1:-1] - 2.0 * center + Vc[:, :-2, 1:-1, 1:-1]
    hyy = Vc[:, 1:-1, 2:, 1:-1] - 2.0 * center + Vc[:, 1:-1, :-2, 1:-1]
    hxx = Vc[:, 1:-1, 1:-1, 2:] - 2.0 * center + Vc[:, 1:-1, 1:-1, :-2]

    hzy = 0.25 * (
        Vc[:, 2:, 2:, 1:-1] - Vc[:, 2:, :-2, 1:-1]
        - Vc[:, :-2, 2:, 1:-1] + Vc[:, :-2, :-2, 1:-1]
    )
    hzx = 0.25 * (
        Vc[:, 2:, 1:-1, 2:] - Vc[:, 2:, 1:-1, :-2]
        - Vc[:, :-2, 1:-1, 2:] + Vc[:, :-2, 1:-1, :-2]
    )
    hyx = 0.25 * (
        Vc[:, 1:-1, 2:, 2:] - Vc[:, 1:-1, 2:, :-2]
        - Vc[:, 1:-1, :-2, 2:] + Vc[:, 1:-1, :-2, :-2]
    )
    return hzz, hyy, hxx, hzy, hzx, hyx


def compute_exact_3x3_symmetric_hessian_eigenvalues(
    hzz: np.ndarray,
    hyy: np.ndarray,
    hxx: np.ndarray,
    hzy: np.ndarray,
    hzx: np.ndarray,
    hyx: np.ndarray,
):
    a11 = hzz.astype(np.float64)
    a22 = hyy.astype(np.float64)
    a33 = hxx.astype(np.float64)
    a12 = hzy.astype(np.float64)
    a13 = hzx.astype(np.float64)
    a23 = hyx.astype(np.float64)

    p1 = a12 ** 2 + a13 ** 2 + a23 ** 2
    q = (a11 + a22 + a33) / 3.0
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))

    inv_p = np.where(p > 1e-12, 1.0 / p, 0.0)
    b11 = (a11 - q) * inv_p
    b22 = (a22 - q) * inv_p
    b33 = (a33 - q) * inv_p
    b12 = a12 * inv_p
    b13 = a13 * inv_p
    b23 = a23 * inv_p

    det_B = (
        b11 * (b22 * b33 - b23 * b23)
        - b12 * (b12 * b33 - b23 * b13)
        + b13 * (b12 * b23 - b22 * b13)
    )
    r = np.clip(0.5 * det_B, -1.0, 1.0)
    phi = np.arccos(r) / 3.0

    eig_max = q + 2.0 * p * np.cos(phi)
    eig_min = q + 2.0 * p * np.cos(phi + (2.0 * np.pi / 3.0))
    eig_mid = 3.0 * q - eig_max - eig_min

    lam1 = np.minimum(eig_min, np.minimum(eig_mid, eig_max)).astype(np.float32)
    lam3 = np.maximum(eig_min, np.maximum(eig_mid, eig_max)).astype(np.float32)
    lam2 = ((a11 + a22 + a33) - lam1 - lam3).astype(np.float32)
    return lam1, lam2, lam3


def verify_hessian_finite_diff_and_eigensolver_unit_test():
    grid = np.arange(10, dtype=np.float64) - 4.5
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing='ij')
    H_true = np.array([
        [-3.4,  0.7, -0.5],
        [ 0.7, -2.1,  0.4],
        [-0.5,  0.4, -0.8],
    ], dtype=np.float64)
    V_quad = (
        0.5 * H_true[0, 0] * zz ** 2
        + 0.5 * H_true[1, 1] * yy ** 2
        + 0.5 * H_true[2, 2] * xx ** 2
        + H_true[0, 1] * zz * yy
        + H_true[0, 2] * zz * xx
        + H_true[1, 2] * yy * xx
        + 1.2 * zz - 0.9 * yy + 0.5 * xx + 3.0
    )[None, ...].astype(np.float32)

    hzz, hyy, hxx, hzy, hzx, hyx = compute_finite_difference_3d_hessian_components(V_quad)
    assert np.allclose(hzz, H_true[0, 0], atol=1e-4)
    assert np.allclose(hyy, H_true[1, 1], atol=1e-4)
    assert np.allclose(hxx, H_true[2, 2], atol=1e-4)
    assert np.allclose(hzy, H_true[0, 1], atol=1e-4)
    assert np.allclose(hzx, H_true[0, 2], atol=1e-4)
    assert np.allclose(hyx, H_true[1, 2], atol=1e-4)

    lam1, lam2, lam3 = compute_exact_3x3_symmetric_hessian_eigenvalues(hzz, hyy, hxx, hzy, hzx, hyx)
    eig_np = np.linalg.eigvalsh(H_true)
    max_err = max(
        float(np.max(np.abs(lam1 - eig_np[0]))),
        float(np.max(np.abs(lam2 - eig_np[1]))),
        float(np.max(np.abs(lam3 - eig_np[2]))),
    )
    assert max_err < 1e-4, f"3x3 Hessian eigensolver unit test failed: max_err={max_err}"
    return max_err


_HESSIAN_UNIT_TEST_MAX_ERR = verify_hessian_finite_diff_and_eigensolver_unit_test()


# ==============================================================================
# 3. INTRA-SCAN CONTEXTUAL FEATURE ENGINE & UNIT TEST FOR `second_margin`
# ==============================================================================
def _select_cardinality_matched_anchor_local_indices(
    star_core_scan: np.ndarray,
    v75_sph_scan: np.ndarray,
    fallback_score_scan: np.ndarray = None,
    k_ref: int = SCAN_CONTEXT_ANCHOR_K_REF,
) -> np.ndarray:
    """
    Selects at most `k_ref` (`24`) reference anchor indices within a scan ranked by
    intrinsic single-patch physical core score `0.65 * star_core + 0.35 * v75_sph`
    (which depends ONLY on each patch's own 48x48x48 voxels, independent of scan
    candidate population size!). When `len <= k_ref` (as in curated HDF5), returns
    all `0..len-1` indices. When `len > k_ref` (as in `.mhd` `candidates_V2.csv`),
    returns the top `k_ref` anchor indices so scan-context statistics have zero
    population-size covariate shift!
    """
    k = len(star_core_scan)
    if k <= k_ref:
        return np.arange(k, dtype=np.int64)
    intrinsic = 0.65 * star_core_scan.astype(np.float64) + 0.35 * v75_sph_scan.astype(np.float64)
    if float(np.std(intrinsic)) < 1e-8 and fallback_score_scan is not None:
        intrinsic = fallback_score_scan.astype(np.float64)
    ord_desc = np.argsort(-intrinsic, kind='mergesort')
    return np.sort(ord_desc[:k_ref])


def compute_intra_scan_contextual_features(
    logit_vec: np.ndarray,
    suids: np.ndarray,
    star_core: np.ndarray,
    v75_sph: np.ndarray,
    v77_pi: np.ndarray,
    anchor_k_ref: int = SCAN_CONTEXT_ANCHOR_K_REF,
) -> np.ndarray:
    n = len(logit_vec)
    p_vec = expit(np.clip(logit_vec, -20.0, 20.0))

    z_scan = np.zeros(n, dtype=np.float32)
    p_rank = np.zeros(n, dtype=np.float32)
    gap_to_max = np.zeros(n, dtype=np.float32)
    second_margin = np.zeros(n, dtype=np.float32)
    p_share = np.zeros(n, dtype=np.float32)
    l_minus_med = np.zeros(n, dtype=np.float32)
    l_minus_q75 = np.zeros(n, dtype=np.float32)
    n_cands_log = np.zeros(n, dtype=np.float32)
    n_high_in_scan = np.zeros(n, dtype=np.float32)
    star_scan_z = np.zeros(n, dtype=np.float32)
    sph_scan_z = np.zeros(n, dtype=np.float32)
    pi_scan_z = np.zeros(n, dtype=np.float32)

    df_g = pd.DataFrame({'s': suids, 'idx': np.arange(n)})
    for _, grp in df_g.groupby('s', sort=False):
        idxs = grp['idx'].values
        k = len(idxs)
        l_g = logit_vec[idxs]
        p_g = p_vec[idxs]

        if k <= anchor_k_ref:
            ref_loc = np.arange(k, dtype=np.int64)
        else:
            ref_loc = _select_cardinality_matched_anchor_local_indices(
                star_core[idxs], v75_sph[idxs], l_g, k_ref=anchor_k_ref
            )
        k_eff = len(ref_loc)
        l_ref = l_g[ref_loc]
        p_ref = p_g[ref_loc]

        n_cands_log[idxs] = np.log1p(k_eff)
        n_high_in_scan[idxs] = float(np.sum(p_ref > 0.25))

        if k_eff == 1:
            z_scan[idxs] = 0.0
            p_rank[idxs] = 1.0
            gap_to_max[idxs] = 0.0
            second_margin[idxs] = 2.0
            p_share[idxs] = 1.0
            l_minus_med[idxs] = 0.0
            l_minus_q75[idxs] = 0.0
        else:
            mu_l = float(np.mean(l_ref))
            sd_l = float(np.std(l_ref)) + 1e-4
            med_l = float(np.median(l_ref))
            q75_l = float(np.percentile(l_ref, 75))
            z_scan[idxs] = (l_g - mu_l) / sd_l
            l_minus_med[idxs] = l_g - med_l
            l_minus_q75[idxs] = l_g - q75_l

            if k <= anchor_k_ref:
                order = np.argsort(l_g)
                ranks = np.empty(k, dtype=np.float32)
                ranks[order] = np.arange(k, dtype=np.float32) / float(k - 1)
                p_rank[idxs] = ranks
                max_idx_local = order[-1]
                max_l = l_g[max_idx_local]
                second_l = l_g[order[-2]]
                gap_to_max[idxs] = l_g - max_l
                sm = l_g - max_l
                sm[max_idx_local] = max_l - second_l
                second_margin[idxs] = sm
            else:
                l_ref_sorted = np.sort(l_ref)
                max_l = l_ref_sorted[-1]
                second_l = l_ref_sorted[-2]
                gap_to_max[idxs] = l_g - max_l
                sm = np.where(l_g >= max_l - 1e-12, max_l - second_l, l_g - max_l)
                second_margin[idxs] = sm
                pos_in_ref = np.clip(np.searchsorted(l_ref_sorted, l_g, side='right') - 1, 0, k_eff - 1)
                p_rank[idxs] = pos_in_ref.astype(np.float32) / float(k_eff - 1)

            p_share[idxs] = p_g / (float(np.sum(p_ref)) + 1e-5)

            for arr_in, arr_out in [(star_core, star_scan_z), (v75_sph, sph_scan_z), (v77_pi, pi_scan_z)]:
                v_g = arr_in[idxs]
                v_ref = v_g[ref_loc]
                arr_out[idxs] = (v_g - float(np.mean(v_ref))) / (float(np.std(v_ref)) + 1e-4)

    return np.nan_to_num(np.column_stack([
        z_scan, p_rank, gap_to_max, second_margin, p_share,
        l_minus_med, l_minus_q75, n_cands_log, n_high_in_scan,
        star_scan_z, sph_scan_z, pi_scan_z,
    ]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def verify_second_margin_unit_test():
    test_logits = np.array([1.0, 4.5, 2.0, -1.0], dtype=np.float64)
    test_suids = np.array(['scanA', 'scanA', 'scanA', 'scanA'])
    zeros4 = np.zeros(4, dtype=np.float32)
    ctx = compute_intra_scan_contextual_features(test_logits, test_suids, zeros4, zeros4, zeros4)
    sm = ctx[:, 3]
    expected = np.array([-3.5, +2.5, -2.5, -5.5], dtype=np.float32)
    assert np.allclose(sm, expected, atol=1e-5), f"second_margin unit test failed: {sm} vs {expected}"


verify_second_margin_unit_test()


def verify_scan_context_population_invariance_unit_test():
    """
    Audit Gate 10 (Resolves Audit Point 3): Proves that Cardinality-Matched
    Reference-Anchor Scan-Context (`SCAN_CONTEXT_ANCHOR_K_REF = 24`) eliminates
    population-size covariate shift between curated scans (`k = 24`) and
    unscreened `.mhd` candidate pools (`k = 224` = 24 anchors + 200 background)!
    """
    rng = np.random.RandomState(SEED)
    k_a = SCAN_CONTEXT_ANCHOR_K_REF
    k_bg = 200
    l_a = rng.uniform(-1.5, 4.0, size=k_a)
    sc_a = rng.uniform(0.35, 0.90, size=k_a).astype(np.float32)
    sph_a = rng.uniform(0.25, 0.85, size=k_a).astype(np.float32)
    pi_a = rng.uniform(-0.2, 0.5, size=k_a).astype(np.float32)
    suids_a = np.array(['scan_test'] * k_a)

    ctx_small = compute_intra_scan_contextual_features(l_a, suids_a, sc_a, sph_a, pi_a)

    l_bg = rng.uniform(-6.0, -2.0, size=k_bg)
    sc_bg = rng.uniform(0.01, 0.20, size=k_bg).astype(np.float32)
    sph_bg = rng.uniform(0.01, 0.15, size=k_bg).astype(np.float32)
    pi_bg = rng.uniform(-0.5, -0.1, size=k_bg).astype(np.float32)

    ctx_large = compute_intra_scan_contextual_features(
        np.concatenate([l_a, l_bg]),
        np.array(['scan_test'] * (k_a + k_bg)),
        np.concatenate([sc_a, sc_bg]),
        np.concatenate([sph_a, sph_bg]),
        np.concatenate([pi_a, pi_bg]),
    )
    max_shift = float(np.max(np.abs(ctx_small - ctx_large[:k_a])))
    assert max_shift < 1e-5, f"Scan-context population invariance failed: max_shift={max_shift}"
    return max_shift


_SCAN_CTX_INVARIANCE_ERR = verify_scan_context_population_invariance_unit_test()


# ==============================================================================
# 4. DISCRETE ANGULAR & SPHERICAL-HARMONIC SUBSPACE PROJECTION (WITH BOTH
#    WITHIN-DEGREE `Q_l` AND GLOBAL-25 `Q_global25` QR BASES + ROTATION GATES)
# ==============================================================================
def _build_26_direction_unit_vectors_and_antipodal_pairs():
    dirs = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                norm = math.sqrt(dz * dz + dy * dy + dx * dx)
                dirs.append((dz / norm, dy / norm, dx / norm, dz, dy, dx))

    used = set()
    ordered_26 = []
    for i, d in enumerate(dirs):
        if i in used:
            continue
        dz, dy, dx = d[3], d[4], d[5]
        for j, d2 in enumerate(dirs):
            if d2[3] == -dz and d2[4] == -dy and d2[5] == -dx:
                used.add(i)
                used.add(j)
                ordered_26.append((d[0], d[1], d[2]))
                ordered_26.append((d2[0], d2[1], d2[2]))
                break
    return np.array(ordered_26, dtype=np.float64)


_U26_UNIT_DIRS = _build_26_direction_unit_vectors_and_antipodal_pairs()


def _build_real_spherical_harmonic_qr_subspaces_26(u_dirs: np.ndarray):
    z = u_dirs[:, 0]
    y = u_dirs[:, 1]
    x = u_dirs[:, 2]
    r2 = x * x + y * y + z * z

    B0 = np.ones((len(u_dirs), 1), dtype=np.float64)
    B1 = np.column_stack([x, y, z])
    B2 = np.column_stack([
        x * y, y * z, z * x,
        x * x - y * y,
        3.0 * z * z - r2,
    ])
    B3 = np.column_stack([
        y * (3.0 * x * x - y * y),
        x * y * z,
        y * (5.0 * z * z - r2),
        z * (5.0 * z * z - 3.0 * r2),
        x * (5.0 * z * z - r2),
        z * (x * x - y * y),
        x * (x * x - 3.0 * y * y),
    ])
    B4 = np.column_stack([
        x * y * (x * x - y * y),
        y * z * (3.0 * x * x - y * y),
        x * y * (7.0 * z * z - r2),
        y * z * (7.0 * z * z - 3.0 * r2),
        35.0 * (z ** 4) - 30.0 * (z ** 2) * r2 + 3.0 * (r2 ** 2),
        x * z * (7.0 * z * z - 3.0 * r2),
        (x * x - y * y) * (7.0 * z * z - r2),
        x * z * (x * x - 3.0 * y * y),
        x ** 4 - 6.0 * (x * x) * (y * y) + y ** 4,
    ])

    raw_blocks = [B0, B1, B2, B3, B4]
    P_within_list = []
    Q_within_list = []
    for B_l in raw_blocks:
        Q_l, _ = np.linalg.qr(B_l)
        Q_within_list.append(Q_l)
        P_within_list.append(Q_l @ Q_l.T)

    B_all25 = np.hstack(raw_blocks)
    Q_global25, _ = np.linalg.qr(B_all25)
    col_slices = [(0, 1), (1, 4), (4, 9), (9, 16), (16, 25)]
    Q_global_list = [Q_global25[:, s:e] for s, e in col_slices]
    P_global_list = [Q_g @ Q_g.T for Q_g in Q_global_list]
    return P_within_list, Q_within_list, P_global_list, Q_global_list, Q_global25


_SH_P_LIST, _SH_Q_LIST, _SH_P_GLOBAL_LIST, _SH_Q_GLOBAL_LIST, _SH_Q_GLOBAL25 = (
    _build_real_spherical_harmonic_qr_subspaces_26(_U26_UNIT_DIRS)
)


def audit_discrete_angular_subspace_properties(sample_volumes_128: np.ndarray, sample_labels_128: np.ndarray) -> dict:
    within_qr_err = max(
        float(np.max(np.abs(Q_l.T @ Q_l - np.eye(Q_l.shape[1]))))
        for Q_l in _SH_Q_LIST
    )
    global_qr_err = float(np.max(np.abs(_SH_Q_GLOBAL25.T @ _SH_Q_GLOBAL25 - np.eye(25))))

    cross_gram = {}
    for l1 in range(5):
        for l2 in range(l1 + 1, 5):
            ov = float(np.max(np.abs(_SH_Q_LIST[l1].T @ _SH_Q_LIST[l2])))
            if ov > 1e-6:
                cross_gram[f"({l1},{l2})"] = round(ov, 4)

    rot_mats_24 = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    R = np.zeros((3, 3), dtype=np.float64)
                    R[0, perm[0]] = s0
                    R[1, perm[1]] = s1
                    R[2, perm[2]] = s2
                    if round(np.linalg.det(R)) == 1:
                        rot_mats_24.append(R)
    assert len(rot_mats_24) == 24

    radii = [2.0, 3.5, 5.0, 7.0, 9.5, 12.5, 15.5]
    n_p = len(sample_volumes_128)
    assert n_p >= 64, f"Expected >=64 sample volumes, got {n_p}"

    def _sample_rays_batch(vols: np.ndarray, u_dirs_rot: np.ndarray) -> np.ndarray:
        B = len(vols)
        vt = torch.from_numpy(vols).to(device, dtype=torch.float32).unsqueeze(1)
        pts = np.zeros((1, 26, len(radii), 1, 3), dtype=np.float32)
        for idir in range(26):
            uz, uy, ux = u_dirs_rot[idir]
            for ir, rv in enumerate(radii):
                pts[0, idir, ir, 0, :] = [float(rv * ux) / 23.5, float(rv * uy) / 23.5, float(rv * uz) / 23.5]
        grid_t = torch.from_numpy(pts).to(device).expand(B, -1, -1, -1, -1)
        with torch.no_grad():
            sampled = F.grid_sample(vt, grid_t, mode='bilinear', padding_mode='border', align_corners=True)
        return sampled[:, 0, :, :, 0].cpu().numpy().astype(np.float64)

    F_orig = _sample_rays_batch(sample_volumes_128, _U26_UNIT_DIRS)
    E_orig_within = np.zeros((n_p, 5, len(radii)), dtype=np.float64)
    E_orig_global = np.zeros((n_p, 5, len(radii)), dtype=np.float64)
    for l_deg in range(5):
        proj_w = np.einsum('ij,bjr->bir', _SH_P_LIST[l_deg], F_orig)
        E_orig_within[:, l_deg, :] = np.sum(proj_w ** 2, axis=1)
        proj_g = np.einsum('ij,bjr->bir', _SH_P_GLOBAL_LIST[l_deg], F_orig)
        E_orig_global[:, l_deg, :] = np.sum(proj_g ** 2, axis=1)

    oh_within_corrs = {l_deg: [] for l_deg in range(5)}
    oh_global_corrs = {l_deg: [] for l_deg in range(5)}
    oh_global_rel_l1 = {l_deg: [] for l_deg in range(5)}

    for R in rot_mats_24:
        u_rot = _U26_UNIT_DIRS @ R.T
        F_rot = _sample_rays_batch(sample_volumes_128, u_rot)
        for l_deg in range(5):
            proj_w = np.einsum('ij,bjr->bir', _SH_P_LIST[l_deg], F_rot)
            E_w = np.sum(proj_w ** 2, axis=1)
            r_w = np.corrcoef(E_orig_within[:, l_deg, :].ravel(), E_w.ravel())[0, 1]
            oh_within_corrs[l_deg].append(float(r_w))

            proj_g = np.einsum('ij,bjr->bir', _SH_P_GLOBAL_LIST[l_deg], F_rot)
            E_g = np.sum(proj_g ** 2, axis=1)
            r_g = np.corrcoef(E_orig_global[:, l_deg, :].ravel(), E_g.ravel())[0, 1]
            oh_global_corrs[l_deg].append(float(r_g))
            rel_err = float(np.mean(np.abs(E_orig_global[:, l_deg, :] - E_g)) / (np.mean(E_orig_global[:, l_deg, :]) + 1e-8))
            oh_global_rel_l1[l_deg].append(rel_err)

    euler_angles_12 = [
        (17.0, (0, 1)), (28.0, (1, 2)), (33.0, (0, 2)), (-22.0, (0, 1)),
        (-37.0, (1, 2)), (41.0, (0, 2)), (19.0, (1, 2)), (-29.0, (0, 2)),
        (23.0, (0, 1)), (-31.0, (1, 2)), (38.0, (0, 1)), (-18.0, (0, 2)),
    ]
    euler_within_corrs = {l_deg: [] for l_deg in range(5)}
    euler_within_rel_l1 = {l_deg: [] for l_deg in range(5)}
    euler_global_corrs = {l_deg: [] for l_deg in range(5)}
    euler_global_rel_l1 = {l_deg: [] for l_deg in range(5)}

    for ang_deg, axes_pair in euler_angles_12:
        V_rot_128 = np.stack([
            ndimage_rotate(sample_volumes_128[i], angle=ang_deg, axes=axes_pair, reshape=False, order=3, mode='nearest')
            for i in range(n_p)
        ], axis=0)
        F_eul = _sample_rays_batch(V_rot_128, _U26_UNIT_DIRS)
        for l_deg in range(5):
            proj_w = np.einsum('ij,bjr->bir', _SH_P_LIST[l_deg], F_eul)
            E_eul_w = np.sum(proj_w ** 2, axis=1)
            r_eul_w = np.corrcoef(E_orig_within[:, l_deg, :].ravel(), E_eul_w.ravel())[0, 1]
            euler_within_corrs[l_deg].append(float(r_eul_w))
            rel_eul_w = float(np.mean(np.abs(E_orig_within[:, l_deg, :] - E_eul_w)) / (np.mean(E_orig_within[:, l_deg, :]) + 1e-8))
            euler_within_rel_l1[l_deg].append(rel_eul_w)

            proj_g = np.einsum('ij,bjr->bir', _SH_P_GLOBAL_LIST[l_deg], F_eul)
            E_eul_g = np.sum(proj_g ** 2, axis=1)
            r_eul_g = np.corrcoef(E_orig_global[:, l_deg, :].ravel(), E_eul_g.ravel())[0, 1]
            euler_global_corrs[l_deg].append(float(r_eul_g))
            rel_eul_g = float(np.mean(np.abs(E_orig_global[:, l_deg, :] - E_eul_g)) / (np.mean(E_orig_global[:, l_deg, :]) + 1e-8))
            euler_global_rel_l1[l_deg].append(rel_eul_g)

    oh_within_thresholds = {0: 0.995, 1: 0.975, 2: 0.950, 3: 0.930, 4: 0.880}
    oh_global_thresholds = {0: 0.995, 1: 0.975, 2: 0.950, 3: 0.920, 4: 0.900}
    euler_within_thresholds = {0: 0.990, 1: 0.940, 2: 0.880, 3: 0.800, 4: 0.850}
    euler_global_thresholds = {0: 0.990, 1: 0.940, 2: 0.880, 3: 0.780, 4: 0.820}

    deg_stats = {}
    for l_deg in range(5):
        mw_oh = float(np.mean(oh_within_corrs[l_deg]))
        mg_oh = float(np.mean(oh_global_corrs[l_deg]))
        mw_eu = float(np.mean(euler_within_corrs[l_deg]))
        mg_eu = float(np.mean(euler_global_corrs[l_deg]))

        assert mw_oh >= oh_within_thresholds[l_deg], f"O_h within_r gate failed for l={l_deg}: {mw_oh:.4f}"
        assert mg_oh >= oh_global_thresholds[l_deg], f"O_h globalQR_r gate failed for l={l_deg}: {mg_oh:.4f}"
        assert mw_eu >= euler_within_thresholds[l_deg], f"Euler within_r gate failed for l={l_deg}: {mw_eu:.4f}"
        assert mg_eu >= euler_global_thresholds[l_deg], f"Euler globalQR_r gate failed for l={l_deg}: {mg_eu:.4f}"

        deg_stats[l_deg] = {
            'n_modes': 2 * l_deg + 1,
            'oh_within_r': mw_oh,
            'oh_global_r': mg_oh,
            'oh_global_rel_l1': float(np.mean(oh_global_rel_l1[l_deg])),
            'euler_within_r': mw_eu,
            'euler_within_rel_l1': float(np.mean(euler_within_rel_l1[l_deg])),
            'euler_global_r': mg_eu,
            'euler_global_rel_l1': float(np.mean(euler_global_rel_l1[l_deg])),
        }

    return {
        'n_sample_patches': n_p,
        'n_pos_patches': int(np.sum(sample_labels_128 == 1)),
        'n_neg_patches': int(np.sum(sample_labels_128 == 0)),
        'within_qr_err': within_qr_err,
        'global_qr_err': global_qr_err,
        'cross_gram_offdiag': cross_gram,
        'deg_stats': deg_stats,
    }


def extract_discrete_angular_and_sh_subspace_features(
    polar_grid: np.ndarray,
    geom: np.ndarray,
    star: np.ndarray,
    v75: np.ndarray,
    v76: np.ndarray,
    v77: np.ndarray,
) -> np.ndarray:
    N, n_dirs, n_r = polar_grid.shape
    assert n_dirs == 26 and n_r == 7

    E_lr = np.zeros((N, 5, n_r), dtype=np.float32)
    E_global_lr = np.zeros((N, 5, n_r), dtype=np.float32)
    for l_deg in range(5):
        proj_w = np.einsum('ij,njr->nir', _SH_P_LIST[l_deg], polar_grid)
        E_lr[:, l_deg, :] = np.sum(proj_w ** 2, axis=1)
        proj_g = np.einsum('ij,njr->nir', _SH_P_GLOBAL_LIST[l_deg], polar_grid)
        E_global_lr[:, l_deg, :] = np.sum(proj_g ** 2, axis=1)

    eps = 1e-6
    E_tot_r = np.sum(E_lr[:, 1:, :], axis=1) + E_lr[:, 0, :] + eps
    frac_lr = E_lr / E_tot_r[:, None, :]

    E_glob_tot_r = np.sum(E_global_lr, axis=1) + eps
    frac_global_lr = E_global_lr / E_glob_tot_r[:, None, :]

    r_pos = polar_grid[:, 0::2, :]
    r_neg = polar_grid[:, 1::2, :]
    r_even = 0.5 * (r_pos + r_neg)
    r_odd = 0.5 * (r_pos - r_neg)
    E_even_r = np.mean(r_even ** 2, axis=1)
    E_odd_r = np.mean(r_odd ** 2, axis=1)
    parity_r = (E_even_r - E_odd_r) / (E_even_r + E_odd_r + eps)

    bands = [
        ("core", [0, 1]),
        ("inner", [1, 2, 3]),
        ("boundary", [2, 3, 4]),
        ("outer", [4, 5, 6]),
        ("full", [0, 1, 2, 3, 4, 5, 6]),
    ]
    band_feats = []
    for _, r_idxs in bands:
        e_b = np.mean(E_lr[:, :, r_idxs], axis=2)
        f_b = np.mean(frac_lr[:, :, r_idxs], axis=2)
        fg_b = np.mean(frac_global_lr[:, :, r_idxs], axis=2)
        l0_b = e_b[:, 0]
        aniso_b = np.sum(e_b[:, 1:], axis=1)
        even_aniso = e_b[:, 2] + e_b[:, 4]
        odd_aniso = e_b[:, 1] + e_b[:, 3]

        iso_ratio = l0_b / (l0_b + aniso_b + eps)
        l2_to_l0 = e_b[:, 2] / (l0_b + eps)
        l1_to_l0 = e_b[:, 1] / (l0_b + eps)
        l4_to_l2 = e_b[:, 4] / (e_b[:, 2] + eps)
        even_odd_sh = (even_aniso - odd_aniso) / (even_aniso + odd_aniso + eps)

        p_prob = np.clip(f_b, 1e-6, 1.0)
        p_prob = p_prob / np.sum(p_prob, axis=1, keepdims=True)
        sh_entropy = -np.sum(p_prob * np.log(p_prob), axis=1) / np.log(5.0)
        glob_high_aniso = fg_b[:, 3] + fg_b[:, 4]

        band_feats.append(np.column_stack([
            f_b, iso_ratio, l2_to_l0, l1_to_l0, l4_to_l2,
            even_odd_sh, sh_entropy, np.log1p(aniso_b),
            fg_b[:, 2], glob_high_aniso,
        ]))
    X_bands = np.hstack(band_feats)

    d_frac = np.diff(frac_lr, axis=2)
    radial_grad_feats = np.column_stack([
        np.mean(d_frac[:, 0, :3], axis=1),
        np.mean(d_frac[:, 0, 3:], axis=1),
        np.mean(d_frac[:, 1, :3], axis=1),
        np.mean(d_frac[:, 2, :3], axis=1),
        np.mean(d_frac[:, 2, 3:], axis=1),
        np.mean(d_frac[:, 3, :], axis=1),
        np.mean(d_frac[:, 4, :], axis=1),
        frac_lr[:, 0, 0] - frac_lr[:, 0, 3],
        frac_lr[:, 0, 1] - frac_lr[:, 0, 5],
        frac_lr[:, 2, 3] - frac_lr[:, 2, 0],
        frac_lr[:, 1, 2] - frac_lr[:, 1, 5],
        frac_global_lr[:, 0, 0] - frac_global_lr[:, 0, 4],
        frac_global_lr[:, 2, 3] - frac_global_lr[:, 2, 0],
        frac_global_lr[:, 4, 2] - frac_global_lr[:, 4, 5],
    ])

    vessel_180 = np.mean(r_even[:, :, 1:5], axis=2) - np.mean(polar_grid[:, :, 1:5], axis=(1, 2), keepdims=True)[:, :, 0]
    vessel_tube_score = np.max(vessel_180, axis=1) - np.median(vessel_180, axis=1)
    vessel_top2_mean = np.mean(np.sort(vessel_180, axis=1)[:, -2:], axis=1)

    core_iso = np.mean(frac_lr[:, 0, :2], axis=1)
    mid_l2 = np.mean(frac_lr[:, 2, 1:4], axis=1)
    mid_l1 = np.mean(frac_lr[:, 1, 1:4], axis=1)
    star_c = star[:, -2]
    v75_s = v75[:, 0]
    v77_p = v77[:, 0]

    cross_feats = np.column_stack([
        parity_r,
        vessel_tube_score,
        vessel_top2_mean,
        core_iso * star_c,
        core_iso * v75_s,
        core_iso * v77_p,
        mid_l2 * (1.0 - v75_s),
        mid_l1 * (1.0 - v77_p),
        (core_iso - mid_l2) * star_c,
        frac_lr[:, 0, :5].reshape(N, -1),
        frac_lr[:, 2, 1:6].reshape(N, -1),
        frac_global_lr[:, 0, :].reshape(N, -1),
        frac_global_lr[:, 2, :].reshape(N, -1),
        frac_global_lr[:, 4, :5].reshape(N, -1),
    ])

    X_harm6 = np.hstack([X_bands, radial_grad_feats, cross_feats]).astype(np.float32)
    X_harm6 = np.nan_to_num(X_harm6, nan=0.0, posinf=0.0, neginf=0.0)
    assert X_harm6.shape[1] == 128, f"Expected 128 features, got {X_harm6.shape[1]}"
    return X_harm6


def extract_v77_bifurcation_and_antipodal_orderstat_features(polar_grid: np.ndarray) -> np.ndarray:
    N = len(polar_grid)
    r_pos = polar_grid[:, 0::2, :]
    r_neg = polar_grid[:, 1::2, :]

    pair_mean = 0.5 * (r_pos + r_neg)
    pair_diff = np.abs(r_pos - r_neg)
    pair_min = np.minimum(r_pos, r_neg)

    m_core = np.mean(polar_grid[:, :, :2], axis=2)
    m_mid = np.mean(polar_grid[:, :, 2:5], axis=2)
    m_out = np.mean(polar_grid[:, :, 4:], axis=2)

    s_core = np.sort(m_core, axis=1)[:, ::-1]
    s_mid = np.sort(m_mid, axis=1)[:, ::-1]
    s_out = np.sort(m_out, axis=1)[:, ::-1]

    pm_mid = np.sort(np.mean(pair_mean[:, :, 1:5], axis=2), axis=1)[:, ::-1]
    pd_mid = np.sort(np.mean(pair_diff[:, :, 1:5], axis=2), axis=1)[:, ::-1]
    pmin_mid = np.sort(np.mean(pair_min[:, :, 1:5], axis=2), axis=1)[:, ::-1]

    pi_core = np.mean(s_core[:, -6:], axis=1) - np.mean(s_out[:, :6], axis=1)
    bifurc_3 = np.mean(s_mid[:, :3], axis=1) - np.mean(s_mid[:, 6:16], axis=1)
    tube_2 = np.mean(pm_mid[:, :2], axis=1) - np.mean(pm_mid[:, 4:10], axis=1)
    stalk_1 = s_mid[:, 0] - s_mid[:, 3]

    radial_drop = m_core - m_out
    s_drop = np.sort(radial_drop, axis=1)[:, ::-1]

    X_v77 = np.column_stack([
        pi_core, bifurc_3, tube_2, stalk_1,
        np.mean(m_core, axis=1), np.std(m_core, axis=1),
        np.mean(m_mid, axis=1), np.std(m_mid, axis=1),
        np.mean(m_out, axis=1), np.std(m_out, axis=1),
        s_core[:, 0], s_core[:, 2], s_core[:, 5], s_core[:, 12], s_core[:, 20], s_core[:, 25],
        s_mid[:, 0], s_mid[:, 1], s_mid[:, 2], s_mid[:, 4], s_mid[:, 8], s_mid[:, 13], s_mid[:, 19], s_mid[:, 25],
        s_out[:, 0], s_out[:, 2], s_out[:, 5], s_out[:, 12], s_out[:, 20], s_out[:, 25],
        pm_mid[:, 0], pm_mid[:, 1], pm_mid[:, 2], pm_mid[:, 6], pm_mid[:, 12],
        pd_mid[:, 0], pd_mid[:, 1], pd_mid[:, 6], pd_mid[:, 12],
        pmin_mid[:, 0], pmin_mid[:, 1], pmin_mid[:, 6], pmin_mid[:, 12],
        s_drop[:, 0], s_drop[:, 2], s_drop[:, 6], s_drop[:, 12], s_drop[:, 18], s_drop[:, 23], s_drop[:, 25],
        np.mean(s_drop, axis=1), np.std(s_drop, axis=1),
        np.percentile(m_mid, [10, 25, 50, 75, 90], axis=1).T,
    ]).astype(np.float32)

    assert X_v77.shape == (N, 57)
    return np.nan_to_num(X_v77, nan=0.0, posinf=0.0, neginf=0.0)


# ==============================================================================
# 5. FAMILY-8 (84D) & NEW V10.6 FAMILY-9 (64D): 3D KOENDERINK SHAPE INDEX,
#    HOLLOW-LUMEN BRONCHIAL WALL VETO, 3-BRANCH Y-BIFURCATION TENSOR &
#    MULTI-SCALE GEODESIC CORE-TO-SHELL CONTRAST (`9-FAMILY = 1,166D` PURE!)
# ==============================================================================
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


def extract_all_9families_from_3d_patches_array(
    V_patches: np.ndarray,
    series_arr_sub: np.ndarray,
    coords_arr_sub: np.ndarray,
):
    n_sub = len(V_patches)
    grid_1d = np.arange(48, dtype=np.float32) - 23.5
    zz, yy, xx = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    r3d = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)

    shell_bounds = [(0.0, 2.2), (2.2, 4.2), (4.2, 6.5), (6.5, 9.5), (9.5, 13.0), (13.0, 17.5), (17.5, 23.5)]
    shell_masks = [(r3d >= r_in) & (r3d < r_out) for r_in, r_out in shell_bounds]

    zz_c, yy_c, xx_c = zz[13:35, 13:35, 13:35], yy[13:35, 13:35, 13:35], xx[13:35, 13:35, 13:35]
    r_c22 = np.sqrt(zz_c ** 2 + yy_c ** 2 + xx_c ** 2)
    uz_c, uy_c, ux_c = zz_c / np.maximum(r_c22, 1e-4), yy_c / np.maximum(r_c22, 1e-4), xx_c / np.maximum(r_c22, 1e-4)
    c22_bounds = [(0.0, 2.2), (2.2, 4.0), (4.0, 6.0), (6.0, 8.5), (8.5, 11.5), (11.5, 15.5)]
    c22_masks = [(r_c22 >= r_in) & (r_c22 < r_out) for r_in, r_out in c22_bounds]

    octant_masks = []
    for sz in [-1, 1]:
        for sy in [-1, 1]:
            for sx in [-1, 1]:
                m_oct = ((zz * sz >= 0) & (yy * sy >= 0) & (xx * sx >= 0) & (r3d >= 2.0) & (r3d <= 10.5))
                octant_masks.append(m_oct)

    y2d, x2d = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    r2d = np.sqrt(y2d ** 2 + x2d ** 2)
    m2d_core = (r2d <= 3.5)
    m2d_Inner = (r2d > 3.5) & (r2d <= 6.5)
    m2d_ann = (r2d > 6.5) & (r2d <= 11.5)
    m2d_bg = (r2d > 11.5) & (r2d <= 18.0)

    radii_7 = [1.6, 3.0, 4.6, 6.5, 9.0, 12.0, 15.5]
    sample_pts_zyx = np.zeros((1, 26, 7, 1, 3), dtype=np.float32)
    for idir in range(26):
        uz, uy, ux = _U26_UNIT_DIRS[idir]
        for ir, rv in enumerate(radii_7):
            sample_pts_zyx[0, idir, ir, 0, :] = [float(rv * ux) / 23.5, float(rv * uy) / 23.5, float(rv * uz) / 23.5]
    sample_grid_t = torch.from_numpy(sample_pts_zyx).to(device) if TORCH_AVAILABLE else sample_pts_zyx

    polar_grid_direct = np.zeros((n_sub, 26, 7), dtype=np.float32)
    geom_direct = np.zeros((n_sub, 126), dtype=np.float32)
    star_direct = np.zeros((n_sub, 125), dtype=np.float32)
    v75_direct = np.zeros((n_sub, 229), dtype=np.float32)
    v76_direct = np.zeros((n_sub, 131), dtype=np.float32)
    X_vol7 = np.zeros((n_sub, 148), dtype=np.float32)
    V3D_ms = np.zeros((n_sub, 4, 24, 24, 24), dtype=np.float16)
    V25D_mip = np.zeros((n_sub, 9, 32, 32), dtype=np.float16)

    m_fov = (r3d <= 5.5)
    m_mid = (r3d <= 9.5)
    m_out = (r3d > 6.5) & (r3d <= 14.5)
    zz_fov, yy_fov, xx_fov = zz[m_fov], yy[m_fov], xx[m_fov]
    r_fov = r3d[m_fov]

    batch_size = 128
    for start_i in range(0, n_sub, batch_size):
        end_i = min(n_sub, start_i + batch_size)
        V = V_patches[start_i:end_i].astype(np.float32)
        B = len(V)
        if TORCH_AVAILABLE:
            with torch.no_grad():
                Vt_gpu = torch.from_numpy(V).to(device, dtype=torch.float32).unsqueeze(1)
                grid_b = sample_grid_t.expand(B, -1, -1, -1, -1)
                pg_b = F.grid_sample(Vt_gpu, grid_b, mode='bilinear', padding_mode='border', align_corners=True)
                pg_np = pg_b[:, 0, :, :, 0].cpu().numpy().astype(np.float32)
                polar_grid_direct[start_i:end_i] = pg_np

                v_core12_up24 = F.interpolate(Vt_gpu[:, :, 18:30, 18:30, 18:30], size=(24, 24, 24), mode='trilinear', align_corners=True)[:, 0].cpu().numpy()
                V_s1_t = F.avg_pool3d(Vt_gpu, kernel_size=3, stride=1, padding=1)[:, 0]
                V_s2_t = F.avg_pool3d(Vt_gpu, kernel_size=5, stride=1, padding=2)[:, 0]
                V_s3_t = F.avg_pool3d(Vt_gpu, kernel_size=7, stride=1, padding=3)[:, 0]
                v_box_diff24 = (V_s1_t - V_s3_t)[:, 12:36, 12:36, 12:36].cpu().numpy()
                V_s1 = V_s1_t.cpu().numpy()
                V_s2 = V_s2_t.cpu().numpy()
                V_s3 = V_s3_t.cpu().numpy()
        else:
            # Exact SciPy trilinear & box-filter execution when running on CPU without PyTorch
            coords_zyx = np.zeros((3, 26, 7), dtype=np.float32)
            for idir in range(26):
                uz, uy, ux = _U26_UNIT_DIRS[idir]
                for ir, rv in enumerate(radii_7):
                    coords_zyx[:, idir, ir] = [23.5 + rv * uz, 23.5 + rv * uy, 23.5 + rv * ux]
            pg_np = np.stack([ndimage.map_coordinates(V[b], coords_zyx, order=1, mode='nearest') for b in range(B)], axis=0).astype(np.float32)
            polar_grid_direct[start_i:end_i] = pg_np
            v_core12_up24 = ndimage.zoom(V[:, 18:30, 18:30, 18:30], (1, 2, 2, 2), order=1).astype(np.float32)
            V_s1 = ndimage.uniform_filter(V, size=(1, 3, 3, 3), mode='nearest').astype(np.float32)
            V_s2 = ndimage.uniform_filter(V, size=(1, 5, 5, 5), mode='nearest').astype(np.float32)
            V_s3 = ndimage.uniform_filter(V, size=(1, 7, 7, 7), mode='nearest').astype(np.float32)
            v_box_diff24 = (V_s1 - V_s3)[:, 12:36, 12:36, 12:36]

        V3D_ms[start_i:end_i, 0] = v_core12_up24.astype(np.float16)
        V3D_ms[start_i:end_i, 1] = V[:, 12:36, 12:36, 12:36].astype(np.float16)
        V3D_ms[start_i:end_i, 2] = V.reshape(B, 24, 2, 24, 2, 24, 2).mean(axis=(2, 4, 6)).astype(np.float16)
        V3D_ms[start_i:end_i, 3] = v_box_diff24.astype(np.float16)

        Vc32 = V[:, 8:40, 8:40, 8:40]
        ax_0 = 0.5 * (Vc32[:, 15, :, :] + Vc32[:, 16, :, :])
        ax_1 = np.max(Vc32[:, 13:19, :, :], axis=1)
        ax_2 = np.max(Vc32[:, 9:23, :, :], axis=1)
        co_0 = 0.5 * (Vc32[:, :, 15, :] + Vc32[:, :, 16, :])
        co_1 = np.max(Vc32[:, :, 13:19, :], axis=2)
        co_2 = np.max(Vc32[:, :, 9:23, :], axis=2)
        sa_0 = 0.5 * (Vc32[:, :, :, 15] + Vc32[:, :, :, 16])
        sa_1 = np.max(Vc32[:, :, :, 13:19], axis=3)
        sa_2 = np.max(Vc32[:, :, :, 9:23], axis=3)
        V25D_mip[start_i:end_i] = np.stack([ax_0, ax_1, ax_2, co_0, co_1, co_2, sa_0, sa_1, sa_2], axis=1).astype(np.float16)

        geom_cols = []
        V_fov, V_mid, V_out = V[:, m_fov], V[:, m_mid], V[:, m_out]
        for tau in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            bw_fov = (V_fov >= tau).astype(np.float32)
            bw_mid = (V_mid >= tau).astype(np.float32)
            bw_out = (V_out >= tau).astype(np.float32)
            vol_f, vol_m, vol_o = np.mean(bw_fov, axis=1), np.mean(bw_mid, axis=1), np.mean(bw_out, axis=1)
            w_sum = np.sum(bw_fov, axis=1) + 1e-5
            cz = np.sum(bw_fov * zz_fov[None, :], axis=1) / w_sum
            cy = np.sum(bw_fov * yy_fov[None, :], axis=1) / w_sum
            cx = np.sum(bw_fov * xx_fov[None, :], axis=1) / w_sum
            c_dist = np.sqrt(cz ** 2 + cy ** 2 + cx ** 2)
            r_mean_t = np.sum(bw_fov * r_fov[None, :], axis=1) / w_sum
            r_std_t = np.sqrt(np.sum(bw_fov * ((r_fov[None, :] - r_mean_t[:, None]) ** 2), axis=1) / w_sum + 1e-6)
            izz = np.sum(bw_fov * ((zz_fov[None, :] - cz[:, None]) ** 2), axis=1) / w_sum
            iyy = np.sum(bw_fov * ((yy_fov[None, :] - cy[:, None]) ** 2), axis=1) / w_sum
            ixx = np.sum(bw_fov * ((xx_fov[None, :] - cx[:, None]) ** 2), axis=1) / w_sum
            i_stack = np.sort(np.column_stack([izz, iyy, ixx]), axis=1)
            elong = np.sqrt((i_stack[:, 1] + 1e-4) / (i_stack[:, 2] + 1e-4))
            flatn = np.sqrt((i_stack[:, 0] + 1e-4) / (i_stack[:, 2] + 1e-4))
            geom_cols.append(np.column_stack([
                vol_f, vol_m, vol_o, cz, cy, cx, c_dist, r_mean_t, r_std_t,
                i_stack[:, 0], i_stack[:, 1], i_stack[:, 2], elong, flatn,
                vol_f / (vol_m + 0.02), vol_f / (vol_o + 0.02),
            ]))
        w_int = np.maximum(V_fov, 1e-4)
        w_int_sum = np.sum(w_int, axis=1)
        cz_w = np.sum(w_int * zz_fov[None, :], axis=1) / w_int_sum
        cy_w = np.sum(w_int * yy_fov[None, :], axis=1) / w_int_sum
        cx_w = np.sum(w_int * xx_fov[None, :], axis=1) / w_int_sum
        cdist_w = np.sqrt(cz_w ** 2 + cy_w ** 2 + cx_w ** 2)
        rmean_w = np.sum(w_int * r_fov[None, :], axis=1) / w_int_sum
        q10_f, q25_f, q50_f, q75_f, q90_f = np.percentile(V_fov, [10, 25, 50, 75, 90], axis=1)
        q25_o, q50_o, q75_o, q90_o = np.percentile(V_out, [25, 50, 75, 90], axis=1)
        geom_cols.append(np.column_stack([
            cz_w, cy_w, cx_w, cdist_w, rmean_w,
            q10_f, q25_f, q50_f, q75_f, q90_f,
            q25_o, q50_o, q75_o, q90_o,
        ]))
        geom_direct[start_i:end_i] = np.nan_to_num(np.hstack(geom_cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        ray_means = np.mean(pg_np, axis=2)
        ray_Drops = pg_np[:, :, 0] - pg_np[:, :, 4]
        ray_slopes = (pg_np[:, :, 1] - pg_np[:, :, 5]) / 4.0
        ray_max = np.max(pg_np[:, :, :4], axis=2)
        r_pos_p, r_neg_p = pg_np[:, 0::2, :], pg_np[:, 1::2, :]
        anti_diff = np.mean(np.abs(r_pos_p - r_neg_p), axis=1)
        anti_min = np.mean(np.minimum(r_pos_p, r_neg_p), axis=1)
        star_core_vec = 0.55 * np.mean(pg_np[:, :, 0], axis=1) + 0.45 * np.mean(pg_np[:, :, 1], axis=1)
        star_outer_vec = np.mean(pg_np[:, :, 4:], axis=(1, 2))
        star_direct[start_i:end_i] = np.nan_to_num(np.column_stack([
            np.sort(ray_means, axis=1)[:, ::-1],
            np.sort(ray_Drops, axis=1)[:, ::-1],
            np.sort(ray_slopes, axis=1)[:, ::-1],
            np.sort(ray_max, axis=1)[:, ::-1],
            anti_diff, anti_min, np.std(pg_np, axis=1)[:, :4],
            star_outer_vec, star_core_vec, star_core_vec - star_outer_vec,
        ]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        v75_cols = []
        pg_sorted = np.sort(pg_np, axis=1)[:, ::-1, :]
        for ib in range(7):
            s_b = pg_sorted[:, :, ib]
            v75_cols.append(np.column_stack([
                s_b[:, 0], s_b[:, 2], s_b[:, 5], s_b[:, 8], s_b[:, 12],
                s_b[:, 16], s_b[:, 20], s_b[:, 23], s_b[:, 25],
                np.mean(s_b, axis=1), np.std(s_b, axis=1),
                s_b[:, 0] - s_b[:, 12], s_b[:, 3] - s_b[:, 18], s_b[:, 6] - s_b[:, 22],
                np.percentile(s_b, 90, axis=1) - np.percentile(s_b, 10, axis=1),
                np.percentile(s_b, 75, axis=1) - np.percentile(s_b, 25, axis=1),
            ]))
        d1_pg = np.diff(pg_np, axis=2)
        d2_pg = np.diff(d1_pg, axis=2)
        v75_cols.append(np.column_stack([
            np.mean(d1_pg, axis=1), np.std(d1_pg, axis=1),
            np.min(d1_pg, axis=1), np.max(d1_pg, axis=1),
            np.mean(d2_pg, axis=1)[:, :4], np.std(d2_pg, axis=1)[:, :4],
        ]))

        hess_cols = []
        core_sphericity_col = None
        for sc_idx, Vs in enumerate([V_s1, V_s2]):
            Vc = Vs[:, 12:36, 12:36, 12:36]
            hzz, hyy, hxx, hzy, hzx, hyx = compute_finite_difference_3d_hessian_components(Vc)
            lam1, lam2, lam3 = compute_exact_3x3_symmetric_hessian_eigenvalues(hzz, hyy, hxx, hzy, hzx, hyx)
            tr_h = lam1 + lam2 + lam3
            frob_h = np.sqrt(lam1 ** 2 + lam2 ** 2 + lam3 ** 2 + 1e-8)
            blob_3d = np.where(lam3 < 0.0, (-lam3) / (-lam1 + 0.02), 0.0)
            sph_trace = np.clip(-tr_h / (np.sqrt(3.0) * frob_h + 0.02), -1.0, 1.0)
            vessel_3d = np.where(lam2 < 0.0, (-lam2) / (-lam1 + 0.02) * np.exp(-(lam3 ** 2) / (2.0 * (0.5 * lam2 ** 2 + 1e-4))), 0.0)
            if sc_idx == 0:
                core_sphericity_col = np.mean((0.6 * sph_trace[:, c22_masks[0]] + 0.4 * blob_3d[:, c22_masks[0]]) * Vc[:, 1:-1, 1:-1, 1:-1][:, c22_masks[0]], axis=1)
            for cm in c22_masks:
                hess_cols.append(np.column_stack([
                    np.mean(tr_h[:, cm], axis=1), np.mean(frob_h[:, cm], axis=1),
                    np.mean(sph_trace[:, cm], axis=1), np.mean(blob_3d[:, cm], axis=1),
                    np.mean(vessel_3d[:, cm], axis=1), np.percentile(vessel_3d[:, cm], 90, axis=1),
                    np.mean(lam1[:, cm], axis=1),
                ]))
        v75_direct[start_i:end_i] = np.nan_to_num(np.hstack(v75_cols + [core_sphericity_col[:, None]] + hess_cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        v76_cols = []
        for Vs in [V, V_s1, V_s2]:
            Vc = Vs[:, 12:36, 12:36, 12:36]
            gz = 0.5 * (Vc[:, 2:, 1:-1, 1:-1] - Vc[:, :-2, 1:-1, 1:-1])
            gy = 0.5 * (Vc[:, 1:-1, 2:, 1:-1] - Vc[:, 1:-1, :-2, 1:-1])
            gx = 0.5 * (Vc[:, 1:-1, 1:-1, 2:] - Vc[:, 1:-1, 1:-1, :-2])
            gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-8)
            inward_flux = -(gz * uz_c[None, ...] + gy * uy_c[None, ...] + gx * ux_c[None, ...])
            norm_flux = inward_flux / (gmag + 0.02)
            for cm in c22_masks:
                v76_cols.append(np.column_stack([
                    np.mean(inward_flux[:, cm], axis=1), np.std(inward_flux[:, cm], axis=1),
                    np.percentile(inward_flux[:, cm], 75, axis=1), np.mean(norm_flux[:, cm], axis=1),
                    np.mean(gmag[:, cm], axis=1),
                ]))
        dog1, dog2, dog3 = V - V_s1, V_s1 - V_s2, V_s2 - V_s3
        for sm in shell_masks:
            v76_cols.append(np.column_stack([
                np.mean(dog1[:, sm], axis=1), np.std(dog1[:, sm], axis=1),
                np.mean(dog2[:, sm], axis=1), np.std(dog2[:, sm], axis=1),
                np.mean(dog3[:, sm], axis=1),
            ]))
        v76_cols.append(np.column_stack([
            np.mean(dog1[:, shell_masks[0]], axis=1) - np.mean(dog1[:, shell_masks[3]], axis=1),
            np.mean(dog2[:, shell_masks[0]], axis=1) - np.mean(dog2[:, shell_masks[3]], axis=1),
            np.mean(dog2[:, shell_masks[1]], axis=1) - np.mean(dog2[:, shell_masks[4]], axis=1),
            np.mean(dog3[:, shell_masks[0]], axis=1) - np.mean(dog3[:, shell_masks[4]], axis=1),
            np.mean(dog3[:, shell_masks[1]], axis=1) - np.mean(dog3[:, shell_masks[5]], axis=1),
            np.mean(dog2[:, shell_masks[2]], axis=1) - np.mean(dog2[:, shell_masks[5]], axis=1),
        ]))
        v76_direct[start_i:end_i] = np.nan_to_num(np.hstack(v76_cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        cols = []
        shell_means, shell_q75s = [], []
        for sm in shell_masks:
            vox_s = V[:, sm]
            m_s, sd_s = np.mean(vox_s, axis=1), np.std(vox_s, axis=1)
            q25_s, q50_s, q75_s, q90_s = np.percentile(vox_s, [25, 50, 75, 90], axis=1)
            f40_s = np.mean((vox_s > 0.40).astype(np.float32), axis=1)
            f65_s = np.mean((vox_s > 0.65).astype(np.float32), axis=1)
            shell_means.append(m_s)
            shell_q75s.append(q75_s)
            cols.append(np.column_stack([m_s, sd_s, q25_s, q50_s, q75_s, q90_s, f40_s, f65_s]))
        sm_mat, sq_mat = np.column_stack(shell_means), np.column_stack(shell_q75s)
        d1_sm = np.diff(sm_mat, axis=1)
        d2_sm = np.diff(d1_sm, axis=1)
        d1_sq = np.diff(sq_mat, axis=1)
        core_to_bg = (0.5 * (sm_mat[:, 0] + sm_mat[:, 1]) + 0.02) / (0.5 * (sm_mat[:, 4] + sm_mat[:, 5]) + 0.05)
        q75_core_to_bg = (0.5 * (sq_mat[:, 0] + sq_mat[:, 1]) + 0.02) / (0.5 * (sq_mat[:, 4] + sq_mat[:, 5]) + 0.05)
        cols.append(np.column_stack([d1_sm, d2_sm, d1_sq, core_to_bg, q75_core_to_bg, sm_mat[:, 0] - sm_mat[:, 3]]))

        for half_w in [1, 3, 6, 10]:
            s_lo, s_hi = 24 - half_w, 24 + half_w + 1
            mip_ax = np.max(V[:, s_lo:s_hi, :, :], axis=1)
            mip_cor = np.max(V[:, :, s_lo:s_hi, :], axis=2)
            mip_sag = np.max(V[:, :, :, s_lo:s_hi], axis=3)
            c_ax = np.mean(mip_ax[:, m2d_core], axis=1) - np.mean(mip_ax[:, m2d_ann], axis=1)
            c_cor = np.mean(mip_cor[:, m2d_core], axis=1) - np.mean(mip_cor[:, m2d_ann], axis=1)
            c_sag = np.mean(mip_sag[:, m2d_core], axis=1) - np.mean(mip_sag[:, m2d_ann], axis=1)
            in_ax = np.mean(mip_ax[:, m2d_Inner], axis=1) - np.mean(mip_ax[:, m2d_bg], axis=1)
            in_cor = np.mean(mip_cor[:, m2d_Inner], axis=1) - np.mean(mip_cor[:, m2d_bg], axis=1)
            in_sag = np.mean(mip_sag[:, m2d_Inner], axis=1) - np.mean(mip_sag[:, m2d_bg], axis=1)
            c_stack = np.column_stack([c_ax, c_cor, c_sag])
            in_stack = np.column_stack([in_ax, in_cor, in_sag])
            mip_min_c, mip_max_c = np.min(c_stack, axis=1), np.max(c_stack, axis=1)
            mip_min_in = np.min(in_stack, axis=1)
            cols.append(np.column_stack([
                c_ax, c_cor, c_sag, in_ax, in_cor, in_sag,
                mip_min_c, mip_max_c, mip_max_c - mip_min_c, np.std(c_stack, axis=1),
                mip_min_in, np.max(in_stack, axis=1) - mip_min_in,
            ]))
        oct_means = np.column_stack([np.mean(V[:, om], axis=1) for om in octant_masks])
        oct_sorted = np.sort(oct_means, axis=1)[:, ::-1]
        Vc = V[:, 12:36, 12:36, 12:36]
        gz = Vc[:, 2:, 1:-1, 1:-1] - Vc[:, :-2, 1:-1, 1:-1]
        gy = Vc[:, 1:-1, 2:, 1:-1] - Vc[:, 1:-1, :-2, 1:-1]
        gx = Vc[:, 1:-1, 1:-1, 2:] - Vc[:, 1:-1, 1:-1, :-2]
        gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-6)
        lap3d = (
            Vc[:, 2:, 1:-1, 1:-1] + Vc[:, :-2, 1:-1, 1:-1]
            + Vc[:, 1:-1, 2:, 1:-1] + Vc[:, 1:-1, :-2, 1:-1]
            + Vc[:, 1:-1, 1:-1, 2:] + Vc[:, 1:-1, 1:-1, :-2]
            - 6.0 * Vc[:, 1:-1, 1:-1, 1:-1]
        )
        g_core = np.mean(gmag[:, 7:15, 7:15, 7:15], axis=(1, 2, 3))
        g_rim = np.mean(gmag[:, 4:18, 4:18, 4:18], axis=(1, 2, 3))
        g_outer = np.mean(gmag, axis=(1, 2, 3))
        g_q90 = np.percentile(gmag.reshape(B, -1), 90, axis=1)
        lap_core = np.mean(lap3d[:, 8:14, 8:14, 8:14], axis=(1, 2, 3))
        lap_rim = np.mean(lap3d[:, 5:17, 5:17, 5:17], axis=(1, 2, 3))
        ez = np.mean(np.abs(gz[:, 5:17, 5:17, 5:17]), axis=(1, 2, 3))
        ey = np.mean(np.abs(gy[:, 5:17, 5:17, 5:17]), axis=(1, 2, 3))
        ex = np.mean(np.abs(gx[:, 5:17, 5:17, 5:17]), axis=(1, 2, 3))
        e_stack = np.column_stack([ez, ey, ex])
        cols.append(np.column_stack([
            oct_sorted, np.std(oct_means, axis=1), oct_sorted[:, 0] - oct_sorted[:, -1],
            np.mean(oct_sorted[:, :4], axis=1) - np.mean(oct_sorted[:, 4:], axis=1),
            np.mean(oct_sorted[:, :2], axis=1) - np.mean(oct_sorted[:, 2:], axis=1),
            np.mean(oct_sorted[:, :3], axis=1) - np.mean(oct_sorted[:, 3:], axis=1),
            g_core, g_rim, g_outer, g_q90, lap_core, lap_rim, lap_rim - lap_core,
            ez, ey, ex, (np.max(e_stack, axis=1) - np.min(e_stack, axis=1)) / (np.mean(e_stack, axis=1) + 0.02),
        ]))
        X_vol7[start_i:end_i] = np.nan_to_num(np.hstack(cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    v77_direct = extract_v77_bifurcation_and_antipodal_orderstat_features(polar_grid_direct)
    harm6_direct = extract_discrete_angular_and_sh_subspace_features(
        polar_grid_direct, geom_direct, star_direct, v75_direct, v76_direct, v77_direct
    )
    X_fam8 = extract_family8_intrascan_pointcloud_and_3d_compactness_features(
        series_arr_sub, coords_arr_sub, polar_grid_direct, geom_direct, star_direct,
        v75_direct, v76_direct, v77_direct, harm6_direct, X_vol7,
    )
    X_fam9 = extract_family9_koenderink_bronchus_yjunction_features(
        series_arr_sub, coords_arr_sub, polar_grid_direct, geom_direct, star_direct,
        v75_direct, v76_direct, v77_direct, harm6_direct, X_vol7, X_fam8,
    )
    return (
        polar_grid_direct, geom_direct, star_direct, v75_direct, v76_direct,
        v77_direct, harm6_direct, X_vol7, X_fam8, X_fam9, V3D_ms, V25D_mip
    )


FAMILY_DIMENSIONS_1166D = {
    "coords_2d": 2,           # (r_xy, z_norm) — Eq. (2)
    "fam1_geom_direct": 126,  # Multi-threshold 3D volume ratios, centroids, inertia I_zz, I_yy, I_xx
    "fam2_star_direct": 125,  # 26-ray Star-Ray profiles, slopes, antipodal minima a_m(r) — Eq. (7)
    "fam3_v75_direct": 229,   # 7-shell ray order stats (145D) + 3x3 Cardano Hessian B_3D, V_3D (84D) — Eqs. (2)-(3)
    "fam4_v76_direct": 131,   # Inward radial-gradient flux Phi_s, Phi_norm,s (90D) + 3D DoG shells (41D) — Eq. (6)
    "fam5_v77_polar": 129,    # Antipodal order stats (57D) + polar core-annulus contrasts (72D) — Eq. (7)
    "fam6_harm6_direct": 128, # Within-degree & Global-25 Dual-QR spherical harmonics E_l(r), R_iso — Eqs. (4)-(5)
    "fam7_vol7": 148,         # 3D concentric shells, 8 octants, and tri-planar core/annulus moments
    "fam8_veto": 84,          # Vessel escape, pleural asymmetry (20D), 3D NN distances (24D), K=24 context (40D)
    "fam9_koenderink": 64,    # Bronchus veto H_bronch, r_FWHM (16D), Y_3(r) (16D), Hessian/SH cap (16D), K=24 (16D) — Eq. (7)
}

compute_exact_3x3_symmetric_eigvals_3d = compute_exact_3x3_symmetric_hessian_eigenvalues
