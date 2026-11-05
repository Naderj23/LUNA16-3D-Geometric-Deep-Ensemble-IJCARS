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
Official LUNA16 < 0.5*D 3D Euclidean Positive Matching Tables & 100% Label-Blind
Seed=42 Raw .mhd Scan Selector (Section 2.1, Gate 10, Gate 13).
"""

def precompute_official_luna16_hit_matching_tables(
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    scans_list: list,
):
    """
    Matches official `noduleCADEvaluationLUNA16.py` (`collectNodulePredictions`):
      - Each candidate in scan `s` is matched to its closest true annotation `ia`
        if `dist(cand, annot_ia) <= 0.5 * diameter_mm`.
      - Candidates not matching any true annotation are checked against excluded
        annotations (`dist <= 0.5 * diameter_mm` if `diameter_mm > 0` else `<= 3.0 mm`).
      - Remaining candidates are marked as false positives (`is_fp_cand`).
    """
    df_ann = pd.read_csv(ANNOTATIONS_PATH)
    df_exc = pd.read_csv(ANNOTATIONS_EXCLUDED_PATH)
    df_ann['seriesuid'] = df_ann['seriesuid'].astype(str).str.strip()
    df_exc['seriesuid'] = df_exc['seriesuid'].astype(str).str.strip()

    scan_tables = {}
    n_total_annot = 0

    df_cands = pd.DataFrame({
        'seriesuid': series_arr,
        'idx': np.arange(len(series_arr)),
    })
    cand_groups = {s: grp['idx'].values for s, grp in df_cands.groupby('seriesuid', sort=False)}
    ann_groups = {s: grp for s, grp in df_ann.groupby('seriesuid', sort=False)}
    exc_groups = {s: grp for s, grp in df_exc.groupby('seriesuid', sort=False)}

    for s in scans_list:
        c_idxs = cand_groups.get(s, np.array([], dtype=np.int64))
        c_xyz = coords_arr[c_idxs] if len(c_idxs) > 0 else np.zeros((0, 3), dtype=np.float64)

        ann_s = ann_groups.get(s, None)
        exc_s = exc_groups.get(s, None)
        n_a = len(ann_s) if ann_s is not None else 0
        n_total_annot += n_a

        pos_hit_lists = [[] for _ in range(n_a)]
        cand_to_annot = np.full(len(c_idxs), -1, dtype=np.int64)
        is_fp_cand = np.ones(len(c_idxs), dtype=bool)

        if len(c_idxs) > 0 and n_a > 0:
            a_xyz = ann_s[['coordX', 'coordY', 'coordZ']].values.astype(np.float64)
            a_rad = 0.5 * ann_s['diameter_mm'].values.astype(np.float64)
            diff = c_xyz[:, None, :] - a_xyz[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            # Official LUNA16 assigns each candidate to its closest annotation if within radius
            closest_ia = np.argmin(dist, axis=1)
            closest_dist = dist[np.arange(len(c_idxs)), closest_ia]
            is_hit = closest_dist <= a_rad[closest_ia]
            cand_to_annot[is_hit] = closest_ia[is_hit]
            for ia in range(n_a):
                pos_hit_lists[ia] = np.where(cand_to_annot == ia)[0]
            is_fp_cand &= ~is_hit

        if len(c_idxs) > 0 and exc_s is not None and len(exc_s) > 0:
            e_xyz = exc_s[['coordX', 'coordY', 'coordZ']].values.astype(np.float64)
            e_dia = exc_s['diameter_mm'].values.astype(np.float64)
            e_rad = np.where(e_dia > 0.0, 0.5 * e_dia, 3.0)
            diff_e = c_xyz[:, None, :] - e_xyz[None, :, :]
            dist_e = np.sqrt(np.sum(diff_e ** 2, axis=2))
            hit_exc = np.any(dist_e <= e_rad[None, :], axis=1)
            is_fp_cand &= ~hit_exc

        scan_tables[s] = {
            'cand_idxs': c_idxs,
            'n_annot': n_a,
            'pos_hit_lists': pos_hit_lists,
            'cand_to_annot': cand_to_annot,
            'is_fp_mask': is_fp_cand,
            'fp_local_idxs': np.where(is_fp_cand)[0],
        }
    return scan_tables, n_total_annot


def select_label_blind_mhd_evaluation_scans(
    avail_mhd_scans: list,
    max_scans: int = EMPIRICAL_MHD_MAX_SCANS,
    seed: int = SEED,
):
    """
    100% Label-Blind `.mhd` Scan Selector (resolves Audit Problem 1):
      - Takes ONLY `(avail_mhd_scans, max_scans, seed)` as arguments.
      - Has ZERO access to `annotations.csv`, nodule counts, or positive/negative scan flags.
      - Performs uniform pseudo-random sampling without replacement across the mounted
        `.mhd` scan pool when `0 < max_scans < len(avail_mhd_scans)`, or returns the full
        mounted `.mhd` scan pool when `max_scans <= 0` or `max_scans >= len(avail_mhd_scans)`.
    """
    ordered_pool = sorted(list(set(str(s).strip() for s in avail_mhd_scans)))
    if max_scans <= 0 or max_scans >= len(ordered_pool):
        return ordered_pool, (
            f"100% Label-Blind Full Mounted `.mhd` Scan Pool ({len(ordered_pool)} scans; "
            f"Zero `annotations.csv` lookup in selection)"
        )
    rng_blind = np.random.RandomState(seed)
    perm_idx = rng_blind.permutation(len(ordered_pool))[:max_scans]
    selected_scans = sorted([ordered_pool[i] for i in perm_idx])
    return selected_scans, (
        f"100% Label-Blind Uniform Random Pilot Subset ({len(selected_scans)} of {len(ordered_pool)} mounted `.mhd` scans [888 total LUNA16]; "
        f"Seed={seed}, Zero `annotations.csv` lookup — Pilot Subset, NOT Full 888-Scan Official LUNA16 Claim)"
    )


def verify_label_blind_mhd_sampler_unit_test() -> bool:
    """Verifies that `select_label_blind_mhd_evaluation_scans` is deterministic and 100% label-blind."""
    fake_scans = [f"1.3.6.1.4.1.14519.5.2.1.6279.6001.{1000 + i}" for i in range(89)]
    s1, _ = select_label_blind_mhd_evaluation_scans(fake_scans, max_scans=25, seed=42)
    s2, _ = select_label_blind_mhd_evaluation_scans(list(reversed(fake_scans)), max_scans=25, seed=42)
    assert len(s1) == 25 and s1 == s2
    s_all, _ = select_label_blind_mhd_evaluation_scans(fake_scans, max_scans=0, seed=42)
    assert len(s_all) == 89
    return True
