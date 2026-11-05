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
Official Candidate Coverage & Prior-Shift Sensitivity Projection Audit (84.15x V2 / 61.34x V1,
Eq. 10), 7-Point FROC Profiles (Table 5), and Diameter-Stratified Recall (Table 6).
"""

def run_part_b_official_candidates_coverage_and_prior_shift_audit(
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
    diam_arr: np.ndarray,
    oof_a1_e2e: np.ndarray,
    all_cur_scans: list,
    discovered_cand_v2_path: str = None,
    scan_to_mhd_path: dict = None,
    X_9fam_h5: np.ndarray = None,
    X_spec9_h5: np.ndarray = None,
    X_aux_h5: np.ndarray = None,
    V3D_h5_gpu: torch.Tensor = None,
    V25D_h5_gpu: torch.Tensor = None,
    veto_h5: np.ndarray = None,
    star_core_h5: np.ndarray = None,
    v75_sph_h5: np.ndarray = None,
    v77_pi_h5: np.ndarray = None,
    eval_py3_script: str = None,
    polar_grid_h5: np.ndarray = None,
    vol7_h5: np.ndarray = None,
    fam8_h5: np.ndarray = None,
):
    print("\n" + "=" * 104, flush=True)
    print("PART B: OFFICIAL `candidates_V2.csv` (754,975) & `candidates.csv` (551,065) EXCLUDED-AWARE COVERAGE AUDIT,", flush=True)
    print("        CONSISTENT CURATED MASK PROJECTION & V10.6 STAGE-0 + 3D RE-CENTERED `.MHD` EMPIRICAL EVALUATION", flush=True)
    print("=" * 104, flush=True)

    curated_pos_mask, curated_excl_mask, curated_eval_neg_mask = audit_curated_hdf5_against_official_annotations(
        series_arr, coords_arr, labels_arr
    )
    n_curated_pos = int(np.sum(curated_pos_mask))
    n_curated_excl = int(np.sum(curated_excl_mask))
    n_curated_eval_neg = int(np.sum(curated_eval_neg_mask))
    print(
        f" -> [Curated 10,066-Patch Official Annotation/Excluded Mask Audit]:\n"
        f"    * Official `< 0.5*D` Positive / Excluded / Eval-Negative rows : "
        f"{n_curated_pos:,} pos | {n_curated_excl:,} excl | {n_curated_eval_neg:,} eval-neg (Total = {len(labels_arr):,})",
        flush=True,
    )

    pos_e2e = oof_a1_e2e[curated_pos_mask]
    neg_e2e = np.sort(oof_a1_e2e[curated_eval_neg_mask])[::-1]
    n_scans_all = len(all_cur_scans)

    def _project_prior_shift(neg_shift: float, rec_ceil: float):
        rates_proj = {}
        for r in FROC_FP_RATES:
            eff_fp = r / neg_shift
            k_allow = eff_fp * n_scans_all
            idx_lo = int(np.floor(k_allow))
            idx_hi = min(len(neg_e2e) - 1, idx_lo + 1)
            frac = k_allow - idx_lo
            if idx_lo <= 0:
                thr = neg_e2e[0] + (1.0 - k_allow) * 0.02
            else:
                thr = (1.0 - frac) * neg_e2e[min(idx_lo, len(neg_e2e) - 1)] + frac * neg_e2e[idx_hi]
            sens_cur = float(np.mean(pos_e2e >= thr))
            rates_proj[r] = min(rec_ceil, sens_cur * rec_ceil)
        return float(np.mean(list(rates_proj.values()))), rates_proj

    cpm_proj_v2, _ = _project_prior_shift(84.15, 1166.0 / 1186.0)
    cpm_proj_v1, _ = _project_prior_shift(61.34, 1120.0 / 1186.0)
    print(
        f" -> Unscreened Analytical Worst-Case Prior-Shift Projection (without Stage-0 Screener): "
        f"V2 Pool (`84.15x`) = {cpm_proj_v2 * 100:.2f}% CPM | V1 Pool (`61.34x`) = {cpm_proj_v1 * 100:.2f}% CPM",
        flush=True,
    )

    if scan_to_mhd_path is not None and len(scan_to_mhd_path) > 0 and X_9fam_h5 is not None:
        run_v106_empirical_mhd_stage0_and_recentered_candidates_v2_benchmark(
            cand_v2_path=discovered_cand_v2_path or CANDIDATES_V2_CSV_PATH,
            scan_to_mhd_path=scan_to_mhd_path,
            series_arr_h5=series_arr,
            X_9fam_h5=X_9fam_h5,
            X_spec9_h5=X_spec9_h5,
            X_aux_h5=X_aux_h5,
            V3D_h5_gpu=V3D_h5_gpu,
            V25D_h5_gpu=V25D_h5_gpu,
            labels_h5=labels_arr,
            diam_h5=diam_arr,
            veto_h5=veto_h5,
            star_core_h5=star_core_h5,
            v75_sph_h5=v75_sph_h5,
            v77_pi_h5=v77_pi_h5,
            eval_py3_script=eval_py3_script,
            max_scans=EMPIRICAL_MHD_MAX_SCANS,
            polar_grid_h5=polar_grid_h5,
            vol7_h5=vol7_h5,
            fam8_h5=fam8_h5,
            coords_h5=coords_arr,
        )
