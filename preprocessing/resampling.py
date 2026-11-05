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
Exact 3x3 Affine MetaImage (.mhd) TransformMatrix World-to-Voxel Inversion (Eq. 1, Gate 01).
"""

def compute_general_mhd_world_to_voxel_affine_inverse(
    transform_matrix_3x3: np.ndarray,
    spacing_xyz: np.ndarray,
) -> np.ndarray:
    """
    Computes the exact general 3x3 world-to-voxel linear operator (resolves Audit Point 5):
      `A_vox_to_world = D @ diag(s)`
      `A_world_to_vox = (D @ diag(s))^{-1}`
    so that for ANY general 3x3 rotation, reflection, or axis-permutation matrix `D`:
      `i_xyz = A_world_to_vox @ (x_world - origin_xyz)`!
    """
    D = np.asarray(transform_matrix_3x3, dtype=np.float64).reshape(3, 3)
    s = np.asarray(spacing_xyz, dtype=np.float64).reshape(3)
    A_vox_to_world = D @ np.diag(s)
    return np.linalg.inv(A_vox_to_world)


def verify_general_mhd_affine_transform_unit_test() -> float:
    """
    Audit Gate 11 (Resolves Audit Point 5): Tests a non-diagonal rotated + axis-permuted
    3x3 `TransformMatrix` `D` and verifies that `i = (D @ diag(s))^{-1} (x_world - o)`
    recovers the exact continuous voxel coordinates `(ix, iy, iz)` to `< 1e-10`!
    """
    theta = 0.37
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)
    P_perm = np.array([
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    D_gen = R_z @ P_perm
    spacing = np.array([0.74, 0.74, 1.25], dtype=np.float64)
    origin = np.array([-180.5, -145.2, -290.0], dtype=np.float64)
    true_vox = np.array([[123.4, 256.7, 88.9], [45.1, 310.2, 142.6]], dtype=np.float64)
    world_pts = origin[None, :] + (true_vox @ (D_gen @ np.diag(spacing)).T)

    A_inv = compute_general_mhd_world_to_voxel_affine_inverse(D_gen, spacing)
    rec_vox = (world_pts - origin[None, :]) @ A_inv.T
    err = float(np.max(np.abs(rec_vox - true_vox)))
    assert err < 1e-9, f"General MHD affine transform inversion failed: {err}"
    return err
