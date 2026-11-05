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
3D Squeeze-and-Excitation (SE) Channel Attention, Core-vs.-Outer Spherical Volumetric Pooling,
Analytical Bottleneck SNR Dilution Model (Figure 3b, Section 2.3), and 8-View Cube Equivariance.
"""
from .backbone import apply_consistent_3d_physical_augmentation, verify_3d_mip_augmentation_geometric_consistency


def compute_analytical_bottleneck_snr_db(diameters_mm: np.ndarray) -> dict:
    """
    Analytical 8x8x8 (512-cell) bottleneck volumetric dilution SNR model (Section 2.3, Figure 3b):
      V_act(D) = clip((4/3) * pi * (D / 8)^3, 0.25, 64.0)
      SNR_dB(D) = 20 * log10(Delta_mu_pool(D) / sigma_bg)
    At D = 3.0 mm:
      - Standard 512-cell Global Average Pooling (GAP): -39.29 dB (below +6.0 dB Rose criterion)
      - Proposed Core-vs.-Outer Spherical Pooling:      +27.51 dB (+66.80 dB analytical recovery)
    """
    d = np.asarray(diameters_mm, dtype=np.float64)
    v_act = np.clip((4.0 / 3.0) * np.pi * ((d / 8.0) ** 3), 0.25, 64.0)
    delta_mu_gap = (v_act / 512.0) * 1.0
    sigma_bg_gap = 0.045
    snr_gap_db = 20.0 * np.log10(delta_mu_gap / sigma_bg_gap)

    core_fill = np.clip(v_act / 0.25, 0.0, 1.0) * 0.45 * 0.95 + np.maximum(v_act - 0.25, 0.0) * 0.015
    sigma_bg_core = 0.018
    snr_core_db = 20.0 * np.log10(core_fill / sigma_bg_core)
    return {
        "diameters_mm": d,
        "v_act_cells": v_act,
        "snr_gap_db": np.round(snr_gap_db, 2),
        "snr_core_outer_db": np.round(snr_core_db, 2),
        "recovery_gain_db": np.round(snr_core_db - snr_gap_db, 2),
    }


def apply_8view_cube_equivariance(volume_batch: np.ndarray) -> list:
    """Generates all 8 axis-aligned cube reflection views ({0,1}^3 flips) for O_h equivariance (Gate 08)."""
    views = []
    for fz in (False, True):
        for fy in (False, True):
            for fx in (False, True):
                v = volume_batch
                if fz:
                    v = np.flip(v, axis=-3)
                if fy:
                    v = np.flip(v, axis=-2)
                if fx:
                    v = np.flip(v, axis=-1)
                views.append(v)
    return views
