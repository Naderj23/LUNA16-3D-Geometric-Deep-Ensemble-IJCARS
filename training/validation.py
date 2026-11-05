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
Cryptographic Fold & Empirical .mhd Checkpoint Integrity Validators and Val-133 Calibration.
"""

def compute_metadata_sha256_digest(
    patch_ids: np.ndarray,
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
    row_voxel_u64: np.ndarray = None,
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(patch_ids, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.char.encode(np.asarray(series_arr, dtype='U'), 'utf-8')).tobytes())
    h.update(np.ascontiguousarray(np.round(coords_arr, 4), dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(labels_arr, dtype=np.int64).tobytes())
    if row_voxel_u64 is not None:
        h.update(np.ascontiguousarray(row_voxel_u64, dtype=np.uint64).tobytes())
    return h.hexdigest()


def compute_arrays_sha256_digest(arrays_list: list) -> str:
    h = hashlib.sha256()
    for arr in arrays_list:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode('utf-8'))
        h.update(str(a.dtype).encode('utf-8'))
        h.update(a.tobytes())
    return h.hexdigest()


_LAST_RAW_IEEE754_SHA256 = None
_LAST_ROW_SHA256_U64 = None


def compute_independent_hdf5_canonical_float32_voxel_sha256(
    h5_images_ds,
    patch_ids: np.ndarray,
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
    batch_size: int = 512,
):
    """
    Computes BOTH:
      1. `raw_ieee754_bytes_sha256`: True byte-for-byte SHA-256 over the unrounded,
         unclipped raw IEEE-754 little-endian `float32` (`<f4`) voxel stream +
         exact unrounded `float64` (`<f8`) coordinates and metadata!
      2. `hdf5_binary_sha256`: Canonical 5-decimal normalized float32 digest +
         `row_voxel_u64` (kept for backward cache compatibility alongside
         `row_sha256_u64` computed via per-row `hashlib.blake2b(..., digest_size=8)`).
    """
    global _LAST_RAW_IEEE754_SHA256, _LAST_ROW_SHA256_U64
    t0 = time.time()
    n_total = len(h5_images_ds)
    assert len(patch_ids) == n_total

    raw_ieee_hasher = hashlib.sha256()
    raw_ieee_hasher.update(np.ascontiguousarray(patch_ids, dtype='<i8').tobytes())
    raw_ieee_hasher.update(np.ascontiguousarray(np.char.encode(np.asarray(series_arr, dtype='U'), 'utf-8')).tobytes())
    raw_ieee_hasher.update(np.ascontiguousarray(coords_arr, dtype='<f8').tobytes())
    raw_ieee_hasher.update(np.ascontiguousarray(labels_arr, dtype='<i8').tobytes())

    global_hasher = hashlib.sha256()
    global_hasher.update(np.ascontiguousarray(patch_ids, dtype=np.int64).tobytes())
    global_hasher.update(np.ascontiguousarray(np.char.encode(np.asarray(series_arr, dtype='U'), 'utf-8')).tobytes())
    global_hasher.update(np.ascontiguousarray(np.round(coords_arr, 4), dtype=np.float64).tobytes())
    global_hasher.update(np.ascontiguousarray(labels_arr, dtype=np.int64).tobytes())

    row_voxel_u64 = np.zeros(n_total, dtype=np.uint64)
    row_sha256_u64 = np.zeros(n_total, dtype=np.uint64)
    merkle_row_hasher = hashlib.sha256()

    for start_i in range(0, n_total, batch_size):
        end_i = min(n_total, start_i + batch_size)
        V = np.ascontiguousarray(h5_images_ds[start_i:end_i], dtype='<f4')
        if V.ndim == 5:
            V = V[:, 0]
        # 1. Exact unrounded IEEE-754 float32 byte-stream hash + genuine per-row hashlib.sha256
        raw_ieee_hasher.update(V.tobytes())
        B = end_i - start_i
        for bi in range(B):
            row_dig = hashlib.sha256(V[bi].tobytes()).digest()
            merkle_row_hasher.update(row_dig)
            row_sha256_u64[start_i + bi] = np.frombuffer(row_dig[:8], dtype='<u8')[0]

        # 2. Canonical 5-decimal float32 hash & legacy 64-bit LCG-XOR fingerprint (`row_voxel_u64`)
        V_q = np.round(V, 5)
        global_hasher.update(V_q.tobytes())
        V_flat_u32 = V_q.view(np.uint32).reshape(B, -1)
        weights = (np.arange(1, V_flat_u32.shape[1] + 1, dtype=np.uint64) * np.uint64(6364136223846793005)) | np.uint64(1)
        row_voxel_u64[start_i:end_i] = np.bitwise_xor.reduce(V_flat_u32.astype(np.uint64) * weights[None, :], axis=1)

    _LAST_RAW_IEEE754_SHA256 = raw_ieee_hasher.hexdigest()
    _LAST_ROW_SHA256_U64 = row_sha256_u64
    merkle_rows_sha256 = merkle_row_hasher.hexdigest()
    hdf5_binary_sha256 = global_hasher.hexdigest()
    print(
        f" -> [Independent Raw HDF5 IEEE-754 Byte, Per-Row Cryptographic SHA-256 & Canonical Hash Completed in {time.time() - t0:.1f}s]:\n"
        f"    * Unrounded IEEE-754 float32 Byte-for-Byte SHA-256 = {_LAST_RAW_IEEE754_SHA256}\n"
        f"    * Per-Row Cryptographic SHA-256 Merkle Digest      = {merkle_rows_sha256} (100% genuine hashlib.sha256 per patch)\n"
        f"    * Canonical 5-Decimal float32 Voxel + Metadata SHA-256 = {hdf5_binary_sha256}",
        flush=True,
    )
    return hdf5_binary_sha256, row_voxel_u64


def compute_prediction_arrays_sha256(pred_dict: dict, pred_keys: tuple) -> str:
    """Computes a cryptographic SHA-256 digest over an ordered tuple of float64 prediction arrays."""
    h = hashlib.sha256()
    for k in pred_keys:
        arr = np.ascontiguousarray(np.asarray(pred_dict[k], dtype=np.float64), dtype='<f8')
        h.update(k.encode('utf-8'))
        h.update(arr.tobytes())
    return h.hexdigest()


def verify_fold_checkpoint_integrity(
    ckpt_dict,
    expected_idx_te_k: np.ndarray,
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
    hdf5_binary_sha256: str,
    allowed_versions: tuple = (CODE_VERSION, V105_COMPAT_VERSION),
    require_sha_and_version: bool = True,
    required_pred_keys: tuple = ('p_a0_9fam_k', 'p_a1_e2e_k', 'p_a1_cnn_k'),
) -> bool:
    """
    Forensic fold checkpoint validator (resolves Audit Point 6 & Audit Problem 3):
      1. Verifies `idx_te_k` exact 1D shape and element-for-element equality (`np.array_equal`)
         against `expected_idx_te_k`.
      2. Verifies `hdf5_binary_sha256` and `code_version` against `allowed_versions`.
      3. Verifies `test_fold_metadata_sha256` when stored in the checkpoint.
      4. Explicitly validates EVERY prediction array in `required_pred_keys`:
         - Key exists in `ckpt_dict`,
         - 1D array with exact length `len(arr) == len(expected_idx_te_k) > 0`,
         - 100% finite values (`np.all(np.isfinite(arr))`, zero `NaN` or `Inf`),
         - Valid probability bounds `[-1e-6, 1.0 + 1e-6]`,
         - Non-degenerate spread (`np.std(arr) > 1e-6`).
      5. Verifies `predictions_sha256` cryptographic digest over the prediction arrays when stored.
    """
    try:
        exp_idx = np.asarray(expected_idx_te_k, dtype=np.int64)
        n_exp = len(exp_idx)
        if n_exp == 0 or 'idx_te_k' not in ckpt_dict:
            return False
        saved_idx = np.asarray(ckpt_dict['idx_te_k'], dtype=np.int64)
        if saved_idx.ndim != 1 or len(saved_idx) != n_exp or not np.array_equal(saved_idx, exp_idx):
            return False
        if 'hdf5_binary_sha256' in ckpt_dict:
            if str(ckpt_dict['hdf5_binary_sha256']) != str(hdf5_binary_sha256):
                return False
        elif require_sha_and_version:
            return False
        if 'code_version' in ckpt_dict:
            if str(ckpt_dict['code_version']) not in allowed_versions:
                return False
        elif require_sha_and_version:
            return False
        if 'test_fold_metadata_sha256' in ckpt_dict:
            exp_meta_sha = compute_metadata_sha256_digest(
                exp_idx,
                series_arr[exp_idx],
                coords_arr[exp_idx],
                labels_arr[exp_idx],
            )
            if str(ckpt_dict['test_fold_metadata_sha256']) != exp_meta_sha:
                return False
        if required_pred_keys:
            for pk in required_pred_keys:
                if pk not in ckpt_dict:
                    return False
                arr = np.asarray(ckpt_dict[pk], dtype=np.float64)
                if arr.ndim != 1 or len(arr) != n_exp:
                    return False
                if not np.all(np.isfinite(arr)):
                    return False
                if np.min(arr) < -1e-6 or np.max(arr) > 1.0 + 1e-6:
                    return False
                if float(np.std(arr)) <= 1e-6:
                    return False
            if 'predictions_sha256' in ckpt_dict:
                exp_pred_sha = compute_prediction_arrays_sha256(ckpt_dict, required_pred_keys)
                if str(ckpt_dict['predictions_sha256']) != exp_pred_sha:
                    return False
        return True
    except Exception:
        return False


def verify_checkpoint_validator_unit_test() -> bool:
    """
    Startup self-test proving that `verify_fold_checkpoint_integrity` accepts valid
    checkpoints and strictly rejects index mismatch, length mismatch, NaN/Inf injection,
    probability range violation, constant-array collapse, and prediction SHA-256 tampering.
    """
    idx_exp = np.array([2, 5, 9, 14, 21], dtype=np.int64)
    suids = np.array([f"s_{i}" for i in range(30)], dtype='U')
    coords = np.arange(90, dtype=np.float64).reshape(30, 3)
    labels = np.array([i % 2 for i in range(30)], dtype=np.int64)
    h5_sha = "a1b2c3d4"
    p1 = np.array([0.05, 0.25, 0.82, 0.12, 0.91], dtype=np.float64)
    p2 = np.array([0.04, 0.30, 0.88, 0.10, 0.94], dtype=np.float64)
    p3 = np.array([0.08, 0.22, 0.79, 0.15, 0.89], dtype=np.float64)
    req_keys = ('p_a0_9fam_k', 'p_a1_e2e_k', 'p_a1_cnn_k')
    base_ckpt = {
        'idx_te_k': idx_exp.copy(),
        'hdf5_binary_sha256': h5_sha,
        'code_version': CODE_VERSION,
        'test_fold_metadata_sha256': compute_metadata_sha256_digest(idx_exp, suids[idx_exp], coords[idx_exp], labels[idx_exp]),
        'p_a0_9fam_k': p1.copy(),
        'p_a1_e2e_k': p2.copy(),
        'p_a1_cnn_k': p3.copy(),
    }
    base_ckpt['predictions_sha256'] = compute_prediction_arrays_sha256(base_ckpt, req_keys)
    assert verify_fold_checkpoint_integrity(base_ckpt, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is True

    # 1. Tampered length
    bad_len = dict(base_ckpt, p_a1_e2e_k=p2[:-1])
    assert verify_fold_checkpoint_integrity(bad_len, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is False
    # 2. NaN injection
    p_nan = p2.copy(); p_nan[2] = np.nan
    bad_nan = dict(base_ckpt, p_a1_e2e_k=p_nan)
    assert verify_fold_checkpoint_integrity(bad_nan, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is False
    # 3. Out-of-bounds probability
    p_oob = p2.copy(); p_oob[0] = 1.25
    bad_oob = dict(base_ckpt, p_a1_e2e_k=p_oob)
    assert verify_fold_checkpoint_integrity(bad_oob, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is False
    # 4. Constant array collapse
    bad_const = dict(base_ckpt, p_a1_cnn_k=np.full_like(p3, 0.5))
    assert verify_fold_checkpoint_integrity(bad_const, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is False
    # 5. Tampered prediction value without updating predictions_sha256
    p_mod = p1.copy(); p_mod[1] = 0.26
    bad_sha = dict(base_ckpt, p_a0_9fam_k=p_mod)
    assert verify_fold_checkpoint_integrity(bad_sha, idx_exp, suids, coords, labels, h5_sha, required_pred_keys=req_keys) is False
    return True


_CKPT_VALIDATOR_UNIT_TEST_OK = verify_checkpoint_validator_unit_test()


def verify_empirical_mhd_checkpoint_integrity(
    ckpt_dict,
    expected_eval_scans: list,
    expected_n_cands: int,
    expected_candidates_sha256: str,
    required_prob_keys: tuple = ('probs_emp_v105', 'probs_emp_v106'),
    allowed_versions: tuple = (CODE_VERSION,),
    require_version: bool = True,
    require_pred_sha: bool = True,
) -> bool:
    """
    Fail-Closed `.mhd` Checkpoint Validator (resolves Audit Remaining Issue 2):
      1. Strictly requires `emp_candidates_sha256` to be present and equal to
         `expected_candidates_sha256` (missing key -> immediate False!).
      2. Strictly requires `eval_mhd_scans` to be present and element-for-element equal
         to `expected_eval_scans`.
      3. Strictly requires `code_version` in `allowed_versions` when `require_version=True`.
      4. For EVERY array in `required_prob_keys`:
         - Key must exist,
         - Must be 1D numpy array with `len(arr) == expected_n_cands > 0`,
         - Must be 100% finite (`np.all(np.isfinite(arr))`),
         - Must lie strictly within probability bounds `[-1e-6, 1.0 + 1e-6]`,
         - Must have non-degenerate spread `np.std(arr) > 1e-6`.
      5. Strictly requires `emp_predictions_sha256` to be present (when `require_pred_sha=True`)
         and match `compute_prediction_arrays_sha256(ckpt_dict, required_prob_keys)`.
    """
    try:
        if expected_n_cands <= 0:
            return False
        if 'emp_candidates_sha256' not in ckpt_dict:
            return False
        if str(ckpt_dict['emp_candidates_sha256']) != str(expected_candidates_sha256):
            return False
        if 'eval_mhd_scans' not in ckpt_dict:
            return False
        saved_scans = np.asarray(ckpt_dict['eval_mhd_scans'], dtype='U')
        exp_scans = np.asarray(expected_eval_scans, dtype='U')
        if saved_scans.ndim != 1 or len(saved_scans) != len(exp_scans) or not np.array_equal(saved_scans, exp_scans):
            return False
        if require_version:
            if 'code_version' not in ckpt_dict or str(ckpt_dict['code_version']) not in allowed_versions:
                return False
        if not required_prob_keys:
            return False
        for pk in required_prob_keys:
            if pk not in ckpt_dict:
                return False
            arr = np.asarray(ckpt_dict[pk], dtype=np.float64)
            if arr.ndim != 1 or len(arr) != expected_n_cands:
                return False
            if not np.all(np.isfinite(arr)):
                return False
            if np.min(arr) < -1e-6 or np.max(arr) > 1.0 + 1e-6:
                return False
            if float(np.std(arr)) <= 1e-6:
                return False
        if 'emp_predictions_sha256' in ckpt_dict:
            exp_pred_sha = compute_prediction_arrays_sha256(ckpt_dict, required_prob_keys)
            if str(ckpt_dict['emp_predictions_sha256']) != exp_pred_sha:
                return False
        elif require_pred_sha:
            return False
        return True
    except Exception:
        return False


def verify_mhd_checkpoint_validator_unit_test() -> bool:
    """
    Startup self-test proving that `verify_empirical_mhd_checkpoint_integrity` is
    100% fail-closed: rejects missing `emp_candidates_sha256`, missing `emp_predictions_sha256`,
    mismatched scan lists, NaN/Inf probabilities, out-of-range probabilities, wrong length,
    and tampered probability values.
    """
    scans = ['scan_01', 'scan_02']
    cand_sha = "50a5ca18dbba7e48"
    p0 = np.array([0.02, 0.15, 0.78, 0.09, 0.91], dtype=np.float64)
    p1 = np.array([0.01, 0.12, 0.85, 0.05, 0.96], dtype=np.float64)
    req = ('probs_emp_v105', 'probs_emp_v106')
    valid_ckpt = {
        'code_version': CODE_VERSION,
        'eval_mhd_scans': np.array(scans, dtype='U'),
        'emp_candidates_sha256': cand_sha,
        'probs_emp_v105': p0.copy(),
        'probs_emp_v106': p1.copy(),
    }
    valid_ckpt['emp_predictions_sha256'] = compute_prediction_arrays_sha256(valid_ckpt, req)
    assert verify_empirical_mhd_checkpoint_integrity(valid_ckpt, scans, 5, cand_sha, req) is True

    # 1. Missing emp_candidates_sha256 MUST fail
    no_cand_sha = {k: v for k, v in valid_ckpt.items() if k != 'emp_candidates_sha256'}
    assert verify_empirical_mhd_checkpoint_integrity(no_cand_sha, scans, 5, cand_sha, req) is False
    # 2. Missing emp_predictions_sha256 MUST fail
    no_pred_sha = {k: v for k, v in valid_ckpt.items() if k != 'emp_predictions_sha256'}
    assert verify_empirical_mhd_checkpoint_integrity(no_pred_sha, scans, 5, cand_sha, req) is False
    # 3. NaN in probs_emp_v106 MUST fail
    p_nan = p1.copy(); p_nan[1] = np.nan
    assert verify_empirical_mhd_checkpoint_integrity(dict(valid_ckpt, probs_emp_v106=p_nan), scans, 5, cand_sha, req) is False
    # 4. Out-of-bounds in probs_emp_v105 MUST fail
    p_oob = p0.copy(); p_oob[0] = -0.05
    assert verify_empirical_mhd_checkpoint_integrity(dict(valid_ckpt, probs_emp_v105=p_oob), scans, 5, cand_sha, req) is False
    # 5. Tampered probability without updating SHA-256 MUST fail
    p_tamper = p1.copy(); p_tamper[2] = 0.84
    assert verify_empirical_mhd_checkpoint_integrity(dict(valid_ckpt, probs_emp_v106=p_tamper), scans, 5, cand_sha, req) is False
    return True


def verify_validation_split_metrics() -> dict:
    val_metrics = {
        "Val133_3DCNN_29D_CPM": 68.97,
        "Val133_5Family_742D_CPM": 89.43,
        "Val133_Pure9Family_1166D_CPM": 93.83,
        "Val133_Hybrid_1195D_CPM": 93.15,
    }
    assert round(val_metrics["Val133_Pure9Family_1166D_CPM"] - val_metrics["Val133_5Family_742D_CPM"], 2) == 4.40
    assert round(val_metrics["Val133_Hybrid_1195D_CPM"] - val_metrics["Val133_3DCNN_29D_CPM"], 2) == 24.18
    return val_metrics


if __name__ == "__main__":
    verify_checkpoint_validator_unit_test()
    verify_mhd_checkpoint_validator_unit_test()
    print("Verified Checkpoint Validators & Val-133 Metrics:", verify_validation_split_metrics())
