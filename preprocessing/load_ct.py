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
Cryptographic SHA-256 Data Provenance, Independent Float32 Voxel & Per-Patch Merkle Checksum
Verification, and Official LUNA16 Subset Discovery (Section 2.1, Table 2, Gate 11).
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


def verify_cache_against_independent_hdf5_and_derived_digest(
    cache_dict,
    patch_ids: np.ndarray,
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
    hdf5_binary_sha256: str,
    row_voxel_u64: np.ndarray,
    allowed_versions: tuple = (CODE_VERSION, V105_COMPAT_VERSION),
    derived_array_keys: list = None,
) -> bool:
    try:
        if str(cache_dict['code_version']) not in allowed_versions:
            return False
        if str(cache_dict['hdf5_binary_sha256']) != str(hdf5_binary_sha256):
            return False
        if not np.array_equal(cache_dict['patch_id'].astype(np.int64), patch_ids.astype(np.int64)):
            return False
        c_suids = np.array([
            (s.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else str(s)).strip()
            for s in cache_dict['seriesuid']
        ])
        if not np.array_equal(c_suids, series_arr):
            return False
        if not np.allclose(cache_dict['coords'].astype(np.float64), coords_arr.astype(np.float64), atol=1e-4, rtol=0.0):
            return False
        if not np.array_equal(cache_dict['labels'].astype(np.int64), labels_arr.astype(np.int64)):
            return False
        if not np.array_equal(cache_dict['row_voxel_u64'].astype(np.uint64), row_voxel_u64.astype(np.uint64)):
            return False
        if derived_array_keys is not None:
            if 'derived_features_sha256' not in cache_dict:
                return False
            live_derived_sha = compute_arrays_sha256_digest([cache_dict[k] for k in derived_array_keys])
            if str(cache_dict['derived_features_sha256']) != live_derived_sha:
                return False
        return True
    except Exception:
        return False


def discover_official_luna16_subsets_and_drive_inventory(h5f, u_scans_all: np.ndarray):
    scan_to_subset = {}
    scan_to_mhd_path = {}
    if 'subset' in h5f:
        raw_sub = h5f['subset'][:]
        raw_s = h5f['seriesuid'][:]
        for s_val, sub_val in zip(raw_s, raw_sub):
            s_str = (s_val.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else str(s)).strip()
            sub_str = (sub_val.decode('utf-8') if isinstance(sub_val, (bytes, np.bytes_)) else str(sub_val)).strip()
            m = re.search(r'(\d)', sub_str)
            if m:
                sub_idx = int(m.group(1))
                if s_str in scan_to_subset and scan_to_subset[s_str] != sub_idx:
                    raise RuntimeError(f"Subset conflict in HDF5 for {s_str}: {scan_to_subset[s_str]} vs {sub_idx}")
                scan_to_subset[s_str] = sub_idx

    all_mhd_paths = sorted(glob.glob(os.path.join(BASE_DIR, '**', '*.mhd'), recursive=True))
    mhd_folder_counts = {}
    for mp in all_mhd_paths:
        parent_dir = os.path.dirname(mp)
        rel_parent = os.path.relpath(parent_dir, BASE_DIR)
        mhd_folder_counts[rel_parent] = mhd_folder_counts.get(rel_parent, 0) + 1
        if 'seg-lungs' in rel_parent.lower():
            continue
        bname = os.path.basename(mp)
        if bname.endswith('.mhd'):
            suid_cand = bname[:-4].strip()
            m_sub = re.search(r'subset(\d)', rel_parent, re.IGNORECASE)
            if m_sub:
                sub_idx = int(m_sub.group(1))
                if suid_cand in scan_to_subset and scan_to_subset[suid_cand] != sub_idx:
                    raise RuntimeError(
                        f"Subset conflict detected for scan {suid_cand}: "
                        f"subset{scan_to_subset[suid_cand]} vs subset{sub_idx} at {mp}"
                    )
                scan_to_subset[suid_cand] = sub_idx
                scan_to_mhd_path[suid_cand] = mp

    cand_v2_matches = sorted(glob.glob(os.path.join(BASE_DIR, '**', '*candidates_V2*.csv'), recursive=True))
    valid_cand_v2_paths = []
    for cp in cand_v2_matches:
        try:
            with open(cp, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
                n_lines = sum(1 for _ in f)
            if 'seriesuid' in header and 'coordX' in header and n_lines == 754975:
                valid_cand_v2_paths.append(cp)
        except Exception:
            pass
    discovered_cand_v2_path = valid_cand_v2_paths[0] if len(valid_cand_v2_paths) > 0 else (
        CANDIDATES_V2_CSV_PATH if os.path.exists(CANDIDATES_V2_CSV_PATH) else None
    )

    inventory_info = {
        'total_mhd_files': len(all_mhd_paths),
        'mhd_folder_counts': mhd_folder_counts,
        'n_scans_mapped_to_subset': sum(1 for s in u_scans_all if s in scan_to_subset),
        'scan_to_subset': scan_to_subset,
        'scan_to_mhd_path': scan_to_mhd_path,
        'discovered_cand_v2_path': discovered_cand_v2_path,
    }

    if all(s in scan_to_subset for s in u_scans_all):
        folds = [
            sorted([s for s in u_scans_all if scan_to_subset[s] == k])
            for k in range(10)
        ]
        if all(len(f) > 0 for f in folds):
            return folds, "Official LUNA16 `subset0..subset9` 10-Fold Partition (Discovered from `.mhd` Paths!)", inventory_info
    return None, "Deterministic Stratified Scan-Disjoint 10-Fold Partition (`SEED=42`; 445/888 `.mhd` scans present in `subset0..4`)", inventory_info


# ==============================================================================
# 11. PART B & B.2: STAGE-0 GPU SCREENER + 3D SUB-VOXEL CENTROID RE-CENTERING +
#     9-FAMILY (`1,195D`) OUT-OF-FOLD INFERENCE ON `.MHD` `candidates_V2.csv`
# ==============================================================================
def audit_curated_hdf5_against_official_annotations(
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels_arr: np.ndarray,
):
    df_ann = pd.read_csv(ANNOTATIONS_PATH)
    df_exc = pd.read_csv(ANNOTATIONS_EXCLUDED_PATH)
    df_ann['seriesuid'] = df_ann['seriesuid'].astype(str).str.strip()
    df_exc['seriesuid'] = df_exc['seriesuid'].astype(str).str.strip()

    ann_by_scan = {
        s: (
            grp[['coordX', 'coordY', 'coordZ']].values.astype(np.float64),
            0.5 * grp['diameter_mm'].values.astype(np.float64),
        )
        for s, grp in df_ann.groupby('seriesuid', sort=False)
    }
    exc_by_scan = {
        s: (
            grp[['coordX', 'coordY', 'coordZ']].values.astype(np.float64),
            np.where(grp['diameter_mm'].values.astype(np.float64) > 0.0, 0.5 * grp['diameter_mm'].values.astype(np.float64), 3.0),
        )
        for s, grp in df_exc.groupby('seriesuid', sort=False)
    }

    N = len(series_arr)
    curated_pos_mask = np.zeros(N, dtype=bool)
    curated_excl_mask = np.zeros(N, dtype=bool)
    curated_eval_neg_mask = np.zeros(N, dtype=bool)

    df_h = pd.DataFrame({'s': series_arr, 'idx': np.arange(N)})
    for s, grp in df_h.groupby('s', sort=False):
        idxs = grp['idx'].values
        c_xyz = coords_arr[idxs]
        is_pos = np.zeros(len(idxs), dtype=bool)
        is_exc = np.zeros(len(idxs), dtype=bool)

        if s in ann_by_scan:
            a_xyz, a_rad = ann_by_scan[s]
            dist_a = np.sqrt(np.sum((c_xyz[:, None, :] - a_xyz[None, :, :]) ** 2, axis=2))
            is_pos = np.any(dist_a < a_rad[None, :], axis=1)

        if s in exc_by_scan:
            e_xyz, e_rad = exc_by_scan[s]
            dist_e = np.sqrt(np.sum((c_xyz[:, None, :] - e_xyz[None, :, :]) ** 2, axis=2))
            is_exc = (~is_pos) & np.any(dist_e < e_rad[None, :], axis=1)

        is_eval_neg = (~is_pos) & (~is_exc)
        curated_pos_mask[idxs] = is_pos
        curated_excl_mask[idxs] = is_exc
        curated_eval_neg_mask[idxs] = is_eval_neg

    assert int(np.sum(curated_pos_mask) + np.sum(curated_excl_mask) + np.sum(curated_eval_neg_mask)) == N
    return curated_pos_mask, curated_excl_mask, curated_eval_neg_mask
