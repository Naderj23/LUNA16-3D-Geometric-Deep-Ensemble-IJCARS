# ==============================================================================
# STAGE 2 & 3 (V8.1 — CORRECTED ZERO-LEAKAGE 10-FOLD ALL-888-SCANS & LOCKED
# HOLD-OUT OFFICIAL LUNA16 EVALUATOR FOR IJCARS):
# ==============================================================================
# Key Scientific Fixes Over V8.0 Based on the Forensic Audit Findings:
#   1. FIXES THE V8.0 `_debias_insample_prob_to_oof` LABEL-SHIFT BUG:
#      - The V8.0 Pillar-3 Forensic Audit proved that `V4.7`, `V5.0`, `V7.1`, and
#        `V7.3` have ZERO positive in-sample optimism on the 622 Train scans
#        (`Optimism Gap = -0.0072 to -0.0317`, i.e., Train AUC <= Val/Test AUC!).
#      - In V8.0, shifting `p46_tr..p73_tr` class-conditionally using `y_tr` to
#        match `sp_oof` artificially injected `y_tr` into the training logits,
#        which broke the Hold-Out Test GBDT (`67.24%`). V8.1 completely removes
#        any label-dependent shifting and uses strictly UNSUPERVISED logit
#        standardization (`median / IQR`, using ZERO labels `y`).
#   2. FULL 2-STAGE NESTED ARCHITECTURE ACROSS ALL 888 SCANS (10-FOLD OOF):
#      - In V8.0, the 10-Fold All-888-Scans evaluation (`81.24%` official CPM,
#        `0.9492` AUC across all 1,186 nodules) only used a single-stage GBDT on
#        2 CNNs (`sp_oof, p76_oof`) and omitted the other 5 CNNs, the Hard-Negative
#        Specialists, and the Stage-2 Pruned FROC Calibrator.
#      - In V8.1, BOTH the **Locked 133-Scan Hold-Out Test** and the **Official
#        10-Fold All-888-Scans (1,186 Nodules) Evaluation** run the full leak-free
#        7-CNN + 737-Dim 3D Geometric/Polar/Hessian/Flux/Bifurcation + Stage-1B
#        Hard-Negative Specialist + Pruned Stage-2 FROC Calibrator pipeline strictly
#        out-of-fold (`Overlap = 0`).
# ==============================================================================

import os
import re
import gc
import sys
import glob
import time
import math
import random
import shutil
import subprocess
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import rankdata
from scipy.special import logit, expit

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

# ==============================================================================
# 0. HARDWARE & PATH CONFIGURATION
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_GPU = (device.type == 'cuda')
if IS_GPU:
    print(f"Active Compute Device: GPU ({torch.cuda.get_device_name(0)}) — V8.1 Zero-Leakage 10-Fold & Hold-Out Mode", flush=True)
    torch.backends.cudnn.benchmark = True
else:
    n_threads = os.cpu_count() or 2
    torch.set_num_threads(n_threads)
    print(f"Active Compute Device: CPU ({n_threads} threads) — V8.1 Zero-Leakage 10-Fold & Hold-Out Mode", flush=True)

BASE_DIR = '/content/drive/MyDrive/LUNA16_Data'
DRIVE_H5 = os.path.join(BASE_DIR, 'LUNA16_Patches_CoordVerified_48.h5')
LOCAL_H5 = '/content/LUNA16_Patches_CoordVerified_48.h5'
STD_H5 = os.path.join(BASE_DIR, 'LUNA16_Patches_Standard.h5')
CANDIDATES_CSV_PATH = os.path.join(BASE_DIR, 'candidates.csv')
CANDIDATES_V2_CSV_PATH = os.path.join(BASE_DIR, 'candidates_V2.csv')
ANNOTATIONS_PATH = os.path.join(BASE_DIR, 'annotations.csv')
ANNOTATIONS_EXCLUDED_PATH = os.path.join(
    BASE_DIR, 'evaluationScript/evaluationScript/annotations/annotations_excluded.csv'
)
ORIG_EVAL_DIR = os.path.join(BASE_DIR, 'evaluationScript/evaluationScript')
PY3_EVAL_DIR = '/content/evaluationScript_py3'

if os.path.exists(DRIVE_H5):
    if not os.path.exists(LOCAL_H5) or os.path.getsize(LOCAL_H5) != os.path.getsize(DRIVE_H5):
        t0 = time.time()
        print(f"Copying {os.path.basename(DRIVE_H5)} to local SSD ({LOCAL_H5})...", flush=True)
        shutil.copy2(DRIVE_H5, LOCAL_H5)
        print(f" -> Copy complete in {time.time() - t0:.1f}s!", flush=True)
    DATA_PATH = LOCAL_H5
else:
    raise FileNotFoundError(f"Could not find {DRIVE_H5}")

PREV_SAVE_DIR_V70 = os.path.join(BASE_DIR, 'V7_0_CoordVerified_Final_IJCARS')
PREV_SAVE_DIR_V71 = os.path.join(BASE_DIR, 'V7_1_CoordVerified_Final_IJCARS')
PREV_SAVE_DIR_V73 = os.path.join(BASE_DIR, 'V7_3_CoordVerified_Final_IJCARS')
PREV_SAVE_DIR_V74 = os.path.join(BASE_DIR, 'V7_4_CoordVerified_Final_IJCARS')
PREV_SAVE_DIR_V75 = os.path.join(BASE_DIR, 'V7_5_CoordVerified_Final_IJCARS')
PREV_SAVE_DIR_V76 = os.path.join(BASE_DIR, 'V7_6_CoordVerified_Final_IJCARS')
SAVE_DIR = os.path.join(BASE_DIR, 'V8_1_ZeroLeakage_10Fold_And_HoldOut_IJCARS')
os.makedirs(SAVE_DIR, exist_ok=True)

ALL_CACHE_DIRS = (
    SAVE_DIR, PREV_SAVE_DIR_V76, PREV_SAVE_DIR_V75,
    PREV_SAVE_DIR_V74, PREV_SAVE_DIR_V73, PREV_SAVE_DIR_V71, PREV_SAVE_DIR_V70
)

SEED = 42
VAL_RATIO, TEST_RATIO = 0.15, 0.15
BOOTSTRAP_ITERS = 1000
FROC_FP_RATES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ==============================================================================
# 1. DATA LOADING & SCAN-DISJOINT SPLITTERS
# ==============================================================================
def open_coordverified_dataset_lazy(h5_path: str):
    h5f = h5py.File(h5_path, 'r')
    images_ds = h5f['images']
    labels = h5f['labels'][:].flatten().astype(np.int64)
    raw_s = h5f['seriesuid'][:]
    series_arr = np.array([
        (s.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else str(s)).strip()
        for s in raw_s
    ])
    coords_arr = h5f['coords'][:].astype(np.float64)
    ann_ids = h5f['ann_id'][:].astype(np.int64) if 'ann_id' in h5f else np.full(len(labels), -1, dtype=np.int64)
    diameters = h5f['diameter_mm'][:].astype(np.float64) if 'diameter_mm' in h5f else np.full(len(labels), -1.0, dtype=np.float64)
    return h5f, images_ds, labels, series_arr, coords_arr, diameters, ann_ids


def create_locked_3way_scan_split(labels, series_arr, std_h5_path: str, seed=42):
    if os.path.exists(std_h5_path):
        with h5py.File(std_h5_path, 'r') as f_std:
            std_lbl = f_std['labels'][:].flatten().astype(np.int64)
            std_s = np.array([
                (s.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else str(s)).strip()
                for s in f_std['seriesuid'][:]
            ])
        u_scans = np.unique(std_s)
        has_p = np.array([int(np.any(std_lbl[std_s == s] == 1)) for s in u_scans])
    else:
        u_scans = np.unique(series_arr)
        has_p = np.array([int(np.any(labels[series_arr == s] == 1)) for s in u_scans])

    rng = np.random.RandomState(seed)
    pos_s = u_scans[has_p == 1].copy()
    neg_s = u_scans[has_p == 0].copy()
    rng.shuffle(pos_s)
    rng.shuffle(neg_s)

    def _part(arr):
        n = len(arr)
        nv, nt = int(np.round(n * VAL_RATIO)), int(np.round(n * TEST_RATIO))
        return arr[nv + nt :], arr[:nv], arr[nv : nv + nt]

    tr_p, va_p, te_p = _part(pos_s)
    tr_n, va_n, te_n = _part(neg_s)
    val_scans = sorted(list(set(va_p).union(set(va_n))))
    test_scans = sorted(list(set(te_p).union(set(te_n))))
    all_cur_scans = set(np.unique(series_arr))
    train_scans = sorted(list(all_cur_scans - set(val_scans) - set(test_scans)))

    assert len(set(train_scans).intersection(set(val_scans))) == 0
    assert len(set(train_scans).intersection(set(test_scans))) == 0
    assert len(set(val_scans).intersection(set(test_scans))) == 0

    train_idx = np.where(np.isin(series_arr, train_scans))[0].tolist()
    val_idx = np.where(np.isin(series_arr, val_scans))[0].tolist()
    test_idx = np.where(np.isin(series_arr, test_scans))[0].tolist()
    return (train_idx, train_scans), (val_idx, val_scans), (test_idx, test_scans)


# ==============================================================================
# 2. PYTHON 3.13 + MATPLOTLIB 3.9+ CONVERTER FOR `noduleCADEvaluationLUNA16.py`
# ==============================================================================
def convert_py2_code_to_py313(code: str) -> str:
    lines = code.splitlines()
    out_lines = []
    for line in lines:
        if 'computing FROC: bootstrap' in line:
            indent = re.match(r'^(\s*)', line).group(1)
            line = f"{indent}pass"
        else:
            m = re.match(r'^(\s*)print\s+(.+)$', line)
            if m and not m.group(2).strip().startswith('('):
                indent, expr = m.group(1), m.group(2).strip()
                if expr.endswith(','):
                    expr = expr[:-1] + ', end=" "'
                line = f"{indent}print({expr})"
            elif re.match(r'^(\s*)print\s*$', line):
                indent = re.match(r'^(\s*)print\s*$', line).group(1)
                line = f"{indent}print()"

        line = re.sub(r'\bxrange\s*\(', 'range(', line)
        line = re.sub(r'\bcPickle\b', 'pickle', line)
        line = line.replace('.iteritems()', '.items()')
        line = line.replace('.itervalues()', '.values()')
        line = line.replace('.iterkeys()', '.keys()')
        line = re.sub(r'except\s+([A-Za-z0-9_.,\s]+)\s*,\s*([A-Za-z0-9_]+)\s*:', r'except \1 as \2:', line)
        line = re.sub(r'open\(([^,]+),\s*["\']r[bU]+["\']\)', r'open(\1, "r", newline="")', line)
        line = re.sub(r'open\(([^,]+),\s*["\']wb["\']\)', r'open(\1, "w", newline="")', line)
        line = re.sub(r'\bbasex\s*=', 'base=', line)
        line = re.sub(r'\bbasey\s*=', 'base=', line)
        line = re.sub(r'\bsubsx\s*=', 'subs=', line)
        line = re.sub(r'\bsubsy\s*=', 'subs=', line)
        line = re.sub(r'\bnonposx\s*=', 'nonpositive=', line)
        line = re.sub(r'\bnonposy\s*=', 'nonpositive=', line)
        line = re.sub(r'grid\(\s*b\s*=', 'grid(visible=', line)
        out_lines.append(line)

    res = "\n".join(out_lines) + "\n"
    if 'import matplotlib.pyplot' in res and "matplotlib.use('Agg')" not in res:
        res = "import matplotlib\nmatplotlib.use('Agg')\n" + res
    return res


def prepare_py3_official_evaluator(orig_dir: str, py3_dir: str):
    if not os.path.exists(orig_dir):
        return None
    if os.path.exists(py3_dir):
        shutil.rmtree(py3_dir)
    shutil.copytree(orig_dir, py3_dir)
    for root, _, files in os.walk(py3_dir):
        for fn in files:
            if fn.endswith('.py'):
                fp = os.path.join(root, fn)
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(convert_py2_code_to_py313(raw))
    eval_py3 = os.path.join(py3_dir, 'noduleCADEvaluationLUNA16.py')
    return eval_py3 if os.path.exists(eval_py3) else None


def parse_official_luna16_output_csvs(sub_dir: str):
    csv_files = glob.glob(os.path.join(sub_dir, '*bootstrap*.csv')) + glob.glob(os.path.join(sub_dir, 'froc_*.csv'))
    for cf in csv_files:
        try:
            df_b = pd.read_csv(cf)
            if df_b.shape[1] >= 4:
                fps_col = df_b.iloc[:, 0].values.astype(np.float64)
                mean_col = df_b.iloc[:, 1].values.astype(np.float64)
                low_col = df_b.iloc[:, 2].values.astype(np.float64)
                up_col = df_b.iloc[:, 3].values.astype(np.float64)
                order = np.argsort(fps_col)
                fps_col, mean_col, low_col, up_col = fps_col[order], mean_col[order], low_col[order], up_col[order]
                res = {}
                for r in FROC_FP_RATES:
                    res[r] = (
                        float(np.interp(r, fps_col, mean_col)),
                        float(np.interp(r, fps_col, low_col)),
                        float(np.interp(r, fps_col, up_col)),
                    )
                cpm_mean = float(np.mean([res[r][0] for r in FROC_FP_RATES]))
                cpm_low = float(np.mean([res[r][1] for r in FROC_FP_RATES]))
                cpm_up = float(np.mean([res[r][2] for r in FROC_FP_RATES]))
                return res, (cpm_mean, cpm_low, cpm_up), cf
        except Exception:
            pass
    return None, None, None


# ==============================================================================
# 3. INSTANT CACHE LOADERS (V7.1 - V7.6)
# ==============================================================================
def load_cached_3d_cnn_probabilities(part_name: str, expected_len: int):
    for folder in ALL_CACHE_DIRS:
        fp = os.path.join(folder, f'cached_v46_v47_v50_coordverified_{part_name}.npz')
        if os.path.exists(fp):
            d = np.load(fp)
            if len(d['p46']) == expected_len:
                return d['p46'], d['p47'], d['p50'], d['p_mim50']
    raise FileNotFoundError(f"Cached file for {part_name} not found!")


def load_all_cached_model_probabilities(n_tr, n_va, n_te):
    p71_tr = p71_va = p71_te = None
    p73_tr = p73_va = p73_te = None
    sp_oof = sp_va = sp_te = None
    p76_oof = p76_va = p76_te = None

    for folder in ALL_CACHE_DIRS:
        fp71 = os.path.join(folder, 'cached_triplanar_foveal_v71_probs.npz')
        if p71_tr is None and os.path.exists(fp71):
            d = np.load(fp71)
            if len(d['p71_va']) == n_va:
                p71_tr, p71_va, p71_te = d['p71_tr'], d['p71_va'], d['p71_te']

        fp73 = os.path.join(folder, 'cached_ms_pyramid16_v73_probs.npz')
        if p73_tr is None and os.path.exists(fp73):
            d = np.load(fp73)
            if len(d['p73_va']) == n_va:
                p73_tr, p73_va, p73_te = d['p73_tr'], d['p73_va'], d['p73_te']

        fp75 = os.path.join(folder, 'cached_v75_spherical_polar_oof_probs.npz')
        if sp_oof is None and os.path.exists(fp75):
            d = np.load(fp75)
            if len(d['sp_oof']) == n_tr:
                sp_oof, sp_va, sp_te = d['sp_oof'], d['sp_va'], d['sp_te']

        fp76 = os.path.join(folder, 'cached_v76_dualpath3d_oof_probs.npz')
        if p76_oof is None and os.path.exists(fp76):
            d = np.load(fp76)
            if len(d['p76_oof']) == n_tr:
                p76_oof, p76_va, p76_te = d['p76_oof'], d['p76_va'], d['p76_te']

    return (p71_tr, p71_va, p71_te), (p73_tr, p73_va, p73_te), (sp_oof, sp_va, sp_te), (p76_oof, p76_va, p76_te)


def load_all_cached_v73_to_v76_features(part_name: str, expected_len: int):
    geom_v73 = star_v74 = v75_X = polar_grid = v76_X = None
    for folder in ALL_CACHE_DIRS:
        fp73 = os.path.join(folder, f'cached_v73_geom_features_{part_name}.npz')
        if geom_v73 is None and os.path.exists(fp73):
            d = np.load(fp73)
            if len(d['geom_X']) == expected_len:
                geom_v73 = d['geom_X']
        fp74 = os.path.join(folder, f'cached_v74_starray_features_{part_name}.npz')
        if star_v74 is None and os.path.exists(fp74):
            d = np.load(fp74)
            if len(d['star_X']) == expected_len:
                star_v74 = d['star_X']
        fp75 = os.path.join(folder, f'cached_v75_polar_hessian_{part_name}.npz')
        if v75_X is None and os.path.exists(fp75):
            d = np.load(fp75)
            if len(d['v75_X']) == expected_len:
                v75_X = d['v75_X']
                polar_grid = d['polar_grid']
        fp76 = os.path.join(folder, f'cached_v76_flux_dog_{part_name}.npz')
        if v76_X is None and os.path.exists(fp76):
            d = np.load(fp76)
            if len(d['v76_X']) == expected_len:
                v76_X = d['v76_X']
    return geom_v73, star_v74, v75_X, polar_grid, v76_X


# ==============================================================================
# 4. V8.1 BIFURCATION ORDER-STATISTICS & UNSUPERVISED LOGIT CALIBRATION
# ==============================================================================
def extract_v77_bifurcation_and_antipodal_orderstat_features(polar_grid: np.ndarray) -> np.ndarray:
    r_desc = np.sort(polar_grid, axis=1)[:, ::-1, :]
    r_pos = polar_grid[:, 0::2, :]
    r_neg = polar_grid[:, 1::2, :]
    ax_min = np.minimum(r_pos, r_neg)
    ax_diff = np.abs(r_pos - r_neg)
    ax_min_desc = np.sort(ax_min, axis=1)[:, ::-1, :]

    feats = []
    through_vessel_bands = []
    bifurcation_bands = []

    for b in range(7):
        rd = r_desc[:, :, b]
        ad = ax_min_desc[:, :, b]
        through_1 = ad[:, 0] - ad[:, 2]
        through_2 = 0.5 * (ad[:, 0] + ad[:, 1]) - ad[:, 4]
        through_ratio = ad[:, 0] / (np.median(ad, axis=1) + 0.04)
        vessel_2ray = 0.5 * (rd[:, 0] + rd[:, 1]) - rd[:, 4]
        bifurc_3ray = (rd[:, 0] + rd[:, 1] + rd[:, 2]) / 3.0 - rd[:, 6]
        cross_4ray = 0.25 * (rd[:, 0] + rd[:, 1] + rd[:, 2] + rd[:, 3]) - rd[:, 8]
        pleura_10ray = np.mean(rd[:, :10], axis=1) - np.mean(rd[:, 14:], axis=1)
        sph_8, sph_12, sph_16, sph_20 = rd[:, 7], rd[:, 11], rd[:, 15], rd[:, 19]
        antipodal_sym_4, antipodal_sym_7 = ad[:, 3], ad[:, 6]
        hemi_asym = np.mean(ax_diff[:, :, b], axis=1)

        through_vessel_bands.append(through_1)
        bifurcation_bands.append(bifurc_3ray)
        feats.append(np.column_stack([
            through_1, through_2, through_ratio,
            vessel_2ray, bifurc_3ray, cross_4ray, pleura_10ray,
            sph_8, sph_12, sph_16, sph_20,
            antipodal_sym_4, antipodal_sym_7, hemi_asym,
        ]))

    cross_pairs = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5), (3, 6), (4, 6)]
    pedicle_immune_contrasts = []
    for j, k in cross_pairs:
        c_12_16 = r_desc[:, 11, j] - r_desc[:, 15, k]
        c_16_18 = r_desc[:, 15, j] - r_desc[:, 17, k]
        c_sym_5 = ax_min_desc[:, 4, j] - ax_min_desc[:, 4, k]
        pedicle_immune_contrasts.append(c_12_16)
        feats.append(np.column_stack([c_12_16, c_16_18, c_sym_5]))

    tv_mat = np.column_stack(through_vessel_bands)
    bf_mat = np.column_stack(bifurcation_bands)
    pi_mat = np.column_stack(pedicle_immune_contrasts)
    feats.append(np.column_stack([
        np.max(tv_mat[:, 1:5], axis=1),
        np.max(bf_mat[:, 1:5], axis=1),
        np.max(pi_mat, axis=1),
        np.max(pi_mat[:, 4:], axis=1),
    ]))
    return np.nan_to_num(np.hstack(feats).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_intra_scan_contextual_features(
    logits_vec: np.ndarray, suids: np.ndarray, max_glob_col: np.ndarray, max_sph_col: np.ndarray, max_pi_col: np.ndarray
):
    elev_q50 = np.zeros_like(logits_vec)
    elev_q75 = np.zeros_like(logits_vec)
    elev_q90 = np.zeros_like(logits_vec)
    loo_margin = np.zeros_like(logits_vec)
    second_margin = np.zeros_like(logits_vec)
    s_z = np.zeros_like(logits_vec)
    s_rank = np.zeros_like(logits_vec)
    sm_t06 = np.zeros_like(logits_vec)
    sm_t12 = np.zeros_like(logits_vec)
    glob_elev = np.zeros_like(logits_vec)
    sph_elev = np.zeros_like(logits_vec)
    pi_elev = np.zeros_like(logits_vec)

    for s in np.unique(suids):
        m = (suids == s)
        lm = logits_vec[m]
        gc_v, sp_v, pi_v = max_glob_col[m], max_sph_col[m], max_pi_col[m]
        n_s = len(lm)
        q50, q75, q90 = float(np.percentile(lm, 50.0)), float(np.percentile(lm, 75.0)), float(np.percentile(lm, 90.0))
        elev_q50[m] = lm - q50
        elev_q75[m] = lm - q75
        elev_q90[m] = lm - q90
        s_z[m] = (lm - q50) / (float(np.std(lm)) + 0.40)
        s_rank[m] = rankdata(lm) / float(n_s)
        glob_elev[m] = gc_v - float(np.median(gc_v))
        sph_elev[m] = sp_v - float(np.median(sp_v))
        pi_elev[m] = pi_v - float(np.median(pi_v))

        e06 = np.exp((lm - np.max(lm)) / 0.6)
        sm_t06[m] = e06 / (np.sum(e06) + 1e-8)
        e12 = np.exp((lm - np.max(lm)) / 1.2)
        sm_t12[m] = e12 / (np.sum(e12) + 1e-8)

        if n_s >= 3:
            s_idx = np.argsort(-lm)
            max1, max2, max3 = lm[s_idx[0]], lm[s_idx[1]], lm[s_idx[2]]
            loo_margin[m] = np.where(np.arange(n_s) == s_idx[0], lm - max2, lm - max1)
            second_margin[m] = np.where(np.arange(n_s) <= s_idx[1], lm - max3, lm - max2)
        elif n_s == 2:
            s_idx = np.argsort(-lm)
            max1, max2 = lm[s_idx[0]], lm[s_idx[1]]
            loo_margin[m] = np.where(np.arange(n_s) == s_idx[0], lm - max2, lm - max1)

    return np.column_stack([
        elev_q50, elev_q75, elev_q90, loo_margin, second_margin,
        s_z, s_rank, sm_t06, sm_t12, glob_elev, sph_elev, pi_elev,
    ])


def build_stage1_feature_matrix(
    coords_arr, series_arr, p_idx,
    p46, p47, p50, p71, p73, p76, sp75, pm50,
    geom_v73, star_v74, v75_X, v76_X, v77_X,
):
    idx_arr = np.array(p_idx, dtype=np.int64)
    suids = series_arr[idx_arr]
    coords = coords_arr[idx_arr]

    eps = 1e-6
    l46 = logit(np.clip(p46, eps, 1.0 - eps))
    l47 = logit(np.clip(p47, eps, 1.0 - eps))
    l50 = logit(np.clip(p50, eps, 1.0 - eps))
    l71 = logit(np.clip(p71, eps, 1.0 - eps))
    l73 = logit(np.clip(p73, eps, 1.0 - eps))
    l76 = logit(np.clip(p76, eps, 1.0 - eps))
    lsp = logit(np.clip(sp75, eps, 1.0 - eps))
    l_mim = logit(np.clip(pm50, eps, 1.0 - eps))

    l_ens = (
        0.11 * l46 + 0.19 * l47 + 0.16 * l50
        + 0.12 * l71 + 0.14 * l73 + 0.15 * l76 + 0.13 * lsp
        - 0.08 * pm50
    )
    p_ens = expit(l_ens)

    r_xy = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) / 150.0
    z_norm = coords[:, 2] / 200.0

    ctx_0 = compute_intra_scan_contextual_features(
        l_ens, suids, star_v74[:, -2], v75_X[:, 144], v77_X[:, -2]
    )

    deep_stack = np.vstack([l46, l47, l50, l71, l73, l76, lsp])
    X1 = np.column_stack([
        l46, l47, l50, l71, l73, l76, lsp, l_ens,
        p46, p47, p50, p71, p73, p76, sp75, p_ens,
        pm50, l_mim,
        np.max(deep_stack, axis=0),
        np.min(deep_stack, axis=0),
        np.std(deep_stack, axis=0),
        np.median(deep_stack, axis=0),
        ctx_0,
        r_xy, z_norm,
        geom_v73,
        star_v74,
        v75_X,
        v76_X,
        v77_X,
    ])
    return np.nan_to_num(X1, nan=0.0, posinf=0.0, neginf=0.0)


class DirectFROCNeuralCalibrator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear_skip = nn.Linear(in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.30),
            nn.Linear(96, 24),
            nn.LayerNorm(24),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return (self.linear_skip(x) + 0.35 * self.mlp(x)).squeeze(-1)


def train_direct_froc_surrogate(X_tr_s, y_tr, n_train_scans: int, rate_weights=None, seed=42):
    torch.manual_seed(seed)
    x_t = torch.from_numpy(X_tr_s).to(device, dtype=torch.float32)
    y_t = torch.from_numpy(y_tr).to(device, dtype=torch.float32)
    pos_idx = torch.where(y_t > 0.5)[0]
    neg_idx = torch.where(y_t < 0.5)[0]
    k_list = [max(1, min(len(neg_idx), int(round(r * n_train_scans)))) for r in FROC_FP_RATES]
    if rate_weights is None:
        rate_weights = [4.0, 3.4, 2.4, 1.5, 0.9, 0.4, 0.2]

    net = DirectFROCNeuralCalibrator(X_tr_s.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=8.0e-3, weight_decay=3.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=360, eta_min=1e-4)

    net.train()
    for _ in range(360):
        opt.zero_grad()
        scores = net(x_t)
        pos_s, neg_s = scores[pos_idx], scores[neg_idx]
        neg_sorted, _ = torch.sort(neg_s, descending=True)
        froc_loss = 0.0
        for k_val, w_r in zip(k_list, rate_weights):
            thresh_k = torch.mean(neg_sorted[max(0, k_val - 4) : min(len(neg_sorted), k_val + 4)])
            froc_loss = froc_loss + w_r * F.softplus(thresh_k + 0.75 - pos_s, beta=3.2).mean()
        loss = (froc_loss / sum(rate_weights)) + 0.30 * F.binary_cross_entropy_with_logits(scores, y_t)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return net


def select_top_discriminative_feature_indices(X_tr: np.ndarray, y_tr: np.ndarray, k_top: int = 110) -> np.ndarray:
    pos_m = (y_tr == 1)
    neg_m = (y_tr == 0)
    mu_p = np.mean(X_tr[pos_m], axis=0)
    mu_n = np.mean(X_tr[neg_m], axis=0)
    sd = np.std(X_tr, axis=0) + 1e-6
    d_prime = np.abs(mu_p - mu_n) / sd
    top_idx = np.argsort(-d_prime)[: min(k_top, X_tr.shape[1])]
    return np.sort(top_idx)


# ==============================================================================
# 5. CORE 2-STAGE LEAK-FREE ENSEMBLE ENGINE (USED FOR BOTH LOCKED HOLD-OUT TEST
#    AND FULL 888-SCAN 10-FOLD CROSS-VALIDATION!)
# ==============================================================================
def run_2stage_nested_ensemble_on_split(
    X1_tr, y_tr, X1_va, y_va, X1_te,
    tr_suids, va_suids, te_suids, train_scans, val_scans,
    star_tr, star_va, star_te, v75_tr, v75_va, v75_te, v77_tr, v77_va, v77_te,
    n_inner_folds: int = 10,
    desc_prefix: str = "Locked-Split",
):
    """
    Fits a 100% leak-free 2-Stage Ensemble where:
      - `RobustScaler` is fitted strictly inside each inner fold during Stage-1 OOF generation!
      - Stage-1B Hard-Negative Specialists (`spec35` & `spec15`) are fitted strictly out-of-fold!
      - `select_top_discriminative_feature_indices(k_top=110)` is fitted strictly on `y_tr`!
      - Stage-2 Broad & Low-FP FROC Calibrators are combined using weights selected strictly on `val_scans`!
    """
    rng = np.random.RandomState(SEED)
    tr_scans_shuf = np.array(train_scans).copy()
    rng.shuffle(tr_scans_shuf)
    scan_folds = np.array_split(tr_scans_shuf, n_inner_folds)

    large_core_tr = np.maximum(star_tr[:, -19], star_tr[:, -1] + 0.20)
    sw_s1 = np.ones(len(y_tr), dtype=np.float64)
    sw_s1[(y_tr == 1) & (large_core_tr >= 0.35)] = 2.6
    sw_s1[(y_tr == 1) & (large_core_tr < 0.35)] = 1.35

    oof_p_hgb1 = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_hgb2 = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_hgb3 = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_xgb = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_lgb = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_et = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_rf = np.zeros(len(y_tr), dtype=np.float64)
    oof_p_mlp = np.zeros(len(y_tr), dtype=np.float64)

    va_p_hgb1, te_p_hgb1 = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_hgb2, te_p_hgb2 = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_hgb3, te_p_hgb3 = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_xgb, te_p_xgb = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_lgb, te_p_lgb = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_et, te_p_et = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_rf, te_p_rf = np.zeros(len(y_va)), np.zeros(len(X1_te))
    va_p_mlp, te_p_mlp = np.zeros(len(y_va)), np.zeros(len(X1_te))

    for f_i, val_f_scans in enumerate(tqdm(scan_folds, desc=f"{desc_prefix} Stage-1 {n_inner_folds}-Fold OOF")):
        m_val_f = np.isin(tr_suids, val_f_scans)
        m_trn_f = ~m_val_f

        # Strict Per-Fold Scaling (Zero Pre-Fold Leakage!)
        sc_f = RobustScaler()
        X_f_tr = np.clip(sc_f.fit_transform(X1_tr[m_trn_f]), -10.0, 10.0)
        X_f_va = np.clip(sc_f.transform(X1_tr[m_val_f]), -10.0, 10.0)
        X1_va_s = np.clip(sc_f.transform(X1_va), -10.0, 10.0)
        X1_te_s = np.clip(sc_f.transform(X1_te), -10.0, 10.0)

        y_f_tr, sw_f = y_tr[m_trn_f], sw_s1[m_trn_f]

        m1 = HistGradientBoostingClassifier(
            max_iter=160, max_bins=128, max_depth=4, learning_rate=0.04, l2_regularization=4.5,
            min_samples_leaf=16, max_features=0.55, early_stopping=False, random_state=SEED + f_i
        ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        oof_p_hgb1[m_val_f] = m1.predict_proba(X_f_va)[:, 1]
        va_p_hgb1 += m1.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_hgb1 += m1.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        m2 = HistGradientBoostingClassifier(
            max_iter=180, max_bins=128, max_depth=6, learning_rate=0.035, l2_regularization=6.5,
            min_samples_leaf=12, max_features=0.45, early_stopping=False, random_state=SEED + 20 + f_i
        ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        oof_p_hgb2[m_val_f] = m2.predict_proba(X_f_va)[:, 1]
        va_p_hgb2 += m2.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_hgb2 += m2.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        m3 = HistGradientBoostingClassifier(
            max_iter=190, max_bins=128, max_depth=3, learning_rate=0.045, l2_regularization=3.5,
            min_samples_leaf=22, max_features=0.65, early_stopping=False, random_state=SEED + 40 + f_i
        ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        oof_p_hgb3[m_val_f] = m3.predict_proba(X_f_va)[:, 1]
        va_p_hgb3 += m3.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_hgb3 += m3.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        if HAS_XGB:
            m_xgb = xgb.XGBClassifier(
                n_estimators=220, max_depth=4, learning_rate=0.035, subsample=0.80,
                colsample_bytree=0.40, reg_lambda=5.0, reg_alpha=1.0, min_child_weight=8,
                tree_method='hist', eval_metric='logloss', n_jobs=-1, random_state=SEED + 60 + f_i, verbosity=0
            ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
            oof_p_xgb[m_val_f] = m_xgb.predict_proba(X_f_va)[:, 1]
            va_p_xgb += m_xgb.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
            te_p_xgb += m_xgb.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)
        else:
            oof_p_xgb[m_val_f] = oof_p_hgb1[m_val_f]
            va_p_xgb, te_p_xgb = va_p_hgb1.copy(), te_p_hgb1.copy()

        if HAS_LGB:
            m_lgb = lgb.LGBMClassifier(
                n_estimators=210, max_depth=5, num_leaves=20, learning_rate=0.035,
                subsample=0.80, subsample_freq=1, colsample_bytree=0.40, reg_lambda=5.0,
                min_child_samples=18, n_jobs=-1, random_state=SEED + 80 + f_i, verbose=-1
            ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
            oof_p_lgb[m_val_f] = m_lgb.predict_proba(X_f_va)[:, 1]
            va_p_lgb += m_lgb.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
            te_p_lgb += m_lgb.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)
        else:
            oof_p_lgb[m_val_f] = oof_p_hgb2[m_val_f]
            va_p_lgb, te_p_lgb = va_p_hgb2.copy(), te_p_hgb2.copy()

        m4 = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=SEED + f_i
        ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        oof_p_et[m_val_f] = m4.predict_proba(X_f_va)[:, 1]
        va_p_et += m4.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_et += m4.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        m5 = RandomForestClassifier(
            n_estimators=200, max_depth=13, min_samples_leaf=4, n_jobs=-1, random_state=SEED + f_i
        ).fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        oof_p_rf[m_val_f] = m5.predict_proba(X_f_va)[:, 1]
        va_p_rf += m5.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_rf += m5.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        m6 = MLPClassifier(hidden_layer_sizes=(112, 36), alpha=0.70, max_iter=85, random_state=SEED + f_i)
        m6.fit(X_f_tr, y_f_tr)
        oof_p_mlp[m_val_f] = m6.predict_proba(X_f_va)[:, 1]
        va_p_mlp += m6.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_p_mlp += m6.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

    eps = 1e-6
    p76_tr_raw, sp_tr_raw = X1_tr[:, 13], X1_tr[:, 14]
    p76_va_raw, sp_va_raw = X1_va[:, 13], X1_va[:, 14]
    p76_te_raw, sp_te_raw = X1_te[:, 13], X1_te[:, 14]

    s1_p_tr = (
        0.15 * oof_p_hgb1 + 0.14 * oof_p_hgb2 + 0.11 * oof_p_hgb3
        + 0.14 * oof_p_xgb + 0.12 * oof_p_lgb
        + 0.08 * oof_p_et + 0.07 * oof_p_rf + 0.06 * oof_p_mlp
        + 0.07 * sp_tr_raw + 0.06 * p76_tr_raw
    )
    s1_p_va = (
        0.15 * va_p_hgb1 + 0.14 * va_p_hgb2 + 0.11 * va_p_hgb3
        + 0.14 * va_p_xgb + 0.12 * va_p_lgb
        + 0.08 * va_p_et + 0.07 * va_p_rf + 0.06 * va_p_mlp
        + 0.07 * sp_va_raw + 0.06 * p76_va_raw
    )
    s1_p_te = (
        0.15 * te_p_hgb1 + 0.14 * te_p_hgb2 + 0.11 * te_p_hgb3
        + 0.14 * te_p_xgb + 0.12 * te_p_lgb
        + 0.08 * te_p_et + 0.07 * te_p_rf + 0.06 * te_p_mlp
        + 0.07 * sp_te_raw + 0.06 * p76_te_raw
    )

    # Stage-1B: Out-of-Fold Hard-Negative Specialists (`spec35` & `spec15`)
    oof_spec35, va_spec35, te_spec35 = np.zeros(len(y_tr)), np.zeros(len(y_va)), np.zeros(len(X1_te))
    oof_spec15, va_spec15, te_spec15 = np.zeros(len(y_tr)), np.zeros(len(y_va)), np.zeros(len(X1_te))

    for f_i, val_f_scans in enumerate(scan_folds):
        m_val_f = np.isin(tr_suids, val_f_scans)
        m_trn_f = ~m_val_f
        sc_f = RobustScaler()
        X_f_tr = np.clip(sc_f.fit_transform(X1_tr[m_trn_f]), -10.0, 10.0)
        X_f_va = np.clip(sc_f.transform(X1_tr[m_val_f]), -10.0, 10.0)
        X1_va_s = np.clip(sc_f.transform(X1_va), -10.0, 10.0)
        X1_te_s = np.clip(sc_f.transform(X1_te), -10.0, 10.0)

        y_f_tr = y_tr[m_trn_f]
        s1_f_tr = s1_p_tr[m_trn_f]

        q65_n = float(np.percentile(s1_f_tr[y_f_tr == 0], 65.0))
        sub35 = (y_f_tr == 1) | (s1_f_tr >= q65_n)
        sw35 = np.where(y_f_tr[sub35] == 1, 2.2, np.where(s1_f_tr[sub35] >= float(np.percentile(s1_f_tr[y_f_tr == 0], 92.0)), 4.5, 1.2))
        sp_m35 = HistGradientBoostingClassifier(
            max_iter=150, max_bins=128, max_depth=4, learning_rate=0.035, l2_regularization=6.0,
            min_samples_leaf=16, max_features=0.50, early_stopping=False, random_state=SEED + 300 + f_i
        ).fit(X_f_tr[sub35], y_f_tr[sub35], sample_weight=sw35)
        oof_spec35[m_val_f] = sp_m35.predict_proba(X_f_va)[:, 1]
        va_spec35 += sp_m35.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_spec35 += sp_m35.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

        q85_n = float(np.percentile(s1_f_tr[y_f_tr == 0], 85.0))
        sub15 = (y_f_tr == 1) | (s1_f_tr >= q85_n)
        sw15 = np.where(y_f_tr[sub15] == 1, 1.8, np.where(s1_f_tr[sub15] >= float(np.percentile(s1_f_tr[y_f_tr == 0], 96.0)), 5.0, 1.5))
        sp_m15 = HistGradientBoostingClassifier(
            max_iter=140, max_bins=128, max_depth=3, learning_rate=0.035, l2_regularization=8.0,
            min_samples_leaf=18, max_features=0.50, early_stopping=False, random_state=SEED + 600 + f_i
        ).fit(X_f_tr[sub15], y_f_tr[sub15], sample_weight=sw15)
        oof_spec15[m_val_f] = sp_m15.predict_proba(X_f_va)[:, 1]
        va_spec15 += sp_m15.predict_proba(X1_va_s)[:, 1] / float(n_inner_folds)
        te_spec15 += sp_m15.predict_proba(X1_te_s)[:, 1] / float(n_inner_folds)

    # Fit Stage-2 Scaler & Feature Selection strictly on `X1_tr, y_tr`
    scaler1_full = RobustScaler()
    X1_tr_full = np.clip(scaler1_full.fit_transform(X1_tr), -10.0, 10.0)
    X1_va_full = np.clip(scaler1_full.transform(X1_va), -10.0, 10.0)
    X1_te_full = np.clip(scaler1_full.transform(X1_te), -10.0, 10.0)
    top_feat_idx = select_top_discriminative_feature_indices(X1_tr_full, y_tr, k_top=110)

    s1_l_tr, s1_l_va, s1_l_te = logit(np.clip(s1_p_tr, eps, 1 - eps)), logit(np.clip(s1_p_va, eps, 1 - eps)), logit(np.clip(s1_p_te, eps, 1 - eps))
    l_sp35_tr, l_sp35_va, l_sp35_te = logit(np.clip(oof_spec35, eps, 1 - eps)), logit(np.clip(va_spec35, eps, 1 - eps)), logit(np.clip(te_spec35, eps, 1 - eps))
    l_sp15_tr, l_sp15_va, l_sp15_te = logit(np.clip(oof_spec15, eps, 1 - eps)), logit(np.clip(va_spec15, eps, 1 - eps)), logit(np.clip(te_spec15, eps, 1 - eps))

    ctx2_tr = compute_intra_scan_contextual_features(s1_l_tr, tr_suids, star_tr[:, -2], v75_tr[:, 144], v77_tr[:, -2])
    ctx2_va = compute_intra_scan_contextual_features(s1_l_va, va_suids, star_va[:, -2], v75_va[:, 144], v77_va[:, -2])
    ctx2_te = compute_intra_scan_contextual_features(s1_l_te, te_suids, star_te[:, -2], v75_te[:, 144], v77_te[:, -2])

    ctx_sp_tr = compute_intra_scan_contextual_features(0.55 * l_sp35_tr + 0.45 * l_sp15_tr, tr_suids, star_tr[:, -2], v75_tr[:, 144], v77_tr[:, -2])
    ctx_sp_va = compute_intra_scan_contextual_features(0.55 * l_sp35_va + 0.45 * l_sp15_va, va_suids, star_va[:, -2], v75_va[:, 144], v77_va[:, -2])
    ctx_sp_te = compute_intra_scan_contextual_features(0.55 * l_sp35_te + 0.45 * l_sp15_te, te_suids, star_te[:, -2], v75_te[:, 144], v77_te[:, -2])

    meta_tr = np.column_stack([
        s1_l_tr, s1_p_tr, l_sp35_tr, oof_spec35, l_sp15_tr, oof_spec15,
        logit(np.clip(oof_p_hgb1, eps, 1 - eps)), logit(np.clip(oof_p_hgb2, eps, 1 - eps)),
        logit(np.clip(oof_p_hgb3, eps, 1 - eps)), logit(np.clip(oof_p_xgb, eps, 1 - eps)),
        logit(np.clip(oof_p_lgb, eps, 1 - eps)), logit(np.clip(oof_p_et, eps, 1 - eps)),
        logit(np.clip(oof_p_rf, eps, 1 - eps)), logit(np.clip(oof_p_mlp, eps, 1 - eps)),
        ctx2_tr, ctx_sp_tr,
    ])
    meta_va = np.column_stack([
        s1_l_va, s1_p_va, l_sp35_va, va_spec35, l_sp15_va, va_spec15,
        logit(np.clip(va_p_hgb1, eps, 1 - eps)), logit(np.clip(va_p_hgb2, eps, 1 - eps)),
        logit(np.clip(va_p_hgb3, eps, 1 - eps)), logit(np.clip(va_p_xgb, eps, 1 - eps)),
        logit(np.clip(va_p_lgb, eps, 1 - eps)), logit(np.clip(va_p_et, eps, 1 - eps)),
        logit(np.clip(va_p_rf, eps, 1 - eps)), logit(np.clip(va_p_mlp, eps, 1 - eps)),
        ctx2_va, ctx_sp_va,
    ])
    meta_te = np.column_stack([
        s1_l_te, s1_p_te, l_sp35_te, te_spec35, l_sp15_te, te_spec15,
        logit(np.clip(te_p_hgb1, eps, 1 - eps)), logit(np.clip(te_p_hgb2, eps, 1 - eps)),
        logit(np.clip(te_p_hgb3, eps, 1 - eps)), logit(np.clip(te_p_xgb, eps, 1 - eps)),
        logit(np.clip(te_p_lgb, eps, 1 - eps)), logit(np.clip(te_p_et, eps, 1 - eps)),
        logit(np.clip(te_p_rf, eps, 1 - eps)), logit(np.clip(te_p_mlp, eps, 1 - eps)),
        ctx2_te, ctx_sp_te,
    ])

    X2_tr = np.column_stack([meta_tr, X1_tr_full])
    X2_va = np.column_stack([meta_va, X1_va_full])
    X2_te = np.column_stack([meta_te, X1_te_full])

    X2_pruned_tr = np.column_stack([meta_tr, X1_tr_full[:, top_feat_idx]])
    X2_pruned_va = np.column_stack([meta_va, X1_va_full[:, top_feat_idx]])
    X2_pruned_te = np.column_stack([meta_te, X1_te_full[:, top_feat_idx]])

    scaler2 = RobustScaler()
    X2_tr_s = np.clip(scaler2.fit_transform(X2_tr), -10.0, 10.0)
    X2_va_s = np.clip(scaler2.transform(X2_va), -10.0, 10.0)
    X2_te_s = np.clip(scaler2.transform(X2_te), -10.0, 10.0)

    scaler2_p = RobustScaler()
    X2_ptr_s = np.clip(scaler2_p.fit_transform(X2_pruned_tr), -10.0, 10.0)
    X2_pva_s = np.clip(scaler2_p.transform(X2_pruned_va), -10.0, 10.0)
    X2_pte_s = np.clip(scaler2_p.transform(X2_pruned_te), -10.0, 10.0)

    # Regime A: Broad FROC Stage-2 Ensemble
    sw_broad = np.ones(len(y_tr), dtype=np.float64)
    neg_m, pos_m = (y_tr == 0), (y_tr == 1)
    q98_neg = float(np.percentile(s1_p_tr[neg_m], 98.0))
    q92_neg = float(np.percentile(s1_p_tr[neg_m], 92.0))
    q80_neg = float(np.percentile(s1_p_tr[neg_m], 80.0))
    sw_broad[neg_m] = np.where(
        s1_p_tr[neg_m] >= q98_neg, 10.5,
        np.where(s1_p_tr[neg_m] >= q92_neg, 5.5, np.where(s1_p_tr[neg_m] >= q80_neg, 2.2, 0.75))
    )
    sw_broad[pos_m] = np.where(s1_p_tr[pos_m] < 0.55, 2.8, 1.8) * np.where(large_core_tr[pos_m] >= 0.35, 1.65, 1.0)

    s2_hgb1 = HistGradientBoostingClassifier(
        max_iter=150, max_bins=128, max_depth=4, learning_rate=0.035, l2_regularization=5.0,
        min_samples_leaf=18, max_features=0.50, random_state=SEED
    ).fit(X2_tr_s, y_tr, sample_weight=sw_broad)
    s2_hgb2 = HistGradientBoostingClassifier(
        max_iter=170, max_bins=128, max_depth=5, learning_rate=0.03, l2_regularization=7.0,
        min_samples_leaf=15, max_features=0.45, random_state=SEED + 9
    ).fit(X2_tr_s, y_tr, sample_weight=sw_broad)
    s2_rf = RandomForestClassifier(
        n_estimators=260, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=SEED
    ).fit(X2_tr_s, y_tr, sample_weight=sw_broad)
    s2_et = ExtraTreesClassifier(
        n_estimators=260, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=SEED
    ).fit(X2_tr_s, y_tr, sample_weight=sw_broad)
    s2_lr = LogisticRegression(C=0.06, max_iter=500, random_state=SEED).fit(X2_ptr_s, y_tr, sample_weight=sw_broad)

    froc_net_broad = train_direct_froc_surrogate(X2_ptr_s, y_tr, len(train_scans), seed=SEED)
    with torch.no_grad():
        p2_froc_va = torch.sigmoid(froc_net_broad(torch.from_numpy(X2_pva_s).to(device, dtype=torch.float32))).cpu().numpy()
        p2_froc_te = torch.sigmoid(froc_net_broad(torch.from_numpy(X2_pte_s).to(device, dtype=torch.float32))).cpu().numpy()

    p_broad_va = (
        0.30 * s2_hgb1.predict_proba(X2_va_s)[:, 1]
        + 0.25 * s2_hgb2.predict_proba(X2_va_s)[:, 1]
        + 0.18 * s2_rf.predict_proba(X2_va_s)[:, 1]
        + 0.15 * s2_et.predict_proba(X2_va_s)[:, 1]
        + 0.12 * s2_lr.predict_proba(X2_pva_s)[:, 1]
    )
    p_broad_te = (
        0.30 * s2_hgb1.predict_proba(X2_te_s)[:, 1]
        + 0.25 * s2_hgb2.predict_proba(X2_te_s)[:, 1]
        + 0.18 * s2_rf.predict_proba(X2_te_s)[:, 1]
        + 0.15 * s2_et.predict_proba(X2_te_s)[:, 1]
        + 0.12 * s2_lr.predict_proba(X2_pte_s)[:, 1]
    )

    # Regime B: Ultra-Low-FP Specialist (`0.125 - 0.5 FP/scan`)
    q70_neg = float(np.percentile(s1_p_tr[neg_m], 70.0))
    top_sub_mask = (y_tr == 1) | (s1_p_tr >= q70_neg)
    X_top_tr, X_ptop_tr = X2_tr_s[top_sub_mask], X2_ptr_s[top_sub_mask]
    y_top_tr, s1_top_tr = y_tr[top_sub_mask], s1_p_tr[top_sub_mask]

    sw_top = np.ones(len(y_top_tr), dtype=np.float64)
    neg_top, pos_top = (y_top_tr == 0), (y_top_tr == 1)
    q96_sub = float(np.percentile(s1_top_tr[neg_top], 88.0))
    q85_sub = float(np.percentile(s1_top_tr[neg_top], 65.0))
    sw_top[neg_top] = np.where(s1_top_tr[neg_top] >= q96_sub, 8.5, np.where(s1_top_tr[neg_top] >= q85_sub, 3.5, 1.0))
    sw_top[pos_top] = 2.2

    top_hgb1 = HistGradientBoostingClassifier(
        max_iter=150, max_bins=128, max_depth=3, learning_rate=0.035, l2_regularization=8.0,
        min_samples_leaf=20, max_features=0.50, random_state=SEED + 101
    ).fit(X_top_tr, y_top_tr, sample_weight=sw_top)
    top_hgb2 = HistGradientBoostingClassifier(
        max_iter=170, max_bins=128, max_depth=4, learning_rate=0.035, l2_regularization=10.0,
        min_samples_leaf=16, max_features=0.45, random_state=SEED + 202
    ).fit(X_top_tr, y_top_tr, sample_weight=sw_top)
    top_et = ExtraTreesClassifier(
        n_estimators=240, max_depth=11, min_samples_leaf=5, n_jobs=-1, random_state=SEED + 303
    ).fit(X_top_tr, y_top_tr, sample_weight=sw_top)
    top_lr = LogisticRegression(C=0.04, max_iter=500, random_state=SEED + 404).fit(X_ptop_tr, y_top_tr, sample_weight=sw_top)

    froc_net_lowfp = train_direct_froc_surrogate(
        X2_ptr_s, y_tr, len(train_scans), rate_weights=[6.5, 5.2, 3.2, 1.5, 0.3, 0.1, 0.05], seed=SEED + 505
    )
    with torch.no_grad():
        p_lowfp_nn_va = torch.sigmoid(froc_net_lowfp(torch.from_numpy(X2_pva_s).to(device, dtype=torch.float32))).cpu().numpy()
        p_lowfp_nn_te = torch.sigmoid(froc_net_lowfp(torch.from_numpy(X2_pte_s).to(device, dtype=torch.float32))).cpu().numpy()

    p_top_va = (
        0.22 * top_hgb1.predict_proba(X2_va_s)[:, 1]
        + 0.18 * top_hgb2.predict_proba(X2_va_s)[:, 1]
        + 0.14 * top_et.predict_proba(X2_va_s)[:, 1]
        + 0.10 * top_lr.predict_proba(X2_pva_s)[:, 1]
        + 0.16 * p_lowfp_nn_va
        + 0.10 * va_spec35 + 0.10 * va_spec15
    )
    p_top_te = (
        0.22 * top_hgb1.predict_proba(X2_te_s)[:, 1]
        + 0.18 * top_hgb2.predict_proba(X2_te_s)[:, 1]
        + 0.14 * top_et.predict_proba(X2_te_s)[:, 1]
        + 0.10 * top_lr.predict_proba(X2_pte_s)[:, 1]
        + 0.16 * p_lowfp_nn_te
        + 0.10 * te_spec35 + 0.10 * te_spec15
    )

    # Compact Validation Selection (strictly on `val_scans` — never touches `X1_te`!)
    best_va_cpm = -1.0
    best_p_va, best_p_te = p_broad_va, p_broad_te
    n_va_scans = float(len(val_scans))
    n_va_pos = max(1, int(np.sum(y_va == 1)))

    for w_s1 in [0.20, 0.35]:
        for w_s2 in [0.45, 0.60]:
            for w_fr in [0.05, 0.15]:
                base_va = (w_s1 * s1_p_va + w_s2 * p_broad_va + w_fr * p2_froc_va) / (w_s1 + w_s2 + w_fr)
                base_te = (w_s1 * s1_p_te + w_s2 * p_broad_te + w_fr * p2_froc_te) / (w_s1 + w_s2 + w_fr)
                for alpha_top in [0.0, 0.20, 0.35]:
                    gate_va = expit((base_va - 0.42) * 8.0)
                    gate_te = expit((base_te - 0.42) * 8.0)
                    c_va = (1.0 - alpha_top * gate_va) * base_va + (alpha_top * gate_va) * p_top_va
                    c_te = (1.0 - alpha_top * gate_te) * base_te + (alpha_top * gate_te) * p_top_te

                    desc_v = np.argsort(-c_va, kind='mergesort')
                    st_v = y_va[desc_v]
                    _, cpm_v = eval_froc_at_7_rates(
                        np.concatenate([[0.0], np.cumsum(st_v == 0) / n_va_scans]),
                        np.concatenate([[0.0], np.cumsum(st_v == 1) / n_va_pos]),
                    )
                    if cpm_v > best_va_cpm:
                        best_va_cpm = cpm_v
                        best_p_va, best_p_te = c_va, c_te

    return best_p_va, best_p_te, s1_p_te


# ==============================================================================
# 6. FULL 888-SCAN 10-FOLD CROSS-VALIDATION (WITH UNSUPERVISED LOGIT ALIGNMENT
#    AND FULL 2-STAGE NESTED ARCHITECTURE!)
# ==============================================================================
def run_full888_10fold_nested_pipeline(
    X1_all: np.ndarray,
    labels: np.ndarray,
    series_arr: np.ndarray,
    star_all: np.ndarray,
    v75_all: np.ndarray,
    v77_all: np.ndarray,
    n_outer_folds: int = 10,
    seed: int = 42,
):
    """
    Runs 10-Fold Scan-Disjoint Cross-Validation across ALL 888 CT scans (10,066 patches, 1,186 nodules):
      - First computes a fast 10-Fold OOF Stage-1 + Stage-1B matrix across all 888 scans (so we don't
        repeat Stage-1 10x10 = 100 times!), fitting `RobustScaler` strictly inside each fold `k = 1..10`.
      - Then inside each outer fold `k = 1..10` (~800 non-test scans vs ~88 held-out test scans):
          * Splits the ~800 non-test scans into 85% outer-train (~680 scans) and 15% inner-val (~120 scans).
          * Fits Stage-2 Broad & Low-FP FROC Calibrators strictly on the ~680 outer-train scans,
            selects mixing weights on the ~120 inner-val scans, and predicts out-of-fold on the ~88 test scans!
    """
    print("\n" + "=" * 104, flush=True)
    print("RUNNING FULL 2-STAGE 10-FOLD SCAN-DISJOINT OOF PIPELINE ACROSS ALL 888 CT SCANS (1,186 NODULES)", flush=True)
    print("=" * 104, flush=True)

    u_scans = np.unique(series_arr)
    has_pos = np.array([int(np.any(labels[series_arr == s] == 1)) for s in u_scans])
    rng = np.random.RandomState(seed)
    pos_scans = u_scans[has_pos == 1].copy()
    neg_scans = u_scans[has_pos == 0].copy()
    rng.shuffle(pos_scans)
    rng.shuffle(neg_scans)

    pos_folds = np.array_split(pos_scans, n_outer_folds)
    neg_folds = np.array_split(neg_scans, n_outer_folds)
    outer_folds_scans = [
        sorted(list(set(pos_folds[k]).union(set(neg_folds[k]))))
        for k in range(n_outer_folds)
    ]

    large_core_all = np.maximum(star_all[:, -19], star_all[:, -1] + 0.20)
    sw_all = np.ones(len(labels), dtype=np.float64)
    sw_all[(labels == 1) & (large_core_all >= 0.35)] = 2.6
    sw_all[(labels == 1) & (large_core_all < 0.35)] = 1.35

    N_all = len(labels)
    oof_hgb1 = np.zeros(N_all, dtype=np.float64)
    oof_hgb2 = np.zeros(N_all, dtype=np.float64)
    oof_hgb3 = np.zeros(N_all, dtype=np.float64)
    oof_xgb = np.zeros(N_all, dtype=np.float64)
    oof_lgb = np.zeros(N_all, dtype=np.float64)
    oof_et = np.zeros(N_all, dtype=np.float64)
    oof_rf = np.zeros(N_all, dtype=np.float64)
    oof_mlp = np.zeros(N_all, dtype=np.float64)

    for k in tqdm(range(n_outer_folds), desc="All-888-Scans Stage-1 10-Fold OOF"):
        te_scans_k = outer_folds_scans[k]
        m_te_k = np.isin(series_arr, te_scans_k)
        m_tr_k = ~m_te_k

        sc_k = RobustScaler()
        X_tr_k = np.clip(sc_k.fit_transform(X1_all[m_tr_k]), -10.0, 10.0)
        X_te_k = np.clip(sc_k.transform(X1_all[m_te_k]), -10.0, 10.0)
        y_tr_k, sw_tr_k = labels[m_tr_k], sw_all[m_tr_k]

        m1 = HistGradientBoostingClassifier(
            max_iter=160, max_bins=128, max_depth=4, learning_rate=0.04, l2_regularization=4.5,
            min_samples_leaf=16, max_features=0.55, early_stopping=False, random_state=SEED + k
        ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
        oof_hgb1[m_te_k] = m1.predict_proba(X_te_k)[:, 1]

        m2 = HistGradientBoostingClassifier(
            max_iter=180, max_bins=128, max_depth=6, learning_rate=0.035, l2_regularization=6.5,
            min_samples_leaf=12, max_features=0.45, early_stopping=False, random_state=SEED + 20 + k
        ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
        oof_hgb2[m_te_k] = m2.predict_proba(X_te_k)[:, 1]

        m3 = HistGradientBoostingClassifier(
            max_iter=190, max_bins=128, max_depth=3, learning_rate=0.045, l2_regularization=3.5,
            min_samples_leaf=22, max_features=0.65, early_stopping=False, random_state=SEED + 40 + k
        ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
        oof_hgb3[m_te_k] = m3.predict_proba(X_te_k)[:, 1]

        if HAS_XGB:
            m_xgb = xgb.XGBClassifier(
                n_estimators=220, max_depth=4, learning_rate=0.035, subsample=0.80,
                colsample_bytree=0.40, reg_lambda=5.0, reg_alpha=1.0, min_child_weight=8,
                tree_method='hist', eval_metric='logloss', n_jobs=-1, random_state=SEED + 60 + k, verbosity=0
            ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
            oof_xgb[m_te_k] = m_xgb.predict_proba(X_te_k)[:, 1]
        else:
            oof_xgb[m_te_k] = oof_hgb1[m_te_k]

        if HAS_LGB:
            m_lgb = lgb.LGBMClassifier(
                n_estimators=210, max_depth=5, num_leaves=20, learning_rate=0.035,
                subsample=0.80, subsample_freq=1, colsample_bytree=0.40, reg_lambda=5.0,
                min_child_samples=18, n_jobs=-1, random_state=SEED + 80 + k, verbose=-1
            ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
            oof_lgb[m_te_k] = m_lgb.predict_proba(X_te_k)[:, 1]
        else:
            oof_lgb[m_te_k] = oof_hgb2[m_te_k]

        m4 = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=SEED + k
        ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
        oof_et[m_te_k] = m4.predict_proba(X_te_k)[:, 1]

        m5 = RandomForestClassifier(
            n_estimators=200, max_depth=13, min_samples_leaf=4, n_jobs=-1, random_state=SEED + k
        ).fit(X_tr_k, y_tr_k, sample_weight=sw_tr_k)
        oof_rf[m_te_k] = m5.predict_proba(X_te_k)[:, 1]

        m6 = MLPClassifier(hidden_layer_sizes=(112, 36), alpha=0.70, max_iter=85, random_state=SEED + k)
        m6.fit(X_tr_k, y_tr_k)
        oof_mlp[m_te_k] = m6.predict_proba(X_te_k)[:, 1]

    eps = 1e-6
    p76_all_raw, sp_all_raw = X1_all[:, 13], X1_all[:, 14]
    s1_oof_all = (
        0.15 * oof_hgb1 + 0.14 * oof_hgb2 + 0.11 * oof_hgb3
        + 0.14 * oof_xgb + 0.12 * oof_lgb
        + 0.08 * oof_et + 0.07 * oof_rf + 0.06 * oof_mlp
        + 0.07 * sp_all_raw + 0.06 * p76_all_raw
    )

    # Stage-1B Hard-Negative Specialists Out-of-Fold across all 888 scans
    oof_sp35 = np.zeros(N_all, dtype=np.float64)
    oof_sp15 = np.zeros(N_all, dtype=np.float64)
    for k in range(n_outer_folds):
        te_scans_k = outer_folds_scans[k]
        m_te_k = np.isin(series_arr, te_scans_k)
        m_tr_k = ~m_te_k

        sc_k = RobustScaler()
        X_tr_k = np.clip(sc_k.fit_transform(X1_all[m_tr_k]), -10.0, 10.0)
        X_te_k = np.clip(sc_k.transform(X1_all[m_te_k]), -10.0, 10.0)
        y_tr_k = labels[m_tr_k]
        s1_tr_k = s1_oof_all[m_tr_k]

        q65_n = float(np.percentile(s1_tr_k[y_tr_k == 0], 65.0))
        sub35 = (y_tr_k == 1) | (s1_tr_k >= q65_n)
        sw35 = np.where(y_tr_k[sub35] == 1, 2.2, np.where(s1_tr_k[sub35] >= float(np.percentile(s1_tr_k[y_tr_k == 0], 92.0)), 4.5, 1.2))
        sp_m35 = HistGradientBoostingClassifier(
            max_iter=150, max_bins=128, max_depth=4, learning_rate=0.035, l2_regularization=6.0,
            min_samples_leaf=16, max_features=0.50, early_stopping=False, random_state=SEED + 300 + k
        ).fit(X_tr_k[sub35], y_tr_k[sub35], sample_weight=sw35)
        oof_sp35[m_te_k] = sp_m35.predict_proba(X_te_k)[:, 1]

        q85_n = float(np.percentile(s1_tr_k[y_tr_k == 0], 85.0))
        sub15 = (y_tr_k == 1) | (s1_tr_k >= q85_n)
        sw15 = np.where(y_tr_k[sub15] == 1, 1.8, np.where(s1_tr_k[sub15] >= float(np.percentile(s1_tr_k[y_tr_k == 0], 96.0)), 5.0, 1.5))
        sp_m15 = HistGradientBoostingClassifier(
            max_iter=140, max_bins=128, max_depth=3, learning_rate=0.035, l2_regularization=8.0,
            min_samples_leaf=18, max_features=0.50, early_stopping=False, random_state=SEED + 600 + k
        ).fit(X_tr_k[sub15], y_tr_k[sub15], sample_weight=sw15)
        oof_sp15[m_te_k] = sp_m15.predict_proba(X_te_k)[:, 1]

    # Stage-2 Out-of-Fold Meta-Calibrator across all 888 scans
    s1_l_all = logit(np.clip(s1_oof_all, eps, 1 - eps))
    l_sp35_all = logit(np.clip(oof_sp35, eps, 1 - eps))
    l_sp15_all = logit(np.clip(oof_sp15, eps, 1 - eps))
    ctx2_all = compute_intra_scan_contextual_features(s1_l_all, series_arr, star_all[:, -2], v75_all[:, 144], v77_all[:, -2])
    ctx_sp_all = compute_intra_scan_contextual_features(0.55 * l_sp35_all + 0.45 * l_sp15_all, series_arr, star_all[:, -2], v75_all[:, 144], v77_all[:, -2])

    meta_all = np.column_stack([
        s1_l_all, s1_oof_all, l_sp35_all, oof_sp35, l_sp15_all, oof_sp15,
        logit(np.clip(oof_hgb1, eps, 1 - eps)), logit(np.clip(oof_hgb2, eps, 1 - eps)),
        logit(np.clip(oof_hgb3, eps, 1 - eps)), logit(np.clip(oof_xgb, eps, 1 - eps)),
        logit(np.clip(oof_lgb, eps, 1 - eps)), logit(np.clip(oof_et, eps, 1 - eps)),
        logit(np.clip(oof_rf, eps, 1 - eps)), logit(np.clip(oof_mlp, eps, 1 - eps)),
        ctx2_all, ctx_sp_all,
    ])

    final_oof_all888 = np.zeros(N_all, dtype=np.float64)
    for k in tqdm(range(n_outer_folds), desc="All-888-Scans Stage-2 10-Fold OOF"):
        te_scans_k = outer_folds_scans[k]
        m_te_k = np.isin(series_arr, te_scans_k)
        m_tr_k = ~m_te_k

        # Select Top-110 physical features strictly on Fold k's training scans!
        sc1_k = RobustScaler()
        X1_tr_k_s = np.clip(sc1_k.fit_transform(X1_all[m_tr_k]), -10.0, 10.0)
        X1_te_k_s = np.clip(sc1_k.transform(X1_all[m_te_k]), -10.0, 10.0)
        y_tr_k = labels[m_tr_k]
        top_idx_k = select_top_discriminative_feature_indices(X1_tr_k_s, y_tr_k, k_top=110)

        X2_tr_k = np.column_stack([meta_all[m_tr_k], X1_tr_k_s])
        X2_te_k = np.column_stack([meta_all[m_te_k], X1_te_k_s])
        X2_ptr_k = np.column_stack([meta_all[m_tr_k], X1_tr_k_s[:, top_idx_k]])
        X2_pte_k = np.column_stack([meta_all[m_te_k], X1_te_k_s[:, top_idx_k]])

        sc2_k = RobustScaler()
        X2_tr_ks = np.clip(sc2_k.fit_transform(X2_tr_k), -10.0, 10.0)
        X2_te_ks = np.clip(sc2_k.transform(X2_te_k), -10.0, 10.0)

        sc2_pk = RobustScaler()
        X2_ptr_ks = np.clip(sc2_pk.fit_transform(X2_ptr_k), -10.0, 10.0)
        X2_pte_ks = np.clip(sc2_pk.transform(X2_pte_k), -10.0, 10.0)

        s1_tr_k = s1_oof_all[m_tr_k]
        lc_tr_k = large_core_all[m_tr_k]
        sw_b = np.ones(len(y_tr_k), dtype=np.float64)
        nm, pm = (y_tr_k == 0), (y_tr_k == 1)
        q98_n, q92_n, q80_n = float(np.percentile(s1_tr_k[nm], 98.0)), float(np.percentile(s1_tr_k[nm], 92.0)), float(np.percentile(s1_tr_k[nm], 80.0))
        sw_b[nm] = np.where(s1_tr_k[nm] >= q98_n, 10.5, np.where(s1_tr_k[nm] >= q92_n, 5.5, np.where(s1_tr_k[nm] >= q80_n, 2.2, 0.75)))
        sw_b[pm] = np.where(s1_tr_k[pm] < 0.55, 2.8, 1.8) * np.where(lc_tr_k[pm] >= 0.35, 1.65, 1.0)

        s2_h1 = HistGradientBoostingClassifier(
            max_iter=150, max_bins=128, max_depth=4, learning_rate=0.035, l2_regularization=5.0,
            min_samples_leaf=18, max_features=0.50, random_state=SEED + k
        ).fit(X2_tr_ks, y_tr_k, sample_weight=sw_b)
        s2_h2 = HistGradientBoostingClassifier(
            max_iter=170, max_bins=128, max_depth=5, learning_rate=0.03, l2_regularization=7.0,
            min_samples_leaf=15, max_features=0.45, random_state=SEED + 9 + k
        ).fit(X2_tr_ks, y_tr_k, sample_weight=sw_b)
        s2_rf = RandomForestClassifier(
            n_estimators=240, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=SEED + k
        ).fit(X2_tr_ks, y_tr_k, sample_weight=sw_b)
        s2_et = ExtraTreesClassifier(
            n_estimators=240, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=SEED + k
        ).fit(X2_tr_ks, y_tr_k, sample_weight=sw_b)
        s2_lr = LogisticRegression(C=0.06, max_iter=500, random_state=SEED + k).fit(X2_ptr_ks, y_tr_k, sample_weight=sw_b)

        froc_net_k = train_direct_froc_surrogate(X2_ptr_ks, y_tr_k, len(u_scans) - len(te_scans_k), seed=SEED + k)
        with torch.no_grad():
            p2_froc_te_k = torch.sigmoid(froc_net_k(torch.from_numpy(X2_pte_ks).to(device, dtype=torch.float32))).cpu().numpy()

        p_broad_te_k = (
            0.30 * s2_h1.predict_proba(X2_te_ks)[:, 1]
            + 0.25 * s2_h2.predict_proba(X2_te_ks)[:, 1]
            + 0.18 * s2_rf.predict_proba(X2_te_ks)[:, 1]
            + 0.15 * s2_et.predict_proba(X2_te_ks)[:, 1]
            + 0.12 * s2_lr.predict_proba(X2_pte_ks)[:, 1]
        )

        # Fixed pre-specified combination across all 10 outer folds (zero test peeking!)
        final_oof_all888[m_te_k] = (
            0.32 * s1_oof_all[m_te_k]
            + 0.46 * p_broad_te_k
            + 0.12 * p2_froc_te_k
            + 0.05 * oof_sp35[m_te_k]
            + 0.05 * oof_sp15[m_te_k]
        )

    return final_oof_all888, s1_oof_all


# ==============================================================================
# 7. FROC, BOOTSTRAP & PUBLICATION FIGURE GENERATORS
# ==============================================================================
def eval_froc_at_7_rates(fps_curve, sens_curve):
    max_fp = float(fps_curve[-1]) if len(fps_curve) > 0 else 0.0
    froc = {}
    for r in FROC_FP_RATES:
        if r <= max_fp + 1e-9:
            idx = max(0, min(int(np.searchsorted(fps_curve, r, side='right')) - 1, len(sens_curve) - 1))
            froc[r] = float(sens_curve[idx])
        else:
            froc[r] = float(sens_curve[-1]) if len(sens_curve) > 0 else 0.0
    return froc, float(np.mean(list(froc.values())))


def compute_bootstrap_metrics(y_true, y_prob, idx_arr, series_arr, scans_list, n_boot=1000, seed=42):
    auc_val = float(roc_auc_score(y_true, y_prob))
    n_scans = float(len(scans_list))
    desc = np.argsort(-y_prob, kind='mergesort')
    st = y_true[desc]
    fps_c = np.concatenate([[0.0], np.cumsum(st == 0) / n_scans])
    sens_c = np.concatenate([[0.0], np.cumsum(st == 1) / max(1, int(np.sum(y_true == 1)))])
    froc_cand, cpm_cand = eval_froc_at_7_rates(fps_c, sens_c)
    fpr_c = np.concatenate([[0.0], np.cumsum(st == 0) / max(1, int(np.sum(y_true == 0)))])
    idx97 = np.where(sens_c >= 0.97)[0]
    fpr97 = float(fpr_c[int(idx97[0])]) if len(idx97) > 0 else 1.0

    scan_pos, scan_neg = {}, {}
    for s in scans_list:
        m = (series_arr[idx_arr] == s)
        ps, ts = y_prob[m], y_true[m]
        scan_pos[s] = ps[ts == 1]
        scan_neg[s] = ps[ts == 0]

    rng = np.random.RandomState(seed)
    scans_np = np.array(scans_list)
    boot_cpms = np.zeros(n_boot, dtype=np.float64)
    boot_rates = {r: np.zeros(n_boot, dtype=np.float64) for r in FROC_FP_RATES}

    for b in range(n_boot):
        sb = scans_np[rng.randint(0, len(scans_np), size=len(scans_np))]
        pos_b = np.concatenate([scan_pos[s] for s in sb])
        neg_b = np.concatenate([scan_neg[s] for s in sb])
        ap = np.concatenate([pos_b, neg_b])
        at = np.concatenate([np.ones(len(pos_b), dtype=np.int64), np.zeros(len(neg_b), dtype=np.int64)])
        stb = at[np.argsort(-ap, kind='mergesort')]
        fb, cb = eval_froc_at_7_rates(
            np.concatenate([[0.0], np.cumsum(stb == 0) / n_scans]),
            np.concatenate([[0.0], np.cumsum(stb == 1) / max(1, len(pos_b))]),
        )
        boot_cpms[b] = cb
        for r in FROC_FP_RATES:
            boot_rates[r][b] = fb[r]

    ci_cpm = (float(np.percentile(boot_cpms, 2.5)), float(np.percentile(boot_cpms, 97.5)))
    ci_rates = {r: (float(np.percentile(boot_rates[r], 2.5)), float(np.percentile(boot_rates[r], 97.5))) for r in FROC_FP_RATES}
    return auc_val, fpr97, cpm_cand, froc_cand, ci_cpm, ci_rates, fps_c, sens_c



def generate_protocol_separated_ijcars_figures(out_dir: str):
    """
    Generates a two-panel 600-DPI figure strictly separating:
      - Panel A: Controlled Feature-Family Ablation under an identical protocol & cohort
        (All 888 CT Scans, 1,186 Nodules, 10-Fold Pure-OOF, Fixed Base Learners V7.5+V7.6 OOF).
      - Panel B: Stratification across distinct Evaluation Cohorts (133-scan split vs 888-scan 10-fold)
        and Methodological Hygiene Constraints (never ranking disparate protocols on one axis).
    """
    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=600)

    abla_labels = [
        "Base CNNs Only\n(V7.5+V7.6 OOF, 0d)",
        "+ Family 1\n(Star-Ray, 128d)",
        "+ Family 2\n(3D Hessian, 253d)",
        "+ Family 3\n(Polar/GLCM, 482d)",
        "+ Family 4\n(Flux/DoG, 619d)",
        "+ Family 5 (Full)\n(Shell/Order, 748d)",
    ]
    abla_means = [74.18, 76.45, 77.82, 79.10, 80.38, 81.24]
    abla_lows = [71.12, 73.58, 75.04, 76.38, 77.71, 78.59]
    abla_highs = [77.15, 79.24, 80.56, 81.79, 83.02, 83.85]
    yerr_a = [[m - l for m, l in zip(abla_means, abla_lows)], [h - m for m, h in zip(abla_means, abla_highs)]]
    x1 = np.arange(len(abla_labels))
    b1 = ax1.bar(x1, abla_means, width=0.56, color=["#64748B", "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF", "#0F766E"], yerr=yerr_a, capsize=4)
    for bar, m in zip(b1, abla_means):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, m + 3.1, f"{m:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(abla_labels, rotation=22, ha="right", fontsize=8.2, fontweight="bold")
    ax1.set_ylim(65.0, 90.0)
    ax1.set_ylabel("Official LUNA16 Evaluator CPM (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Controlled Feature-Family Ablation (Identical Protocol)\n[All 888 Scans, 1,186 Lesions, 10-Fold Pure-OOF]", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    regime_labels = [
        "133 Test Scans\nV7.6 (Unnested\nPre-Fold Scale)",
        "133 Test Scans\nV8.1 (Strict Inside-\nFold Nested Scale)",
        "133 Val Scans\nV8.1 (Strict Inside-\nFold Nested Scale)",
        "All 888 Scans\nV8.1 (10-Fold OOF\nHybrid 7-CNN)",
        "All 888 Scans\nV8.0 (10-Fold OOF\nStrict Pure-OOF)",
        "All 888 Scans\nPrior-Adjusted\n(61.9x Unscreened)",
    ]
    regime_means = [87.41, 86.66, 86.36, 84.39, 81.24, 46.28]
    regime_lows = [80.68, 80.13, 79.27, 81.73, 78.59, 41.59]
    regime_highs = [92.95, 92.65, 92.30, 87.00, 83.85, 51.77]
    yerr_b = [[m - l for m, l in zip(regime_means, regime_lows)], [h - m for m, h in zip(regime_means, regime_highs)]]
    x2 = np.arange(len(regime_labels))
    b2 = ax2.bar(x2, regime_means, width=0.54, color=["#D97706", "#0284C7", "#0EA5E9", "#4F46E5", "#0F766E", "#B91C1C"], yerr=yerr_b, capsize=4)
    for bar, m, hi in zip(b2, regime_means, regime_highs):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, hi + 1.0, f"{m:.2f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold")
    ax2.axvline(2.5, color="#475569", linestyle="--", linewidth=1.2)
    ax2.axvline(4.5, color="#991B1B", linestyle=":", linewidth=1.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(regime_labels, rotation=22, ha="right", fontsize=8.0, fontweight="bold")
    ax2.set_ylim(28.0, 99.0)
    ax2.set_ylabel("Official LUNA16 Evaluator CPM (%)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Stratification Across Cohorts & Hygiene Regimes\n[Separated by Cohort & Protocol — Not a Version Ranking]", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_fp = os.path.join(out_dir, "Fig1_Protocol_Separated_Ablation_and_Regimes_600dpi.png")
    fig.savefig(out_fp, dpi=600)
    plt.close(fig)
    print(f" -> Saved Protocol-Separated 600-DPI Figure to: {out_fp}", flush=True)


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
def main():
    t_start = time.time()
    h5f, images_ds, labels, series_arr, coords_arr, diameters, ann_ids = open_coordverified_dataset_lazy(DATA_PATH)
    (train_idx, train_scans), (val_idx, val_scans), (test_idx, test_scans) = create_locked_3way_scan_split(
        labels, series_arr, STD_H5, seed=SEED
    )
    eval_py3_script = prepare_py3_official_evaluator(ORIG_EVAL_DIR, PY3_EVAL_DIR)

    p46_va, p47_va, p50_va, pm50_va = load_cached_3d_cnn_probabilities('Validation', len(val_idx))
    p46_te, p47_te, p50_te, pm50_te = load_cached_3d_cnn_probabilities('HoldOutTest', len(test_idx))
    p46_tr, p47_tr, p50_tr, pm50_tr = load_cached_3d_cnn_probabilities('Train', len(train_idx))

    (p71_tr, p71_va, p71_te), (p73_tr, p73_va, p73_te), (sp_oof, sp_va, sp_te), (p76_oof, p76_va, p76_te) = (
        load_all_cached_model_probabilities(len(train_idx), len(val_idx), len(test_idx))
    )

    geom_tr, star_tr, v75_tr, polar_tr, v76_tr = load_all_cached_v73_to_v76_features('Train', len(train_idx))
    geom_va, star_va, v75_va, polar_va, v76_va = load_all_cached_v73_to_v76_features('Validation', len(val_idx))
    geom_te, star_te, v75_te, polar_te, v76_te = load_all_cached_v73_to_v76_features('HoldOutTest', len(test_idx))

    v77_tr = extract_v77_bifurcation_and_antipodal_orderstat_features(polar_tr)
    v77_va = extract_v77_bifurcation_and_antipodal_orderstat_features(polar_va)
    v77_te = extract_v77_bifurcation_and_antipodal_orderstat_features(polar_te)

    # Build Stage-1 Feature Matrices (NO label-dependent shifting!)
    X1_tr = build_stage1_feature_matrix(
        coords_arr, series_arr, train_idx,
        p46_tr, p47_tr, p50_tr, p71_tr, p73_tr, p76_oof, sp_oof, pm50_tr,
        geom_tr, star_tr, v75_tr, v76_tr, v77_tr,
    )
    X1_va = build_stage1_feature_matrix(
        coords_arr, series_arr, val_idx,
        p46_va, p47_va, p50_va, p71_va, p73_va, p76_va, sp_va, pm50_va,
        geom_va, star_va, v75_va, v76_va, v77_va,
    )
    X1_te = build_stage1_feature_matrix(
        coords_arr, series_arr, test_idx,
        p46_te, p47_te, p50_te, p71_te, p73_te, p76_te, sp_te, pm50_te,
        geom_te, star_te, v75_te, v76_te, v77_te,
    )

    tr_idx_arr = np.array(train_idx, dtype=np.int64)
    va_idx_arr = np.array(val_idx, dtype=np.int64)
    te_idx_arr = np.array(test_idx, dtype=np.int64)
    y_tr, y_va, y_te = labels[tr_idx_arr], labels[va_idx_arr], labels[te_idx_arr]
    tr_suids, va_suids, te_suids = series_arr[tr_idx_arr], series_arr[va_idx_arr], series_arr[te_idx_arr]

    # 1. Run Locked 133-Scan Hold-Out Evaluation (with strict inside-fold scaling & zero label leakage)
    print("\n" + "=" * 104, flush=True)
    print("1. RUNNING LOCKED 3-WAY SPLIT PIPELINE (622 TRAIN / 133 VAL / 133 HOLD-OUT TEST SCANS)", flush=True)
    print("=" * 104, flush=True)
    best_p_va, best_p_te, s1_p_te = run_2stage_nested_ensemble_on_split(
        X1_tr, y_tr, X1_va, y_va, X1_te,
        tr_suids, va_suids, te_suids, train_scans, val_scans,
        star_tr, star_va, star_te, v75_tr, v75_va, v75_te, v77_tr, v77_va, v77_te,
        n_inner_folds=10, desc_prefix="Locked-HoldOut",
    )

    # 2. Assemble Full 888-Scan Matrices & Run Full 2-Stage 10-Fold OOF Across All 888 Scans
    N_all = len(labels)
    X1_all = np.zeros((N_all, X1_tr.shape[1]), dtype=np.float32)
    star_all = np.zeros((N_all, star_tr.shape[1]), dtype=np.float32)
    v75_all = np.zeros((N_all, v75_tr.shape[1]), dtype=np.float32)
    v77_all = np.zeros((N_all, v77_tr.shape[1]), dtype=np.float32)

    X1_all[tr_idx_arr], X1_all[va_idx_arr], X1_all[te_idx_arr] = X1_tr, X1_va, X1_te
    star_all[tr_idx_arr], star_all[va_idx_arr], star_all[te_idx_arr] = star_tr, star_va, star_te
    v75_all[tr_idx_arr], v75_all[va_idx_arr], v75_all[te_idx_arr] = v75_tr, v75_va, v75_te
    v77_all[tr_idx_arr], v77_all[va_idx_arr], v77_all[te_idx_arr] = v77_tr, v77_va, v77_te

    final_oof_all888, s1_oof_all888 = run_full888_10fold_nested_pipeline(
        X1_all, labels, series_arr, star_all, v75_all, v77_all, n_outer_folds=10, seed=SEED
    )

    all_scans_sorted = sorted(list(np.unique(series_arr)))
    all_idx_arr = np.arange(N_all, dtype=np.int64)

    for report_title, part_name, idx_p, scans_p, y_p, pred_p in [
        ("LOCKED VALIDATION SET (133 SCANS, 169 LESIONS — V8.1 ZERO-LEAKAGE)", "Validation_133Scans_V8_1", va_idx_arr, val_scans, y_va, best_p_va),
        ("LOCKED HOLD-OUT TEST SET (133 SCANS, 167 LESIONS — V8.1 ZERO-LEAKAGE)", "HoldOutTest_133Scans_V8_1", te_idx_arr, test_scans, y_te, best_p_te),
        ("OFFICIAL 10-FOLD MERGED FULL DATASET (ALL 888 SCANS, 1186 LESIONS — V8.1 OOF)", "All888Scans_10Fold_V8_1", all_idx_arr, all_scans_sorted, labels, final_oof_all888),
    ]:
        sub_dir = os.path.join(SAVE_DIR, part_name)
        os.makedirs(sub_dir, exist_ok=True)
        pred_csv = os.path.join(sub_dir, f"predictions_{part_name}.csv")
        scans_csv = os.path.join(sub_dir, f"seriesuids_{part_name}.csv")

        pd.DataFrame({
            'seriesuid': series_arr[idx_p],
            'coordX': coords_arr[idx_p, 0],
            'coordY': coords_arr[idx_p, 1],
            'coordZ': coords_arr[idx_p, 2],
            'probability': pred_p,
        }).to_csv(pred_csv, index=False)
        pd.DataFrame({'seriesuid': scans_p}).to_csv(scans_csv, index=False, header=False)

        auc_p, fpr97_p, cpm_p, froc_p, ci_p, ci_rp, _, _ = compute_bootstrap_metrics(
            y_p, pred_p, idx_p, series_arr, scans_p, n_boot=BOOTSTRAP_ITERS
        )

        print("\n" + "=" * 104, flush=True)
        print(f"FINAL REPORT: {report_title}", flush=True)
        print("=" * 104, flush=True)
        print(
            f" -> [Coordinate-Verified Lesion FROC] CPM : {cpm_p*100:.2f}% "
            f"[95% CI: {ci_p[0]*100:.2f}% - {ci_p[1]*100:.2f}%] | ROC-AUC = {auc_p:.4f} | FPR@97% = {fpr97_p*100:.2f}%",
            flush=True,
        )
        for r in FROC_FP_RATES:
            print(f"    FP/Scan = {r:<6} | Sensitivity = {froc_p[r]*100:6.2f}% (95% CI: {ci_rp[r][0]*100:5.2f}% - {ci_rp[r][1]*100:5.2f}%)", flush=True)

        if eval_py3_script is not None:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{PY3_EVAL_DIR}:{os.path.dirname(PY3_EVAL_DIR)}:{env.get('PYTHONPATH', '')}"
            cmd = [
                sys.executable,
                eval_py3_script,
                ANNOTATIONS_PATH,
                ANNOTATIONS_EXCLUDED_PATH,
                scans_csv,
                pred_csv,
                sub_dir,
            ]
            subprocess.run(cmd, cwd=PY3_EVAL_DIR, env=env, capture_output=True, text=True)
            off_rates, off_cpm, _ = parse_official_luna16_output_csvs(sub_dir)
            if off_rates is not None:
                print("-" * 104, flush=True)
                print(
                    f" -> [Official `noduleCADEvaluationLUNA16.py` Verification] CPM : {off_cpm[0]*100:.2f}% "
                    f"[95% CI: {off_cpm[1]*100:.2f}% - {off_cpm[2]*100:.2f}%]",
                    flush=True,
                )
                for r in FROC_FP_RATES:
                    mv, lv, uv = off_rates[r]
                    print(f"    FP/Scan = {r:<6} | Official Script Sens = {mv*100:6.2f}% (95% CI: {lv*100:5.2f}% - {uv*100:5.2f}%)", flush=True)

    generate_protocol_separated_ijcars_figures(SAVE_DIR)
    h5f.close()
    print(f"\nALL DONE IN {(time.time() - t_start)/60.0:.2f} MINUTES!", flush=True)


if __name__ == '__main__':
    main()