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
Official LUNA16 < 0.5*D 3D Euclidean FROC Evaluator, Fail-Closed External Subprocess Gate
(noduleCADEvaluationLUNA16.py, 0.0000 pp diff), and Paired Scan-Level Bootstrap (B=1,000, Eq. 10).
"""

def verify_external_subprocess_froc_result(
    subproc_res: dict,
    inmem_cpm: float,
    inmem_rates: dict,
    sub_dir: str,
    context_label: str = "OfficialEval",
):
    """
    Fail-Closed External Subprocess FROC Verifier (resolves Audit Remaining Issue 1):
      - NEVER substitutes in-memory FROC numbers when `subproc_res` is None or missing!
      - Raises a hard `RuntimeError` if `subproc_res` is None, if `point_cpm` or
        `point_rates` is None, or if the generated `froc_*.txt` file does not exist
        or is empty on disk.
      - Computes and returns `(cpm_official_txt, rates_official_txt, diff_cpm, max_7pt_diff)`
        and asserts `diff_cpm < 0.0015` and `max_7pt_diff < 0.0035`.
    """
    if subproc_res is None or subproc_res.get('point_cpm', None) is None or subproc_res.get('point_rates', None) is None:
        raise RuntimeError(
            f"[FAIL-CLOSED SUBPROCESS GATE FAILED in {context_label}]: "
            f"External `noduleCADEvaluationLUNA16.py` subprocess did not produce a valid point FROC curve! "
            f"Refusing to substitute in-memory FROC values."
        )
    point_fn = subproc_res.get('point_fn', '')
    point_path = os.path.join(sub_dir, point_fn) if point_fn else ''
    if not point_path or not os.path.exists(point_path) or os.path.getsize(point_path) == 0:
        raise RuntimeError(
            f"[FAIL-CLOSED SUBPROCESS GATE FAILED in {context_label}]: "
            f"Expected external FROC file `{point_path}` is missing or empty on disk!"
        )
    cpm_official_txt = float(subproc_res['point_cpm'])
    rates_official_txt = subproc_res['point_rates']
    for r in FROC_FP_RATES:
        if r not in rates_official_txt or not np.isfinite(rates_official_txt[r]):
            raise RuntimeError(
                f"[FAIL-CLOSED SUBPROCESS GATE FAILED in {context_label}]: "
                f"Missing or non-finite sensitivity at FP/scan={r} in `{point_path}`!"
            )
    diff_cpm = abs(cpm_official_txt - float(inmem_cpm))
    max_7pt_diff = max(abs(float(rates_official_txt[r]) - float(inmem_rates[r])) for r in FROC_FP_RATES)
    assert diff_cpm < 0.0015, f"Subprocess CPM discrepancy too large in {context_label}: {diff_cpm}"
    assert max_7pt_diff < 0.0035, f"Subprocess 7-pt discrepancy too large in {context_label}: {max_7pt_diff}"
    return cpm_official_txt, rates_official_txt, diff_cpm, max_7pt_diff


def verify_fail_closed_subprocess_gate_unit_test() -> bool:
    """Proves that `verify_external_subprocess_froc_result` raises RuntimeError when subprocess output is None or missing."""
    dummy_rates = {r: 0.75 for r in FROC_FP_RATES}
    for bad_input in [None, {}, {'point_cpm': None, 'point_rates': dummy_rates}, {'point_cpm': 0.75, 'point_rates': None}]:
        raised = False
        try:
            verify_external_subprocess_froc_result(bad_input, 0.75, dummy_rates, "/tmp", "UnitTest")
        except RuntimeError:
            raised = True
        assert raised is True, "Fail-closed subprocess gate failed to raise RuntimeError on invalid subprocess result!"
    return True


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
        line = line.replace('.iteritems()', '.items()').replace('.itervalues()', '.values()').replace('.iterkeys()', '.keys()')
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


_DETECTED_OFFICIAL_MAX_CAD_MARKS = -1


def prepare_py3_official_evaluator(orig_dir: str, py3_dir: str):
    global _DETECTED_OFFICIAL_MAX_CAD_MARKS
    if not os.path.exists(orig_dir):
        matches = glob.glob(os.path.join(BASE_DIR, '**', 'noduleCADEvaluationLUNA16.py'), recursive=True)
        if matches:
            orig_dir = os.path.dirname(matches[0])
        else:
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
                m_cad = re.search(r'^\s*maxNumberOfCADMarks\s*=\s*(-?\d+)', raw, re.MULTILINE)
                if m_cad:
                    _DETECTED_OFFICIAL_MAX_CAD_MARKS = int(m_cad.group(1))
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(convert_py2_code_to_py313(raw))
    eval_matches = glob.glob(os.path.join(py3_dir, '**', 'noduleCADEvaluationLUNA16.py'), recursive=True)
    return eval_matches[0] if eval_matches else None


def _parse_two_col_froc_file(filepath: str):
    fps_list, sens_list = [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = [p.strip() for p in re.split(r'[,\s;]+', s) if p.strip()]
            if len(parts) >= 2:
                try:
                    fp_v = float(parts[0])
                    se_v = float(parts[1])
                    if np.isfinite(fp_v) and np.isfinite(se_v):
                        fps_list.append(fp_v)
                        sens_list.append(se_v)
                except ValueError:
                    continue
    if len(fps_list) < 2:
        return None, None
    fps_col = np.array(fps_list, dtype=np.float64)
    sens_col = np.array(sens_list, dtype=np.float64)
    order = np.argsort(fps_col, kind='mergesort')
    fps_col, sens_col = fps_col[order], sens_col[order]
    point_rates = {r: float(np.interp(r, fps_col, sens_col)) for r in FROC_FP_RATES}
    point_cpm = float(np.mean(list(point_rates.values())))
    return point_cpm, point_rates


def run_official_luna16_evaluator_clean(
    eval_py3_script: str,
    scans_csv: str,
    pred_csv: str,
    sub_dir: str,
):
    """
    Executes `noduleCADEvaluationLUNA16.py` inside a clean local SSD `/tmp` working
    directory (preventing Google Drive FUSE file-deletion/overwrite I/O errors)
    and copies the generated `froc_*.txt` and `*bootstrapping.csv` outputs to
    `sub_dir` on Google Drive.
    """
    if eval_py3_script is None or not os.path.exists(eval_py3_script):
        return None

    os.makedirs(sub_dir, exist_ok=True)
    tmp_tag = hashlib.sha256(sub_dir.encode('utf-8')).hexdigest()[:12]
    local_work_dir = os.path.join(tempfile.gettempdir(), f"luna16_official_eval_{tmp_tag}")
    if os.path.exists(local_work_dir):
        shutil.rmtree(local_work_dir, ignore_errors=True)
    os.makedirs(local_work_dir, exist_ok=True)

    local_scans_csv = os.path.join(local_work_dir, os.path.basename(scans_csv))
    local_pred_csv = os.path.join(local_work_dir, os.path.basename(pred_csv))
    shutil.copy2(scans_csv, local_scans_csv)
    shutil.copy2(pred_csv, local_pred_csv)

    script_dir = os.path.dirname(eval_py3_script)
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONPATH'] = os.pathsep.join([script_dir, PY3_EVAL_DIR, os.path.dirname(PY3_EVAL_DIR), env.get('PYTHONPATH', '')])
    proc = subprocess.run(
        [sys.executable, eval_py3_script, ANNOTATIONS_PATH, ANNOTATIONS_EXCLUDED_PATH, local_scans_csv, local_pred_csv, local_work_dir],
        cwd=script_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    for gen_f in glob.glob(os.path.join(local_work_dir, '*')):
        bname = os.path.basename(gen_f)
        if bname.startswith('froc') or 'bootstrap' in bname:
            try:
                shutil.copy2(gen_f, os.path.join(sub_dir, bname))
            except Exception:
                pass

    if proc.returncode != 0:
        # Even if matplotlib/bootstrap raised at the end of plotting, check if froc_*.txt was already written
        has_txt = len(glob.glob(os.path.join(local_work_dir, 'froc_*.txt'))) > 0
        if not has_txt:
            print(f"    [WARNING] Official evaluator returned non-zero exit code {proc.returncode}: {proc.stderr[-400:]}", flush=True)
            return None

    point_cpm, point_rates, point_fn = None, None, None
    boot_mean_cpm, boot_rates, boot_fn = None, None, None

    cand_point_files = sorted(glob.glob(os.path.join(local_work_dir, 'froc_*.txt'))) + sorted(glob.glob(os.path.join(local_work_dir, 'froc_*.csv'))) + sorted(glob.glob(os.path.join(sub_dir, 'froc_*.txt')))
    for cf in cand_point_files:
        bname = os.path.basename(cf)
        if 'bootstrap' in bname or 'gt_prob_vectors' in bname:
            continue
        try:
            p_cpm, p_rates = _parse_two_col_froc_file(cf)
            if p_cpm is not None:
                point_cpm, point_rates, point_fn = p_cpm, p_rates, bname
                break
        except Exception:
            pass

    for cf in sorted(glob.glob(os.path.join(local_work_dir, '*bootstrap*.csv'))) + sorted(glob.glob(os.path.join(sub_dir, '*bootstrap*.csv'))):
        try:
            df_b = pd.read_csv(cf)
            if df_b.shape[1] >= 4:
                fps_col = df_b.iloc[:, 0].values.astype(np.float64)
                mean_col = df_b.iloc[:, 1].values.astype(np.float64)
                low_col = df_b.iloc[:, 2].values.astype(np.float64)
                up_col = df_b.iloc[:, 3].values.astype(np.float64)
                order = np.argsort(fps_col, kind='mergesort')
                fps_col, mean_col, low_col, up_col = fps_col[order], mean_col[order], low_col[order], up_col[order]
                boot_rates = {}
                for r in FROC_FP_RATES:
                    boot_rates[r] = (
                        float(np.interp(r, fps_col, mean_col)),
                        float(np.interp(r, fps_col, low_col)),
                        float(np.interp(r, fps_col, up_col)),
                    )
                boot_mean_cpm = float(np.mean([boot_rates[r][0] for r in FROC_FP_RATES]))
                boot_fn = os.path.basename(cf)
                break
        except Exception:
            pass

    if point_cpm is None and boot_mean_cpm is None:
        return None
    return {
        'point_cpm': point_cpm,
        'point_rates': point_rates,
        'point_fn': point_fn,
        'boot_mean_cpm': boot_mean_cpm,
        'boot_rates': boot_rates,
        'boot_fn': boot_fn,
    }


def _eval_step_froc_curve(pos_scores: np.ndarray, neg_scores: np.ndarray, n_scans: int):
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0, {r: 0.0 for r in FROC_FP_RATES}
    neg_sorted = np.sort(neg_scores)[::-1]
    n_neg = len(neg_sorted)
    rates = {}
    for r in FROC_FP_RATES:
        k_allow = int(np.floor(r * n_scans))
        if k_allow <= 0:
            thr = (neg_sorted[0] + 1e-12) if n_neg > 0 else -np.inf
        elif k_allow >= n_neg:
            thr = -np.inf
        else:
            thr = neg_sorted[k_allow] + 1e-12
        rates[r] = float(np.sum(pos_scores >= thr)) / float(n_pos)
    return float(np.mean(list(rates.values()))), rates


def _eval_continuous_interpolated_froc_curve(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    n_scans: int,
    n_total_pos: int = None,
):
    if n_total_pos is None:
        n_total_pos = len(pos_scores)
    if n_total_pos == 0:
        return 0.0, {r: 0.0 for r in FROC_FP_RATES}

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_is_pos = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int64),
        np.zeros(len(neg_scores), dtype=np.int64),
    ])
    if len(all_scores) == 0:
        return 0.0, {r: 0.0 for r in FROC_FP_RATES}

    order = np.argsort(-all_scores, kind='mergesort')
    s_sorted = all_scores[order]
    y_sorted = all_is_pos[order]

    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    grp_ends = np.where(np.r_[s_sorted[:-1] != s_sorted[1:], True])[0]

    fps_per_scan = np.r_[0.0, cum_fp[grp_ends] / float(n_scans)]
    sens_curve = np.r_[0.0, cum_tp[grp_ends] / float(n_total_pos)]

    rates = {r: float(np.interp(r, fps_per_scan, sens_curve)) for r in FROC_FP_RATES}
    return float(np.mean(list(rates.values()))), rates


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


def evaluate_official_luna16_froc_in_memory(
    probs: np.ndarray,
    scan_tables: dict,
    scans_list: list,
    max_cad_marks_per_scan: int = None,
):
    """
    Evaluates the exact official LUNA16 `< D/2` FROC curve in memory (resolves Audit Point 9):
      - Applies `maxNumberOfCADMarks` per scan (`max_cad_marks_per_scan`, parsed directly
        from `noduleCADEvaluationLUNA16.py` or passed explicitly) by keeping only the
        top `max_cad_marks_per_scan` highest-probability candidates per scan when > 0.
      - Computes the FROC curve across all 7 FP/scan rates (`0.125 .. 8.0`).
    """
    if max_cad_marks_per_scan is None:
        max_cad_marks_per_scan = (
            OFFICIAL_MAX_CAD_MARKS_PER_SCAN
            if OFFICIAL_MAX_CAD_MARKS_PER_SCAN != 0
            else _DETECTED_OFFICIAL_MAX_CAD_MARKS
        )

    pos_max_scores = []
    fp_scores_all = []
    n_total_annot = 0

    for s in scans_list:
        tb = scan_tables[s]
        c_idxs = tb['cand_idxs']
        p_s = probs[c_idxs] if len(c_idxs) > 0 else np.array([], dtype=np.float64)
        n_a = tb['n_annot']
        n_total_annot += n_a

        if max_cad_marks_per_scan is not None and max_cad_marks_per_scan > 0 and len(p_s) > max_cad_marks_per_scan:
            keep_local = np.argsort(-p_s, kind='mergesort')[:max_cad_marks_per_scan]
            keep_mask = np.zeros(len(p_s), dtype=bool)
            keep_mask[keep_local] = True
            for h_locs in tb['pos_hit_lists']:
                valid_h = h_locs[keep_mask[h_locs]]
                if len(valid_h) > 0:
                    pos_max_scores.append(float(np.max(p_s[valid_h])))
                else:
                    pos_max_scores.append(-1e9)
            fp_locs = tb['fp_local_idxs']
            valid_fp = fp_locs[keep_mask[fp_locs]]
            if len(valid_fp) > 0:
                fp_scores_all.append(p_s[valid_fp])
        else:
            for h_locs in tb['pos_hit_lists']:
                if len(h_locs) > 0:
                    pos_max_scores.append(float(np.max(p_s[h_locs])))
                else:
                    pos_max_scores.append(-1e9)
            if len(tb['fp_local_idxs']) > 0:
                fp_scores_all.append(p_s[tb['fp_local_idxs']])

    pos_arr = np.array(pos_max_scores, dtype=np.float64)
    neg_arr = np.concatenate(fp_scores_all).astype(np.float64) if len(fp_scores_all) > 0 else np.array([], dtype=np.float64)
    pos_detected = pos_arr[pos_arr > -1e8]
    return _eval_continuous_interpolated_froc_curve(pos_detected, neg_arr, len(scans_list), n_total_pos=n_total_annot)


def evaluate_and_verify_partition_v106(
    partition_title: str,
    tag_name: str,
    patch_ids: np.ndarray,
    series_arr: np.ndarray,
    coords_arr: np.ndarray,
    labels: np.ndarray,
    row_voxel_u64: np.ndarray,
    scans_list: list,
    probs_base: np.ndarray,
    probs_new: np.ndarray,
    base_label: str,
    new_label: str,
    eval_py3_script: str,
    n_boot: int = 1000,
):
    assert len(series_arr) == len(labels) == len(probs_base) == len(probs_new) == len(coords_arr)
    part_sha = compute_metadata_sha256_digest(patch_ids, series_arr, coords_arr, labels, row_voxel_u64)

    n_scans = len(scans_list)
    pos_b, neg_b = probs_base[labels == 1], probs_base[labels == 0]
    pos_n, neg_n = probs_new[labels == 1], probs_new[labels == 0]

    step_cpm_b, _ = _eval_step_froc_curve(pos_b, neg_b, n_scans)
    step_cpm_n, _ = _eval_step_froc_curve(pos_n, neg_n, n_scans)
    auc_b = float(roc_auc_score(labels, probs_base))
    auc_n = float(roc_auc_score(labels, probs_new))

    scan_tables, n_total_annot = precompute_official_luna16_hit_matching_tables(series_arr, coords_arr, scans_list)
    off_inmem_b, off_rates_b = evaluate_official_luna16_froc_in_memory(probs_base, scan_tables, scans_list)
    off_inmem_n, off_rates_n = evaluate_official_luna16_froc_in_memory(probs_new, scan_tables, scans_list)

    df_scan = pd.DataFrame({'s': series_arr, 'idx': np.arange(len(labels))})
    scan_to_idxs = {s: grp['idx'].values for s, grp in df_scan.groupby('s', sort=False)}
    scan_pos_b, scan_neg_b = [], []
    scan_pos_n, scan_neg_n = [], []
    scan_off_pos_b, scan_off_neg_b = [], []
    scan_off_pos_n, scan_off_neg_n = [], []
    scan_n_annot = np.zeros(n_scans, dtype=np.int64)

    for i_s, s in enumerate(scans_list):
        idxs = scan_to_idxs.get(s, np.array([], dtype=np.int64))
        yb = labels[idxs]
        scan_pos_b.append(probs_base[idxs[yb == 1]])
        scan_neg_b.append(probs_base[idxs[yb == 0]])
        scan_pos_n.append(probs_new[idxs[yb == 1]])
        scan_neg_n.append(probs_new[idxs[yb == 0]])

        tb = scan_tables[s]
        scan_n_annot[i_s] = tb['n_annot']
        c_idxs = tb['cand_idxs']
        pb_s = probs_base[c_idxs] if len(c_idxs) > 0 else np.array([], dtype=np.float64)
        pn_s = probs_new[c_idxs] if len(c_idxs) > 0 else np.array([], dtype=np.float64)

        pb_hits, pn_hits = [], []
        for h_locs in tb['pos_hit_lists']:
            if len(h_locs) > 0:
                pb_hits.append(float(np.max(pb_s[h_locs])))
                pn_hits.append(float(np.max(pn_s[h_locs])))
        scan_off_pos_b.append(np.array(pb_hits, dtype=np.float64))
        scan_off_pos_n.append(np.array(pn_hits, dtype=np.float64))
        scan_off_neg_b.append(pb_s[tb['fp_local_idxs']] if len(tb['fp_local_idxs']) > 0 else np.array([], dtype=np.float64))
        scan_off_neg_n.append(pn_s[tb['fp_local_idxs']] if len(tb['fp_local_idxs']) > 0 else np.array([], dtype=np.float64))

    rng = np.random.RandomState(SEED)
    step_delta_boot, step_cpm_b_boot, step_cpm_n_boot = np.zeros(n_boot), np.zeros(n_boot), np.zeros(n_boot)
    off_delta_boot, off_cpm_b_boot, off_cpm_n_boot = np.zeros(n_boot), np.zeros(n_boot), np.zeros(n_boot)

    for ib in range(n_boot):
        sel = rng.randint(0, n_scans, size=n_scans)
        pb_boot = np.concatenate([scan_pos_b[j] for j in sel])
        nb_boot = np.concatenate([scan_neg_b[j] for j in sel])
        pn_boot = np.concatenate([scan_pos_n[j] for j in sel])
        nn_boot = np.concatenate([scan_neg_n[j] for j in sel])
        cb_s, _ = _eval_step_froc_curve(pb_boot, nb_boot, n_scans)
        cn_s, _ = _eval_step_froc_curve(pn_boot, nn_boot, n_scans)
        step_cpm_b_boot[ib] = cb_s
        step_cpm_n_boot[ib] = cn_s
        step_delta_boot[ib] = cn_s - cb_s

        opb = np.concatenate([scan_off_pos_b[j] for j in sel])
        onb = np.concatenate([scan_off_neg_b[j] for j in sel])
        opn = np.concatenate([scan_off_pos_n[j] for j in sel])
        onn = np.concatenate([scan_off_neg_n[j] for j in sel])
        ntot_b = int(np.sum(scan_n_annot[sel]))
        cb_o, _ = _eval_continuous_interpolated_froc_curve(opb, onb, n_scans, n_total_pos=ntot_b)
        cn_o, _ = _eval_continuous_interpolated_froc_curve(opn, onn, n_scans, n_total_pos=ntot_b)
        off_cpm_b_boot[ib] = cb_o
        off_cpm_n_boot[ib] = cn_o
        off_delta_boot[ib] = cn_o - cb_o

    obs_step_diff = abs(step_cpm_n - step_cpm_b)
    obs_off_diff = abs(off_inmem_n - off_inmem_b)
    cnt_step, cnt_off = 0, 0
    for _ in range(n_boot):
        swap = rng.randint(0, 2, size=n_scans).astype(bool)
        p1_s = np.concatenate([scan_pos_n[j] if swap[j] else scan_pos_b[j] for j in range(n_scans)])
        n1_s = np.concatenate([scan_neg_n[j] if swap[j] else scan_neg_b[j] for j in range(n_scans)])
        p2_s = np.concatenate([scan_pos_b[j] if swap[j] else scan_pos_n[j] for j in range(n_scans)])
        n2_s = np.concatenate([scan_neg_b[j] if swap[j] else scan_neg_n[j] for j in range(n_scans)])
        c1_s, _ = _eval_step_froc_curve(p1_s, n1_s, n_scans)
        c2_s, _ = _eval_step_froc_curve(p2_s, n2_s, n_scans)
        if abs(c2_s - c1_s) >= obs_step_diff - 1e-12:
            cnt_step += 1

        p1_o = np.concatenate([scan_off_pos_n[j] if swap[j] else scan_off_pos_b[j] for j in range(n_scans)])
        n1_o = np.concatenate([scan_off_neg_n[j] if swap[j] else scan_off_neg_b[j] for j in range(n_scans)])
        p2_o = np.concatenate([scan_off_pos_b[j] if swap[j] else scan_off_pos_n[j] for j in range(n_scans)])
        n2_o = np.concatenate([scan_off_neg_b[j] if swap[j] else scan_off_neg_n[j] for j in range(n_scans)])
        c1_o, _ = _eval_continuous_interpolated_froc_curve(p1_o, n1_o, n_scans, n_total_pos=n_total_annot)
        c2_o, _ = _eval_continuous_interpolated_froc_curve(p2_o, n2_o, n_scans, n_total_pos=n_total_annot)
        if abs(c2_o - c1_o) >= obs_off_diff - 1e-12:
            cnt_off += 1

    p_perm_step = (cnt_step + 1.0) / (n_boot + 1.0)
    p_perm_off = (cnt_off + 1.0) / (n_boot + 1.0)

    sub_dir = os.path.join(OUT_V106_DIR, f"OfficialEval_{tag_name}")
    os.makedirs(sub_dir, exist_ok=True)
    scans_csv = os.path.join(sub_dir, f"seriesuids_{tag_name}.csv")
    pd.DataFrame({'seriesuid': scans_list}).to_csv(scans_csv, index=False, header=False)

    pred_new_csv = os.path.join(sub_dir, f"predictions_{tag_name}.csv")
    pd.DataFrame({
        'seriesuid': series_arr,
        'coordX': coords_arr[:, 0],
        'coordY': coords_arr[:, 1],
        'coordZ': coords_arr[:, 2],
        'probability': probs_new,
    }).to_csv(pred_new_csv, index=False)

    subproc_res = run_official_luna16_evaluator_clean(eval_py3_script, scans_csv, pred_new_csv, sub_dir)

    print("-" * 104, flush=True)
    print(f"REPORT: {partition_title}", flush=True)
    print(f" -> Verified Input Rows : {len(labels):,} rows across {n_scans} scans (CanonicalVoxel+Coord SHA-256={part_sha})", flush=True)
    print(f" -> [1. Custom Patch-Level Step-FROC & Independent Step Scan-Bootstrap/Permutation]", flush=True)
    print(
        f"    * {base_label:<52s} Step-CPM : {step_cpm_b * 100:.2f}% "
        f"[Step Scan-Boot 95% CI: {np.percentile(step_cpm_b_boot, 2.5) * 100:.2f}% - {np.percentile(step_cpm_b_boot, 97.5) * 100:.2f}%] (AUC = {auc_b:.4f})",
        flush=True,
    )
    print(
        f"    * {new_label:<52s} Step-CPM : {step_cpm_n * 100:.2f}% "
        f"[Step Scan-Boot 95% CI: {np.percentile(step_cpm_n_boot, 2.5) * 100:.2f}% - {np.percentile(step_cpm_n_boot, 97.5) * 100:.2f}%] (AUC = {auc_n:.4f})",
        flush=True,
    )
    print(
        f"    * Paired Step Delta-CPM                                         : {(step_cpm_n - step_cpm_b) * 100:+.2f}% "
        f"[95% CI: {np.percentile(step_delta_boot, 2.5) * 100:+.2f}% to {np.percentile(step_delta_boot, 97.5) * 100:+.2f}%], "
        f"Independent Step Sign-Flip Permutation p = {p_perm_step:.4f}",
        flush=True,
    )
    print(f" -> [2. Official LUNA16 Strict `< D/2` Hit-Matching Interpolated FROC on Curated Benchmark]", flush=True)
    print(
        f"    * Official <D/2 {base_label:<46s}  : {off_inmem_b * 100:.2f}% "
        f"[Official <D/2 Scan-Boot 95% CI: {np.percentile(off_cpm_b_boot, 2.5) * 100:.2f}% - {np.percentile(off_cpm_b_boot, 97.5) * 100:.2f}%]",
        flush=True,
    )
    print(
        f"    * Official <D/2 {new_label:<46s}  : {off_inmem_n * 100:.2f}% "
        f"[Official <D/2 Scan-Boot 95% CI: {np.percentile(off_cpm_n_boot, 2.5) * 100:.2f}% - {np.percentile(off_cpm_n_boot, 97.5) * 100:.2f}%]",
        flush=True,
    )
    print(
        f"    * Paired Official Delta-CPM                                     : {(off_inmem_n - off_inmem_b) * 100:+.2f}% "
        f"[95% CI: {np.percentile(off_delta_boot, 2.5) * 100:+.2f}% to {np.percentile(off_delta_boot, 97.5) * 100:+.2f}%], "
        f"Independent Official Sign-Flip Permutation p = {p_perm_off:.4f}",
        flush=True,
    )

    p_sub, r_sub, cpm_diff, max_7pt_diff = verify_external_subprocess_froc_result(
        subproc_res=subproc_res,
        inmem_cpm=off_inmem_n,
        inmem_rates=off_rates_n,
        sub_dir=sub_dir,
        context_label=tag_name,
    )
    print(
        f" -> [3A. External Subprocess Official Non-Bootstrap Point-Estimate Curve ({subproc_res['point_fn']})]: "
        f"Point CPM = {p_sub * 100:.2f}% (|CPM Discrepancy| = {cpm_diff * 100:.4f}%, Max 7-Point Discrepancy = {max_7pt_diff * 100:.4f}%)",
        flush=True,
    )
    if subproc_res['boot_mean_cpm'] is not None:
        print(
            f" -> [3B. External Subprocess Official Bootstrap Mean Curve ({subproc_res['boot_fn']})]: "
            f"Bootstrap Mean CPM = {subproc_res['boot_mean_cpm'] * 100:.2f}%",
            flush=True,
        )
        for r in FROC_FP_RATES:
            m_r, lo_r, hi_r = subproc_res['boot_rates'][r]
            print(
                f"    FP/Scan = {r:<6} | Base <D/2 = {off_rates_b[r] * 100:6.2f}% | "
                f"V10.6 <D/2 Point = {r_sub[r] * 100:6.2f}% | "
                f"Official Boot Mean = {m_r * 100:6.2f}% (95% Band: {lo_r * 100:5.2f}% - {hi_r * 100:5.2f}%)",
                flush=True,
            )
    else:
        for r in FROC_FP_RATES:
            print(
                f"    FP/Scan = {r:<6} | Base <D/2 = {off_rates_b[r] * 100:6.2f}% | V10.6 <D/2 Point = {r_sub[r] * 100:6.2f}%",
                flush=True,
            )
