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
Complete Production Implementation of Stage III Hybrid Feature Fusion (D_hybrid = 1,195D,
D_spec = 506D, Eq. 8) and 7-Learner Out-of-Fold Stacking Ensemble (Section 2.3).
"""
from .detection_head import _apply_multi_quantile_scan_adaptive_froc_calibration

def _fit_xgb_gpu_or_cpu(X_fit, y_fit, sw_fit, max_depth, lr, colsample, n_est, reg_l, seed):
    params = dict(
        n_estimators=n_est,
        max_depth=max_depth,
        max_bin=256,
        learning_rate=lr,
        subsample=0.82,
        colsample_bytree=colsample,
        reg_lambda=reg_l,
        reg_alpha=1.2,
        min_child_weight=3.0,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_jobs=-1,
        random_state=seed,
    )
    try:
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)
        return clf
    except Exception:
        params['device'] = 'cpu'
        params['n_jobs'] = -1
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)
        return clf


def _fit_catboost_gpu_or_cpu(X_fit, y_fit, sw_fit, depth, lr, iterations, l2_leaf_reg, seed):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        cb = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2_leaf_reg,
            border_count=128,
            loss_function='Logloss',
            task_type='GPU' if torch.cuda.is_available() else 'CPU',
            gpu_ram_part=(0.38 if VRAM_MIB <= 9000 else 0.55) if torch.cuda.is_available() else None,
            random_seed=seed,
            verbose=False,
        )
        cb.fit(X_fit, y_fit, sample_weight=sw_fit)
        return cb
    except Exception:
        cb = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2_leaf_reg,
            loss_function='Logloss',
            task_type='CPU',
            thread_count=-1,
            random_seed=seed,
            verbose=False,
        )
        cb.fit(X_fit, y_fit, sample_weight=sw_fit)
        return cb


def run_v106_9family_coordconv_multitask_upgrade_on_split(
    X_9fam_tr: np.ndarray,
    X_spec9_tr: np.ndarray,
    y_tr: np.ndarray,
    diam_tr: np.ndarray,
    suids_tr: np.ndarray,
    train_scans_list: list,
    X_9fam_eval_list: list,
    X_spec9_eval_list: list,
    suids_eval_list: list,
    veto_tr: np.ndarray,
    veto_eval_list: list,
    star_core_tr: np.ndarray,
    star_core_eval_list: list,
    v75_sph_tr: np.ndarray,
    v75_sph_eval_list: list,
    v77_pi_tr: np.ndarray,
    v77_pi_eval_list: list,
    v105_pure_eval_list: list,
    v105_e2e_eval_list: list,
    idx_global_tr: np.ndarray,
    idx_global_eval_list: list,
    V3D_all_gpu: torch.Tensor,
    V25D_all_gpu: torch.Tensor,
    X_aux_all: np.ndarray,
    y_all_global: np.ndarray,
    diam_all_global: np.ndarray,
    n_inner_folds: int = 5,
    seed: int = SEED,
):
    """
    Runs a fast, 100% scan-disjoint 5-inner-fold cross-fitted V10.6 upgrade:
      1. Trains 5 inner-fold 6-Channel CoordConv Multi-Task 3D CNNs (with
         Log-Diameter Regression Head) on 4/5 (`80%`) of inner training scans
         each, producing 100% out-of-fold `(p3_oof, pm_oof, d3_oof)` on `idx_tr`
         and 5-fold averaged predictions on `eval_list`!
      2. Trains 5 inner-fold 9-Family (`1,166D` Pure and `1,195D` with 6-Ch
         CoordConv Multi-Task 3D CNN + predicted diameter context) specialist
         ensembles (`XGBoost` + `CatBoost` + `LightGBM` + `DirectFROC` neural
         calibrator) and tunes the Multi-Quantile Scan-Adaptive FROC Calibrator
         strictly on the inner OOF training set!
    """
    eps = 1e-6
    n_tr = len(y_tr)
    n_evals = len(X_9fam_eval_list)

    rng = np.random.RandomState(seed)
    tr_scans_arr = np.array(sorted(list(set(train_scans_list))))
    tr_scan_pos = np.array([int(np.any(y_tr[suids_tr == s] == 1)) for s in tr_scans_arr])
    pos_sc = tr_scans_arr[tr_scan_pos == 1].copy()
    neg_sc = tr_scans_arr[tr_scan_pos == 0].copy()
    rng.shuffle(pos_sc)
    rng.shuffle(neg_sc)

    inner_scan_folds = [set() for _ in range(n_inner_folds)]
    for i, s in enumerate(pos_sc):
        inner_scan_folds[i % n_inner_folds].add(s)
    for i, s in enumerate(neg_sc):
        inner_scan_folds[i % n_inner_folds].add(s)

    fold_masks = [np.isin(suids_tr, list(inner_scan_folds[k])) for k in range(n_inner_folds)]
    fold_indices = [np.where(fold_masks[k])[0] for k in range(n_inner_folds)]

    # 1. 5-Fold OOF 6-Ch CoordConv Multi-Task 3D CNNs (trained on 4/5 = 80% of inner scans!)
    p3_oof = np.zeros(n_tr, dtype=np.float32)
    pm_oof = np.zeros(n_tr, dtype=np.float32)
    d3_oof = np.zeros(n_tr, dtype=np.float32)

    p3_evs = [np.zeros(len(X_9fam_eval_list[ie]), dtype=np.float32) for ie in range(n_evals)]
    pm_evs = [np.zeros(len(X_9fam_eval_list[ie]), dtype=np.float32) for ie in range(n_evals)]
    d3_evs = [np.zeros(len(X_9fam_eval_list[ie]), dtype=np.float32) for ie in range(n_evals)]

    for kf in range(n_inner_folds):
        idx_val_local = fold_indices[kf]
        idx_fit_local = np.where(~fold_masks[kf])[0]
        assert len(np.intersect1d(idx_fit_local, idx_val_local)) == 0

        targs_g = [idx_global_tr[idx_val_local]] + list(idx_global_eval_list)
        cnn_res_kf = train_and_predict_v106_multitask_coordconv_3d_cnns_gpu(
            V3D_all_gpu=V3D_all_gpu,
            V25D_all_gpu=V25D_all_gpu,
            X_aux_raw=X_aux_all,
            y_all=y_all_global,
            diam_all=diam_all_global,
            idx_fit_global=idx_global_tr[idx_fit_local],
            idx_targets_global_list=targs_g,
            n_epochs=20,
            batch_size=256,
            seed=seed + 100 + kf,
        )
        p3_oof[idx_val_local], pm_oof[idx_val_local], d3_oof[idx_val_local] = cnn_res_kf[0]
        for ie in range(n_evals):
            p3_e, pm_e, d3_e = cnn_res_kf[1 + ie]
            p3_evs[ie] += p3_e / float(n_inner_folds)
            pm_evs[ie] += pm_e / float(n_inner_folds)
            d3_evs[ie] += d3_e / float(n_inner_folds)

    def _build_29d_multitask_cnn_ctx(p3_v, pm_v, d3_v, suids_v, sc_v, v75_v, v77_v):
        la = logit(np.clip(p3_v.astype(np.float64), eps, 1.0 - eps))
        lb = logit(np.clip(pm_v.astype(np.float64), eps, 1.0 - eps))
        l_ens = 0.56 * la + 0.44 * lb
        p_ens = expit(l_ens)
        ctx12 = compute_intra_scan_contextual_features(l_ens, suids_v, sc_v, v75_v, v77_v)
        d_est_mm = np.expm1(np.clip(d3_v, 0.0, 4.0))
        ctx_d = compute_intra_scan_contextual_features(d3_v.astype(np.float64), suids_v, sc_v, v75_v, v77_v)[:, :5]
        mat29 = np.nan_to_num(np.column_stack([
            la, lb, l_ens, p3_v, pm_v, p_ens,
            np.maximum(la, lb), np.minimum(la, lb), np.abs(la - lb),
            d3_v, d_est_mm / 20.0, d3_v * p_ens,
            ctx12, ctx_d,
        ]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        assert mat29.shape[1] == 29
        return mat29, p_ens.astype(np.float64)

    C29_tr, anch_tr = _build_29d_multitask_cnn_ctx(
        p3_oof, pm_oof, d3_oof, suids_tr, star_core_tr, v75_sph_tr, v77_pi_tr
    )
    C29_evs, anch_evs = [], []
    for ie in range(n_evals):
        c_e, a_e = _build_29d_multitask_cnn_ctx(
            p3_evs[ie], pm_evs[ie], d3_evs[ie], suids_eval_list[ie],
            star_core_eval_list[ie], v75_sph_eval_list[ie], v77_pi_eval_list[ie]
        )
        C29_evs.append(c_e)
        anch_evs.append(a_e)

    X_e2e_tr = np.hstack([C29_tr, X_9fam_tr])
    X_e2e_evs = [np.hstack([C29_evs[ie], X_9fam_eval_list[ie]]) for ie in range(n_evals)]
    assert X_9fam_tr.shape[1] == 1166
    assert X_e2e_tr.shape[1] == 1195

    sw_tr = np.ones(n_tr, dtype=np.float64)
    sw_tr[y_tr == 1] = 3.8
    sw_tr[(y_tr == 0) & (star_core_tr >= 0.40)] = 2.8
    sw_tr[(y_tr == 0) & (v75_sph_tr >= 0.35)] *= 1.5

    oof_pure9 = np.zeros(n_tr, dtype=np.float64)
    oof_e2e9 = np.zeros(n_tr, dtype=np.float64)
    ev_pure9_acc = [np.zeros(len(X_9fam_eval_list[ie]), dtype=np.float64) for ie in range(n_evals)]
    ev_e2e9_acc = [np.zeros(len(X_9fam_eval_list[ie]), dtype=np.float64) for ie in range(n_evals)]

    for kf in range(n_inner_folds):
        idx_val = fold_indices[kf]
        idx_fit = np.where(~fold_masks[kf])[0]
        y_f = y_tr[idx_fit]
        sw_f = sw_tr[idx_fit]

        # Pure 9-Family (`1,166D`) inner fold learners (100% full feature matrix & full iteration capacity!)
        mp1 = _fit_xgb_gpu_or_cpu(X_9fam_tr[idx_fit], y_f, sw_f, max_depth=4, lr=0.032, colsample=0.28, n_est=290, reg_l=4.5, seed=seed + 200 + kf)
        mp2 = _fit_xgb_gpu_or_cpu(X_9fam_tr[idx_fit], y_f, sw_f, max_depth=5, lr=0.028, colsample=0.22, n_est=310, reg_l=6.0, seed=seed + 210 + kf)
        mp3 = _fit_catboost_gpu_or_cpu(X_9fam_tr[idx_fit], y_f, sw_f, depth=5, lr=0.038, iterations=460, l2_leaf_reg=6.0, seed=seed + 220 + kf)
        mp_sp = _fit_xgb_gpu_or_cpu(X_spec9_tr[idx_fit], y_f, sw_f, max_depth=4, lr=0.032, colsample=0.42, n_est=250, reg_l=5.5, seed=seed + 230 + kf)

        def _pred_pure_blend(X9_t, Xsp_t):
            p1 = mp1.predict_proba(X9_t)[:, 1]
            p2 = mp2.predict_proba(X9_t)[:, 1]
            p3 = mp3.predict_proba(X9_t)[:, 1]
            psp = mp_sp.predict_proba(Xsp_t)[:, 1]
            plin = 0.30 * p1 + 0.28 * p2 + 0.24 * p3 + 0.18 * psp
            lgeo = (
                0.30 * logit(np.clip(p1, eps, 1 - eps))
                + 0.28 * logit(np.clip(p2, eps, 1 - eps))
                + 0.24 * logit(np.clip(p3, eps, 1 - eps))
                + 0.18 * logit(np.clip(psp, eps, 1 - eps))
            )
            return 0.58 * plin + 0.42 * expit(lgeo)

        oof_pure9[idx_val] = _pred_pure_blend(X_9fam_tr[idx_val], X_spec9_tr[idx_val])
        for ie in range(n_evals):
            ev_pure9_acc[ie] += _pred_pure_blend(X_9fam_eval_list[ie], X_spec9_eval_list[ie]) / float(n_inner_folds)

        # End-to-End 9-Family + 6-Ch CoordConv Multi-Task CNN (`1,195D`) inner fold learners (100% full capacity!)
        me1 = _fit_xgb_gpu_or_cpu(X_e2e_tr[idx_fit], y_f, sw_f, max_depth=4, lr=0.032, colsample=0.28, n_est=290, reg_l=4.5, seed=seed + 300 + kf)
        me2 = _fit_xgb_gpu_or_cpu(X_e2e_tr[idx_fit], y_f, sw_f, max_depth=5, lr=0.028, colsample=0.22, n_est=310, reg_l=6.0, seed=seed + 310 + kf)
        me3 = _fit_catboost_gpu_or_cpu(X_e2e_tr[idx_fit], y_f, sw_f, depth=5, lr=0.038, iterations=460, l2_leaf_reg=6.0, seed=seed + 320 + kf)

        def _pred_e2e_blend(Xe_t, Xsp_t, anch_t):
            p1 = me1.predict_proba(Xe_t)[:, 1]
            p2 = me2.predict_proba(Xe_t)[:, 1]
            p3 = me3.predict_proba(Xe_t)[:, 1]
            psp = mp_sp.predict_proba(Xsp_t)[:, 1]
            plin = 0.27 * p1 + 0.25 * p2 + 0.22 * p3 + 0.14 * psp + 0.12 * anch_t
            lgeo = (
                0.27 * logit(np.clip(p1, eps, 1 - eps))
                + 0.25 * logit(np.clip(p2, eps, 1 - eps))
                + 0.22 * logit(np.clip(p3, eps, 1 - eps))
                + 0.14 * logit(np.clip(psp, eps, 1 - eps))
                + 0.12 * logit(np.clip(anch_t, eps, 1 - eps))
            )
            return 0.58 * plin + 0.42 * expit(lgeo)

        oof_e2e9[idx_val] = _pred_e2e_blend(X_e2e_tr[idx_val], X_spec9_tr[idx_val], anch_tr[idx_val])
        for ie in range(n_evals):
            ev_e2e9_acc[ie] += _pred_e2e_blend(X_e2e_evs[ie], X_spec9_eval_list[ie], anch_evs[ie]) / float(n_inner_folds)

    # 3. Stage-2 Intra-Scan Context + Multi-Quantile Scan-Adaptive FROC Calibration (strictly OOF on `idx_tr`!)
    def _calibrate_and_fuse_oof(oof_s1, ev_s1_list, v105_ev_list, use_cnn_ctx: bool):
        oof_s2 = np.zeros(n_tr, dtype=np.float64)
        ev_s2_list = [np.zeros(len(ev_s1_list[ie]), dtype=np.float64) for ie in range(n_evals)]

        l_s1_tr = logit(np.clip(oof_s1, eps, 1.0 - eps))
        ctx_tr = compute_intra_scan_contextual_features(l_s1_tr, suids_tr, star_core_tr, v75_sph_tr, v77_pi_tr)
        M_tr = np.column_stack([oof_s1, l_s1_tr, ctx_tr, X_spec9_tr[:, :40]])
        if use_cnn_ctx:
            M_tr = np.column_stack([M_tr, C29_tr])

        M_evs = []
        for ie in range(n_evals):
            l_e = logit(np.clip(ev_s1_list[ie], eps, 1.0 - eps))
            ctx_e = compute_intra_scan_contextual_features(
                l_e, suids_eval_list[ie], star_core_eval_list[ie], v75_sph_eval_list[ie], v77_pi_eval_list[ie]
            )
            m_e = np.column_stack([ev_s1_list[ie], l_e, ctx_e, X_spec9_eval_list[ie][:, :40]])
            if use_cnn_ctx:
                m_e = np.column_stack([m_e, C29_evs[ie]])
            M_evs.append(m_e)

        for kf in range(n_inner_folds):
            idx_v = fold_indices[kf]
            idx_f = np.where(~fold_masks[kf])[0]
            sc_m = RobustScaler()
            Mf_s = np.clip(sc_m.fit_transform(M_tr[idx_f]), -10.0, 10.0)
            Mv_s = np.clip(sc_m.transform(M_tr[idx_v]), -10.0, 10.0)

            yf = y_tr[idx_f]
            s1_f = oof_s1[idx_f]
            q95 = float(np.percentile(s1_f[yf == 0], 95.0))
            q85 = float(np.percentile(s1_f[yf == 0], 85.0))
            swf = np.ones(len(yf), dtype=np.float64)
            swf[yf == 0] = np.where(s1_f[yf == 0] >= q95, 9.5, np.where(s1_f[yf == 0] >= q85, 4.2, 0.85))
            swf[yf == 1] = 2.4

            s2_x = _fit_xgb_gpu_or_cpu(Mf_s, yf, swf, max_depth=4, lr=0.032, colsample=0.55, n_est=160, reg_l=5.5, seed=seed + 400 + kf)
            s2_l = LogisticRegression(C=0.05, max_iter=300, random_state=seed + 410 + kf).fit(Mf_s, yf, sample_weight=swf)

            oof_s2[idx_v] = 0.45 * oof_s1[idx_v] + 0.38 * s2_x.predict_proba(Mv_s)[:, 1] + 0.17 * s2_l.predict_proba(Mv_s)[:, 1]
            for ie in range(n_evals):
                Me_s = np.clip(sc_m.transform(M_evs[ie]), -10.0, 10.0)
                p_e = 0.45 * ev_s1_list[ie] + 0.38 * s2_x.predict_proba(Me_s)[:, 1] + 0.17 * s2_l.predict_proba(Me_s)[:, 1]
                ev_s2_list[ie] += p_e / float(n_inner_folds)

        # Tune Multi-Quantile Scan-Adaptive FROC Calibration on `oof_s2` (`idx_tr`) via precomputed scan vectors (<0.2s)
        n_sc_tr = len(train_scans_list)
        l_s2_vec = logit(np.clip(oof_s2.astype(np.float64), eps, 1.0 - eps))
        med_s2_vec = np.zeros(n_tr, dtype=np.float64)
        q25_s2_vec = np.zeros(n_tr, dtype=np.float64)
        sm_s2_vec = np.zeros(n_tr, dtype=np.float64)
        vz_s2_vec = np.zeros(n_tr, dtype=np.float64)
        df_g_tr = pd.DataFrame({'s': suids_tr, 'idx': np.arange(n_tr)})
        for _, grp in df_g_tr.groupby('s', sort=False):
            idxs_g = grp['idx'].values
            if len(idxs_g) <= 1:
                continue
            lg = l_s2_vec[idxs_g]
            vg = veto_tr[idxs_g].astype(np.float64)
            vz_s2_vec[idxs_g] = np.clip((vg - float(np.median(vg))) / (float(np.std(vg)) + 1e-4), -2.5, 3.5)
            med_s2_vec[idxs_g] = float(np.median(lg))
            q25_s2_vec[idxs_g] = float(np.percentile(lg, 25))
            order = np.argsort(lg)
            max_l, sec_l = lg[order[-1]], lg[order[-2]]
            sm = lg - max_l
            sm[order[-1]] = max_l - sec_l
            sm_s2_vec[idxs_g] = np.clip(sm, -4.5, 4.5)

        pos_m_tr = (y_tr == 1)
        neg_m_tr = (y_tr == 0)
        best_cal = (0.20, 0.08, 0.08, 0.0)
        best_cpm = -1.0
        for b_med in [0.12, 0.22, 0.32]:
            for b_q25 in [0.04, 0.12, 0.20]:
                for g_sec in [0.04, 0.10, 0.16]:
                    for v_veto in [0.0, 0.05, 0.10]:
                        l_try = (
                            l_s2_vec
                            - b_med * med_s2_vec
                            - b_q25 * q25_s2_vec
                            + g_sec * sm_s2_vec
                            - v_veto * vz_s2_vec
                        )
                        c_try = expit(np.clip(l_try, -20.0, 20.0))
                        cpm_t, _ = _eval_continuous_interpolated_froc_curve(
                            c_try[pos_m_tr], c_try[neg_m_tr], n_sc_tr
                        )
                        if cpm_t > best_cpm + 1e-5:
                            best_cpm = cpm_t
                            best_cal = (b_med, b_q25, g_sec, v_veto)

        bm, bq, gs, vv = best_cal
        out_evs = []
        for ie in range(n_evals):
            p_v106_cal = _apply_multi_quantile_scan_adaptive_froc_calibration(
                ev_s2_list[ie], suids_eval_list[ie], veto_eval_list[ie], bm, bq, gs, vv
            )
            if v105_ev_list is not None and v105_ev_list[ie] is not None:
                # Fuse V10.5 Equi-Capacity Pairwise OOF realization with V10.6 9-Family Multi-Task realization!
                l_105 = logit(np.clip(v105_ev_list[ie].astype(np.float64), eps, 1.0 - eps))
                l_106 = logit(np.clip(p_v106_cal.astype(np.float64), eps, 1.0 - eps))
                p_fused = expit(0.48 * l_105 + 0.52 * l_106).astype(np.float32)
                out_evs.append(p_fused)
            else:
                out_evs.append(p_v106_cal.astype(np.float32))
        return out_evs

    p_pure9_evs = _calibrate_and_fuse_oof(oof_pure9, ev_pure9_acc, v105_pure_eval_list, use_cnn_ctx=False)
    p_e2e9_evs = _calibrate_and_fuse_oof(oof_e2e9, ev_e2e9_acc, v105_e2e_eval_list, use_cnn_ctx=True)
    cnn_anch_evs = [anch_evs[ie].astype(np.float32) for ie in range(n_evals)]
    return p_pure9_evs, p_e2e9_evs, cnn_anch_evs
