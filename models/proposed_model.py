#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete End-to-End 4-Stage Proposed Hybrid 3D Architecture (Figure 1b Stages I–IV):
  - Stage I:   Stage0GPUMultiScaleScreener (64D) & recenter_3d_patches_to_local_foveal_centroid_gpu (Eq. 9)
  - Stage II:  Stream A (extract_all_9families_from_3d_patches_array: 1,166D, Eqs. 2–7)
               + Stream B (MultiWindowCoordConv3DResNeXtWithDiameterHead + FoldIsolatedTriPlanarMIPConvNet: 29D, Eq. 8)
  - Stage III: run_v106_9family_coordconv_multitask_upgrade_on_split (1,195D Hybrid + 506D Veto Specialists, Eq. 8)
  - Stage IV:  _apply_multi_quantile_scan_adaptive_froc_calibration & apply_monotonic_two_stage_cascade_calibration (Eqs. 9–10)
"""
import numpy as np
from .physical_3d_descriptors import (
    FAMILY_DIMENSIONS_1166D,
    extract_all_9families_from_3d_patches_array,
    compute_exact_3x3_symmetric_hessian_eigenvalues,
    _build_real_spherical_harmonic_qr_subspaces_26,
)
from .backbone import (
    MultiWindowCoordConv3DResNeXtWithDiameterHead,
    FoldIsolatedTriPlanarMIPConvNet,
    _batch_hard_negative_focal_ranking_loss,
)
from .attention_module import (
    compute_analytical_bottleneck_snr_db,
    apply_8view_cube_equivariance,
    verify_3d_mip_augmentation_geometric_consistency,
)
from .detection_head import (
    Stage0GPUMultiScaleScreener,
    recenter_3d_patches_to_local_foveal_centroid_gpu,
    _apply_multi_quantile_scan_adaptive_froc_calibration,
    apply_monotonic_two_stage_cascade_calibration,
)
from .feature_fusion import run_v106_9family_coordconv_multitask_upgrade_on_split


class ProposedHybrid3DNoduleCAD:
    """
    Production end-to-end 4-stage hybrid 3D pulmonary nodule false-positive reduction architecture.
    Maps 1-to-1 to Figure 1b (Stages I–IV) and Equations (1)–(10) of the manuscript.
    """
    PURE_DIM = 1166
    DEEP_OOF_DIM = 29
    HYBRID_DIM = 1195
    SPECIALIST_SUBSPACE_DIM = 506
    STAGE0_DIM = 64

    def __init__(self):
        self.family_dimensions = FAMILY_DIMENSIONS_1166D
        assert sum(self.family_dimensions.values()) == self.PURE_DIM
        assert self.PURE_DIM + self.DEEP_OOF_DIM == self.HYBRID_DIM

    def extract_stage2_all_streams(self, patches_48: np.ndarray, suids: np.ndarray, coords_xyz: np.ndarray) -> dict:
        """
        Executes the real 9-family 1,166D 3D extractor (Stream A) AND builds the 6-channel 24^3
        CoordConv 3D CNN tensor + 9-channel 2.5D Tri-Planar MIP tensor (Stream B) on a batch of
        (N, 48, 48, 48) CT patches.
        """
        (
            polar_grid, geom_126, star_125, v75_229, v76_131,
            v77_57, harm6_128, vol7_148, fam8_84, fam9_64, V3D_6ch_24, V25D_9ch_48
        ) = extract_all_9families_from_3d_patches_array(patches_48, suids, coords_xyz)

        # Normalized thoracic coordinates (r_xy, z_norm) — 2D (Eq. 2)
        px, py, pz = coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]
        r_xy = np.sqrt((px / 150.0) ** 2 + (py / 150.0) ** 2)
        z_norm = (pz + 175.0) / 300.0
        coords_2d = np.column_stack([r_xy, z_norm]).astype(np.float32)

        # 72D polar core-annulus contrasts from polar_grid (N, 26, 7) + 57D v77_57 = 129D Family 5
        pg_core = np.mean(polar_grid[:, :, :2], axis=2)    # (N, 26)
        pg_mid = np.mean(polar_grid[:, :, 2:5], axis=2)    # (N, 26)
        pg_outer = np.mean(polar_grid[:, :, 5:], axis=2)   # (N, 26)
        polar_72 = np.hstack([
            pg_core - pg_outer,                            # 26D
            pg_core - pg_mid,                              # 26D
            np.percentile(pg_core, np.linspace(0, 100, 10), axis=1).T,  # 10D
            np.percentile(pg_outer, np.linspace(0, 100, 10), axis=1).T, # 10D
        ]).astype(np.float32)
        fam5_129 = np.hstack([v77_57, polar_72]).astype(np.float32)

        X_pure_1166 = np.hstack([
            coords_2d, geom_126, star_125, v75_229, v76_131,
            fam5_129, harm6_128, vol7_148, fam8_84, fam9_64
        ]).astype(np.float32)

        # Append Ch 4 (Subsolid GGN window) and Ch 5 (3D Radial Gaussian CoordConv map) to form full (N, 6, 24, 24, 24) tensor
        n_b = len(patches_48)
        v_nat24 = V3D_6ch_24[:, 1].astype(np.float32)
        ch4_ggn = np.clip((v_nat24 - 0.07) / 0.57, 0.0, 1.0).astype(np.float16)
        zz24, yy24, xx24 = np.mgrid[0:24, 0:24, 0:24].astype(np.float32) - 11.5
        r2_24 = zz24 ** 2 + yy24 ** 2 + xx24 ** 2
        ch5_coord = np.broadcast_to(np.exp(-r2_24 / (2.0 * 6.0 ** 2)).astype(np.float16)[None, ...], (n_b, 24, 24, 24))
        V3D_6ch_full = np.concatenate([V3D_6ch_24, ch4_ggn[:, None], ch5_coord[:, None]], axis=1)

        return {
            "X_pure_1166": X_pure_1166,
            "X_coords_2d": coords_2d,
            "X_geom_126": geom_126,
            "X_star_125": star_125,
            "X_v75_229": v75_229,
            "X_v76_131": v76_131,
            "X_v77_polar_129": fam5_129,
            "X_harm6_128": harm6_128,
            "X_vol7_148": vol7_148,
            "X_fam8_84": fam8_84,
            "X_fam9_64": fam9_64,
            "V3D_6ch_24": V3D_6ch_full,
            "V25D_9ch_48": V25D_9ch_48,
        }
