#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiological Hounsfield Unit (HU) Dual-Window Normalization, Multi-Scale 6-Channel
CoordConv 24x24x24 Tensor Construction (Section 2.3, Eq. 8), and Fold-Isolated Feature Scaling.
"""
import numpy as np
from scipy import ndimage


def normalize_lung_hu_window(hu_volume: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """
    Maps raw thoracic CT Hounsfield Units [-1000, +400] HU linearly onto [0.0, 1.0] in IEEE-754 float32
    (Section 2.1, Eq. 1). Preserves parenchymal, vascular, and solid nodule attenuation contrast.
    """
    v = np.asarray(hu_volume, dtype=np.float32)
    return np.clip((v - float(hu_min)) / float(hu_max - hu_min), 0.0, 1.0)


def normalize_ggn_subsolid_window(hu_volume: np.ndarray, hu_min: float = -900.0, hu_max: float = -100.0) -> np.ndarray:
    """
    Maps subsolid / ground-glass nodule (GGN) attenuation [-900, -100] HU onto [0.0, 1.0] in float32
    (Channel 4 of the 6-Channel MultiWindowCoordConv3DResNeXtWithDiameterHead, Section 2.3).
    """
    v = np.asarray(hu_volume, dtype=np.float32)
    return np.clip((v - float(hu_min)) / float(hu_max - hu_min), 0.0, 1.0)


def build_6channel_multiscale_coordconv_tensor_24(patches_48: np.ndarray) -> np.ndarray:
    """
    Constructs the 6-channel 24x24x24 input tensor for MultiWindowCoordConv3DResNeXtWithDiameterHead
    from a batch of isotropic 48x48x48 mm^3 CT patches in [0, 1] (Section 2.3, Figure 1b Stage II Stream B):
      - Ch 0: 12x12x12 mm^3 high-resolution core crop (trilinearly upsampled 2x to 24^3)
      - Ch 1: 24x24x24 mm^3 native nodule neighborhood crop
      - Ch 2: 48x48x48 mm^3 full contextual crop (2x2x2 average-pooled to 24^3)
      - Ch 3: 3D multi-scale box-difference map (3x3x3 smoothed minus 7x7x7 smoothed over 24^3)
      - Ch 4: Subsolid ground-glass nodule (GGN) soft-attenuation window [-900, -100] HU over 24^3
      - Ch 5: 3D radial Gaussian CoordConv spatial prior exp(-||x - x_0||_2^2 / (2 * 6.0^2))
    """
    V = np.asarray(patches_48, dtype=np.float32)
    if V.ndim == 3:
        V = V[None, ...]
    assert V.shape[1:] == (48, 48, 48), f"Expected (B, 48, 48, 48), got {V.shape}"
    B = len(V)

    ch0_core24 = ndimage.zoom(V[:, 18:30, 18:30, 18:30], (1, 2, 2, 2), order=1).astype(np.float32)
    ch1_nat24 = V[:, 12:36, 12:36, 12:36]
    ch2_ctx24 = V.reshape(B, 24, 2, 24, 2, 24, 2).mean(axis=(2, 4, 6)).astype(np.float32)

    s1 = ndimage.uniform_filter(V, size=(1, 3, 3, 3), mode="nearest")
    s3 = ndimage.uniform_filter(V, size=(1, 7, 7, 7), mode="nearest")
    ch3_boxdiff = (s1 - s3)[:, 12:36, 12:36, 12:36].astype(np.float32)

    ch4_ggn24 = np.clip((ch1_nat24 - 0.0714) / 0.5714, 0.0, 1.0).astype(np.float32)

    zz, yy, xx = np.mgrid[0:24, 0:24, 0:24].astype(np.float32) - 11.5
    r2 = zz ** 2 + yy ** 2 + xx ** 2
    ch5_coord = np.broadcast_to(np.exp(-r2 / (2.0 * 6.0 ** 2))[None, ...], (B, 24, 24, 24)).astype(np.float32)

    return np.stack([ch0_core24, ch1_nat24, ch2_ctx24, ch3_boxdiff, ch4_ggn24, ch5_coord], axis=1)


class FoldIsolatedRobustNormalizer:
    """
    Leak-free median/interquartile-range (IQR) feature normalizer fit strictly on training scans
    within each outer/inner cross-validation fold (Section 2.3).
    """
    def __init__(self, clip_val: float = 8.0, eps: float = 1e-5):
        self.clip_val = float(clip_val)
        self.eps = float(eps)
        self.median_ = None
        self.iqr_ = None

    def fit(self, X_train: np.ndarray):
        X = np.asarray(X_train, dtype=np.float32)
        self.median_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqr_ = np.maximum(q75 - q25, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.median_ is not None, "Normalizer must be fit on training fold before transform()!"
        X_clean = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Z = (X_clean - self.median_[None, :]) / self.iqr_[None, :]
        return np.clip(Z, -self.clip_val, self.clip_val).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    dummy_hu = rng.uniform(-1000.0, 400.0, size=(2, 48, 48, 48)).astype(np.float32)
    norm_p = normalize_lung_hu_window(dummy_hu)
    t6 = build_6channel_multiscale_coordconv_tensor_24(norm_p)
    assert t6.shape == (2, 6, 24, 24, 24)
    print("Verified preprocessing/normalization.py: 6-Ch 24^3 tensor shape =", t6.shape)
