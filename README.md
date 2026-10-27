# Integrating Multi-Scale Radial and Differential-Geometric 3D Descriptors with Volumetric Deep Ensembles for Pulmonary Nodule Classification on LUNA16

[License: MIT](https://www.google.com/search?q=LICENSE)
[](https://www.python.org/)
[](https://pytorch.org/)
[](https://link.springer.com/journal/11548)

**Author:** Nader Jafari  
**Affiliation:** Member of the Computer Society of Iran (CSI), Tehran, Iran  
**Corresponding Author Email:** [Nader.jafari23@gmail.com](mailto:Nader.jafari23@gmail.com)

---

## 1. Overview & Scientific Contributions

This repository provides the official, end-to-end reproducible Python/PyTorch implementation, 3D differential-geometric feature extraction engine, scan-level cluster bootstrap evaluator, and forensic base-learner provenance audit for our manuscript submitted to the *International Journal of Computer Assisted Radiology and Surgery (IJCARS)*.

### Key Algorithmic & Methodological Contributions
1. **748 Physical 3D Descriptors (`D_1 = 784` with 36 Deep/Contextual Columns):**
   - **Family 1 (`star_v74`, 128d):** 26-ray 3D Star-Ray radial profiles, full-width at half-maximum radius ($r_{\mathrm{FWHM}}$), radial slope ($dI/dr$), and 6 concentric shell moments ($r \in \{1,\dots,14\}\text{ mm}$).
   - **Family 2 (`geom_v73`, 125d):** Scale-normalized 3D Hessian ($\mathbf{H}_\sigma = \sigma^2 \nabla^2 I_\sigma$, $\sigma \in \{1,2,3,4\}\text{ mm}$) Frangi vesselness ($V_\sigma$), blobness ($B_\sigma$), and algebraic Koenderink shape index ($\mathrm{SI}_\sigma$) and curvedness ($\mathrm{CV}_\sigma$).
   - **Family 3 (`v75_X`, 229d):** Spherical-polar variance $\mathrm{Var}_\Omega(I)$, 3D Sobel gradient magnitude $\|\nabla I\|_2$, compact sphericity $\Psi$ (218d), and 13-direction 3D Gray-Level Co-occurrence Matrix (GLCM) Haralick texture features (11d).
   - **Family 4 (`v76_X`, 137d):** Outward radial unit vector $\mathbf{u}_r(\mathbf{x})$ (with $\mathbf{u}_r(\mathbf{x}_0) = \mathbf{0}$ at $\mathbf{x}_0 = (24,24,24)$), inward radial-gradient flux ($\Phi_\sigma(\mathbf{x}) = -\nabla I_\sigma(\mathbf{x}) \cdot \mathbf{u}_r(\mathbf{x})$), and multi-scale Difference-of-Gaussians (DoG) bank.
   - **Family 5 (`v77_X`, 129d):** 13 antipodal ray-pair minima ($a_m$) and 3-ray Y-bifurcation contrast ($C_{\mathrm{bifurc}} = \frac{r_{(1)}+r_{(2)}+r_{(3)}}{3} - r_{(7)}$) across 7 polar radial bands.
2. **Center-Focused Multi-Scale Pyramid Pooling (`CF-MSPP`) & Bottleneck SNR Proof:**
   - Eliminates $1/512$ lung-air dilution in $8^3 \to 1^3$ global average pooling on $3\text{–}6\text{ mm}$ micro-nodules, achieving a **+66.72 dB bottleneck SNR gain** at $D = 3.0\text{ mm}$ ($-39.21\text{ dB} \to +27.51\text{ dB}$).
3. **Direct-FROC Surrogate Neural Calibrator (`DirectFROCNeuralCalibrator`):**
   - Optimizes a differentiable Softplus surrogate loss directly at the 7 official LUNA16 false-positive operating thresholds $\{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$ FP/scan.
4. **Forensic 9-Field Base-Learner Provenance & Analytical $61.9\times$ Prior-Shift Audit:**
   - Strictly separates **Regime I (100% Pure-OOF across all 888 scans: 81.24% Official CPM, +7.06% over CNNs alone, $p < 0.001$)** from **Regime II (Hybrid 7-CNN + 10-Fold Meta-OOF: 84.39% Official CPM across 888 scans; 86.36% on Validation and 86.66% [step: 86.83%, ROC-AUC = 0.9707] on Hold-Out Test)**.

---

## 2. Summary of Audited Results on LUNA16

| Evaluation Cohort & Regime | Base CNN OOF Status | Official LUNA16 CPM (95% CI) | Step FROC CPM | ROC-AUC / FPR@97% |
| :--- | :--- | :---: | :---: | :---: |
| **All 888 Scans — Step 0: OOF CNNs Only (0d)** | 100% Pure-OOF (`V7.5 + V7.6`) | 74.18% (71.12–77.15%) | 74.31% | 0.9399 / 38.40% |
| **All 888 Scans — Regime I: +748 Physical 3D Descriptors** | **100% Pure-OOF (`is_oof = True`)** | **81.24% (78.59–83.85%)** | **81.33%** | **0.9492 / 28.70%** |
| **All 888 Scans — Regime II: Hybrid 7-CNN + 10F Meta-OOF** | 10F Meta-OOF (2 OOF + 5 Split CNNs) | **84.39% (81.73–86.90%)** | **84.33%** | **0.9621 / 24.02%** |
| **Independent Validation Split (133 Scans, 169 Lesions)** | **100% Out-of-Sample (`Overlap = 0`)** | **86.36% (79.53–92.08%)** | **86.98%** | **0.9640 / 23.46%** |
| **Independent Hold-Out Test Split (133 Scans, 167 Lesions)** | **100% Out-of-Sample (`Overlap = 0`)** | **86.66% (80.23–92.28%)** | **86.83%** | **0.9707 / 17.59%** |

- **Large-Nodule ($D \ge 10.0\text{ mm}$, $n=35$ on Test) Recall Recovery at 1.0 FP/scan:** Improves from **65.71% (23/35)** with 3D CNNs alone to **82.86% (29/35)** in nested V8.1 and **88.57% (31/35)** in exploratory V7.6, while preserving **87.84% (65/74)** sensitivity on small nodules ($D < 6.0\text{ mm}$).
- **Clinical Safety-Net Recall on Hold-Out Test:** Reaches **97.62%**, **99.44%**, and **100.00%** sensitivity at `2.0`, `4.0`, and `8.0 FP/scan` (`95.41%`, `99.58%`, and `99.91%` across all 888 scans).

---

## 3. Repository Structure

```text
LUNA16-3D-Geometric-Deep-Ensemble-IJCARS/
├── README.md                                              # Documentation, results, and reproduction guide
├── LICENSE                                                # MIT Open-Source License
├── CITATION.cff                                           # Official GitHub & ORCID citation metadata
├── requirements.txt                                       # Exact pinned Python environment dependencies
├── docs/
│   └── GITHUB_PROFILE_README_TEMPLATE.md                  # Ready-to-use GitHub Profile README for Nader Jafari
├── figures/                                               # 600-DPI publication figures (Fig. 1 to Fig. 7)
│   ├── Fig1_IJCARS_Data_and_Pipeline_600dpi.png
│   ├── Fig2_IJCARS_CT_Feature_Maps_600dpi.png
│   ├── Fig3_IJCARS_Convergence_and_SNR_600dpi.png
│   ├── Fig4_IJCARS_FROC_Curves_600dpi.png
│   ├── Fig5_IJCARS_Ablation_and_Radar_600dpi.png
│   ├── Fig6_IJCARS_Diameter_and_Features_600dpi.png
│   └── Fig7_IJCARS_Regimes_and_PriorShift_600dpi.png
└── src/
    ├── stage1_luna16_isotropic_preprocessing.py           # SimpleITK 1x1x1 mm^3 resampling & 48^3 patch extraction
    ├── stage2_train_3d_resacs_and_748d_descriptors.py     # 3D ResACS-Net (CF-MSPP + OHEM) & 748D feature extraction
    └── stage3_nested_oof_ensemble_and_luna16_evaluator.py # Leak-free 2-stage OOF ensemble, Direct-FROC & bootstrap
```

---

## 4. Quick Start & Reproduction

### 4.1 Environment Setup
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/LUNA16-3D-Geometric-Deep-Ensemble-IJCARS.git
cd LUNA16-3D-Geometric-Deep-Ensemble-IJCARS
pip install -r requirements.txt
```

### 4.2 Three-Stage Pipeline Execution
1. **Stage 1 — Isotropic $1\times 1\times 1\text{ mm}^3$ Resampling & Coordinate-Verified $48^3$ Patch Extraction:**
   ```bash
   python3 src/stage1_luna16_isotropic_preprocessing.py
   ```
2. **Stage 2 — 3D ResACS-Net Training (`CF-MSPP + OHEM`) & 748D Physical Feature Extraction:**
   ```bash
   python3 src/stage2_train_3d_resacs_and_748d_descriptors.py
   ```
3. **Stage 3 — Two-Stage Inside-Fold Nested OOF Ensemble, `DirectFROCNeuralCalibrator` & Official `noduleCADEvaluationLUNA16.py` Verification:**
   ```bash
   python3 src/stage3_nested_oof_ensemble_and_luna16_evaluator.py
   ```

---

## 5. Citation

If you find this code, the 748D 3D differential-geometric descriptor bank, or our forensic LUNA16 benchmark audit useful in your research, please cite:

```bibtex
@article{Jafari2026LUNA16,
  author  = {Jafari, Nader},
  title   = {Integrating Multi-Scale Radial and Differential-Geometric 3D Descriptors with Volumetric Deep Ensembles for Pulmonary Nodule Classification: Forensic Protocol Audit, Base-Learner Provenance, and Prior-Shift Analysis on {LUNA16}},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year    = {2026},
  note    = {Computer Society of Iran (CSI), Tehran, Iran}
}
```
