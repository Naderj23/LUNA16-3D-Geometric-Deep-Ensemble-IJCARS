# 3D Differential-Geometric Descriptors and Volumetric Deep Ensembles for Pulmonary Nodule Classification: A LUNA16 Benchmark Audit

[License: MIT](LICENSE)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)
![Journal: IJCARS](https://img.shields.io/badge/Journal-IJCARS%20(Springer)-003366.svg)

**Author:** Nader Jafari, M.Sc. (M.Sc. in Software Engineering)  
**Affiliation:** Member of the Computer Society of Iran (CSI), Tehran, Iran  
**ORCID iD:** [0009-0008-6849-3207](https://orcid.org/0009-0008-6849-3207)  
**Corresponding Author Email:** [Nader.jafari23@gmail.com](mailto:Nader.jafari23@gmail.com)

---

## 1. Overview & Scientific Contributions

This repository provides the official, end-to-end reproducible Python/PyTorch implementation, 3D differential-geometric feature extraction engine, scan-level cluster bootstrap evaluator, and forensic base-learner provenance audit for our manuscript submitted to the *International Journal of Computer Assisted Radiology and Surgery (IJCARS)*.

### Key Algorithmic & Methodological Contributions
1. **748 Physical 3D Descriptors (`D_1 = 784` with 36 Deep/Contextual Columns):**
   - **Family 1 (`geom_v73`, 128d):** 26-ray 3D Star-Ray radial profiles, full-width at half-maximum radius ($r_{\mathrm{FWHM}}$), radial slope ($dI/dr$), and 6 concentric shell moments ($r \in \{1,\dots,14\}\text{ mm}$).
   - **Family 2 (`star_v74`, 125d):** Scale-normalized 3D Hessian ($\mathbf{H}_\sigma = \sigma^2 \nabla^2 I_\sigma$, $\sigma \in \{1,2,3,4\}\text{ mm}$) Frangi vesselness ($V_\sigma$), blobness ($B_\sigma$), and algebraic Koenderink shape index ($\mathrm{SI}_\sigma$) and curvedness ($\mathrm{CV}_\sigma$).
   - **Family 3 (`v75_X`, 229d):** Spherical-polar variance $\mathrm{Var}_\Omega(I)$, 3D Sobel gradient magnitude $\|\nabla I\|_2$, compact sphericity $\Psi$ (218d), and 13-direction 3D Gray-Level Co-occurrence Matrix (GLCM) Haralick texture features (11d).
   - **Family 4 (`v76_X`, 137d):** Outward radial unit vector $\mathbf{u}_r(\mathbf{x})$ (with $\mathbf{u}_r(\mathbf{x}_0) = \mathbf{0}$ at $\mathbf{x}_0 = (24,24,24)$), inward radial-gradient flux ($\Phi_\sigma(\mathbf{x}) = -\nabla I_\sigma(\mathbf{x}) \cdot \mathbf{u}_r(\mathbf{x})$), and multi-scale Difference-of-Gaussians (DoG) bank.
   - **Family 5 (`v77_X`, 129d):** 13 antipodal ray-pair minima ($a_m$) and 3-ray Y-bifurcation contrast ($C_{\mathrm{bifurc}} = \frac{r_{(1)}+r_{(2)}+r_{(3)}}{3} - r_{(7)}$) across 7 polar radial bands.
2. **Center-Focused Multi-Scale Pyramid Pooling (`CF-MSPP`) & Bottleneck SNR Proof:**
   - Eliminates $1/512$ lung-air dilution in $8^3 \to 1^3$ global average pooling on $3\text{–}6\text{ mm}$ micro-nodules, achieving a **+66.72 dB bottleneck SNR gain** at $D = 3.0\text{ mm}$ ($-39.21\text{ dB} \to +27.51\text{ dB}$).
3. **Direct-FROC Surrogate Neural Calibrator (`DirectFROCNeuralCalibrator`):**
   - Optimizes a differentiable Softplus surrogate loss directly at the 7 LUNA16 false-positive operating thresholds $\{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0\}$ FP/scan.
4. **Forensic 9-Field Base-Learner Provenance & Analytical $61.9\times$ Prior-Shift Audit:**
   - **Curated Patch Benchmark vs. Unscreened Official Track:** All empirical experiments are conducted on the curated 10,066-patch cohort (`LUNA16_Patches_CoordVerified_48.h5`: 1,186 annotation-centered positives + 8,880 negatives at `10.0 neg/scan`), scored with the official `noduleCADEvaluationLUNA16.py` evaluator. Because we do not empirically evaluate all 551,065 (`candidates.csv`) or 754,975 (`candidates_V2.csv`) unscreened detector candidates, our CPM scores represent curated-benchmark performance, complemented by an analytical $61.9\times$ horizontal FP-axis prior-shift projection (`46.28%` CPM).
   - **Non-Equivalence of Regime I and Regime II Across All 888 Scans:**
     - **Regime I (Strict 100% Pure-OOF across all 888 scans: `81.24%` vs. `74.18%` CPM, `+7.06` percentage points, $p < 0.001$)** uses only the two scan-disjoint OOF 3D CNNs (`V7.5` 10-fold OOF and `V7.6` 5-fold OOF; `is_oof = True` for all 10,066 patches) and serves as the **sole confirmatory Pure-OOF baseline across all 888 scans**.
     - **Regime II across all 888 scans (`84.39%` CPM)** is a **Hybrid Meta-OOF** regime: while the Stage-1/Stage-2 meta-learners, scaling, and feature selection are 10-fold scan-disjoint OOF, 5 of the 7 CNN base learners (`V4.6`–`V7.3`) were trained on `Train-622` (`is_oof = False` on `Train-622`). Meta-learner OOF partitioning alone does not remove this upstream prediction provenance difference on `Train-622`.
     - **Independent 133-Scan Validation (`86.36%`) and 133-Scan Hold-Out Test (`86.66%`) Splits:** On these locked 133-scan cohorts, all 7 CNN base learners are **100% scan-disjoint out-of-sample (`is_oof = True`, `Overlap = 0`)**.

---

## 2. Summary of Audited Results on the Curated LUNA16 Benchmark (10,066 Patches)

| Evaluation Cohort & Regime | Base CNN Provenance (`is_oof`) | Evaluator CPM (95% CI) | Step FROC CPM | ROC-AUC / FPR@97% |
| :--- | :--- | :---: | :---: | :---: |
| **All 888 Scans — Step 0: OOF CNNs Only (0d)** | **100% Pure-OOF (`V7.5 + V7.6`, `is_oof = True`)** | 74.18% (71.12–77.15%) | 74.31% | 0.9399 / 38.40% |
| **All 888 Scans — Regime I: +748 Physical 3D Descriptors** | **100% Pure-OOF (`V7.5 + V7.6`, `is_oof = True`)** | **81.24% (78.59–83.85%)** | **81.33%** | **0.9492 / 28.70%** |
| **All 888 Scans — Regime II: Hybrid 7-CNN + 10F Meta-OOF** | Hybrid (2 OOF CNNs + 5 `Train-622` CNNs)* | 84.39% (81.73–86.90%) | 84.33% | 0.9621 / 24.02% |
| **Independent Validation Split (133 Scans, 169 Lesions)** | **100% Out-of-Sample (All 7 CNNs `is_oof = True`)** | **86.36% (79.53–92.08%)** | **86.98%** | **0.9640 / 23.46%** |
| **Independent Hold-Out Test Split (133 Scans, 167 Lesions)** | **100% Out-of-Sample (All 7 CNNs `is_oof = True`)** | **86.66% (80.23–92.28%)** | **86.83%** | **0.9707 / 17.59%** |
| **Analytical $61.9\times$ Prior-Shift Projection (All 888 Scans)** | Analytical Horizontal FP-Axis Scaling (`619.1/scan`) | 46.28% (41.59–51.77%) | 39.35% | 0.9621 / 24.02% |

*Note: On All-888 Scans, Regime I (`81.24%`) and Regime II (`84.39%`) do not provide equivalent statistical evidence because 5/7 CNNs in Regime II were trained on `Train-622`. Strict 7-CNN out-of-sample evaluation holds exclusively on the 133-scan Validation and 133-scan Hold-Out Test splits.

- **Large-Nodule ($D \ge 10.0\text{ mm}$, $n=35$ on Test) Recall Recovery at 1.0 FP/scan:** Improves from **65.71% (23/35)** with 3D CNNs alone to **82.86% (29/35)** in nested V8.1 and **88.57% (31/35)** in exploratory V7.6, while preserving **87.84% (65/74)** sensitivity on small nodules ($D < 6.0\text{ mm}$).
- **Clinical Safety-Net Recall on Hold-Out Test:** Reaches **97.62%**, **99.44%**, and **100.00%** sensitivity at `2.0`, `4.0`, and `8.0 FP/scan`.

---

## 3. Repository Structure

```text
LUNA16-3D-Geometric-Deep-Ensemble-IJCARS/
├── README.md                                              # Documentation, provenance audit, and reproduction guide
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
    ├── physical_3d_descriptors_748d.py                    # Reference implementation of Eqs. (1)-(10) & 748D descriptors
    ├── stage1_luna16_isotropic_preprocessing.py           # SimpleITK 1x1x1 mm^3 resampling & 48^3 patch extraction
    ├── stage2_train_3d_resacs_and_748d_descriptors.py     # Regime I (Pure-OOF), Regime II (Hybrid), & 133-Split pipeline
    └── stage3_evaluate_froc_cpm_and_plots.py              # FROC/CPM bootstrap evaluator & 600-DPI plot generator
```

---

## 4. Quick Start & Reproduction

### 4.1 Environment Setup
```bash
git clone https://github.com/Naderj23/LUNA16-3D-Geometric-Deep-Ensemble-IJCARS.git
cd LUNA16-3D-Geometric-Deep-Ensemble-IJCARS
pip install -r requirements.txt
```

### 4.2 Three-Stage Pipeline Execution
1. **Stage 1 — Isotropic $1\times 1\times 1\text{ mm}^3$ Resampling & Coordinate-Verified $48^3$ Patch Extraction:**
   ```bash
   python3 src/stage1_luna16_isotropic_preprocessing.py
   ```
2. **Stage 2 — 3D ResACS-Net (`CF-MSPP + OHEM`), 748D Physical Feature Extraction, and Regime I / Regime II / 133-Split Evaluation:**
   ```bash
   python3 src/stage2_train_3d_resacs_and_748d_descriptors.py
   ```
3. **Stage 3 — Scan-Level Cluster Bootstrap Verification & 600-DPI Publication Figures:**
   ```bash
   python3 src/stage3_evaluate_froc_cpm_and_plots.py
   ```

---

## 5. Citation

If you find this code, the 748D 3D differential-geometric descriptor bank, or our forensic LUNA16 benchmark audit useful in your research, please cite:

```bibtex
@article{Jafari2026LUNA16,
  author  = {Jafari, Nader},
  title   = {3D Differential-Geometric Descriptors and Volumetric Deep Ensembles for Pulmonary Nodule Classification: A {LUNA16} Benchmark Audit},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year    = {2026},
  note    = {Member of the Computer Society of Iran (CSI), Tehran, Iran; ORCID: 0009-0008-6849-3207}
}
```
