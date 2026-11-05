# 3D Differential-Geometric and Spherical-Harmonic Feature Fusion with Multi-Task CoordConv Networks for Pulmonary Nodule False-Positive Reduction

[![Journal: IJCARS (Springer Nature)](https://img.shields.io/badge/Journal-IJCARS%20(Springer%20Nature)-0F2544)](https://link.springer.com/journal/11548)
[![Verification Suite: 15/15 Gates Passing](https://img.shields.io/badge/Verification%20Gates-15%2F15%20PASSED-15803D)](src/test_final_reproducible_framework.py)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-1D4ED8)](requirements.txt)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-0F766E)](LICENSE)
[![ORCID: 0009-0008-6849-3207](https://img.shields.io/badge/ORCID-0009--0008--6849--3207-A6CE39)](https://orcid.org/0009-0008-6849-3207)

Official Three-Layer Reproducible Research Repository for the manuscript:

> **Nader Jafari\*** (2026). *"3D Differential-Geometric and Spherical-Harmonic Feature Fusion with Multi-Task CoordConv Networks for Pulmonary Nodule False-Positive Reduction: Curated LUNA16 Evaluation and Raw-CT Audit."* Submitted to the **International Journal of Computer Assisted Radiology and Surgery (IJCARS)**, Springer Nature.
> \*Member of the Computer Society of Iran (CSI), Tehran, Iran | Email: `Nader.jafari23@gmail.com` | ORCID: [0009-0008-6849-3207](https://orcid.org/0009-0008-6849-3207)

---

## 1. Architectural Overview & Cohort Stratification (Figure 1)

![Figure 1: LUNA16 Cohort Stratification and Proposed 4-Stage Hybrid 3D Architecture](figures/Fig1_IJCARS_Data_and_Pipeline_600dpi.png)

Our framework resolves two core challenges in thoracic CT false-positive reduction (FPRED):
1. **Strict Separation of Evaluation Regimes & Prior-Shift Accounting (`Eqs. 1, 10`)**: We formally quantify the $84.15\times$ (`candidates_V2.csv`: `754,975` rows, `848.4 neg/scan`) and $61.34\times$ (`candidates.csv`: `551,065` rows, `619.1 neg/scan`) negative-density ratios relative to the curated coordinate-verified cohort (`LUNA16_Patches_CoordVerified_48.h5`: `10,066` patches, `1,186` positives, `8,880` negatives = `10.0 neg/scan`), preventing conflation of curated-cohort CPMs with full-pool FPRED tracks.
2. **Synergistic 3D Geometric-Harmonic (`1,166D`) and Inner-Fold Multi-Task CoordConv (`29D`) Fusion (`Eqs. 2–9`)**: Nine segmentation-free 3D physical, differential-geometric ($3\times 3$ Cardano Hessian invariants), Dual-QR real spherical-harmonic ($\ell \in \{0..4\}$ on the 26-ray lattice), inward radial-gradient flux, and topological veto families (`1,166D`) are fused with a leak-free inner 5-fold 6-channel `MultiWindowCoordConv3DResNeXtWithDiameterHead` (`12³/24³/48³` + box-diff + GGN window + 3D radial CoordConv, with core-vs.-outer pooling and auxiliary Huber $\log(1+D)$ regression) and a 9-channel 2.5D Tri-Planar MIP ConvNet (`29D` OOF context; $D_{\text{hybrid}} = 1,195\text{D}$).

---

## 2. Three-Layer Repository Structure

| Layer | Repository Directory & Key Files | Scientific Role |
| :--- | :--- | :--- |
| **Layer 1: Final Manuscript & Artwork** | `docs/LUNA16_Hybrid_3D_CAD_Final_Manuscript.pdf` (`.docx`)<br>`docs/MS_ResACS_3DNet_V3_Blinded_Only.docx`<br>`figures/Fig1..Fig7` (`600 DPI PNG` + `Vector PDF`) | Complete IJCARS manuscript (`4,984` total words; `222`-word structured abstract; 10 editable OMML equations; 6 Booktabs tables; 7 figures; 30 verified Springer Nature references). |
| **Layer 2: Reproducible Code & 15-Gate Test Suite** | `src/LUNA16_Final_Reproducible_Implementation.py`<br>`src/stage4_full_551065_unscreened_evaluator.py`<br>`src/test_final_reproducible_framework.py`<br>`src/generate_final_7_figures.py` | Self-contained PyTorch/NumPy/SciPy implementation and **15-gate automated verification suite** (`15/15 PASSED` in `< 1.2s`). |
| **Layer 3: Supplementary Material & Ablation Ledger** | `docs/ESM_1.pdf` (`Online_Resource_1_Supplementary_Material.pdf`)<br>`docs/IJCARS_Title_Page.pdf` & `docs/IJCARS_Cover_Letter.pdf` | Extended 7-point FROC comparison (`Table S1`), cryptographic SHA-256 & 286-patch `.mhd` vs. HDF5 audit (`Table S2`), subprocess FROC equivalence (`Table S3`), development & ablation ledger (`Table S4`), and 15-gate inventory (`Table S5`). |

---

## 3. Verified Empirical Results Across Separated Evaluation Regimes

| Evaluation Regime & Protocol | Evaluated Model / Stage | Dimensions | Official Point CPM (%) | Bootstrap Mean [95% CI] | ROC-AUC | External Subprocess Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime 1: Curated All-888 (`10×5 Nested CV`)** | Standalone 6-Ch CoordConv 3D CNN | `29D` | `67.48%` | `67.51% [65.10%–69.85%]` | `0.9312` | `0.0000 pp` |
| **Regime 1: Curated All-888 (`10×5 Nested CV`)** | Classical 5-Family 3D Baseline (`Zero CNN`) | `742D` | `88.19%` | `88.21% [86.52%–89.81%]` | `0.9745` | `0.0000 pp` |
| **Regime 1: Curated All-888 (`10×5 Nested CV`)** | **Proposed Pure 9-Family 3D (`Zero CNN`)** | **`1,166D`** | **`91.94%`** | **`91.97% [90.51%–93.38%]`** | **`0.9835`** | **`0.0000 pp`** |
| **Regime 1: Curated All-888 (`10×5 Nested CV`)** | **Proposed Hybrid End-to-End (`9-Fam + Deep`)** | **`1,195D`** | **`92.33%`** | **`92.34% [90.91%–93.66%]`** | **`0.9841`** | **`0.0000 pp`** |
| **Regime 1: Locked Validation Split (`133 Scans`)** | **Pure 9-Family / Hybrid End-to-End** | `1,166D / 1,195D` | **`93.83% / 93.15%`** | `93.66% / 93.43%` | `0.9869 / 0.9847` | `0.0000 pp` |
| **Regime 1: Locked Hold-Out Test Split (`133 Scans`)** | **Pure 9-Family / Hybrid End-to-End** | `1,166D / 1,195D` | **`89.91% / 90.25%`** | `90.02% / 90.35%` | `0.9795 / 0.9808` | `0.0000 pp` |
| **Regime 3: Raw `.mhd` 25-Scan Blind Pilot (`20,984 Rows`)** | 64D Dual-View Stage-0 Screener (`Top-80: 88.89%`) | `64D` | `53.97%` | `54.02% [45.80%–62.10%]` | — | `0.0000 pp` |
| **Regime 3: Raw `.mhd` 25-Scan Blind Pilot (`20,984 Rows`)** | **Proposed Two-Stage Re-Centered Cascade** | **`64D -> 1,195D`** | **`76.19%`** | **`76.21% [68.40%–83.60%]`** | — | **`0.0000 pp`** |

---

## 4. Quickstart: Running the 15-Gate Verification Suite & Reproducing Figures

### A. Run the 15-Gate Unit, Integration & Manuscript-Reconciliation Test Suite (`< 1.2s`)
```bash
pip install -r requirements.txt
python3 src/test_final_reproducible_framework.py
```

All 15 verification gates (`Gate 00` through `Gate 14`) validate:
- **Gate 00**: AST syntax & architectural symbol presence (`Stage0GPUMultiScaleScreener`, `MultiWindowCoordConv3DResNeXtWithDiameterHead`, `FoldIsolatedTriPlanarMIPConvNet`, etc.).
- **Gate 01**: Exact $3\times 3$ affine `.mhd` world-to-voxel and inverse mapping ($< 10^{-11}\text{ mm}$ error, `Eq. 1`).
- **Gate 02**: Scale-normalized $3\times 3$ Cardano Hessian eigenvalues vs. `numpy.linalg.eigh` ($< 10^{-10}$) and $O(3)$ axis-permutation invariance (`Eqs. 2–3`).
- **Gate 03**: Dual-QR real spherical harmonics ($\ell=0..4$, 25 modes) orthonormality ($< 10^{-12}$) and $O_h$ reflection invariance (`Eqs. 4–5`).
- **Gate 04**: Exact 9-family dimensionality (`126 + 125 + 229 + 131 + 129 = 740D`; $D_{\text{pure}} = 1,166\text{D}$; $D_{\text{hybrid}} = 1,195\text{D}$; $D_{\text{spec}} = 506\text{D}$, `Eq. 8`).
- **Gate 05**: Bit-for-bit `0.00e+00` float32 equivalence of 64D Stage-0 descriptors.
- **Gate 06**: Sphericity-gated 3D foveal centroid shift recovery ($\Delta\mathbf{x}_{\text{fovea}}$, `Eq. 9`).
- **Gate 07**: Cardinality-matched $K=24$ scan-context invariance (`0.00e+00` shift under background candidate expansion, `Eq. 9`).
- **Gate 08**: Exact 8-view cube reflection symmetry of 3D CoordConv and 2.5D MIP tensors.
- **Gate 09**: Strict monotonic cascade ordering ($\min P[\text{Top-80}] > \max P[\text{Rest}]$).
- **Gate 10**: Deterministic `Seed=42` label-blind `.mhd` scan selector without annotation lookup.
- **Gate 11**: Aggregated per-patch `hashlib.sha256` checksum digest and fail-closed cache rejection.
- **Gate 12**: 1-to-1 numerical reconciliation across all manuscript tables (`Tables 1–6`) and supplementary tables (`Tables S1–S4`).
- **Gate 13**: End-to-end $< 0.5 D$ 3D Euclidean FROC matching and fail-closed `RuntimeError` gate on missing subprocess output (`Eq. 10`).
- **Gate 14**: Full manuscript reconciliation (`Eqs. 1–10`, `Tables 1–6`, `Figs. 1–7`, `<= 5,000` total words, and strict sequential citation ordering `[1]..[30]`).

### B. Regenerate All 7 Publication Figures (`600 DPI PNG` & `Vector PDF`)
```bash
python3 src/generate_final_7_figures.py
```

---

## 5. Cryptographic Data Provenance (SHA-256 Digests)

- **`LUNA16_Patches_CoordVerified_48.h5` (`10,066` Patches, $48^3\text{ mm}^3$)**:
  - Unrounded IEEE-754 `float32` Voxel SHA-256: `48f009e851d23da1ae3ee72fa2bd4a37146a975264f0f027415a7fe15a7e97bc`
  - Aggregated Per-Patch SHA-256 Checksum Digest: `90f6167ae46af384a685c9293da86e9b510859d119b05450ad94e3bb564c78f7`
  - Canonical 5-Decimal Voxel + Metadata SHA-256: `a1a2d656047040a9c5e36406649e84250ca5cf93740e14e0b0443a438fa8c3e5`
- **25-Scan Label-Blind Raw `.mhd` Pilot Cohort (`20,984` `candidates_V2.csv` Rows, `Seed=42`)**:
  - Candidate Pool SHA-256 Prefix: `50a5ca18dbba7e48`
