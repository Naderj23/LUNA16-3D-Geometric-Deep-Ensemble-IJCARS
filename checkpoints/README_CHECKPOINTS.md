# Trained Model Checkpoints & Out-of-Fold Prediction Archives

All outer-fold and locked-split checkpoints are cryptographically verified by `verify_fold_checkpoint_integrity()` and `verify_empirical_mhd_checkpoint_integrity()` in `src/LUNA16_Final_Reproducible_Implementation.py`:

- `v106_outer_fold_01.npz` .. `v106_outer_fold_10.npz`: Outer-fold out-of-fold predictions (`oof_a0_9fam`, `oof_a1_e2e`, `oof_a1_cnn`, `oof_v105_5fam`), 506D feature mask, and ensemble weights across all 888 scans (`10x5` nested CV).
- `v106_protocol_a2_predictions.npz`: Locked `622/133/133` split predictions (`p_a2_9fam_va`, `p_a2_e2e_va`, `p_a2_9fam_te`, `p_a2_e2e_te`).
- `v106_part_b2_25scan_mhd_pilot.npz`: Label-blind 25-scan raw `.mhd` pilot Stage-0 (`64D`) and Stage-1/2 (`1,195D`) predictions (`20,984` `candidates_V2.csv` rows).
