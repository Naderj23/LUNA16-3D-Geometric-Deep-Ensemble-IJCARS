# LUNA16 Data Splits & Zero-Leakage Cross-Validation Protocol

1. **Official 10-Fold Scan-Level Assignment (`fold_0.csv` .. `fold_9.csv`)**:
   - Uses the official LUNA16 10-subset partitioning (`subset0` .. `subset9`, `888` thoracic CT scans, `1,186` reference nodules $\ge 3\text{ mm}$).
   - Every candidate patch in `LUNA16_Patches_CoordVerified_48.h5` (`10,066` patches) is strictly grouped by `seriesuid` so **zero candidates from the same patient/scan ever cross between training and validation/testing folds**.
   - Inside each outer training fold (`90%` of scans), all feature standardization, `SelectFromModel` (`506D`) selection, 5-fold inner neural cross-fitting (`C29 = 29D`), and 7-learner OOF stacking weights are fit strictly on the `90%` outer-training scans before scoring the held-out `10%` outer test fold.

2. **Locked `622 / 133 / 133` Train / Validation / Hold-Out Test Split (`locked_split_622_133_133_summary.csv`)**:
   - Deterministic scan-level partition (`Seed = 42`, zero scan overlap):
     * **Train-622**: `622` scans (`850` positives, `6,220` negatives)
     * **Val-133**: `133` scans (`169` positives, `1,330` negatives) — used for ensemble blend calibration
     * **Test-133**: `133` scans (`167` positives, `1,330` negatives) — evaluated once as a locked internal hold-out partition (`89.91%` Pure / `90.25%` Hybrid CPM, `ROC-AUC = 0.9808`).
   - *Note*: Because All-888 nested $10\times 5$ CV spans all 888 scans (which include the 133 scans of Test-133), the two protocols are reported as complementary internal analyses rather than independent external cohorts.
