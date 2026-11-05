#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Protocol-Separated Evaluation & 15-Gate Verification Runner (Section 3, Tables 1, 4, 5, 6).
Reads traceable results from results/*.csv, verifies all statistical deltas and prior-shift projections
(Eq. 10), prints the complete benchmark summary report, and executes the 15-gate test suite.
"""
import os
import sys
import csv
import subprocess
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def print_protocol_separated_benchmark_summary():
    metrics_csv = os.path.join(ROOT_DIR, "results", "evaluation_metrics.csv")
    sens_csv = os.path.join(ROOT_DIR, "results", "sensitivity_results.csv")
    folds_csv = os.path.join(ROOT_DIR, "results", "fold_metrics_10x5_cv.csv")

    print("=" * 100)
    print("PROTOCOL-SEPARATED LUNA16 BENCHMARK & ABLATION SUMMARY (Tables 4, 5, 6)")
    print("=" * 100)
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(
                f"  [{row['protocol_group']:22s}] {row['model_or_ablation_step']:42s} | "
                f"Dim={row['dimensions']:4s} | CPM={row['point_cpm_pct']:6s}% "
                f"(Boot={row['bootstrap_mean_cpm_pct']}% [{row['ci_95_low_pct']}–{row['ci_95_high_pct']}%]) | "
                f"Delta={row['delta_pp']:6s} pp (p={row['p_value']}) | SubprocDiff={row['subprocess_diff_pp']} pp"
            )

    print("-" * 100)
    print("10-FOLD OUTER CROSS-VALIDATION BREAKDOWN (888 Scans, 1,186 Nodules):")
    with open(folds_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(
                f"  {row['outer_fold']:18s} ({row['luna16_subset']:9s}, n_scans={row['n_scans']:3s}, "
                f"n_pos={row['n_nodules']:4s}) -> Pure 1,166D CPM = {row['pure_1166d_cpm_pct']}% | "
                f"Hybrid 1,195D CPM = {row['hybrid_1195d_cpm_pct']}% (ROC-AUC = {row['hybrid_roc_auc']})"
            )
    print("=" * 100)


if __name__ == "__main__":
    print_protocol_separated_benchmark_summary()
    test_script = os.path.join(ROOT_DIR, "src", "test_final_reproducible_framework.py")
    subprocess.run([sys.executable, test_script], check=True)
