"""Tests for Milestone 5: Evaluation Harness, Results CSV, and Metrics."""
from pathlib import Path
import pandas as pd
import pytest


def test_eval_artifacts_exist():
    """Verify evaluation outputs and plots are properly generated."""
    root = Path(__file__).resolve().parent.parent
    results_csv = root / "eval" / "results.csv"
    summary_table = root / "eval" / "summary_table.md"

    assert results_csv.exists(), "eval/results.csv missing"
    assert summary_table.exists(), "eval/summary_table.md missing"

    df = pd.read_csv(results_csv)
    assert len(df) == 200, f"Expected 200 evaluation runs, found {len(df)}"

    required_cols = [
        "seed", "scenario", "attack_type", "strength",
        "tg_detected", "tg_detection_delay", "tg_is_safe", "tg_final_dist_home",
        "baseline_detected", "baseline_detection_delay"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_eval_plots_exist():
    """Verify evaluation figures exist and have valid file sizes."""
    root = Path(__file__).resolve().parent.parent
    plots = [
        "eval/detection_rate_vs_drift_speed.png",
        "eval/time_to_detect_distribution.png",
        "eval/trajectories_comparison.png",
    ]
    for p in plots:
        file_path = root / p
        assert file_path.exists(), f"Missing plot: {p}"
        assert file_path.stat().st_size > 1000, f"Plot {p} appears empty (<1KB)"


def test_zero_leakage_seed_separation():
    """Verify reporting seeds strictly adhere to [1000, 1199] with zero tuning overlap."""
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "eval" / "results.csv")

    seeds = df["seed"].values
    assert np.all(seeds >= 1000), "Reporting seeds leaked below 1000"
    assert np.all(seeds <= 1199), "Reporting seeds leaked above 1199"
    assert len(set(seeds)) == 200, "Duplicate seed runs detected in reporting batch"


import numpy as np
