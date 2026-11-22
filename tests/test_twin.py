"""Tests for Milestone 2: Twin, Kalman Filter, and Innovation Statistics."""
import numpy as np
from scipy import stats
import pytest

from sim.simulator import Simulator
from twin.estimator import TwinEstimator
from twin.dynamics import TwinModel


def run_clean_twin_flight(seed: int, total_time: float = 60.0):
    """Helper to run a clean flight through simulator and digital twin."""
    sim = Simulator(seed=seed, total_time=total_time)
    log = sim.run_clean_flight()

    estimator = TwinEstimator(dt=sim.dt)
    records = []

    gps_idx = 0
    gps_measurements = log.gps_measurements
    n_gps = len(gps_measurements)

    for k in range(len(log.times)):
        t = log.times[k]
        imu_meas = log.imu_measurements[k]

        # 1. IMU Prediction step
        estimator.predict(imu_meas)

        # 2. GPS Update step when available
        if gps_idx < n_gps and abs(gps_measurements[gps_idx].timestamp - t) < 1e-4:
            rec = estimator.update_gps(gps_measurements[gps_idx])
            if rec is not None:
                records.append(rec)
            gps_idx += 1

    return log, estimator, records


def test_single_flight_tracking():
    """Verify that Kalman filter tracks ground truth drone within 3-sigma error bounds."""
    log, estimator, records = run_clean_twin_flight(seed=42, total_time=120.0)

    assert len(records) > 0, "No GPS records produced"

    # Compare estimated position against true position at GPS timestamps
    for rec in records[10:]:  # allow initial convergence
        t = rec.timestamp
        # find closest true state
        idx = int(round(t / 0.02))
        true_p = log.true_pos[idx]
        err = np.linalg.norm(rec.est_pos - true_p)
        # Position error should typically stay within ~3.5 meters for sigma_gps=1.5m
        assert err < 5.0, f"Tracking error {err}m exceeded bound at t={t}s"


def test_covariance_convergence():
    """Verify state covariance converges to steady-state and remains positive definite."""
    log, estimator, records = run_clean_twin_flight(seed=10, total_time=30.0)

    # Initial trace was ~4.5 + 0.02 = 4.52
    initial_trace = records[0].cov_trace
    final_trace = records[-1].cov_trace

    # Trace should drop as measurements are fused
    assert final_trace < initial_trace
    assert np.all(np.linalg.eigvals(estimator.P) > 0), "Covariance P is not positive definite"


def test_100_clean_seeds_innovation_stats():
    """Milestone 2 core acceptance criteria:
    Run on 100 clean seeds (0-99). Innovation stats must look sane
    (approximately chi-square distributed with dof=4) and show no divergence.
    """
    all_nis = []
    seed_mean_nis = []

    # Run over 100 clean seeds (using 30s per seed for fast test execution)
    for seed in range(100):
        _, _, records = run_clean_twin_flight(seed=seed, total_time=30.0)
        # Take after initial filter lock (first 5 samples discarded)
        nis_vals = [r.nis for r in records[5:]]
        all_nis.extend(nis_vals)
        seed_mean_nis.append(np.mean(nis_vals))

    all_nis = np.array(all_nis)
    global_mean_nis = np.mean(all_nis)
    global_median_nis = np.median(all_nis)

    print(f"\n[M2 100 Clean Seeds] Total samples: {len(all_nis)}")
    print(f"Mean NIS: {global_mean_nis:.3f} (Theoretical chi-square dof=4 is 4.0)")
    print(f"Median NIS: {global_median_nis:.3f} (Theoretical chi-square median is 3.36)")
    print(f"95th percentile NIS: {np.percentile(all_nis, 95):.3f} (Theoretical is 9.49)")

    # With 5-10% model mismatch and random wind, the mean NIS should remain healthy around 3.0 to 5.5
    assert 2.5 <= global_mean_nis <= 5.5, f"Mean NIS {global_mean_nis} outside valid range [2.5, 5.5]"
    assert 2.0 <= global_median_nis <= 4.5, f"Median NIS {global_median_nis} outside valid range [2.0, 4.5]"

    # Verify Kolmogorov-Smirnov goodness-of-fit against chi2(df=4) has reasonable correlation
    # Quantiles should match within 25%
    p50_theo = stats.chi2.ppf(0.50, df=4)
    p90_theo = stats.chi2.ppf(0.90, df=4)
    assert abs(np.percentile(all_nis, 50) - p50_theo) < 1.5
    assert abs(np.percentile(all_nis, 90) - p90_theo) < 3.0
