"""Tests for Milestone 3: Cyber Attacks and Anomaly Detectors."""
import numpy as np
import pytest

from sim.simulator import Simulator
from twin.estimator import TwinEstimator
from attacks.spoof_jump import SpoofJumpAttack
from attacks.spoof_drift import SpoofDriftAttack
from attacks.jamming import GPSJammingAttack
from detect.fusion import AnomalyDetectorSuite


def simulate_with_attack(seed: int, attack, total_time: float = 120.0):
    """Helper to simulate flight under a specific attack and evaluate detectors."""
    sim = Simulator(seed=seed, total_time=total_time)
    log = sim.run_clean_flight()

    estimator = TwinEstimator(dt=sim.dt)
    detector_suite = AnomalyDetectorSuite()
    statuses = []

    gps_idx = 0
    gps_measurements = log.gps_measurements
    n_gps = len(gps_measurements)

    for k in range(len(log.times)):
        t = log.times[k]
        raw_imu = log.imu_measurements[k]

        # Apply attack to IMU
        imu_meas = attack.apply_imu(t, raw_imu) if attack else raw_imu
        estimator.predict(imu_meas)

        # GPS step
        if gps_idx < n_gps and abs(gps_measurements[gps_idx].timestamp - t) < 1e-4:
            raw_gps = gps_measurements[gps_idx]
            gps_idx += 1

            # Apply attack to GPS
            gps_meas = attack.apply_gps(t, raw_gps) if attack else raw_gps
            rec = estimator.update_gps(gps_meas)

            status = detector_suite.process(t, gps_meas, rec)
            statuses.append(status)

    return log, statuses


def test_spoof_jump_detection():
    """Verify Chi-Square detector fires quickly on 40m GPS jump attack."""
    attack = SpoofJumpAttack(start_time=60.0, jump_distance=40.0)
    _, statuses = simulate_with_attack(seed=42, attack=attack)

    # Filter statuses after attack start
    post_attack = [s for s in statuses if s.timestamp >= 60.0]
    fired = [s for s in post_attack if s.chi2_alarm]

    assert len(fired) > 0, "Chi-Square failed to detect GPS jump attack"
    first_detect_time = fired[0].timestamp
    detection_delay = first_detect_time - 60.0
    print(f"\n[Spoof Jump] Detected in {detection_delay:.2f} s")
    assert detection_delay <= 1.5, f"Detection took too long: {detection_delay} s"


def test_spoof_drift_detection_vs_baseline():
    """Core hypothesis verification:
    CUSUM detects slow ramp drift (0.8 m/s), whereas Naive Baseline completely misses it!
    """
    attack = SpoofDriftAttack(start_time=60.0, drift_rate=0.8, stealth_velocity=True)
    _, statuses = simulate_with_attack(seed=42, attack=attack)

    post_attack = [s for s in statuses if s.timestamp >= 60.0]

    cusum_fired = [s for s in post_attack if s.cusum_alarm]
    baseline_fired = [s for s in post_attack if s.baseline_alarm]

    assert len(cusum_fired) > 0, "CUSUM failed to detect slow spoof drift"
    first_cusum_time = cusum_fired[0].timestamp
    cusum_delay = first_cusum_time - 60.0
    print(f"\n[Spoof Drift 0.8m/s] CUSUM detected in {cusum_delay:.2f} s")
    assert cusum_delay <= 10.0, f"CUSUM took too long: {cusum_delay} s"

    # Baseline MUST NOT detect stealthy drift
    assert len(baseline_fired) == 0, f"Baseline should miss stealthy drift, but fired {len(baseline_fired)} times"


def test_jamming_dropout_detection():
    """Verify GPS jamming / loss of signal is detected promptly."""
    attack = GPSJammingAttack(start_time=60.0, duration=15.0, mode="dropout")
    _, statuses = simulate_with_attack(seed=42, attack=attack)

    post_attack = [s for s in statuses if s.timestamp >= 60.0]
    jamming_fired = [s for s in post_attack if s.jamming_alarm]

    assert len(jamming_fired) > 0, "Jamming detector failed to fire on GPS dropout"
    first_jam_time = jamming_fired[0].timestamp
    jam_delay = first_jam_time - 60.0
    print(f"\n[GPS Jamming] Detected in {jam_delay:.2f} s")
    assert jam_delay <= 1.2, f"Jamming detection took too long: {jam_delay} s"


def test_clean_seeds_false_alarm_rate():
    """Evaluate false alarms on clean flights across tuning seeds 0-49."""
    false_alarm_count = 0
    total_flights = 50

    for seed in range(total_flights):
        _, statuses = simulate_with_attack(seed=seed, attack=None, total_time=60.0)
        # Check if any twin alarm fired during clean flight
        alarms = [s for s in statuses[5:] if s.twin_active_alarm]
        if len(alarms) > 0:
            false_alarm_count += 1

    fa_rate = false_alarm_count / total_flights
    print(f"\n[M3 Clean Seeds False Alarm Rate] {false_alarm_count}/{total_flights} = {fa_rate * 100:.1f}%")
    assert fa_rate <= 0.02, f"False alarm rate too high: {fa_rate * 100:.1f}%"
