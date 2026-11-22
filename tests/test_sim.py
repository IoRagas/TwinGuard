"""Tests for Milestone 1: Simulation, sensors, wind, and mission."""
import numpy as np
import pytest
from pathlib import Path

from sim.simulator import Simulator
from sim.drone import Drone
from sim.wind import WindModel
from sim.sensors import SensorSuite


def test_determinism():
    """Verify that identical seeds produce identical simulation outputs."""
    sim1 = Simulator(seed=42, total_time=30.0)
    log1 = sim1.run_clean_flight()

    sim2 = Simulator(seed=42, total_time=30.0)
    log2 = sim2.run_clean_flight()

    assert np.allclose(log1.true_pos, log2.true_pos, atol=1e-12)
    assert np.allclose(log1.true_vel, log2.true_vel, atol=1e-12)
    assert np.allclose(log1.wind, log2.wind, atol=1e-12)
    assert len(log1.gps_measurements) == len(log2.gps_measurements)
    assert np.allclose(log1.gps_measurements[0].pos, log2.gps_measurements[0].pos, atol=1e-12)


def test_seed_variance():
    """Verify that different seeds produce different wind and noise trajectories."""
    sim1 = Simulator(seed=42, total_time=10.0)
    log1 = sim1.run_clean_flight()

    sim2 = Simulator(seed=999, total_time=10.0)
    log2 = sim2.run_clean_flight()

    assert not np.allclose(log1.wind, log2.wind)
    assert not np.allclose(log1.gps_measurements[0].pos, log2.gps_measurements[0].pos)


def test_speed_and_accel_limits():
    """Verify drone speed and acceleration limits are strictly obeyed."""
    sim = Simulator(seed=123, total_time=60.0, max_speed=10.0, max_accel=4.0)
    log = sim.run_clean_flight()

    speeds = np.linalg.norm(log.true_vel, axis=1)
    assert np.all(speeds <= 10.0 + 1e-4), f"Max speed violated: {np.max(speeds)}"

    # Check that control commands respect max_accel
    cmd_accels = np.linalg.norm(log.cmd_accel, axis=1)
    assert np.all(cmd_accels <= 4.0 + 1e-4), f"Max command accel violated: {np.max(cmd_accels)}"


def test_sensor_sampling_frequencies():
    """Verify that IMU runs at 50 Hz and GPS runs at 5 Hz."""
    total_time = 20.0
    dt = 0.02
    sim = Simulator(seed=42, dt=dt, total_time=total_time)
    log = sim.run_clean_flight()

    expected_imu_count = int(total_time / dt)
    assert len(log.imu_measurements) == expected_imu_count

    expected_gps_count = int(total_time * 5.0)
    assert abs(len(log.gps_measurements) - expected_gps_count) <= 1


def test_patrol_route_completion():
    """Verify that drone traverses the 400x300m rectangle and approaches home."""
    sim = Simulator(seed=42, total_time=180.0)
    log = sim.run_clean_flight()

    # Max extent in X should reach ~400m, Y should reach ~300m
    max_x = np.max(log.true_pos[:, 0])
    max_y = np.max(log.true_pos[:, 1])
    assert max_x >= 380.0, f"Drone did not reach eastern patrol limit: {max_x}m"
    assert max_y >= 280.0, f"Drone did not reach northern patrol limit: {max_y}m"

    # Final position at t=180s should be close to home (0,0)
    final_pos = log.true_pos[-1]
    final_dist_home = np.linalg.norm(final_pos)
    assert final_dist_home < 15.0, f"Drone failed to return home: distance is {final_dist_home}m"
