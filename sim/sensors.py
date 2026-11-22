"""Sensor models for 5 Hz GPS and 50 Hz IMU."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GPSMeasurement:
    """GPS measurement packet."""
    timestamp: float
    pos: np.ndarray      # [x, y] in meters
    vel: np.ndarray      # [vx, vy] in m/s
    valid: bool = True   # False if dropped or jammed


@dataclass
class IMUMeasurement:
    """IMU measurement packet."""
    timestamp: float
    accel: np.ndarray    # [ax, ay] in m/s^2


class SensorSuite:
    """Simulates 50 Hz IMU and 5 Hz GPS sensors with realistic noise and bias."""

    def __init__(
        self,
        seed: int = 42,
        gps_rate_hz: float = 5.0,
        imu_rate_hz: float = 50.0,
        gps_pos_noise_std: float = 1.5,
        gps_vel_noise_std: float = 0.1,
        imu_accel_noise_std: float = 0.05,
        imu_bias_std: float = 0.02,
    ):
        self.rng = np.random.default_rng(seed)
        self.gps_period = 1.0 / gps_rate_hz
        self.imu_period = 1.0 / imu_rate_hz

        self.gps_pos_noise_std = gps_pos_noise_std
        self.gps_vel_noise_std = gps_vel_noise_std
        self.imu_accel_noise_std = imu_accel_noise_std

        # Small constant IMU bias per flight
        self.imu_bias = self.rng.normal(0.0, imu_bias_std, size=2)

        self._last_gps_time = -1.0
        self._last_imu_time = -1.0

    def sample_imu(self, timestamp: float, true_accel: np.ndarray) -> IMUMeasurement:
        """Sample IMU accelerometer at 50 Hz."""
        noise = self.rng.normal(0.0, self.imu_accel_noise_std, size=2)
        measured_accel = true_accel + self.imu_bias + noise
        self._last_imu_time = timestamp
        return IMUMeasurement(timestamp=timestamp, accel=measured_accel)

    def sample_gps(
        self,
        timestamp: float,
        true_pos: np.ndarray,
        true_vel: np.ndarray,
        force: bool = False,
    ) -> Optional[GPSMeasurement]:
        """Sample GPS receiver at 5 Hz. Returns None between GPS epochs."""
        if not force and (timestamp - self._last_gps_time) < (self.gps_period - 1e-6):
            return None

        pos_noise = self.rng.normal(0.0, self.gps_pos_noise_std, size=2)
        vel_noise = self.rng.normal(0.0, self.gps_vel_noise_std, size=2)

        self._last_gps_time = timestamp
        return GPSMeasurement(
            timestamp=timestamp,
            pos=true_pos + pos_noise,
            vel=true_vel + vel_noise,
            valid=True,
        )
