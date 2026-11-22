"""IMU Accelerometer Drift Attack: Ramping sensor bias."""
from typing import Optional
import numpy as np

from attacks.base import Attack
from sim.sensors import GPSMeasurement, IMUMeasurement


class IMUDriftAttack(Attack):
    """Simulates internal hardware fault or sensor spoofing on IMU."""

    def __init__(
        self,
        start_time: float = 60.0,
        duration: Optional[float] = None,
        bias_drift_rate: float = 0.05,  # (m/s^2)/s
    ):
        super().__init__(name="imu_drift", start_time=start_time, duration=duration)
        self.bias_drift_rate = bias_drift_rate

    def apply_gps(self, timestamp: float, meas: Optional[GPSMeasurement]) -> Optional[GPSMeasurement]:
        return meas

    def apply_imu(self, timestamp: float, meas: IMUMeasurement) -> IMUMeasurement:
        if not self.is_active(timestamp):
            return meas

        elapsed = timestamp - self.start_time
        bias_offset = np.array([self.bias_drift_rate * elapsed, 0.0])
        return IMUMeasurement(
            timestamp=meas.timestamp,
            accel=meas.accel + bias_offset,
        )
