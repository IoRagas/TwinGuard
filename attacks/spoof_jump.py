"""GPS Spoof Jump Attack: Sudden offset injection."""
from typing import Optional
import numpy as np

from attacks.base import Attack
from sim.sensors import GPSMeasurement, IMUMeasurement


class SpoofJumpAttack(Attack):
    """Injects an abrupt position step jump into GPS telemetry."""

    def __init__(
        self,
        start_time: float = 60.0,
        duration: Optional[float] = None,
        offset: Optional[np.ndarray] = None,
        jump_distance: float = 40.0,
        jump_heading_deg: float = 45.0,
    ):
        super().__init__(name="spoof_jump", start_time=start_time, duration=duration)
        if offset is not None:
            self.offset = np.array(offset, dtype=float)
        else:
            rad = np.radians(jump_heading_deg)
            self.offset = np.array([
                jump_distance * np.cos(rad),
                jump_distance * np.sin(rad)
            ], dtype=float)

    def apply_gps(self, timestamp: float, meas: Optional[GPSMeasurement]) -> Optional[GPSMeasurement]:
        if meas is None or not self.is_active(timestamp):
            return meas

        return GPSMeasurement(
            timestamp=meas.timestamp,
            pos=meas.pos + self.offset,
            vel=meas.vel.copy(),
            valid=meas.valid,
        )

    def apply_imu(self, timestamp: float, meas: IMUMeasurement) -> IMUMeasurement:
        # GPS spoofing does not alter IMU readings
        return meas
