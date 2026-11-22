"""GPS Spoof Drift Attack: Stealthy slow linear ramp offset."""
from typing import Optional
import numpy as np

from attacks.base import Attack
from sim.sensors import GPSMeasurement, IMUMeasurement


class SpoofDriftAttack(Attack):
    """Injects a slowly ramping position offset to mimic stealthy spoofing."""

    def __init__(
        self,
        start_time: float = 60.0,
        duration: Optional[float] = None,
        drift_rate: float = 0.8,              # m/s drift velocity magnitude
        drift_heading_deg: float = 90.0,      # degrees
        stealth_velocity: bool = True,        # spoof velocity as well so delta-vel baseline is fooled
    ):
        super().__init__(name="spoof_drift", start_time=start_time, duration=duration)
        self.drift_rate = drift_rate
        rad = np.radians(drift_heading_deg)
        self.drift_vel = np.array([
            drift_rate * np.cos(rad),
            drift_rate * np.sin(rad)
        ], dtype=float)
        self.stealth_velocity = stealth_velocity

    def apply_gps(self, timestamp: float, meas: Optional[GPSMeasurement]) -> Optional[GPSMeasurement]:
        if meas is None or not self.is_active(timestamp):
            return meas

        elapsed = timestamp - self.start_time
        pos_offset = self.drift_vel * elapsed

        new_vel = meas.vel.copy()
        if self.stealth_velocity:
            new_vel = new_vel + self.drift_vel

        return GPSMeasurement(
            timestamp=meas.timestamp,
            pos=meas.pos + pos_offset,
            vel=new_vel,
            valid=meas.valid,
        )

    def apply_imu(self, timestamp: float, meas: IMUMeasurement) -> IMUMeasurement:
        return meas
