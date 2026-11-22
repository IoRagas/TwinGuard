"""GPS Jamming Attack: Complete signal loss or massive noise burst."""
from typing import Optional
import numpy as np

from attacks.base import Attack
from sim.sensors import GPSMeasurement, IMUMeasurement


class GPSJammingAttack(Attack):
    """Simulates GPS jamming via denial-of-service (dropout) or barrage noise."""

    def __init__(
        self,
        start_time: float = 60.0,
        duration: float = 15.0,
        mode: str = "dropout",           # "dropout" or "noise_burst"
        noise_burst_std: float = 50.0,   # 50m noise if in noise_burst mode
        seed: int = 42,
    ):
        super().__init__(name="jamming", start_time=start_time, duration=duration)
        self.mode = mode
        self.noise_burst_std = noise_burst_std
        self.rng = np.random.default_rng(seed)

    def apply_gps(self, timestamp: float, meas: Optional[GPSMeasurement]) -> Optional[GPSMeasurement]:
        if meas is None or not self.is_active(timestamp):
            return meas

        if self.mode == "dropout":
            # Signal denial: measurement packet marked invalid or dropped
            return GPSMeasurement(
                timestamp=meas.timestamp,
                pos=meas.pos.copy(),
                vel=meas.vel.copy(),
                valid=False,
            )
        elif self.mode == "noise_burst":
            # Extreme radio interference causing huge variance
            noise = self.rng.normal(0.0, self.noise_burst_std, size=2)
            return GPSMeasurement(
                timestamp=meas.timestamp,
                pos=meas.pos + noise,
                vel=meas.vel + self.rng.normal(0.0, 5.0, size=2),
                valid=True,
            )
        return meas

    def apply_imu(self, timestamp: float, meas: IMUMeasurement) -> IMUMeasurement:
        return meas
