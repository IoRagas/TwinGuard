"""Naive fixed-threshold baseline detector for direct comparison with TwinGuard."""
from typing import Optional, Tuple
import numpy as np

from sim.sensors import GPSMeasurement


class BaselineDetector:
    """Fixed-threshold detector monitoring point-to-point GPS discontinuities.
    
    Catches large instant step jumps, but is blind to smooth slow drifts
    and correlated spoofing profiles.
    """

    def __init__(
        self,
        max_pos_jump: float = 15.0,  # Alarm if position jump > 15m between 0.2s epochs
        max_vel_jump: float = 4.0,   # Alarm if velocity jump > 4 m/s
    ):
        self.max_pos_jump = max_pos_jump
        self.max_vel_jump = max_vel_jump
        self.last_pos: Optional[np.ndarray] = None
        self.last_vel: Optional[np.ndarray] = None
        self.statistic: float = 0.0

    def reset(self):
        self.last_pos = None
        self.last_vel = None
        self.statistic = 0.0

    def update(self, meas: Optional[GPSMeasurement]) -> Tuple[bool, float]:
        """Check for jump anomaly on raw GPS readings."""
        if meas is None or not meas.valid:
            return False, 0.0

        if self.last_pos is None:
            self.last_pos = meas.pos.copy()
            self.last_vel = meas.vel.copy()
            return False, 0.0

        delta_pos = float(np.linalg.norm(meas.pos - self.last_pos))
        delta_vel = float(np.linalg.norm(meas.vel - self.last_vel))

        self.last_pos = meas.pos.copy()
        self.last_vel = meas.vel.copy()

        # Statistic is maximum normalized jump ratio
        pos_ratio = delta_pos / self.max_pos_jump
        vel_ratio = delta_vel / self.max_vel_jump
        self.statistic = max(pos_ratio, vel_ratio)

        is_alarm = (delta_pos > self.max_pos_jump) or (delta_vel > self.max_vel_jump)
        return is_alarm, self.statistic
