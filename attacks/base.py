"""Base class for sensor attack injection."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from sim.sensors import GPSMeasurement, IMUMeasurement


class Attack(ABC):
    """Abstract base class for all cyber-physical sensor attacks."""

    def __init__(self, name: str, start_time: float, duration: Optional[float] = None):
        self.name = name
        self.start_time = start_time
        self.duration = duration

    def is_active(self, timestamp: float) -> bool:
        """Return True if attack is active at the given timestamp."""
        if timestamp < self.start_time:
            return False
        if self.duration is not None and timestamp > (self.start_time + self.duration):
            return False
        return True

    @abstractmethod
    def apply_gps(self, timestamp: float, meas: Optional[GPSMeasurement]) -> Optional[GPSMeasurement]:
        """Apply attack modification to GPS measurement."""
        pass

    @abstractmethod
    def apply_imu(self, timestamp: float, meas: IMUMeasurement) -> IMUMeasurement:
        """Apply attack modification to IMU measurement."""
        pass
