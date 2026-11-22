"""Chi-Square Statistical Anomaly Detector (Sliding Window NIS)."""
from collections import deque
from typing import Tuple
import numpy as np


class ChiSquareDetector:
    """Sliding-window normalized innovation squared (NIS) detector.
    
    Under H0 (normal operation), the sum of W independent NIS values (dim 4)
    follows a chi-square distribution with W * 4 degrees of freedom.
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 52.0,  # Tuned on clean seeds 0-99 (approx P_FA < 1e-4)
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.statistic = 0.0

    def reset(self):
        self.window.clear()
        self.statistic = 0.0

    def update(self, nis: float) -> Tuple[bool, float]:
        """Update window with new NIS and check for threshold crossing."""
        self.window.append(nis)
        # Compute sliding window sum
        self.statistic = float(sum(self.window))
        is_alarm = len(self.window) >= self.window_size and (self.statistic > self.threshold)
        return is_alarm, self.statistic
