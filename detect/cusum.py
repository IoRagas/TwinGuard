"""Cumulative Sum (CUSUM) Detector for persistent subtle innovation drift."""
from typing import Tuple
import numpy as np


class CUSUMDetector:
    """Vector CUSUM detector that accumulates small persistent innovation biases.
    
    Ideal for catching stealthy GPS spoofing ramps (0.2 - 2.0 m/s) that stay
    below single-sample chi-square alarm thresholds.
    """

    def __init__(
        self,
        allowance: float = 2.2,   # Slack parameter b; chosen above clean E[sqrt(NIS)]
        threshold: float = 9.0,   # Decision threshold h; tuned on seeds 0-99 (max clean was 6.64)
    ):
        self.allowance = allowance
        self.threshold = threshold
        self.cum_sum = 0.0

    def reset(self):
        self.cum_sum = 0.0

    def update(self, nis: float) -> Tuple[bool, float]:
        """Update CUSUM accumulation with latest NIS sample."""
        # Residual score: magnitude of normalized innovation vector
        score = np.sqrt(max(0.0, nis))
        self.cum_sum = max(0.0, self.cum_sum + (score - self.allowance))
        is_alarm = self.cum_sum > self.threshold
        return is_alarm, float(self.cum_sum)
