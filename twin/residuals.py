"""Innovation residuals and statistical tracking."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ResidualRecord:
    """Record of Kalman filter innovation at a measurement epoch."""
    timestamp: float
    innovation: np.ndarray        # y_k = z_k - H x_hat (dimension m)
    cov_innovation: np.ndarray    # S_k = H P H^T + R (dimension m x m)
    nis: float                    # y_k^T S_k^{-1} y_k (scalar)
    est_pos: np.ndarray           # x_hat position [x, y]
    est_vel: np.ndarray           # x_hat velocity [vx, vy]
    est_bias: np.ndarray          # x_hat IMU bias [bx, by]
    cov_trace: float              # trace of state covariance P


class ResidualHistory:
    """Maintains a sliding history of innovation residuals and statistics."""

    def __init__(self, max_history: int = 5000):
        self.max_history = max_history
        self.records: List[ResidualRecord] = []

    def append(self, record: ResidualRecord):
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records.pop(0)

    def __len__(self):
        return len(self.records)

    def get_nis_series(self) -> np.ndarray:
        """Return 1D array of all NIS values."""
        if not self.records:
            return np.array([])
        return np.array([r.nis for r in self.records])

    def get_window_nis(self, window_size: int = 10) -> np.ndarray:
        """Return sliding window cumulative NIS sum."""
        nis = self.get_nis_series()
        if len(nis) < window_size:
            return np.array([])
        # Convolution for rolling sum
        return np.convolve(nis, np.ones(window_size), mode="valid")

    def get_summary_stats(self) -> dict:
        """Compute statistical properties of the innovation sequence."""
        nis = self.get_nis_series()
        if len(nis) == 0:
            return {"mean_nis": 0.0, "median_nis": 0.0, "std_nis": 0.0, "max_nis": 0.0}
        return {
            "mean_nis": float(np.mean(nis)),
            "median_nis": float(np.median(nis)),
            "std_nis": float(np.std(nis)),
            "max_nis": float(np.max(nis)),
            "p95_nis": float(np.percentile(nis, 95)),
            "p99_nis": float(np.percentile(nis, 99)),
        }
