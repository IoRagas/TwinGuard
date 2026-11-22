"""Integrated detection suite combining Chi-Square, CUSUM, Jamming and Baseline."""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from detect.chisquare import ChiSquareDetector
from detect.cusum import CUSUMDetector
from detect.baseline import BaselineDetector
from sim.sensors import GPSMeasurement
from twin.residuals import ResidualRecord


@dataclass
class DetectionStatus:
    """Snapshot of all detector flags and scores at a given epoch."""
    timestamp: float
    chi2_alarm: bool
    chi2_stat: float
    chi2_threshold: float
    cusum_alarm: bool
    cusum_stat: float
    cusum_threshold: float
    jamming_alarm: bool
    baseline_alarm: bool
    baseline_stat: float
    twin_active_alarm: bool    # True if either Chi2 or CUSUM or Jamming fired


class AnomalyDetectorSuite:
    """Multi-layer statistical anomaly detection suite."""

    def __init__(
        self,
        chi2_window: int = 5,
        chi2_threshold: float = 48.0,
        cusum_allowance: float = 2.2,
        cusum_threshold: float = 9.0,
        gps_timeout: float = 1.0,  # Alarm if no valid GPS for >1.0 s
        baseline_pos_jump: float = 15.0,
        baseline_vel_jump: float = 4.0,
    ):
        self.chi2 = ChiSquareDetector(window_size=chi2_window, threshold=chi2_threshold)
        self.cusum = CUSUMDetector(allowance=cusum_allowance, threshold=cusum_threshold)
        self.baseline = BaselineDetector(max_pos_jump=baseline_pos_jump, max_vel_jump=baseline_vel_jump)

        self.gps_timeout = gps_timeout
        self.last_valid_gps_time = 0.0

    def reset(self):
        self.chi2.reset()
        self.cusum.reset()
        self.baseline.reset()
        self.last_valid_gps_time = 0.0

    def process(
        self,
        timestamp: float,
        gps_meas: Optional[GPSMeasurement],
        residual_rec: Optional[ResidualRecord],
    ) -> DetectionStatus:
        """Process latest measurements and return comprehensive detection status."""
        # 1. Check Baseline
        base_alarm, base_stat = self.baseline.update(gps_meas)

        # 2. Check Jamming / Dropout
        jamming_alarm = False
        if gps_meas is not None:
            if gps_meas.valid:
                self.last_valid_gps_time = timestamp
            else:
                jamming_alarm = True

        if (timestamp - self.last_valid_gps_time) > self.gps_timeout and timestamp > self.gps_timeout:
            jamming_alarm = True

        # 3. Check Chi-Square and CUSUM if residual is available
        if residual_rec is not None:
            chi2_alarm, chi2_stat = self.chi2.update(residual_rec.nis)
            cusum_alarm, cusum_stat = self.cusum.update(residual_rec.nis)
        else:
            chi2_alarm = False
            chi2_stat = self.chi2.statistic
            cusum_alarm = False
            cusum_stat = self.cusum.cum_sum

        twin_alarm = chi2_alarm or cusum_alarm or jamming_alarm

        return DetectionStatus(
            timestamp=timestamp,
            chi2_alarm=chi2_alarm,
            chi2_stat=chi2_stat,
            chi2_threshold=self.chi2.threshold,
            cusum_alarm=cusum_alarm,
            cusum_stat=cusum_stat,
            cusum_threshold=self.cusum.threshold,
            jamming_alarm=jamming_alarm,
            baseline_alarm=base_alarm,
            baseline_stat=base_stat,
            twin_active_alarm=twin_alarm,
        )
