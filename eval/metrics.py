"""Evaluation metrics computation for TwinGuard vs. Baseline."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_scenario_metrics(df_scenario: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate detection, latency, safety, and false alarm metrics."""
    total_runs = len(df_scenario)
    if total_runs == 0:
        return {}

    # TwinGuard metrics
    tg_detected = df_scenario["tg_detected"].sum()
    tg_det_rate = (tg_detected / total_runs) * 100.0

    tg_latencies = df_scenario.loc[df_scenario["tg_detected"], "tg_detection_delay"].dropna()
    tg_median_lat = float(tg_latencies.median()) if len(tg_latencies) > 0 else np.nan
    tg_p90_lat = float(tg_latencies.quantile(0.90)) if len(tg_latencies) > 0 else np.nan

    tg_safe_count = df_scenario["tg_is_safe"].sum()
    tg_safe_rate = (tg_safe_count / total_runs) * 100.0
    tg_final_err = float(df_scenario["tg_final_dist_home"].mean())

    # Baseline metrics
    base_detected = df_scenario["baseline_detected"].sum()
    base_det_rate = (base_detected / total_runs) * 100.0

    base_latencies = df_scenario.loc[df_scenario["baseline_detected"], "baseline_detection_delay"].dropna()
    base_median_lat = float(base_latencies.median()) if len(base_latencies) > 0 else np.nan
    base_p90_lat = float(base_latencies.quantile(0.90)) if len(base_latencies) > 0 else np.nan

    return {
        "total_runs": total_runs,
        "tg_detection_rate_pct": tg_det_rate,
        "tg_median_delay_s": tg_median_lat,
        "tg_p90_delay_s": tg_p90_lat,
        "tg_safe_rate_pct": tg_safe_rate,
        "tg_mean_final_dist_m": tg_final_err,
        "baseline_detection_rate_pct": base_det_rate,
        "baseline_median_delay_s": base_median_lat,
        "baseline_p90_delay_s": base_p90_lat,
    }
