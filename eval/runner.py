"""Batch evaluation runner on reporting seeds (1000-1199) with zero test leakage."""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sim.simulator import Simulator
from attacks.spoof_jump import SpoofJumpAttack
from attacks.spoof_drift import SpoofDriftAttack
from attacks.jamming import GPSJammingAttack
from detect.baseline import BaselineDetector
from agent.state_machine import RecoveryAction, DroneSafetyState
from eval.metrics import compute_scenario_metrics
from eval.plotter import plot_detection_rate_vs_drift, plot_time_to_detect, plot_example_trajectories


def evaluate_baseline_on_flight(mission_res, attack_start_time: float) -> Tuple[bool, Optional[float]]:
    """Evaluate naive baseline detector on raw GPS packets from the flight."""
    baseline = BaselineDetector()
    detected = False
    delay = None

    for m in mission_res.gps_measurements:
        alarm, _ = baseline.update(m)
        if alarm and m.timestamp >= attack_start_time:
            detected = True
            delay = m.timestamp - attack_start_time
            break

    return detected, delay



def run_evaluation_suite(
    output_csv: str = "eval/results.csv",
    output_table: str = "eval/summary_table.md",
) -> pd.DataFrame:
    """Execute complete reporting suite on seeds 1000-1199."""
    print("=" * 70)
    print("TWINGUARD EMPIRICAL EVALUATION HARNESS")
    print("Zero-Leakage Reporting Seeds: 1000 to 1199")
    print("Tuning Seeds (Used Only for Detector Thresholds): 0 to 99")
    print("=" * 70)

    records: List[Dict[str, Any]] = []
    start_eval_time = time.time()

    # -------------------------------------------------------------
    # 1. Clean Flights (Seeds 1000-1049, 50 flights)
    # -------------------------------------------------------------
    print("\n[1/4] Running Clean Flights (Seeds 1000-1049)...")
    for seed in range(1000, 1050):
        sim = Simulator(seed=seed, total_time=60.0)
        res = sim.run_clean_flight()

        # TwinGuard false alarm
        tg_fa = any(tr.to_state == DroneSafetyState.UNDER_ATTACK for tr in res.fsm_transitions)
        # Baseline false alarm
        base_det, _ = evaluate_baseline_on_flight(res, attack_start_time=0.0)

        records.append({
            "seed": seed,
            "scenario": "clean",
            "attack_type": "none",
            "strength": 0.0,
            "attack_start_time": np.nan,
            "tg_detected": tg_fa,
            "tg_detection_delay": np.nan,
            "tg_is_safe": res.is_safe,
            "tg_final_dist_home": res.final_distance_to_home,
            "baseline_detected": base_det,
            "baseline_detection_delay": np.nan,
        })

    # -------------------------------------------------------------
    # 2. Spoof Jump Attacks (Seeds 1050-1099, 50 flights)
    # -------------------------------------------------------------
    print("\n[2/4] Running Spoof Jump Attacks (Seeds 1050-1099, offsets 20m - 100m)...")
    jump_distances = [20.0, 40.0, 60.0, 80.0, 100.0]
    seed_idx = 1050
    for dist in jump_distances:
        for _ in range(10):
            seed = seed_idx
            seed_idx += 1
            att = SpoofJumpAttack(start_time=50.0, jump_distance=dist)
            sim = Simulator(seed=seed, total_time=120.0)
            res = sim.run_mission(attack=att, preferred_response=RecoveryAction.RETURN)

            tg_det = res.detection_time is not None
            tg_delay = (res.detection_time - 50.0) if tg_det else np.nan
            base_det, base_delay = evaluate_baseline_on_flight(res, attack_start_time=50.0)

            records.append({
                "seed": seed,
                "scenario": "spoof_jump",
                "attack_type": "jump",
                "strength": dist,
                "attack_start_time": 50.0,
                "tg_detected": tg_det,
                "tg_detection_delay": tg_delay,
                "tg_is_safe": res.is_safe,
                "tg_final_dist_home": res.final_distance_to_home,
                "baseline_detected": base_det,
                "baseline_detection_delay": base_delay,
            })

    # -------------------------------------------------------------
    # 3. Spoof Drift Attacks (Seeds 1100-1169, 70 flights)
    # -------------------------------------------------------------
    print("\n[3/4] Running Spoof Drift Attacks (Seeds 1100-1169, rates 0.2 - 2.0 m/s)...")
    drift_rates = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    seed_idx = 1100
    for rate in drift_rates:
        for _ in range(10):
            seed = seed_idx
            seed_idx += 1
            att = SpoofDriftAttack(start_time=50.0, drift_rate=rate, stealth_velocity=True)
            sim = Simulator(seed=seed, total_time=130.0)
            res = sim.run_mission(attack=att, preferred_response=RecoveryAction.RETURN)

            tg_det = res.detection_time is not None
            tg_delay = (res.detection_time - 50.0) if tg_det else np.nan
            base_det, base_delay = evaluate_baseline_on_flight(res, attack_start_time=50.0)

            records.append({
                "seed": seed,
                "scenario": "spoof_drift",
                "attack_type": "drift",
                "strength": rate,
                "attack_start_time": 50.0,
                "tg_detected": tg_det,
                "tg_detection_delay": tg_delay,
                "tg_is_safe": res.is_safe,
                "tg_final_dist_home": res.final_distance_to_home,
                "baseline_detected": base_det,
                "baseline_detection_delay": base_delay,
            })

    # -------------------------------------------------------------
    # 4. GPS Jamming Attacks (Seeds 1170-1199, 30 flights)
    # -------------------------------------------------------------
    print("\n[4/4] Running GPS Jamming Attacks (Seeds 1170-1199, dropout & noise)...")
    seed_idx = 1170
    for mode in ["dropout", "noise_burst"]:
        for dur in [5.0, 10.0, 15.0]:
            for _ in range(5):
                seed = seed_idx
                seed_idx += 1
                att = GPSJammingAttack(start_time=50.0, duration=dur, mode=mode, seed=seed)
                sim = Simulator(seed=seed, total_time=120.0)
                res = sim.run_mission(attack=att, preferred_response=RecoveryAction.RETURN)

                tg_det = res.detection_time is not None
                tg_delay = (res.detection_time - 50.0) if tg_det else np.nan
                base_det, base_delay = evaluate_baseline_on_flight(res, attack_start_time=50.0)

                records.append({
                    "seed": seed,
                    "scenario": "jamming",
                    "attack_type": f"jamming_{mode}",
                    "strength": dur,
                    "attack_start_time": 50.0,
                    "tg_detected": tg_det,
                    "tg_detection_delay": tg_delay,
                    "tg_is_safe": res.is_safe,
                    "tg_final_dist_home": res.final_distance_to_home,
                    "baseline_detected": base_det,
                    "baseline_detection_delay": base_delay,
                })

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved raw run records to: {output_csv}")

    # -------------------------------------------------------------
    # Compute Aggregate Metrics & Format Markdown Table
    # -------------------------------------------------------------
    clean_summary = compute_scenario_metrics(df[df["scenario"] == "clean"])
    jump_summary = compute_scenario_metrics(df[df["scenario"] == "spoof_jump"])
    drift_summary = compute_scenario_metrics(df[df["scenario"] == "spoof_drift"])
    jam_summary = compute_scenario_metrics(df[df["scenario"] == "jamming"])

    table_md = f"""# TwinGuard Rigorous Empirical Evaluation Results

> **Evaluation Protocol**:
> - Detector Tuning Seeds: `0 - 99` (Used strictly for detector calibration).
> - Reporting Seeds: `1000 - 1199` (Total 200 evaluation runs; zero data leakage).
> - Physics Simulation: 2D Quadrotor with aerodynamic drag, 50 Hz IMU, 5 Hz GPS, random walk wind disturbance.

| Metric | Naive Fixed-Threshold Alarm | TwinGuard Digital Twin (Chi-Square + CUSUM) |
| :--- | :---: | :---: |
| **Spoof Jump Detection Rate (20-100m)** | {jump_summary.get('baseline_detection_rate_pct', 0.0):.1f}% | **{jump_summary.get('tg_detection_rate_pct', 0.0):.1f}%** |
| **Spoof Jump Median Detection Latency** | {jump_summary.get('baseline_median_delay_s', np.nan):.2f} s | **{jump_summary.get('tg_median_delay_s', np.nan):.2f} s** |
| **Spoof Drift Detection Rate (0.2-2.0 m/s)** | {drift_summary.get('baseline_detection_rate_pct', 0.0):.1f}% | **{drift_summary.get('tg_detection_rate_pct', 0.0):.1f}%** |
| **Spoof Drift Median Detection Latency** | {('N/A (Never Detected)' if np.isnan(drift_summary.get('baseline_median_delay_s', np.nan)) else f"{drift_summary.get('baseline_median_delay_s', 0):.2f} s")} | **{drift_summary.get('tg_median_delay_s', np.nan):.2f} s** |
| **GPS Jamming Detection Rate** | {jam_summary.get('baseline_detection_rate_pct', 0.0):.1f}% | **{jam_summary.get('tg_detection_rate_pct', 0.0):.1f}%** |
| **GPS Jamming Median Detection Latency** | {('N/A (Never Detected)' if np.isnan(jam_summary.get('baseline_median_delay_s', np.nan)) else f"{jam_summary.get('baseline_median_delay_s', 0):.2f} s")} | **{jam_summary.get('tg_median_delay_s', np.nan):.2f} s** |
| **False Alarm Rate (Clean Flights)** | {clean_summary.get('baseline_detection_rate_pct', 0.0):.1f}% ({int(clean_summary.get('baseline_detection_rate_pct', 0) * clean_summary.get('total_runs', 50) / 100)}/50) | **{clean_summary.get('tg_detection_rate_pct', 0.0):.1f}% ({int(clean_summary.get('tg_detection_rate_pct', 0) * clean_summary.get('total_runs', 50) / 100)}/50)** |
| **Safe Recovery Rate (Attacked Flights)** | 0.0% (No Recovery Logic) | **{((df[df['scenario'] != 'clean']['tg_is_safe'].sum() / len(df[df['scenario'] != 'clean'])) * 100.0):.1f}%** |
| **Mean Final Distance to Home After Safe Land** | N/A (Drone flies away) | **{df[df['scenario'] != 'clean']['tg_final_dist_home'].mean():.2f} m** |
"""

    with open(output_table, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"Saved markdown summary table to: {output_table}")

    # Generate all plots
    plot_detection_rate_vs_drift(df, save_path="eval/detection_rate_vs_drift_speed.png")
    plot_time_to_detect(df, save_path="eval/time_to_detect_distribution.png")
    plot_example_trajectories(seed=1042, save_path="eval/trajectories_comparison.png")

    elapsed = time.time() - start_eval_time
    print(f"\n[EVAL COMPLETE] 200 flights evaluated in {elapsed:.1f} s")
    print("\n" + table_md)

    return df


if __name__ == "__main__":
    run_evaluation_suite()
