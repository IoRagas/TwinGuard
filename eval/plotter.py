"""Plotting utilities for evaluation metrics and trajectory visualizations."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.simulator import Simulator
from attacks.spoof_jump import SpoofJumpAttack
from attacks.spoof_drift import SpoofDriftAttack
from attacks.jamming import GPSJammingAttack
from agent.state_machine import RecoveryAction


def plot_detection_rate_vs_drift(df_results: pd.DataFrame, save_path: str = "eval/detection_rate_vs_drift_speed.png"):
    """Plot detection rate as a function of drift speed (0.2 - 2.0 m/s)."""
    df_drift = df_results[df_results["scenario"] == "spoof_drift"].copy()
    if df_drift.empty:
        return

    grouped = df_drift.groupby("strength").agg(
        tg_det=("tg_detected", lambda x: (x.sum() / len(x)) * 100.0),
        base_det=("baseline_detected", lambda x: (x.sum() / len(x)) * 100.0),
    ).reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(grouped["strength"], grouped["tg_det"], "ro-", linewidth=2.5, markersize=8, label="TwinGuard (CUSUM + Twin)")
    plt.plot(grouped["strength"], grouped["base_det"], "bs--", linewidth=2.0, markersize=8, label="Naive Baseline Alarm")

    plt.title("GPS Spoof Drift: Detection Rate vs. Drift Speed", fontsize=13, fontweight="bold")
    plt.xlabel("Drift Velocity (m/s)", fontsize=11)
    plt.ylabel("Detection Rate (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="center right")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved drift detection plot to: {save_path}")


def plot_time_to_detect(df_results: pd.DataFrame, save_path: str = "eval/time_to_detect_distribution.png"):
    """Plot time-to-detect distributions across attack scenarios."""
    df_attacks = df_results[df_results["scenario"].isin(["spoof_jump", "spoof_drift", "jamming"]) & df_results["tg_detected"]].copy()
    if df_attacks.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ["spoof_jump", "spoof_drift", "jamming"]
    data = [df_attacks[df_attacks["scenario"] == sc]["tg_detection_delay"].dropna().values for sc in scenarios]

    labels = ["Spoof Jump (20-100m)", "Spoof Drift (0.2-2.0m/s)", "GPS Jamming"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("TwinGuard Time to Detect Distribution (Reporting Seeds 1000-1199)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Latency (seconds)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved time-to-detect distribution plot to: {save_path}")


def plot_example_trajectories(seed: int = 1042, save_path: str = "eval/trajectories_comparison.png"):
    """Generate side-by-side 2D trajectory plots under 3 attack types."""
    scenarios = [
        ("Spoof Jump (+50m @ t=60s)", SpoofJumpAttack(start_time=60.0, jump_distance=50.0)),
        ("Spoof Drift (0.8 m/s @ t=60s)", SpoofDriftAttack(start_time=60.0, drift_rate=0.8, stealth_velocity=True)),
        ("GPS Jamming (Dropout @ t=60s)", GPSJammingAttack(start_time=60.0, duration=20.0, mode="dropout")),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (title, attack) in enumerate(scenarios):
        sim = Simulator(seed=seed, total_time=140.0)
        res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)

        ax = axs[idx]
        ax.plot(res.true_pos[:, 0], res.true_pos[:, 1], "g-", linewidth=2.5, label="True Path (Ground Truth)")
        ax.plot(res.gps_reported_pos[:, 0], res.gps_reported_pos[:, 1], "r--", linewidth=1.5, alpha=0.7, label="GPS Reported Path")
        ax.plot(res.twin_est_pos[:, 0], res.twin_est_pos[:, 1], "b:", linewidth=2.0, label="Twin Estimated Path")
        ax.plot(0, 0, "ks", markersize=8, label="Home Base (0,0)")

        # Mark attack start point
        att_idx = int(round(60.0 / 0.02))
        if att_idx < len(res.true_pos):
            ax.plot(res.true_pos[att_idx, 0], res.true_pos[att_idx, 1], "m*", markersize=12, label="Attack Onset")

        # Mark safe landing point
        ax.plot(res.true_pos[-1, 0], res.true_pos[-1, 1], "go", markersize=10, label=f"Final Safe Landing ({res.states[-1].value})")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("TwinGuard Cyber-Physical Attack & Autonomous Recovery Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved example trajectories to: {save_path}")
