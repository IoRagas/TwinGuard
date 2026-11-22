"""Pre-recorded demo generator: Creates fallback animated GIF and interactive dashboard."""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from sim.simulator import Simulator
from attacks.spoof_drift import SpoofDriftAttack
from agent.state_machine import RecoveryAction, DroneSafetyState


def generate_fallback_gif(
    save_path: str = "demo/fallback_demo.gif",
    seed: int = 1042,
    total_time: float = 120.0,
    num_frames: int = 100,
    fps: int = 10,
):
    """Simulate a flight under stealthy spoof drift and record an animated GIF."""
    print("Simulating flight for fallback demo...")
    sim = Simulator(seed=seed, total_time=total_time)
    attack = SpoofDriftAttack(start_time=50.0, drift_rate=0.8, stealth_velocity=True)
    res = sim.run_mission(attack=attack, preferred_response=RecoveryAction.RETURN)

    # Subsample to exact num_frames for fast rendering & compact file size
    step_skip = max(1, len(res.times) // num_frames)
    frame_indices = list(range(0, len(res.times), step_skip))[:num_frames]

    fig, (ax_map, ax_nis) = plt.subplots(1, 2, figsize=(14, 6))

    # Setup 2D map
    ax_map.set_xlim(-50, 450)
    ax_map.set_ylim(-50, 350)
    ax_map.set_title("TwinGuard: 2D Autonomous Defense", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("X Position (m)")
    ax_map.set_ylabel("Y Position (m)")
    ax_map.grid(True, linestyle="--", alpha=0.5)

    # Waypoints
    wp_x = [0, 400, 400, 0, 0]
    wp_y = [0, 0, 300, 300, 0]
    ax_map.plot(wp_x, wp_y, "k--", alpha=0.4, label="Patrol Route")
    ax_map.plot(0, 0, "ks", markersize=8, label="Home (0,0)")

    true_line, = ax_map.plot([], [], "g-", linewidth=2.5, label="True Path (Ground Truth)")
    gps_line, = ax_map.plot([], [], "r--", linewidth=1.8, alpha=0.7, label="Spoofed GPS Telemetry")
    twin_line, = ax_map.plot([], [], "b:", linewidth=2.2, label="Digital Twin Dead-Reckoning")
    drone_dot, = ax_map.plot([], [], "go", markersize=9)
    status_text = ax_map.text(0.03, 0.92, "", transform=ax_map.transAxes, fontsize=11, fontweight="bold",
                              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax_map.legend(loc="lower right", fontsize=8)

    # Setup CUSUM / Innovation plot
    ax_nis.set_xlim(0, total_time)
    ax_nis.set_ylim(0, 35)
    ax_nis.set_title("Statistical Vector CUSUM Accumulator", fontsize=12, fontweight="bold")
    ax_nis.set_xlabel("Flight Time (s)")
    ax_nis.set_ylabel("CUSUM Statistic")
    ax_nis.grid(True, linestyle="--", alpha=0.5)
    ax_nis.axhline(9.0, color="r", linestyle="--", label="CUSUM Alarm Threshold (9.0)")
    ax_nis.axvline(50.0, color="m", linestyle=":", label="Attack Onset (t=50s)")
    cusum_line, = ax_nis.plot([], [], "orange", linewidth=2.0, label="CUSUM Score")
    ax_nis.legend(loc="upper left", fontsize=8)

    # Pre-extract CUSUM series
    cusum_times = [s.timestamp for s in res.detection_statuses]
    cusum_scores = [s.cusum_stat for s in res.detection_statuses]

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        t = res.times[idx]
        st = res.states[idx]

        # Update paths
        true_line.set_data(res.true_pos[:idx, 0], res.true_pos[:idx, 1])
        gps_line.set_data(res.gps_reported_pos[:idx, 0], res.gps_reported_pos[:idx, 1])
        twin_line.set_data(res.twin_est_pos[:idx, 0], res.twin_est_pos[:idx, 1])
        drone_dot.set_data([res.true_pos[idx, 0]], [res.true_pos[idx, 1]])

        # Status text with state-dependent color
        if st == DroneSafetyState.NORMAL:
            status_text.set_text(f"t={t:5.1f}s | State: NORMAL\nNaive Baseline: NORMAL")
            status_text.get_bbox_patch().set_facecolor("#d4edda")
        elif st == DroneSafetyState.SUSPECT:
            status_text.set_text(f"t={t:5.1f}s | State: SUSPECT (Surveillance Up)\nNaive Baseline: BLIND")
            status_text.get_bbox_patch().set_facecolor("#fff3cd")
        elif st == DroneSafetyState.UNDER_ATTACK:
            status_text.set_text(f"t={t:5.1f}s | State: UNDER_ATTACK! Decoupling GPS\nNaive Baseline: BLIND")
            status_text.get_bbox_patch().set_facecolor("#f8d7da")
        elif st == DroneSafetyState.RESPONSE:
            status_text.set_text(f"t={t:5.1f}s | State: RESPONSE (RETURN HOME)\nNaive Baseline: BLIND")
            status_text.get_bbox_patch().set_facecolor("#e2e3e5")
        elif st == DroneSafetyState.LAND:
            status_text.set_text(f"t={t:5.1f}s | State: LAND (Safe Landing Complete)\nNaive Baseline: BLIND")
            status_text.get_bbox_patch().set_facecolor("#cce5ff")

        # Update CUSUM plot
        c_sub_idx = [i for i, ct in enumerate(cusum_times) if ct <= t]
        if c_sub_idx:
            cusum_line.set_data(
                [cusum_times[i] for i in c_sub_idx],
                [cusum_scores[i] for i in c_sub_idx]
            )

        return true_line, gps_line, twin_line, drone_dot, status_text, cusum_line

    print(f"Rendering {len(frame_indices)} animation frames to {save_path}...")
    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close()
    print(f"Fallback demo GIF successfully rendered to: {save_path}")


if __name__ == "__main__":
    generate_fallback_gif()
