"""Utility to plot and save a clean flight trajectory and telemetry."""
import sys
from pathlib import Path

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np

from sim.simulator import Simulator


def plot_clean_flight(seed: int = 42, save_path: str = "eval/clean_flight.png"):
    """Simulate a clean flight and plot trajectory, velocity, wind, and sensor noise."""
    sim = Simulator(seed=seed, total_time=180.0)
    log = sim.run_clean_flight()

    # Ensure output directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 2D Trajectory
    ax_traj = axs[0, 0]
    ax_traj.plot(log.true_pos[:, 0], log.true_pos[:, 1], "b-", label="True Path (Ground Truth)", linewidth=2)
    gps_positions = np.array([m.pos for m in log.gps_measurements if m.valid])
    ax_traj.scatter(gps_positions[::5, 0], gps_positions[::5, 1], c="orange", s=10, alpha=0.6, label="GPS Samples (5 Hz)")
    waypoints = sim.mission.waypoints
    wp_x = [wp[0] for wp in waypoints]
    wp_y = [wp[1] for wp in waypoints]
    ax_traj.plot(wp_x, wp_y, "rx--", label="Waypoints (400x300m)", markersize=8)
    ax_traj.set_title("2D Drone Trajectory & Waypoint Patrol")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.grid(True, linestyle="--", alpha=0.6)
    ax_traj.legend(loc="upper right")

    # 2. Velocity Profile
    ax_vel = axs[0, 1]
    speeds = np.linalg.norm(log.true_vel, axis=1)
    ax_vel.plot(log.times, speeds, "g-", label="Ground Speed", linewidth=1.5)
    ax_vel.axhline(10.0, color="r", linestyle="--", label="Max Speed Limit (10 m/s)")
    ax_vel.axhline(8.0, color="k", linestyle=":", label="Cruise Speed (8 m/s)")
    ax_vel.set_title("Ground Speed Profile")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.grid(True, linestyle="--", alpha=0.6)
    ax_vel.legend(loc="lower right")

    # 3. Wind Disturbance
    ax_wind = axs[1, 0]
    ax_wind.plot(log.times, log.wind[:, 0], label="Wind X (m/s)", alpha=0.8)
    ax_wind.plot(log.times, log.wind[:, 1], label="Wind Y (m/s)", alpha=0.8)
    wind_speed = np.linalg.norm(log.wind, axis=1)
    ax_wind.plot(log.times, wind_speed, "k--", label="Wind Speed", linewidth=1.2)
    ax_wind.axhline(2.0, color="r", linestyle=":", label="Max Wind Limit (2 m/s)")
    ax_wind.set_title("Wind Disturbance (Constant + Random Walk)")
    ax_wind.set_xlabel("Time (s)")
    ax_wind.set_ylabel("Wind Velocity (m/s)")
    ax_wind.grid(True, linestyle="--", alpha=0.6)
    ax_wind.legend(loc="upper right")

    # 4. GPS Noise Residuals
    ax_gps = axs[1, 1]
    gps_times = np.array([m.timestamp for m in log.gps_measurements if m.valid])
    # Interpolate true position at GPS timestamps
    true_x_at_gps = np.interp(gps_times, log.times, log.true_pos[:, 0])
    true_y_at_gps = np.interp(gps_times, log.times, log.true_pos[:, 1])
    err_x = gps_positions[:, 0] - true_x_at_gps
    err_y = gps_positions[:, 1] - true_y_at_gps
    ax_gps.hist(err_x, bins=30, alpha=0.6, label=f"GPS Err X (std={np.std(err_x):.2f}m)")
    ax_gps.hist(err_y, bins=30, alpha=0.6, label=f"GPS Err Y (std={np.std(err_y):.2f}m)")
    ax_gps.set_title("Clean GPS Position Noise Distribution")
    ax_gps.set_xlabel("Position Error (m)")
    ax_gps.set_ylabel("Sample Count")
    ax_gps.grid(True, linestyle="--", alpha=0.6)
    ax_gps.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Clean flight plot saved to: {save_path}")


if __name__ == "__main__":
    plot_clean_flight()
