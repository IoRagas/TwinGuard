"""Mission and waypoint guidance controller for rectangular patrol route."""
from typing import List, Optional
import numpy as np


class WaypointMission:
    """Manages waypoints and generates PD control commands for patrol mission."""

    def __init__(
        self,
        waypoints: Optional[List[np.ndarray]] = None,
        cruise_speed: float = 8.0,
        acceptance_radius: float = 12.0,
        kp_vel: float = 1.8,
        kp_pos: float = 0.5,
        max_accel: float = 4.0,
    ):
        if waypoints is None:
            # 400m x 300m rectangular patrol route returning home
            self.waypoints = [
                np.array([400.0, 0.0]),
                np.array([400.0, 300.0]),
                np.array([0.0, 300.0]),
                np.array([0.0, 0.0]),
            ]
        else:
            self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]

        self.cruise_speed = cruise_speed
        self.acceptance_radius = acceptance_radius
        self.kp_vel = kp_vel
        self.kp_pos = kp_pos
        self.max_accel = max_accel

        self.current_idx = 0
        self.completed = False

    def reset(self):
        """Reset mission to first waypoint."""
        self.current_idx = 0
        self.completed = False

    def get_current_waypoint(self) -> np.ndarray:
        """Return the active target waypoint."""
        if self.completed:
            return self.waypoints[-1]
        return self.waypoints[self.current_idx]

    def compute_command(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute commanded acceleration towards active waypoint using PD guidance."""
        target = self.get_current_waypoint()
        pos_error = target - pos
        dist = np.linalg.norm(pos_error)

        # Check waypoint arrival
        if not self.completed and dist < self.acceptance_radius:
            if self.current_idx < len(self.waypoints) - 1:
                self.current_idx += 1
                target = self.get_current_waypoint()
                pos_error = target - pos
                dist = np.linalg.norm(pos_error)
            else:
                self.completed = True

        if dist < 1e-4:
            # Hover at destination
            raw_accel = -self.kp_vel * vel
        else:
            # Desired velocity vector (ramped down within deceleration zone)
            decel_distance = 15.0
            if dist < decel_distance:
                speed = self.cruise_speed * (dist / decel_distance)
            else:
                speed = self.cruise_speed

            direction = pos_error / dist
            v_des = direction * speed

            # Velocity tracking PD acceleration command
            raw_accel = self.kp_vel * (v_des - vel) + self.kp_pos * direction

        # Enforce max acceleration limit
        norm = np.linalg.norm(raw_accel)
        if norm > self.max_accel and norm > 1e-6:
            return raw_accel * (self.max_accel / norm)
        return raw_accel
