"""Recovery trajectory guidance controller for safe return, hold, or landing."""
from typing import Optional
import numpy as np

from agent.state_machine import RecoveryAction


class RecoveryController:
    """Computes command acceleration for drone failsafe maneuvers using digital twin state."""

    def __init__(
        self,
        kp_vel: float = 1.8,
        kp_pos: float = 0.5,
        max_speed: float = 8.0,
        max_accel: float = 4.0,
    ):
        self.kp_vel = kp_vel
        self.kp_pos = kp_pos
        self.max_speed = max_speed
        self.max_accel = max_accel

    def compute_command(
        self,
        action: RecoveryAction,
        twin_pos: np.ndarray,
        twin_vel: np.ndarray,
        hold_target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute commanded acceleration based on active recovery action."""
        if action == RecoveryAction.RETURN:
            # Fly towards home base (0,0)
            target = np.array([0.0, 0.0])
            error = target - twin_pos
            dist = np.linalg.norm(error)

            if dist < 1e-3:
                raw_accel = -self.kp_vel * twin_vel
            else:
                speed = min(self.max_speed, 1.2 * dist)
                v_des = (error / dist) * speed
                raw_accel = self.kp_vel * (v_des - twin_vel) + self.kp_pos * (error / dist)

        elif action == RecoveryAction.HOLD:
            target = hold_target if hold_target is not None else twin_pos
            error = target - twin_pos
            v_des = 0.5 * error
            raw_accel = self.kp_vel * (v_des - twin_vel)

        elif action == RecoveryAction.LAND:
            # Active deceleration to hover standstill, then land
            raw_accel = -self.kp_vel * twin_vel

        else:
            raw_accel = np.zeros(2)

        # Clip command to max acceleration limit
        norm = np.linalg.norm(raw_accel)
        if norm > self.max_accel and norm > 1e-6:
            return raw_accel * (self.max_accel / norm)
        return raw_accel
