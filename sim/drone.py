"""2D point-mass quadrotor dynamics model."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DroneState:
    """State vector of the drone."""
    pos: np.ndarray       # [x, y] in meters
    vel: np.ndarray       # [vx, vy] in m/s
    accel: np.ndarray     # [ax, ay] in m/s^2 (kinematic acceleration)


class Drone:
    """2D quadrotor point-mass dynamics with aerodynamic drag and saturation limits."""

    def __init__(
        self,
        dt: float = 0.02,
        mass: float = 1.5,
        drag_coeff: float = 0.15,
        max_speed: float = 10.0,
        max_accel: float = 4.0,
        initial_pos: np.ndarray = None,
        initial_vel: np.ndarray = None,
    ):
        self.dt = dt
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.max_speed = max_speed
        self.max_accel = max_accel

        self.pos = np.zeros(2, dtype=float) if initial_pos is None else np.array(initial_pos, dtype=float)
        self.vel = np.zeros(2, dtype=float) if initial_vel is None else np.array(initial_vel, dtype=float)
        self.accel = np.zeros(2, dtype=float)

    def step(self, cmd_accel: np.ndarray, wind: np.ndarray) -> DroneState:
        """Advance drone state by dt using commanded acceleration and current wind vector."""
        # 1. Enforce max acceleration command limit
        cmd_norm = np.linalg.norm(cmd_accel)
        if cmd_norm > self.max_accel and cmd_norm > 1e-6:
            effective_cmd = cmd_accel * (self.max_accel / cmd_norm)
        else:
            effective_cmd = np.array(cmd_accel, dtype=float)

        # 2. Aerodynamic drag force: F_drag = -c_d * (v - w)
        v_rel = self.vel - wind
        a_drag = -(self.drag_coeff / self.mass) * v_rel

        # 3. Total kinematic acceleration
        total_accel = effective_cmd + a_drag
        self.accel = total_accel

        # 4. Numerical integration (Euler-Cromer / Second-order Taylor)
        self.pos += self.vel * self.dt + 0.5 * total_accel * (self.dt ** 2)
        self.vel += total_accel * self.dt

        # 5. Enforce max speed limit
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed and speed > 1e-6:
            self.vel = self.vel * (self.max_speed / speed)

        return DroneState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            accel=self.accel.copy(),
        )

    def get_state(self) -> DroneState:
        """Return current state copy."""
        return DroneState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            accel=self.accel.copy(),
        )
