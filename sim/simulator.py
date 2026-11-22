"""Full simulation loop combining drone dynamics, wind, sensors, and mission."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from sim.drone import Drone, DroneState
from sim.wind import WindModel
from sim.sensors import SensorSuite, GPSMeasurement, IMUMeasurement
from sim.mission import WaypointMission


@dataclass
class FlightLog:
    """Historical telemetry record of a simulation flight."""
    times: np.ndarray
    true_pos: np.ndarray       # shape (N, 2)
    true_vel: np.ndarray       # shape (N, 2)
    true_accel: np.ndarray     # shape (N, 2)
    wind: np.ndarray           # shape (N, 2)
    cmd_accel: np.ndarray      # shape (N, 2)
    gps_measurements: List[GPSMeasurement] = field(default_factory=list)
    imu_measurements: List[IMUMeasurement] = field(default_factory=list)


class Simulator:
    """Deterministic simulation environment for quadrotor flight."""

    def __init__(
        self,
        seed: int = 42,
        dt: float = 0.02,
        total_time: float = 180.0,
        mass: float = 1.5,
        drag_coeff: float = 0.15,
        max_speed: float = 10.0,
        max_accel: float = 4.0,
    ):
        self.seed = seed
        self.dt = dt
        self.total_time = total_time

        self.drone = Drone(
            dt=dt,
            mass=mass,
            drag_coeff=drag_coeff,
            max_speed=max_speed,
            max_accel=max_accel,
        )
        self.wind_model = WindModel(seed=seed, max_wind=2.0)
        self.sensors = SensorSuite(seed=seed)
        self.mission = WaypointMission(max_accel=max_accel)

    def run_clean_flight(self) -> FlightLog:
        """Execute a full clean flight and record complete telemetry."""
        n_steps = int(self.total_time / self.dt)
        times = np.zeros(n_steps)
        true_pos = np.zeros((n_steps, 2))
        true_vel = np.zeros((n_steps, 2))
        true_accel = np.zeros((n_steps, 2))
        wind_log = np.zeros((n_steps, 2))
        cmd_accel_log = np.zeros((n_steps, 2))

        gps_list = []
        imu_list = []

        for k in range(n_steps):
            t = k * self.dt
            times[k] = t

            # 1. Update wind
            w = self.wind_model.step(self.dt)
            wind_log[k] = w

            # 2. Guidance controller generates command based on current state
            state = self.drone.get_state()
            cmd_a = self.mission.compute_command(state.pos, state.vel)
            cmd_accel_log[k] = cmd_a

            # 3. Advance drone physics
            new_state = self.drone.step(cmd_a, w)
            true_pos[k] = new_state.pos
            true_vel[k] = new_state.vel
            true_accel[k] = new_state.accel

            # 4. Sample sensors
            imu_meas = self.sensors.sample_imu(t, new_state.accel)
            imu_list.append(imu_meas)

            gps_meas = self.sensors.sample_gps(t, new_state.pos, new_state.vel)
            if gps_meas is not None:
                gps_list.append(gps_meas)

        return FlightLog(
            times=times,
            true_pos=true_pos,
            true_vel=true_vel,
            true_accel=true_accel,
            wind=wind_log,
            cmd_accel=cmd_accel_log,
            gps_measurements=gps_list,
            imu_measurements=imu_list,
        )
