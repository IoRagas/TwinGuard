"""Comprehensive closed-loop simulation engine for TwinGuard."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from sim.drone import Drone, DroneState
from sim.wind import WindModel
from sim.sensors import SensorSuite, GPSMeasurement, IMUMeasurement
from sim.mission import WaypointMission
from twin.estimator import TwinEstimator
from attacks.base import Attack
from detect.fusion import AnomalyDetectorSuite, DetectionStatus
from agent.state_machine import SafetyStateMachine, DroneSafetyState, RecoveryAction
from agent.recovery_controller import RecoveryController
from agent.explainer import IncidentExplainer


@dataclass
class MissionResult:
    """Complete time-series telemetry and incident record of a flight mission."""
    times: np.ndarray
    true_pos: np.ndarray              # (N, 2)
    true_vel: np.ndarray              # (N, 2)
    gps_reported_pos: np.ndarray      # (N, 2) (interpolated or forward-filled)
    twin_est_pos: np.ndarray          # (N, 2)
    twin_est_vel: np.ndarray          # (N, 2)
    twin_uncertainty: np.ndarray      # (N,) 3-sigma pos uncertainty
    states: List[DroneSafetyState]    # (N,)
    actions: List[RecoveryAction]     # (N,)
    nis_values: List[float]           # at GPS epochs
    nis_times: List[float]            # GPS epoch times
    detection_statuses: List[DetectionStatus]
    fsm_transitions: list
    incident_report: str
    attack_name: str
    seed: int
    cmd_accel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    wind: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    gps_measurements: List[GPSMeasurement] = field(default_factory=list)
    imu_measurements: List[IMUMeasurement] = field(default_factory=list)

    @property
    def final_true_pos(self) -> np.ndarray:
        return self.true_pos[-1]

    @property
    def final_distance_to_home(self) -> float:
        return float(np.linalg.norm(self.true_pos[-1]))

    @property
    def detection_time(self) -> Optional[float]:
        """Timestamp of first UNDER_ATTACK confirmation, if any."""
        for tr in self.fsm_transitions:
            if tr.to_state == DroneSafetyState.UNDER_ATTACK:
                return tr.timestamp
        return None

    @property
    def is_safe(self) -> bool:
        """True if drone landed safely or completed mission without crashing."""
        final_speed = float(np.linalg.norm(self.true_vel[-1]))
        return final_speed <= 1.0 or self.states[-1] in [DroneSafetyState.LAND, DroneSafetyState.NORMAL]


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
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.max_speed = max_speed
        self.max_accel = max_accel

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

    def run_clean_flight(self):
        """Execute a clean mission and return historical telemetry."""
        result = self.run_mission(attack=None)
        return result

    def run_mission(
        self,
        attack: Optional[Attack] = None,
        preferred_response: RecoveryAction = RecoveryAction.RETURN,
    ) -> MissionResult:
        """Execute full closed-loop mission with optional attack and TwinGuard agent."""
        n_steps = int(self.total_time / self.dt)
        times = np.zeros(n_steps)
        true_pos = np.zeros((n_steps, 2))
        true_vel = np.zeros((n_steps, 2))
        cmd_accel_log = np.zeros((n_steps, 2))
        wind_log = np.zeros((n_steps, 2))
        gps_reported_pos = np.zeros((n_steps, 2))
        twin_est_pos = np.zeros((n_steps, 2))
        twin_est_vel = np.zeros((n_steps, 2))
        twin_uncertainty = np.zeros(n_steps)
        states = []
        actions = []
        gps_measurements_list = []
        imu_measurements_list = []

        estimator = TwinEstimator(dt=self.dt)
        detectors = AnomalyDetectorSuite()
        fsm = SafetyStateMachine(preferred_response=preferred_response)
        recovery_ctrl = RecoveryController(max_speed=self.max_speed, max_accel=self.max_accel)
        explainer = IncidentExplainer()

        nis_values = []
        nis_times = []
        detection_statuses = []

        last_reported_gps = np.zeros(2)

        for k in range(n_steps):
            t = k * self.dt
            times[k] = t

            # 1. Update wind
            w = self.wind_model.step(self.dt)
            wind_log[k] = w

            # 2. Safety Agent Guidance Selection
            current_state = fsm.state
            current_action = fsm.action
            states.append(current_state)
            actions.append(current_action)

            state = self.drone.get_state()
            true_pos[k] = state.pos
            true_vel[k] = state.vel
            twin_est_pos[k] = estimator.pos
            twin_est_vel[k] = estimator.vel
            twin_uncertainty[k] = estimator.pos_uncertainty

            if current_state in [DroneSafetyState.UNDER_ATTACK, DroneSafetyState.RESPONSE, DroneSafetyState.LAND]:
                # Flight controller executes recovery guidance from TwinGuard
                cmd_a = recovery_ctrl.compute_command(
                    action=current_action,
                    twin_pos=estimator.pos,
                    twin_vel=estimator.vel,
                    hold_target=fsm.hold_position,
                )
            else:
                # Normal mission waypoint guidance
                cmd_a = self.mission.compute_command(state.pos, state.vel)

            cmd_accel_log[k] = cmd_a

            # 3. Step Drone Physics
            new_state = self.drone.step(cmd_a, w)

            # 4. Sample Sensors & Apply Attack
            raw_imu = self.sensors.sample_imu(t, new_state.accel)
            imu_measurements_list.append(raw_imu)
            imu_meas = attack.apply_imu(t, raw_imu) if attack else raw_imu
            estimator.predict(imu_meas)

            raw_gps = self.sensors.sample_gps(t, new_state.pos, new_state.vel)
            if raw_gps is not None:
                gps_measurements_list.append(raw_gps)
                # Apply attack to GPS
                gps_meas = attack.apply_gps(t, raw_gps) if attack else raw_gps
                if gps_meas.valid:
                    last_reported_gps = gps_meas.pos.copy()

                rec = estimator.update_gps(gps_meas)
                if rec is not None:
                    nis_values.append(rec.nis)
                    nis_times.append(t)

                status = detectors.process(t, gps_meas, rec)
                detection_statuses.append(status)

                # Update safety state machine
                fsm.update(t, status, estimator)

            gps_reported_pos[k] = last_reported_gps.copy()

        incident_summary = explainer.explain_flight_summary(fsm.transition_history, true_pos[-1])

        return MissionResult(
            times=times,
            true_pos=true_pos,
            true_vel=true_vel,
            gps_reported_pos=gps_reported_pos,
            twin_est_pos=twin_est_pos,
            twin_est_vel=twin_est_vel,
            twin_uncertainty=twin_uncertainty,
            states=states,
            actions=actions,
            nis_values=nis_values,
            nis_times=nis_times,
            detection_statuses=detection_statuses,
            fsm_transitions=fsm.transition_history,
            incident_report=incident_summary,
            attack_name=attack.name if attack else "clean",
            seed=self.seed,
            cmd_accel=cmd_accel_log,
            wind=wind_log,
            gps_measurements=gps_measurements_list,
            imu_measurements=imu_measurements_list,
        )
