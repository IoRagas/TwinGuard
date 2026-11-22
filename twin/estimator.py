"""Extended/Linear Kalman Filter fusing 50 Hz IMU and 5 Hz GPS."""
from typing import Optional, Tuple
import numpy as np

from sim.sensors import GPSMeasurement, IMUMeasurement
from twin.dynamics import TwinModel
from twin.residuals import ResidualRecord, ResidualHistory


class TwinEstimator:
    """Fuses IMU acceleration prediction with GPS observations.
    
    State: x = [pos_x, pos_y, vel_x, vel_y, bias_x, bias_y]^T  (dim 6)
    Measurements: z = [gps_x, gps_y, gps_vx, gps_vy]^T           (dim 4)
    """

    def __init__(
        self,
        dt: float = 0.02,
        twin_model: Optional[TwinModel] = None,
        accel_process_noise: float = 0.35,  # accounts for IMU noise + wind/drag model mismatch
        bias_random_walk: float = 0.001,
        gps_pos_noise_std: float = 1.5,
        gps_vel_noise_std: float = 0.1,
        initial_pos: Optional[np.ndarray] = None,
    ):
        self.dt = dt
        self.twin = twin_model if twin_model is not None else TwinModel(dt=dt)

        # 6-state vector
        self.x = np.zeros(6, dtype=float)
        if initial_pos is not None:
            self.x[0:2] = initial_pos

        # Initial state covariance
        self.P = np.diag([
            gps_pos_noise_std ** 2, gps_pos_noise_std ** 2,
            gps_vel_noise_std ** 2, gps_vel_noise_std ** 2,
            0.01, 0.01
        ])

        # Measurement noise covariance R (4x4)
        self.R = np.diag([
            gps_pos_noise_std ** 2, gps_pos_noise_std ** 2,
            gps_vel_noise_std ** 2, gps_vel_noise_std ** 2
        ])

        # Measurement matrix H (4x6)
        self.H = np.zeros((4, 6), dtype=float)
        self.H[0:2, 0:2] = np.eye(2)
        self.H[2:4, 2:4] = np.eye(2)

        # Discrete process noise Q (6x6)
        q_a = accel_process_noise ** 2
        q_b = bias_random_walk ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        self.Q = np.zeros((6, 6), dtype=float)
        # Position block
        self.Q[0:2, 0:2] = 0.25 * dt4 * q_a * np.eye(2)
        self.Q[0:2, 2:4] = 0.5 * dt3 * q_a * np.eye(2)
        # Velocity block
        self.Q[2:4, 0:2] = 0.5 * dt3 * q_a * np.eye(2)
        self.Q[2:4, 2:4] = dt2 * q_a * np.eye(2)
        # Bias random walk
        self.Q[4:6, 4:6] = dt * q_b * np.eye(2)

        # Continuous state transition Jacobian F (6x6)
        self.F = np.eye(6, dtype=float)
        self.F[0:2, 2:4] = np.eye(2) * dt
        self.F[0:2, 4:6] = -0.5 * (dt ** 2) * np.eye(2)
        self.F[2:4, 4:6] = -dt * np.eye(2)

        self.history = ResidualHistory()
        self.gps_decoupled = False  # Set to True when attack confirmed to dead-reckon

    @property
    def pos(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def bias(self) -> np.ndarray:
        return self.x[4:6].copy()

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self.P))

    @property
    def pos_uncertainty(self) -> float:
        """3-sigma position uncertainty radius in meters."""
        pos_var = self.P[0, 0] + self.P[1, 1]
        return 3.0 * float(np.sqrt(max(0.0, pos_var)))

    def predict(self, imu_meas: IMUMeasurement) -> None:
        """Propagate state and covariance forward using IMU acceleration measurement."""
        corrected_accel = imu_meas.accel - self.x[4:6]

        # State propagation
        self.x[0:2] += self.x[2:4] * self.dt + 0.5 * corrected_accel * (self.dt ** 2)
        self.x[2:4] += corrected_accel * self.dt
        # Bias is assumed constant in propagation step (random walk in Q)

        # Covariance propagation
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, gps_meas: GPSMeasurement) -> Optional[ResidualRecord]:
        """Compute innovation and update filter state with GPS measurement.
        
        If gps_decoupled is True or GPS measurement is invalid, returns None.
        """
        if not gps_meas.valid or self.gps_decoupled:
            return None

        z = np.hstack([gps_meas.pos, gps_meas.vel])  # shape (4,)

        # Innovation: y = z - H x_pred
        z_pred = self.H @ self.x
        y = z - z_pred  # shape (4,)

        # Innovation covariance: S = H P H^T + R
        S = self.H @ self.P @ self.H.T + self.R  # shape (4, 4)

        # Invert S
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Normalized Innovation Squared (NIS): y^T S^-1 y
        nis = float(y.T @ S_inv @ y)

        # Kalman Gain: K = P H^T S^-1
        K = self.P @ self.H.T @ S_inv  # shape (6, 4)

        # State update
        self.x += K @ y

        # Joseph form covariance update: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        record = ResidualRecord(
            timestamp=gps_meas.timestamp,
            innovation=y,
            cov_innovation=S,
            nis=nis,
            est_pos=self.pos,
            est_vel=self.vel,
            est_bias=self.bias,
            cov_trace=self.covariance_trace,
        )
        self.history.append(record)
        return record

    def decouple_gps(self):
        """Sever GPS coupling and enter pure dead-reckoning mode."""
        self.gps_decoupled = True
