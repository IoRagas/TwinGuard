"""Digital Twin internal physics model with deliberately mismatched parameters."""
import numpy as np


class TwinModel:
    """Physics model of the quadrotor used by the digital twin.
    
    Contains deliberate 5-10% parameter errors compared to physical drone
    to represent real-world modeling uncertainty.
    """

    def __init__(
        self,
        dt: float = 0.02,
        nominal_mass: float = 1.5,
        nominal_drag: float = 0.15,
        mass_error_pct: float = 7.0,     # +7% error
        drag_error_pct: float = -8.0,    # -8% error
    ):
        self.dt = dt
        self.mass = nominal_mass * (1.0 + mass_error_pct / 100.0)
        self.drag_coeff = nominal_drag * (1.0 + drag_error_pct / 100.0)

    def predict_acceleration(self, vel: np.ndarray, cmd_accel: np.ndarray) -> np.ndarray:
        """Predict acceleration using twin's internal model without exact wind knowledge."""
        # Twin models nominal drag without micro-wind fluctuations
        a_drag = -(self.drag_coeff / self.mass) * vel
        return cmd_accel + a_drag
