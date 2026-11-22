"""Wind disturbance model with constant baseline and slow random-walk dynamics."""
import numpy as np


class WindModel:
    """Simulates 2D wind with a constant component and an Ornstein-Uhlenbeck random walk."""

    def __init__(self, seed: int = 42, max_wind: float = 2.0):
        self.rng = np.random.default_rng(seed)
        self.max_wind = max_wind

        # Constant base wind (up to 1.2 m/s)
        base_speed = self.rng.uniform(0.5, 1.2)
        base_angle = self.rng.uniform(0, 2 * np.pi)
        self.base_wind = np.array([
            base_speed * np.cos(base_angle),
            base_speed * np.sin(base_angle)
        ], dtype=float)

        # Current turbulent fluctuation
        self.gust = np.zeros(2, dtype=float)
        # OU parameters: correlation time ~ 10 seconds
        self.theta = 0.1  # mean-reversion rate
        self.sigma = 0.25  # diffusion intensity

    def step(self, dt: float) -> np.ndarray:
        """Advance wind state and return current 2D wind vector [w_x, w_y] in m/s."""
        # OU update: dG = -theta * G * dt + sigma * sqrt(dt) * dW
        dw = self.rng.normal(0.0, 1.0, size=2)
        self.gust += -self.theta * self.gust * dt + self.sigma * np.sqrt(dt) * dw

        total_wind = self.base_wind + self.gust
        speed = np.linalg.norm(total_wind)
        if speed > self.max_wind and speed > 1e-6:
            total_wind = total_wind * (self.max_wind / speed)

        return total_wind
