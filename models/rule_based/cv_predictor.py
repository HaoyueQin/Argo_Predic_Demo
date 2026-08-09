"""
Constant Velocity (CV) Trajectory Predictor.

Baseline model: estimates velocity from the last two history frames
and extrapolates linearly for future trajectory prediction.

Serves as a naive rule-based baseline for comparison against
Kalman Filter (rule-based) and LSTM (learning-based) methods.
"""

import numpy as np
from typing import Optional


class ConstantVelocityPredictor:
    """
    Constant velocity trajectory predictor.

    Uses the last two positions in the history to estimate a constant
    velocity vector, then extrapolates linearly to predict future positions.
    """

    def __init__(self, dt: float = 1.0):
        """
        Args:
            dt: Time step between frames (default 1.0 for Argoverse 1).
        """
        self.dt = dt
        self.velocity: Optional[np.ndarray] = None
        self.last_pos: Optional[np.ndarray] = None

    def fit(self, history: np.ndarray) -> "ConstantVelocityPredictor":
        """
        Estimate velocity from history trajectory.

        Args:
            history: numpy array of shape (T, 2) representing (x, y) positions
                     over T timesteps. T >= 1.

        Returns:
            self, for chaining.
        """
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"Expected history of shape (T, 2), got {history.shape}"
            )

        self.last_pos = history[-1].copy()

        if history.shape[0] >= 2:
            # Use last two frames for instantaneous velocity
            self.velocity = (history[-1] - history[-2]) / self.dt
        else:
            # Single frame: assume zero velocity
            self.velocity = np.zeros(2)

        return self

    def forecast(self, steps: int) -> np.ndarray:
        """
        Predict future trajectory via linear extrapolation.

        Args:
            steps: Number of future timesteps to predict.

        Returns:
            numpy array of shape (steps, 2): predicted (x, y) positions.
        """
        if self.velocity is None or self.last_pos is None:
            raise RuntimeError("Call fit() before forecast().")

        t = np.arange(1, steps + 1, dtype=np.float32).reshape(-1, 1) * self.dt
        preds = self.last_pos + t * self.velocity
        return preds
