import numpy as np


class KalmanTrajectoryPredictor:
    def __init__(self, dt: float = 1.0, process_var: float = 1e-2, meas_var: float = 1e-1):
        self.dt = dt
        self.F = np.array([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        q = process_var
        r = meas_var
        self.Q = np.diag([q, q, q, q]).astype(np.float64)
        self.R = np.diag([r, r]).astype(np.float64)
        self.P = np.diag([1e3, 1e3, 1e3, 1e3]).astype(np.float64)
        self.x = None

    def _predict_step(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update_step(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def fit(self, measurements: np.ndarray):
        m = np.asarray(measurements, dtype=np.float64)
        if m.ndim != 2 or m.shape[1] != 2:
            raise ValueError("measurements must be shape (N, 2)")
        if m.shape[0] < 1:
            raise ValueError("at least 1 measurement required")
        if m.shape[0] >= 2:
            v = (m[1] - m[0]) / self.dt
        else:
            v = np.array([0.0, 0.0], dtype=np.float64)
        self.x = np.array([m[0, 0], m[0, 1], v[0], v[1]], dtype=np.float64)
        for i, z in enumerate(m):
            if i > 0:  # first measurement corrects the initial state directly
                self._predict_step()
            self._update_step(z)
        return self

    def forecast(self, steps: int):
        if self.x is None:
            raise RuntimeError("model not fitted")
        preds = []
        x = self.x.copy()
        P = self.P.copy()
        for _ in range(steps):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            z = self.H @ x
            preds.append(z.copy())
        return np.stack(preds, axis=0)

