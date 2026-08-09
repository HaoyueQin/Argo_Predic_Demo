"""规则基线模型测试：Kalman 滤波与匀速（CV）模型。"""
import numpy as np
import pytest

from models.rule_based.kalman_predictor import KalmanTrajectoryPredictor
from models.rule_based.cv_predictor import ConstantVelocityPredictor


class TestConstantVelocityPredictor:
    def test_linear_extrapolation(self):
        """匀速直线运动的两个点 → 速度估计 → 外推与真实轨迹一致。"""
        history = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        model = ConstantVelocityPredictor(dt=1.0).fit(history)
        preds = model.forecast(steps=3)
        assert preds.shape == (3, 2)
        # last_pos=(1,2), velocity=(1,2) → t=1..3: (2,4),(3,6),(4,8)
        expected = np.array([[2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        np.testing.assert_allclose(preds, expected, atol=1e-5)

    def test_single_frame_zero_velocity(self):
        """只有一帧历史时速度为 0，预测保持静止。"""
        history = np.array([[5.0, 5.0]])
        model = ConstantVelocityPredictor(dt=1.0).fit(history)
        preds = model.forecast(steps=4)
        np.testing.assert_allclose(preds, np.full((4, 2), 5.0), atol=1e-6)

    def test_forecast_before_fit_raises(self):
        model = ConstantVelocityPredictor()
        with pytest.raises(RuntimeError):
            model.forecast(5)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            ConstantVelocityPredictor().fit(np.zeros((10, 3)))

    def test_empty_history_raises(self):
        # (0, 2) passes the dimension check but has no rows — must fail clearly
        with pytest.raises(IndexError):
            ConstantVelocityPredictor().fit(np.zeros((0, 2)))

    def test_nonpositive_steps_raises(self):
        p = ConstantVelocityPredictor().fit(np.array([[0.0, 0.0], [1.0, 2.0]]))
        with pytest.raises(ValueError):
            p.forecast(0)


class TestKalmanTrajectoryPredictor:
    def test_linear_trajectory_recovery(self):
        """匀速直线轨迹：卡尔曼滤波应恢复速度并平滑外推。"""
        t = np.arange(20, dtype=np.float64)
        measurements = np.stack([t, 2.0 * t], axis=-1)  # 直线 x=t, y=2t
        model = KalmanTrajectoryPredictor(dt=1.0).fit(measurements)
        preds = model.forecast(steps=30)
        assert preds.shape == (30, 2)
        # 外推最后一点应接近 t=49: (49, 98)，容差内（噪声小）
        np.testing.assert_allclose(preds[-1], [49.0, 98.0], atol=1.5)

    def test_forecast_shape_and_continuity(self):
        """预测从最后一个测量时刻开始连续推进。"""
        t = np.arange(10, dtype=np.float64)
        measurements = np.stack([t, t], axis=-1)
        model = KalmanTrajectoryPredictor(dt=1.0).fit(measurements)
        preds = model.forecast(steps=5)
        # 第一步接近 t=10: (10, 10)
        np.testing.assert_allclose(preds[0], [10.0, 10.0], atol=1.0)

    def test_single_measurement(self):
        """单个测量：速度估计为 0，预测保持原位（缓慢漂移）。"""
        model = KalmanTrajectoryPredictor().fit(np.array([[3.0, 4.0]]))
        preds = model.forecast(steps=3)
        assert preds.shape == (3, 2)
        np.testing.assert_allclose(preds, [[3.0, 4.0]] * 3, atol=2.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            KalmanTrajectoryPredictor().fit(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            KalmanTrajectoryPredictor().fit(np.zeros((0, 2)))
        with pytest.raises(RuntimeError):
            KalmanTrajectoryPredictor().forecast(5)
