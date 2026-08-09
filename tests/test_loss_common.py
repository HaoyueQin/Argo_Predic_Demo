"""损失函数测试：WeightedSmoothL1Loss 手算验证、ADE/FDE 损失与指标一致性。"""
import numpy as np
import pytest
import torch

from models.loss_common.trajectory_loss import (
    WeightedSmoothL1Loss, ade_loss, fde_loss,
)
from models.metrics_common.trajectory_metrics import compute_ade, compute_fde


class TestWeightedSmoothL1Loss:
    def test_beta1_hand_computed(self):
        # smooth_l1(d, beta=1): |d|<=1 → 0.5*d² ; |d|>1 → |d|-0.5
        # 单点 (x,y)=(0.5, 2.0) 误差 → 坐标和: 0.5*0.25 + (2-0.5) = 0.125 + 1.5 = 1.625
        pred = torch.zeros(1, 1, 2)
        target = torch.tensor([[[0.5, 2.0]]])
        loss = WeightedSmoothL1Loss(beta=1.0, reduction='mean')
        assert loss(pred, target).item() == pytest.approx(1.625)

    def test_zero_loss_when_equal(self):
        pred = torch.randn(3, 10, 2)
        loss = WeightedSmoothL1Loss()
        assert loss(pred, pred).item() == pytest.approx(0.0, abs=1e-6)

    def test_reduction_modes(self):
        pred = torch.zeros(2, 3, 2)
        target = torch.ones(2, 3, 2)
        mean = WeightedSmoothL1Loss(reduction='mean')(pred, target)
        summ = WeightedSmoothL1Loss(reduction='sum')(pred, target)
        # loss.sum(dim=-1) 已合并 (x,y) 坐标对 → sum = mean * B*T
        assert summ.item() == pytest.approx(mean.item() * 6)  # 2*3 个坐标对

    def test_weights(self):
        pred = torch.zeros(1, 2, 2)
        target = torch.ones(1, 2, 2)
        weights = torch.tensor([[1.0, 0.0]])
        loss = WeightedSmoothL1Loss()
        weighted = loss(pred, target, weights=weights)
        unweighted = loss(pred, target)
        # 加权均值应等于第一帧的损失（第二帧权重 0）
        first_frame = WeightedSmoothL1Loss()(pred[:, :1], target[:, :1])
        assert weighted.item() == pytest.approx(first_frame.item())
        assert weighted.item() < unweighted.item()


class TestTrajectoryLosses:
    def test_ade_loss_matches_metric(self):
        pred = torch.randn(2, 30, 2)
        target = torch.randn(2, 30, 2)
        assert ade_loss(pred, target).item() == pytest.approx(compute_ade(pred, target))

    def test_fde_loss_matches_metric(self):
        pred = torch.randn(2, 30, 2)
        target = torch.randn(2, 30, 2)
        assert fde_loss(pred, target).item() == pytest.approx(compute_fde(pred, target))

    def test_ade_loss_mask(self):
        pred = torch.zeros(1, 3, 2)
        target = torch.ones(1, 3, 2)
        mask = torch.tensor([[True, False, True]])
        # 只有两帧计入：sqrt(2) 均值
        assert ade_loss(pred, target, mask=mask).item() == pytest.approx(np.sqrt(2), rel=1e-5)
