"""轨迹预测指标测试：ADE / FDE / minADE / minFDE / MR 手算验证与边界。"""
import numpy as np
import pytest
import torch

from models.metrics_common.trajectory_metrics import (
    compute_ade, compute_fde, compute_min_ade, compute_min_fde, compute_miss_rate,
)


class TestComputeADE:
    def test_hand_computed_value(self):
        # 两个时间步：误差 0 和 sqrt(8)≈2.8284 → ADE = 1.4142
        pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        target = torch.tensor([[[0.0, 0.0], [3.0, 3.0]]])
        assert compute_ade(pred, target) == pytest.approx(np.sqrt(8) / 2, rel=1e-5)

    def test_perfect_prediction_is_zero(self):
        pred = torch.rand(2, 30, 2)
        assert compute_ade(pred, pred) == 0.0

    def test_batch_mean(self):
        pred = torch.zeros(2, 3, 2)
        target = torch.tensor([[[1.0, 0.0]] * 3, [[0.0, 2.0]] * 3])
        # batch0: 每步误差 1 → ADE=1; batch1: 每步误差 2 → ADE=2; 均值 1.5
        assert compute_ade(pred, target) == pytest.approx(1.5)

    def test_mask(self):
        pred = torch.zeros(1, 3, 2)
        target = torch.ones(1, 3, 2)
        mask = torch.tensor([[True, True, False]])
        # 仅前两步计入：sqrt(2) 每步 → 均值 sqrt(2)
        assert compute_ade(pred, target, mask=mask) == pytest.approx(np.sqrt(2), rel=1e-5)


class TestComputeFDE:
    def test_hand_computed_value(self):
        pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        target = torch.tensor([[[0.0, 0.0], [3.0, 3.0]]])
        assert compute_fde(pred, target) == pytest.approx(np.sqrt(8), rel=1e-5)

    def test_mask_uses_last_step(self):
        pred = torch.zeros(1, 3, 2)
        target = torch.ones(1, 3, 2)
        mask = torch.tensor([[True, False, True]])
        assert compute_fde(pred, target, mask=mask) == pytest.approx(np.sqrt(2), rel=1e-5)


class TestMinMetrics:
    def test_min_ade_picks_best_mode(self):
        preds = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]],   # mode0 完美
                               [[10.0, 10.0], [10.0, 10.0]]]])  # mode1 很差
        target = torch.zeros(1, 2, 2)
        assert compute_min_ade(preds, target) == pytest.approx(0.0, abs=1e-6)
        assert compute_min_fde(preds, target) == pytest.approx(0.0, abs=1e-6)

    def test_min_ade_official_caliber(self):
        # Official Argoverse 1: minADE = ADE of the trajectory with minimum FDE,
        # NOT the minimum ADE across modes. T=3, target = [(0,0),(0,0),(1,0)]:
        #   mode0: [(0,0),(0,0),(0,0)]  -> ADE=1/3, FDE=1.0
        #   mode1: [(0,0),(2,0),(1,0)]  -> ADE=2/3, FDE=0.0
        # min-ADE caliber would pick mode0 (0.333); official caliber picks
        # mode1 (its FDE is best) and reports its ADE 0.667.
        preds = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                               [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]])
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
        assert compute_min_ade(preds, target) == pytest.approx(2.0 / 3.0, rel=1e-5)
        assert compute_min_fde(preds, target) == pytest.approx(0.0, abs=1e-6)

    def test_min_fde_hand_computed(self):
        # B=2, K=2, T=1：每批取最近模态的末点误差
        preds = torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]]],
                              [[[0.0, 0.0]], [[0.0, 3.0]]]])
        target = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])
        # batch0: min(1, 1)=1; batch1: min(0, 3)=0 → mean=0.5
        assert compute_min_fde(preds, target) == pytest.approx(0.5)


class TestMissRate:
    def test_hand_computed(self):
        # B=2：模态最优末点误差分别为 1.0（命中）和 3.0（未命中）
        preds = torch.tensor([[[[0.0, 0.0]]], [[[0.0, 0.0]]]])
        target = torch.tensor([[[1.0, 0.0]], [[3.0, 0.0]]])
        assert compute_miss_rate(preds, target, threshold=2.0) == pytest.approx(0.5)

    def test_all_hit_and_all_miss(self):
        preds = torch.zeros(3, 1, 1, 2)
        t_hit = torch.zeros(3, 1, 2)
        assert compute_miss_rate(preds, t_hit, threshold=2.0) == 0.0
        t_miss = torch.full((3, 1, 2), 5.0)
        assert compute_miss_rate(preds, t_miss, threshold=2.0) == 1.0

    def test_mask(self):
        preds = torch.zeros(2, 1, 1, 2)
        target = torch.full((2, 1, 2), 3.0)  # 都未命中
        mask = torch.tensor([True, False])   # 只统计第 0 个
        assert compute_miss_rate(preds, target, threshold=2.0, mask=mask) == pytest.approx(1.0, abs=1e-3)
