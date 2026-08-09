"""LSTM 学习基线测试：前向输出形状与确定性。"""
import pytest
import torch

from models.learning_based.lstm_predictor import LSTMTrajectoryModel


@pytest.fixture
def model():
    return LSTMTrajectoryModel(input_size=2, hidden_size=32, num_layers=2, dropout=0.0)


class TestLSTMModel:
    def test_forward_shape(self, model):
        """输入 [B, 20, 2] 历史 → 输出 [B, 30, 2] 未来。"""
        x = torch.randn(2, 20, 2)
        pred = model(x, steps=30)
        assert pred.shape == (2, 30, 2)

    def test_single_obs_history(self, model):
        """单帧历史也可运行（速度为零向量）。"""
        x = torch.randn(1, 1, 2)
        pred = model(x, steps=10)
        assert pred.shape == (1, 10, 2)

    def test_deterministic_in_eval_no_teacher_forcing(self, model):
        """eval 模式 + teacher_forcing_ratio=0 → 两次前向结果一致。"""
        model.eval()
        x = torch.randn(1, 20, 2)
        with torch.no_grad():
            p1 = model(x, steps=30, teacher_forcing_ratio=0.0)
            p2 = model(x, steps=30, teacher_forcing_ratio=0.0)
        assert torch.equal(p1, p2)

    def test_teacher_forcing_changes_input_flow(self, model):
        """teacher forcing 改变解码器输入（目标速度），应产生不同输出。"""
        model.eval()
        x = torch.randn(1, 20, 2)
        target = torch.zeros(1, 30, 2)
        with torch.no_grad():
            pred_tf = model(x, target=target, steps=30, teacher_forcing_ratio=1.0)
            pred_no = model(x, steps=30, teacher_forcing_ratio=0.0)
        assert pred_tf.shape == (1, 30, 2)
        assert not torch.equal(pred_tf, pred_no)

    def test_backward(self, model):
        """训练模式梯度可以回传。"""
        model.train()
        x = torch.randn(2, 20, 2)
        target = torch.zeros(2, 10, 2)
        pred = model(x, target=target, steps=10, teacher_forcing_ratio=0.5)
        loss = pred.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
