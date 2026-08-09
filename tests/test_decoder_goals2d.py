"""decoder.goals_2D_eval top-K fallback 分支测试（review P3-9）。

不依赖编译的 utils_cython：fake 之。覆盖：
- 候选目标 < mode_num 时循环补齐，形状保持 (batch, mode_num, 2)；
- 候选目标充足时取分数最高的 mode_num 个。
"""
import argparse
import os
import sys
import types

import numpy as np
import pytest
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')


def _install_fakes(monkeypatch):
    # utils.py 直接 `import utils_cython`（无 fallback），未编译时需 fake
    uc = types.ModuleType('utils_cython')
    monkeypatch.setitem(sys.modules, 'utils_cython', uc)
    monkeypatch.syspath_prepend(SRC_DIR)
    import utils  # noqa: F401
    import modeling.decoder as decoder
    return decoder


def _make_args(mode_num=6):
    return argparse.Namespace(
        hidden_size=128, future_frame_num=30, mode_num=mode_num,
        other_params={'complete_traj': True, 'goals_2D': True},  # eval 路径要求（上游隐式依赖）
        argoverse=False, nms_threshold=None, do_eval=True, do_train=False,
        output_dir='.', model_recover_path='', distributed_training=0,
        temp_file_dir='.', data_dir_for_val='', train_batch_size=1,
        eval_batch_size=1, visualize=False, debug=False, attention_decay=False,
    )


def _make_decoder(decoder, args):
    """构造 Decoder 并补齐 VectorNet 在 complete_traj 时挂载的属性。"""
    import sys
    utils = sys.modules['utils']
    utils.args = args  # lib.py 的 CrossAttention 等读取 utils.args（模块级全局）
    d = decoder.Decoder(args, None)
    d.complete_traj_cross_attention = decoder.CrossAttention(args.hidden_size)
    d.complete_traj_decoder = decoder.DecoderResCat(
        args.hidden_size, args.hidden_size * 3, out_features=args.future_frame_num * 2)
    return d


class TestGoals2DEvalTopK:
    def test_insufficient_goals_padded_to_mode_num(self, monkeypatch):
        """候选目标（3 个）< mode_num（6）时补齐，形状保持 (1, 6, 2)。"""
        decoder = _install_fakes(monkeypatch)
        d = _make_decoder(decoder, _make_args())
        gs = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)
        ss = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        mapping = [{'goals_2D_scores': (gs, ss)}]
        trajs, probs, _ = d.goals_2D_eval(
            1, mapping, None, torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128), [1], 'cpu')
        assert trajs.shape == (1, 6, 30, 2)
        assert probs.shape == (1, 6)
        # 补齐为循环复制：索引 3 与 0 相同、索引 5 与 2 相同（末帧即目标点）
        np.testing.assert_array_equal(trajs[0, 3, -1], trajs[0, 0, -1])
        np.testing.assert_array_equal(trajs[0, 5, -1], trajs[0, 2, -1])
        # 分数排序保持：最高分目标排在最前
        np.testing.assert_array_equal(trajs[0, 0, -1], [1, 1])
        assert probs[0, 0] == 0.5

    def test_sufficient_goals_take_topk(self, monkeypatch):
        """候选目标充足时取分数最高的 mode_num 个。"""
        decoder = _install_fakes(monkeypatch)
        d = _make_decoder(decoder, _make_args())
        gs = np.array([[i, i] for i in range(10)], dtype=np.float32)
        ss = np.array([float(i) for i in range(10)], dtype=np.float32)
        mapping = [{'goals_2D_scores': (gs, ss)}]
        trajs, probs, _ = d.goals_2D_eval(
            1, mapping, None, torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128), [1], 'cpu')
        assert trajs.shape == (1, 6, 30, 2)
        assert probs.shape == (1, 6)
        # 前 6 个是分数最高的 6 个目标（9..4），落位在末帧
        np.testing.assert_array_equal(trajs[0, 0, -1], [9, 9])
        np.testing.assert_array_equal(trajs[0, -1, -1], [4, 4])
        np.testing.assert_allclose(probs[0], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0])

    def test_batch_with_different_goal_counts(self, monkeypatch):
        """批内样本目标数不同（3 个 vs 8 个）时仍产出统一形状。"""
        decoder = _install_fakes(monkeypatch)
        d = _make_decoder(decoder, _make_args())
        gs1 = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)
        ss1 = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        gs2 = np.array([[i, i] for i in range(8)], dtype=np.float32)
        ss2 = np.array([float(i) for i in range(8)], dtype=np.float32)
        mapping = [
            {'goals_2D_scores': (gs1, ss1)},
            {'goals_2D_scores': (gs2, ss2)},
        ]
        trajs, probs, _ = d.goals_2D_eval(
            2, mapping, None, torch.zeros(2, 1, 128),
            torch.zeros(2, 1, 128), [1, 1], 'cpu')
        assert trajs.shape == (2, 6, 30, 2)
        assert probs.shape == (2, 6)
