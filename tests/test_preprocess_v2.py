"""argoverse_preprocess_v2.process_one 回归测试：短轨迹跳过、正常场景输出。

不依赖真实数据与 argoverse-api，仅需 numpy。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'scripts', 'preprocess'))
from argoverse_preprocess_v2 import process_one  # noqa: E402


def _write_csv(path, av_frames=50, agent_frames=50, others_frames=10):
    with open(path, 'w') as f:
        f.write('TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n')
        for i in range(av_frames):
            f.write(f'{i*100000},AV,AV,{float(i)},0.0,MIA\n')
        for i in range(agent_frames):
            f.write(f'{i*100000},ag1,AGENT,{float(i)},0.0,MIA\n')
        for i in range(others_frames):
            f.write(f'{i*100000},ot1,OTHERS,{float(i)*2},1.0,MIA\n')


def _out_dirs(tmp_path):
    cleaned = tmp_path / 'cleaned'
    processed = tmp_path / 'processed'
    cleaned.mkdir(parents=True)
    processed.mkdir()
    return str(cleaned), str(processed)


class TestProcessOne:
    def test_short_agent_trajectory_skipped(self, tmp_path):
        """AGENT 轨迹 ≥20 帧但 <50 帧时应跳过，而不是越界崩溃（review P1）。"""
        csv_path = tmp_path / '12345.csv'
        _write_csv(csv_path, agent_frames=30)
        cleaned, processed = _out_dirs(tmp_path)
        res = process_one((str(csv_path), 'train', cleaned, processed))
        assert res[0] == 'skip_short', res
        assert not os.path.exists(os.path.join(cleaned, '12345.csv'))
        assert not os.path.exists(os.path.join(processed, '12345.npz'))

    def test_full_trajectory_ok(self, tmp_path):
        """完整 50 帧场景正常处理：CSV 输出 AGENT 50 行，NPZ 形状正确。"""
        csv_path = tmp_path / '99999.csv'
        _write_csv(csv_path)
        cleaned, processed = _out_dirs(tmp_path)
        res = process_one((str(csv_path), 'train', cleaned, processed))
        assert res[0] == 'ok', res
        with open(os.path.join(cleaned, '99999.csv')) as f:
            agent_lines = [l for l in f.read().strip().split('\n') if 'AGENT' in l]
        assert len(agent_lines) == 50
        d = np.load(os.path.join(processed, '99999.npz'))
        assert d['hist'].shape == (20, 2)
        assert d['gt'].shape == (30, 2)

    def test_agent_short_history_skipped(self, tmp_path):
        """AGENT 历史帧 <20 帧时按 skip_short 跳过。"""
        csv_path = tmp_path / '7.csv'
        _write_csv(csv_path, agent_frames=15)
        cleaned, processed = _out_dirs(tmp_path)
        res = process_one((str(csv_path), 'train', cleaned, processed))
        assert res[0] == 'skip_short', res

    def test_missing_agent_skipped(self, tmp_path):
        """无 AGENT 时按 skip_no_agent 跳过。"""
        csv_path = tmp_path / '8.csv'
        with open(csv_path, 'w') as f:
            f.write('TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n')
            for i in range(50):
                f.write(f'{i*100000},AV,AV,{float(i)},0.0,MIA\n')
        cleaned, processed = _out_dirs(tmp_path)
        res = process_one((str(csv_path), 'train', cleaned, processed))
        assert res[0] == 'skip_no_agent', res

    def test_missing_av_skipped(self, tmp_path):
        """无 AV 时按 skip_no_av 跳过。"""
        csv_path = tmp_path / '9.csv'
        with open(csv_path, 'w') as f:
            f.write('TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n')
            for i in range(50):
                f.write(f'{i*100000},ag1,AGENT,{float(i)},0.0,MIA\n')
        cleaned, processed = _out_dirs(tmp_path)
        res = process_one((str(csv_path), 'train', cleaned, processed))
        assert res[0] == 'skip_no_av', res
