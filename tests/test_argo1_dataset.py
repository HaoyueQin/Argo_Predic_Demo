"""ArgoverseV1Dataset 端到端测试：伪造 CSV → process → 加载验证。

不依赖真实数据与 argoverse-api，仅需 pandas/torch。
"""
import os

import numpy as np
import pytest
import torch

from scripts.preprocess.argo1_dataset import ArgoverseV1Dataset, process_argoverse

NUM_TIMESTAMPS = 50  # Argoverse 1: 20 历史 + 30 未来


def _make_fake_csv(path, seq_id=10001):
    """生成一个合法的 Argoverse 1 格式 CSV（AV、AGENT、2 个 OTHERS）。"""
    rows = []
    # actor: (track_id, object_type, 每帧位移向量)
    actors = [
        (1, 'AV', (1.0, 0.0)),
        (2, 'AGENT', (0.5, 0.5)),
        (3, 'OTHERS', (0.0, 1.0)),
        (4, 'OTHERS', (-0.2, 0.3)),
    ]
    for t in range(NUM_TIMESTAMPS):
        for track_id, obj_type, step in actors:
            x = t * step[0]
            y = t * step[1]
            rows.append(f"{track_id},{t},{obj_type},{x:.6f},{y:.6f},PIT")
    with open(path, 'w') as f:
        f.write("TRACK_ID,TIMESTAMP,OBJECT_TYPE,X,Y,CITY_NAME\n")
        f.write("\n".join(rows))
    return seq_id


@pytest.fixture
def fake_data_dir(tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir(parents=True)
    processed.mkdir()
    csv_path = raw / "10001.csv"
    _make_fake_csv(str(csv_path), seq_id=10001)
    return str(raw), str(processed)


class TestProcessArgoverse:
    def test_process_creates_pt(self, fake_data_dir):
        raw_dir, processed_dir = fake_data_dir
        ds = ArgoverseV1Dataset(os.path.dirname(raw_dir),
                                raw_dir=raw_dir, processed_dir=processed_dir)
        assert len(ds.raw_paths) == 1
        ds.process()
        pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        assert pt_files == ['10001.pt']

    def test_sample_tensor_shapes_and_fields(self, fake_data_dir):
        raw_dir, processed_dir = fake_data_dir
        ds = ArgoverseV1Dataset(os.path.dirname(raw_dir),
                                raw_dir=raw_dir, processed_dir=processed_dir)
        ds.process()
        sample = ds.get(0)
        assert sample['seq_id'] == 10001
        assert sample['x'].shape == (4, 20, 2)   # 4 actors × 20 历史帧
        assert sample['y'].shape == (4, 30, 2)   # 4 actors × 30 未来帧
        assert sample['positions'].shape == (4, 50, 2)
        assert sample['edge_index'].shape[0] == 2
        assert sample['padding_mask'].shape == (4, 50)
        assert sample['av_index'] == 0
        assert sample['agent_index'] == 1

    def test_origin_is_av_last_history_position(self, fake_data_dir):
        raw_dir, processed_dir = fake_data_dir
        ds = ArgoverseV1Dataset(os.path.dirname(raw_dir),
                                raw_dir=raw_dir, processed_dir=processed_dir)
        ds.process()
        sample = ds.get(0)
        # AV 最后历史帧在全局坐标 (19, 0)，原点应等于它
        np.testing.assert_allclose(sample['origin'].numpy(), [[19.0, 0.0]], atol=1e-4)

    def test_missing_raw_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ArgoverseV1Dataset(str(tmp_path),
                               raw_dir=str(tmp_path / "nonexistent"),
                               processed_dir=str(tmp_path / "out"))


class TestProcessArgoverseFunction:
    def test_direct_call(self, tmp_path):
        csv_path = tmp_path / "42.csv"
        _make_fake_csv(str(csv_path), seq_id=42)
        out = process_argoverse(str(csv_path))
        assert out['seq_id'] == 42
        assert out['x'].shape == (4, 20, 2)
        assert out['y'].shape == (4, 30, 2)
        # AV 的速度方向被旋转到 x 轴：theta 应为 0（AV 沿 x 轴运动）
        assert abs(out['theta'].item()) < 1e-6

    def test_av_short_history_clamps_origin(self, tmp_path):
        """AV 历史不足 20 帧（如 15 帧）时应 clamp 到最后可用帧，而非越界崩溃。"""
        csv_path = tmp_path / "43.csv"
        rows = []
        av_len = 15
        actors = [
            (1, 'AV', (1.0, 0.0)),
            (2, 'AGENT', (0.5, 0.5)),
        ]
        for t in range(NUM_TIMESTAMPS):
            for track_id, obj_type, step in actors:
                if obj_type == 'AV' and t >= av_len:
                    continue  # AV 只出现前 15 帧
                x = t * step[0]
                y = t * step[1]
                rows.append(f"{track_id},{t},{obj_type},{x:.6f},{y:.6f},PIT")
        with open(csv_path, 'w') as f:
            f.write("TRACK_ID,TIMESTAMP,OBJECT_TYPE,X,Y,CITY_NAME\n")
            f.write("\n".join(rows))

        out = process_argoverse(str(csv_path))
        # origin 应为 AV 最后可用帧 (14, 0)
        np.testing.assert_allclose(out['origin'].numpy(), [[14.0, 0.0]], atol=1e-4)

    def test_av_too_short_raises(self, tmp_path):
        """AV 少于 2 帧时无法计算原点/朝向，应给出明确错误。"""
        csv_path = tmp_path / "44.csv"
        rows = []
        actors = [
            (1, 'AV', (1.0, 0.0)),
            (2, 'AGENT', (0.5, 0.5)),
        ]
        for t in range(NUM_TIMESTAMPS):
            for track_id, obj_type, step in actors:
                if obj_type == 'AV' and t >= 1:
                    continue  # AV 只有 1 帧
                x = t * step[0]
                y = t * step[1]
                rows.append(f"{track_id},{t},{obj_type},{x:.6f},{y:.6f},PIT")
        with open(csv_path, 'w') as f:
            f.write("TRACK_ID,TIMESTAMP,OBJECT_TYPE,X,Y,CITY_NAME\n")
            f.write("\n".join(rows))

        with pytest.raises(ValueError, match="AV trajectory shorter than 2 frames"):
            process_argoverse(str(csv_path))

    def test_av_absent_in_history_raises(self, tmp_path):
        """AV 只在未来帧出现（历史 20 帧内无 AV）时报明确错误。"""
        csv_path = tmp_path / "45.csv"
        rows = []
        actors = [
            (2, 'AGENT', (0.5, 0.5)),
            (3, 'OTHERS', (0.0, 1.0)),
        ]
        for t in range(NUM_TIMESTAMPS):
            for track_id, obj_type, step in actors:
                x = t * step[0]
                y = t * step[1]
                rows.append(f"{track_id},{t},{obj_type},{x:.6f},{y:.6f},PIT")
        # AV 仅在 t=40..49（未来帧）出现
        for t in range(40, NUM_TIMESTAMPS):
            rows.append(f"1,{t},AV,{float(t):.6f},0.000000,PIT")
        with open(csv_path, 'w') as f:
            f.write("TRACK_ID,TIMESTAMP,OBJECT_TYPE,X,Y,CITY_NAME\n")
            f.write("\n".join(rows))

        with pytest.raises(ValueError, match="no AV in the historical frames"):
            process_argoverse(str(csv_path))
