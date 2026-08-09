"""utils.py 修复项回归测试（review P2-1 / P2-2 / P3）。

不依赖编译的 utils_cython：fake 之。覆盖：
- speed_scale_factor 纯 Python 实现（上游 utils_cython.pyx 缺失，见 review）；
- select_goals_by_NMS 在无 Cython 环境下可用（--nms_threshold 路径）；
- run_process 在 cnt_sample 存在、MRminFDE 缺失时 MRratio 默认 1.0
  （原 UnboundLocalError，见 docs/optimization-verification-report.md Bug 3）；
- dataset_argoverse._pool_load_file 返回 (compressed, vector_num) 契约
  （供主进程跨进程聚合 max_vector_num）。
"""
import argparse
import os
import sys
import types

import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')


@pytest.fixture
def utils_mod(monkeypatch):
    """安装空 utils_cython + 最小 utils 全局 args，返回可用的 utils 模块。"""
    uc = types.ModuleType('utils_cython')
    monkeypatch.setitem(sys.modules, 'utils_cython', uc)
    monkeypatch.syspath_prepend(SRC_DIR)
    import utils
    utils.args = argparse.Namespace(
        eval_params=[], other_params={}, do_test=False, do_eval=True,
        do_train=False, debug=False, add_prefix=None, log_dir='.',
    )
    yield utils
    monkeypatch.undo()


class TestSpeedScaleFactor:
    def test_bounds_and_clamp(self, utils_mod):
        utils = utils_mod
        assert utils.speed_scale_factor(0.0) == pytest.approx(1.0)
        assert utils.speed_scale_factor(10.0) == pytest.approx(2.0)
        assert utils.speed_scale_factor(50.0) == pytest.approx(3.0)   # 上限 3.0
        assert utils.speed_scale_factor(-100.0) == pytest.approx(0.5)  # 下限 0.5

    def test_select_goals_by_NMS_without_cython(self, utils_mod):
        """--nms_threshold 路径不再依赖 utils_cython.speed_scale_factor。"""
        utils = utils_mod
        goals = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [10.0, 0.0]], dtype=np.float32)
        scores = np.array([0.9, 0.6, 0.5, 0.4], dtype=np.float32)
        mapping = {}
        utils.select_goals_by_NMS(mapping, goals, scores, 0.5, 2.0)
        assert 'pred_goals' in mapping
        assert mapping['pred_goals'].shape == (6, 2)  # 4 选中 + 随机补齐到 mode_num
        assert mapping['pred_goals'][0].tolist() == [0.0, 0.0]  # 最高分目标排最前


class TestRunProcessMRratio:
    def test_mrratio_defaults_when_missing(self, utils_mod, monkeypatch):
        """cnt_sample 存在、MRminFDE 缺失 → MRratio 默认 1.0，不抛 UnboundLocalError。"""
        utils = utils_mod
        received = {}

        def fake_get_optimal_targets(goals, scores, file_name, objective, opti_time, kwargs=None):
            received['kwargs'] = kwargs
            return 1.0, np.zeros((6, 2)), np.ones(6)

        # 注意：必须修改 utils 模块绑定的 utils_cython 对象（utils.utils_cython），
        # 而不是 sys.modules 中可能被其他测试替换的新对象。
        monkeypatch.setattr(utils.utils_cython, 'get_optimal_targets',
                            fake_get_optimal_targets, raising=False)

        class FakeQueue:
            def __init__(self, items):
                self.items = list(items)
                self.res = []

            def get(self):
                return self.items.pop(0) if self.items else None

            def put(self, v):
                self.res.append(v)

        args = argparse.Namespace(other_params={'cnt_sample': 36}, core_num=1)
        q = FakeQueue([(0, 'f.csv', (np.zeros((4, 2), np.float32), np.ones(4, np.float32)), {})])
        qr = FakeQueue([])
        utils.run_process(q, qr, args)
        assert received['kwargs']['MRratio'] == pytest.approx(1.0)
        assert len(qr.res) == 1


class TestSubdividePointsDegenerate:
    """get_subdivide_points 对退化多边形（<2 个顶点）返回空集，不除零（review P3-4）。"""

    def test_single_point_returns_empty(self, utils_mod):
        utils = utils_mod
        assert utils.get_subdivide_points([(0.0, 0.0)]) == []

    def test_empty_polygon_returns_empty(self, utils_mod):
        utils = utils_mod
        assert utils.get_subdivide_points([]) == []

    def test_return_unit_vectors_keeps_tuple_shape(self, utils_mod):
        utils = utils_mod
        points, unit_vectors = utils.get_subdivide_points([(1.0, 1.0)], return_unit_vectors=True)
        assert points == [] and unit_vectors == []

    def test_normal_polygon_still_subdivides(self, utils_mod):
        utils = utils_mod
        pts = utils.get_subdivide_points([(0.0, 0.0), (10.0, 0.0)], threshold=1.0)
        # 10m 边按 1m 阈值细分 → 10 等分 + include_self=False 时 9 个插入点
        assert len(pts) == 9
        assert pts[0][0] == pytest.approx(1.0)


class TestPoolLoadFileContract:
    def test_returns_compressed_and_vector_num(self, monkeypatch):
        """_pool_load_file 返回 (compressed, vector_num)，供主进程聚合。"""
        # fake argoverse 模块树（dataset_argoverse 顶层 import 需要）
        argoverse = types.ModuleType('argoverse')
        map_rep = types.ModuleType('argoverse.map_representation')
        map_api = types.ModuleType('argoverse.map_representation.map_api')
        map_api.ArgoverseMap = object
        map_rep.map_api = map_api
        argoverse.map_representation = map_rep
        monkeypatch.setitem(sys.modules, 'argoverse', argoverse)
        monkeypatch.setitem(sys.modules, 'argoverse.map_representation', map_rep)
        monkeypatch.setitem(sys.modules, 'argoverse.map_representation.map_api', map_api)
        # fake utils_cython（dataset_argoverse 顶层 import utils_cython 需要）
        uc = types.ModuleType('utils_cython')
        monkeypatch.setitem(sys.modules, 'utils_cython', uc)
        monkeypatch.syspath_prepend(SRC_DIR)

        import dataset_argoverse
        monkeypatch.setattr(dataset_argoverse, 'argoverse_get_instance',
                            lambda lines, file, args: {'vector_num': 123})
        monkeypatch.setattr(dataset_argoverse, 'ArgoverseMap', object)

        import tempfile
        csv = os.path.join(tempfile.mkdtemp(), 'f.csv')
        with open(csv, 'w') as f:
            f.write('TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME\n1,1,AV,0,0,MIA\n')

        compressed, vn = dataset_argoverse._pool_load_file((csv, {'hidden_size': 64}))
        assert compressed is not None and vn == 123

        # 错误路径返回 (None, 0)
        res = dataset_argoverse._pool_load_file((os.path.join(tempfile.mkdtemp(), 'missing.csv'), {}))
        assert res == (None, 0)
