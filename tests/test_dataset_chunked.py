"""dataset_argoverse_chunked 缓存机制测试。

不依赖 argoverse-api 真实安装：用 monkeypatch 伪造 argoverse / utils /
utils_cython 模块树与 Pool，使 Dataset 的缓存加载/重建路径可在任何环境运行。

覆盖（review 修复项）：
- 缓存文件名固定为 'ex_list'（不带 get_name 的 'eval.' 前缀），
  train_v4 / eval_all_models / eval_single 共用同一份验证缓存；
- force_rebuild=True（eval --no-cache）在 reuse 模式下也生效；
- 缓存数据签名校验：数据集变化（文件增删）时不再静默复用旧缓存。
"""
import argparse
import os
import pickle
import sys
import types

import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')


def _install_fake_deps(monkeypatch):
    """伪造 argoverse / utils / utils_cython 模块树，再导入目标模块。"""
    # argoverse.map_representation.map_api.ArgoverseMap
    argoverse = types.ModuleType('argoverse')
    map_rep = types.ModuleType('argoverse.map_representation')
    map_api = types.ModuleType('argoverse.map_representation.map_api')
    map_api.ArgoverseMap = object
    map_rep.map_api = map_api
    argoverse.map_representation = map_rep
    monkeypatch.setitem(sys.modules, 'argoverse', argoverse)
    monkeypatch.setitem(sys.modules, 'argoverse.map_representation', map_rep)
    monkeypatch.setitem(sys.modules, 'argoverse.map_representation.map_api', map_api)

    # utils：提供模块级 import 所需的名字（测试不经过 preprocess 路径）
    utils = types.ModuleType('utils')
    for name in ['get_name', 'get_file_name_int', 'get_angle', 'logging', 'rotate',
                 'round_value', 'get_pad_vector', 'get_dis', 'get_subdivide_polygons',
                 'get_points_remove_repeated', 'get_one_subdivide_polygon',
                 'get_dis_point_2_polygons', 'larger', 'equal', 'assert_',
                 'get_neighbour_points', 'get_subdivide_points', 'get_unit_vector',
                 'get_dis_point_2_points', 'other_errors_to_string']:
        setattr(utils, name, lambda *a, **k: None)
    utils.args = argparse.Namespace()
    monkeypatch.setitem(sys.modules, 'utils', utils)

    utils_cython = types.ModuleType('utils_cython')
    monkeypatch.setitem(sys.modules, 'utils_cython', utils_cython)

    monkeypatch.syspath_prepend(SRC_DIR)
    import dataset_argoverse_chunked as mod
    return mod


class FakePool:
    """同步串行 Pool，记录 imap_unordered 调用次数。"""

    def __init__(self, *a, **k):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a, **k):
        return False

    def imap_unordered(self, fn, args_list, chunksize=1):
        self.calls.append(len(args_list))
        for item in args_list:
            yield fn(item)


class FakeCtx:
    """fake multiprocessing context：每次 get_context 返回独立计数。"""

    def __init__(self):
        self.pool = FakePool()

    def Pool(self, processes=None, initializer=None):
        return self.pool


class FakeMP:
    """fake multiprocessing：get_context 返回 FakeCtx，记录全部调用。"""

    def __init__(self):
        self.ctxs = []

    def get_context(self, method=None):
        ctx = FakeCtx()
        self.ctxs.append(ctx)
        return ctx


def _pool_calls(fake_mp):
    return sum(len(ctx.pool.calls) for ctx in fake_mp.ctxs)


def _make_args(tmp_path, csv_names, reuse=True, do_eval=False, core_num=1):
    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    for name in csv_names:
        (data_dir / name).write_text('dummy\n')
    temp_dir = tmp_path / 'temp_file'
    temp_dir.mkdir(exist_ok=True)
    return argparse.Namespace(
        data_dir=[str(data_dir)],
        temp_file_dir=str(temp_dir),
        reuse_temp_file=reuse,
        core_num=core_num,
        do_eval=do_eval,
        do_test=False,
        debug=False,
        other_params=dict(),
        # 缓存签名相关字段（与 utils.add_argument 默认值一致）
        hidden_size=128,
        max_distance=50.0,
        not_use_api=False,
        no_agents=False,
        use_map=True,
        visualize=False,
        future_frame_num=30,
    ), data_dir, temp_dir


@pytest.fixture()
def mod(monkeypatch):
    return _install_fake_deps(monkeypatch)


class TestCacheHelpers:
    def test_collect_csv_files_top_level_only_sorted(self, mod, tmp_path):
        d = tmp_path / 'd'
        d.mkdir()
        (d / 'b.csv').write_text('')
        (d / 'a.csv').write_text('')
        (d / 'x.txt').write_text('')
        (d / 'sub').mkdir()
        (d / 'sub' / 'c.csv').write_text('')
        files = mod._collect_csv_files([str(d)])
        assert files == [os.path.join(str(d), 'a.csv'), os.path.join(str(d), 'b.csv')]

    def test_data_signature(self, mod):
        assert mod._data_signature([]) == (0, None, None)
        sig = mod._data_signature(['/x/a.csv', '/x/b.csv'])
        assert sig == (2, '/x/a.csv', '/x/b.csv')

    def test_args_signature(self, mod):
        def make_args(**over):
            base = dict(hidden_size=128, max_distance=50.0, not_use_api=False,
                        no_agents=False, use_map=True, visualize=False,
                        future_frame_num=30, other_params=['goals_2D', 'subdivide'])
            base.update(over)
            return argparse.Namespace(**base)

        a1 = make_args()
        # 相同参数（含 other_params 乱序）→ 相同指纹
        assert mod._args_signature(a1) == mod._args_signature(make_args())
        assert mod._args_signature(a1) == mod._args_signature(
            make_args(other_params=['subdivide', 'goals_2D']))
        # 关键参数变化 → 指纹变化（缓存必须重建）
        assert mod._args_signature(make_args(use_map=False)) != mod._args_signature(a1)
        assert mod._args_signature(make_args(other_params=['goals_2D'])) != mod._args_signature(a1)
        assert mod._args_signature(make_args(hidden_size=256)) != mod._args_signature(a1)
        assert mod._args_signature(make_args(future_frame_num=60)) != mod._args_signature(a1)
        # dict 形式与 'k=v' list 形式：带值参数等价
        assert mod._args_signature(
            make_args(other_params={'set_predict': 6, 'goals_2D': True})) == \
            mod._args_signature(make_args(other_params=['set_predict=6', 'goals_2D=True']))

    def test_data_signature_includes_args(self, mod):
        files = ['/x/a.csv', '/x/b.csv']
        args_a = argparse.Namespace(hidden_size=128, max_distance=50.0, not_use_api=False,
                                    no_agents=False, use_map=True, visualize=False,
                                    future_frame_num=30, other_params=['goals_2D'])
        args_b = argparse.Namespace(hidden_size=128, max_distance=50.0, not_use_api=False,
                                    no_agents=False, use_map=False, visualize=False,
                                    future_frame_num=30, other_params=['goals_2D'])
        assert mod._data_signature(files, args_a) != mod._data_signature(files, args_b)
        assert mod._data_signature(files, args_a) == mod._data_signature(files, args_a)

    def test_cache_matches_requires_signature(self, mod):
        sig = (2, 'a.csv', 'b.csv')
        assert mod._cache_matches({'ex_list': [], 'data_signature': sig}, sig)
        # 旧格式缓存（无签名）视为不匹配 → 触发重建
        assert not mod._cache_matches({'ex_list': []}, sig)
        assert not mod._cache_matches(['a.csv', 'b.csv'], sig)
        assert not mod._cache_matches({'ex_list': [], 'data_signature': (1, 'a.csv', 'a.csv')}, sig)


class TestDatasetCache:
    def _build(self, mod, args, batch_size=1):
        return mod.Dataset(args, batch_size, to_screen=False)

    def test_cache_file_name_is_fixed(self, mod, tmp_path):
        """缓存文件名为 'ex_list'，与 do_eval 状态无关（eval 脚本可复用训练缓存）。"""
        args, _, temp_dir = _make_args(tmp_path, ['1.csv', '2.csv'])
        fake_mp = FakeMP()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, 'multiprocessing', fake_mp)
        monkeypatch.setattr(mod, '_pool_load_file',
                            lambda t: (t[0], b'compressed', 5))
        try:
            ds = self._build(mod, args)
            assert len(ds.ex_list) == 2
            cache = os.path.join(str(temp_dir), 'ex_list')
            assert os.path.exists(cache)
            with open(cache, 'rb') as f:
                data = pickle.load(f)
            assert 'data_signature' in data
            # 新签名 = (文件签名, args 指纹)；文件签名首元素为文件数
            assert data['data_signature'][0][0] == 2
        finally:
            monkeypatch.undo()

    def test_reuse_on_same_data(self, mod, tmp_path):
        """数据未变时二次构建复用缓存，不重新处理文件。"""
        args, _, _ = _make_args(tmp_path, ['1.csv', '2.csv'])
        fake_mp = FakeMP()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, 'multiprocessing', fake_mp)
        monkeypatch.setattr(mod, '_pool_load_file',
                            lambda t: (t[0], b'compressed', 5))
        try:
            ds1 = self._build(mod, args)
            assert len(ds1.ex_list) == 2
            first_calls = _pool_calls(fake_mp)
            assert first_calls == 1
            ds2 = self._build(mod, args)
            assert len(ds2.ex_list) == 2
            assert _pool_calls(fake_mp) == first_calls, '同数据不应重建缓存'
        finally:
            monkeypatch.undo()

    def test_rebuild_on_dataset_change(self, mod, tmp_path):
        """数据集变化（文件增删）时签名不匹配 → 重建而不是静默复用。"""
        args, data_dir, _ = _make_args(tmp_path, ['1.csv', '2.csv'])
        fake_mp = FakeMP()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, 'multiprocessing', fake_mp)
        monkeypatch.setattr(mod, '_pool_load_file',
                            lambda t: (t[0], b'compressed', 5))
        try:
            ds1 = self._build(mod, args)
            assert len(ds1.ex_list) == 2
            # 换数据集：删一个文件、加一个新文件
            (data_dir / '1.csv').unlink()
            (data_dir / '3.csv').write_text('dummy\n')
            ds2 = self._build(mod, args)
            assert len(ds2.ex_list) == 2
            assert _pool_calls(fake_mp) == 2, '数据集变化应触发重建'
        finally:
            monkeypatch.undo()

    def test_force_rebuild_with_reuse_temp_file(self, mod, tmp_path):
        """reuse_temp_file=True 时 force_rebuild=True（eval --no-cache）仍重建。"""
        args, _, _ = _make_args(tmp_path, ['1.csv'])
        fake_mp = FakeMP()
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, 'multiprocessing', fake_mp)
        monkeypatch.setattr(mod, '_pool_load_file',
                            lambda t: (t[0], b'compressed', 5))
        try:
            ds1 = self._build(mod, args)
            assert len(ds1.ex_list) == 1
            assert _pool_calls(fake_mp) == 1
            ds2 = mod.Dataset(args, 1, to_screen=False, force_rebuild=True)
            assert len(ds2.ex_list) == 1
            assert _pool_calls(fake_mp) == 2, '--no-cache 应强制重建'
        finally:
            monkeypatch.undo()
