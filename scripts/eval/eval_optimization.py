#!/usr/bin/env python3
"""
DenseTNT 优化后处理验证脚本
============================
对同一批场景分别跑 baseline（NMS top-k）和 optimization 后处理，
对比 minADE / minFDE / MR 指标差异。

用法（在 WSL / Linux 中运行，需先按 README 编译 Cython 扩展）:
    # 只跑 baseline，不跑 optimization
    python scripts/eval/eval_optimization.py --baseline-only

    # 只跑 2000 个场景快速验证
    python scripts/eval/eval_optimization.py --max-scenes 2000

    # 使用已缓存的 temp_file（仅第二次及以后）
    python scripts/eval/eval_optimization.py --max-scenes 2000 --reuse-cache

    # 同时跑 baseline + optimization 全量对比
    python scripts/eval/eval_optimization.py

结果写入 <repo>/outputs/eval_output/optimization_comparison.json。

说明:
    - optimization 依赖 Cython 编译的 utils_cython（get_optimal_targets）；
      未编译时自动降级为仅 baseline（打印 WARN）。
    - 本脚本由开发期验证工具适配入库（原评估日志与运行环境见
      docs/OPTIMIZATION_VERIFICATION_REPORT.md）。
"""

import os, sys, copy, argparse, logging, types, time, json, math
import numpy as np
import torch

# =========================================================================
# 路径配置（基于仓库根，适配公开仓库布局）
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = os.path.join(REPO_ROOT, 'src')
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, 'model_save_full_chunked',
                                  'model_save', 'model.16.bin')
DEFAULT_VAL_DIR = os.path.join(REPO_ROOT, 'val', 'data')

# 模块级默认值；main() 中可被 --model / --data-dir 覆盖
MODEL_PATH = DEFAULT_MODEL_PATH
VAL_DIR = DEFAULT_VAL_DIR

sys.path.insert(0, SRC)
# vendored argoverse-api：不依赖 pip install -e，直接可用
sys.path.insert(0, os.path.join(REPO_ROOT, 'argoverse-api'))

# =========================================================================
# Cython 模块处理
# =========================================================================
try:
    import utils_cython as _uc
    _CYTHON_OK = True
    print(f"[Info] utils_cython loaded: {getattr(_uc, '__file__', 'unknown')}")
except (ImportError, ModuleNotFoundError):
    _uc = types.ModuleType('utils_cython')
    sys.modules['utils_cython'] = _uc
    _CYTHON_OK = False
    print("[WARN] utils_cython not available; optimization disabled")

# =========================================================================
# DenseTNT 模块导入
# =========================================================================
import utils
import structs
from modeling.vectornet import VectorNet
import modeling.decoder as _decoder_mod
import modeling.vectornet as _vectornet_mod

# =========================================================================
# 工具函数
# =========================================================================

def make_args(enable_optimization=True):
    """构建 DenseTNT 推理参数"""
    a = utils.Args()
    a.hidden_size = 128
    a.future_frame_num = 30
    a.mode_num = 6
    a.use_map = True
    a.use_centerline = True
    a.core_num = 4
    a.num_workers = 0
    a.argoverse = True
    a.do_eval = True
    a.do_train = False
    a.do_test = False
    a.debug = False
    a.visualize = False
    a.single_agent = True
    a.nms_threshold = None
    a.not_use_api = False
    a.reuse_temp_file = False
    a.add_prefix = None
    a.autoregression = False
    a.lstm = False
    a.attention_decay = False
    a.placeholder = 0.0
    a.multi = None
    a.method_span = [0, 1]
    a.waymo = False
    a.nuscenes = False
    a.max_distance = 50.0
    a.no_sub_graph = False
    a.no_agents = False
    a.sub_graph_batch_size = 8000
    a.sub_graph_depth = 3
    a.global_graph_depth = 1
    a.train_batch_size = 16
    a.eval_batch_size = 16
    a.hidden_dropout_prob = 0.1
    a.initializer_range = 0.02
    a.seed = 42
    a.no_cuda = False
    a.model_recover_path = MODEL_PATH
    a.train_extra = False
    a.data_dir = [VAL_DIR]
    a.data_dir_for_val = VAL_DIR
    a.output_dir = os.path.join(REPO_ROOT, 'outputs', 'eval_output')
    a.temp_file_dir = os.path.join(a.output_dir, 'temp_file_eval')
    a.model_save_dir = os.path.join(a.output_dir, 'model_save')
    a.log_dir = a.output_dir
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(a.model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(a.temp_file_dir, 'ex'), exist_ok=True)

    params_list = [
        'semantic_lane', 'direction', 'l1_loss', 'goals_2D',
        'enhance_global_graph', 'subdivide', 'goal_scoring',
        'laneGCN', 'point_sub_graph', 'lane_scoring',
        'complete_traj', 'complete_traj-3',
    ]
    if enable_optimization:
        params_list.append('optimization')
    a.other_params = {p: True for p in params_list}
    if enable_optimization:
        a.other_params['cnt_sample'] = 36  # perfect square for Cython
        # MRratio 由 utils.run_process 默认 1.0（纯 MR 目标）；此处显式设置
        # 与历史评估配置保持一致（见 docs/optimization-verification-report.md）。
        a.other_params['MRminFDE'] = 1.0
        # opti_time not set -> defaults to 10000 -> avoids Cython num_step override
    a.eval_params = []
    a.train_params = []
    return a


def load_model(args, device):
    """加载 DenseTNT 模型"""
    model = VectorNet(args).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    model.eval()
    return model


def run_evaluation(args, model, device, tag="", max_scenes=None, reuse_cache=False):
    """
    执行推理流程。
    dataset_argoverse.Dataset 自身支持 reuse_temp_file（temp_file_dir 内缓存），
    baseline 与 optimization 两个 pass 共享同一缓存（缓存名由 get_name 生成，
    两个 pass 的 do_eval/do_test 状态一致）。
    """
    from dataset_argoverse import Dataset
    from torch.utils.data import SequentialSampler, DataLoader

    args.reuse_temp_file = reuse_cache

    print(f"[{tag}] 创建数据集 (首次需要预处理) ...")
    t0 = time.time()
    ds = Dataset(args, args.eval_batch_size, to_screen=True)
    dt_setup = time.time() - t0

    if max_scenes and max_scenes < len(ds.ex_list):
        ds.ex_list = ds.ex_list[:max_scenes]
    print(f"[{tag}] 数据集: {len(ds.ex_list)} 场景, 耗时 {dt_setup:.0f}s")

    dataloader = DataLoader(
        ds, batch_size=args.eval_batch_size,
        sampler=SequentialSampler(ds),
        collate_fn=utils.batch_list_to_batch_tensors,
        pin_memory=False)

    # 推理
    print(f"[{tag}] 开始推理...")
    file2pred = {}
    file2labels = {}
    t_infer = time.time()
    total = 0

    for step, batch in enumerate(dataloader):
        pred_trajectory, pred_score, _ = model(batch, device)
        mapping = batch
        bs = pred_trajectory.shape[0]
        for i in range(bs):
            try:
                fid = int(os.path.split(mapping[i]['file_name'])[1][:-4])
            except (ValueError, KeyError):
                continue
            pred_i = pred_trajectory[i].cpu().numpy() if hasattr(pred_trajectory[i], 'cpu') else pred_trajectory[i]
            file2pred[fid] = pred_i
            try:
                file2labels[fid] = mapping[i]['origin_labels']
            except KeyError:
                file2labels[fid] = mapping[i]['labels']
        total += bs
        if (step + 1) % 20 == 0:
            elapsed = time.time() - t_infer
            print(f"  [{step+1}] {total} scenes, {total/elapsed:.1f}/s")

    t_infer = time.time() - t_infer
    print(f"[{tag}] 推理完成: {len(file2pred)} 场景, {t_infer:.0f}s ({total/t_infer:.1f}/s)")
    return file2pred, file2labels


def compute_metrics(file2pred, file2labels, tag=""):
    """计算 minADE / minFDE / MR"""
    from argoverse.evaluation import eval_forecasting
    mr = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, 30, 2.0)
    results = {
        'minADE': float(mr.get('minADE', -1)),
        'minFDE': float(mr.get('minFDE', -1)),
        'MR': float(mr.get('MR', -1)),
    }
    for k, v in sorted(mr.items()):
        if k not in ('minADE', 'minFDE', 'MR'):
            results[k] = float(v) if hasattr(v, '__float__') else v
    print(f"\n[{tag}] minADE={results['minADE']:.4f}  minFDE={results['minFDE']:.4f}  MR={results['MR']*100:.2f}%")
    return results


# =========================================================================
# 主流程
# =========================================================================

def main():
    global MODEL_PATH, VAL_DIR
    parser = argparse.ArgumentParser(description='DenseTNT 优化后处理验证')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='限制推理场景数')
    parser.add_argument('--baseline-only', action='store_true',
                        help='只跑 baseline（NMS top-k）')
    parser.add_argument('--optim-only', action='store_true',
                        help='只跑 optimization')
    parser.add_argument('--reuse-cache', action='store_true',
                        help='使用已缓存的 temp_file')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH,
                        help=f'model checkpoint (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--data-dir', default=DEFAULT_VAL_DIR,
                        help=f'validation data dir (default: {DEFAULT_VAL_DIR})')
    flags = parser.parse_args()

    MODEL_PATH = flags.model
    VAL_DIR = flags.data_dir

    print("=" * 60)
    print("DenseTNT — Optimization Evaluation")
    print("=" * 60)
    print(f"  Cython: {_CYTHON_OK}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Val data: {VAL_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # 加载模型
    print(f"\n加载模型...")
    init_args = make_args(enable_optimization=False)
    utils.args = init_args
    model = load_model(init_args, device)
    print(f"  模型加载完成")

    # 决定运行哪几个 pass
    run_baseline = not flags.optim_only
    run_optim = not flags.baseline_only and _CYTHON_OK

    results = {}

    # ── Pass 1: Baseline ──
    if run_baseline:
        print(f"\n{'#'*60}")
        print(f"# PASS 1: BASELINE (NMS top-k)")
        print(f"{'#'*60}")
        if hasattr(utils.select_goals_by_optimization, 'processes'):
            utils.select_goals_by_optimization(None, None, close=True)

        bs_args = make_args(enable_optimization=False)
        utils.args = bs_args
        _decoder_mod.args = bs_args
        _vectornet_mod.args = bs_args
        p, l = run_evaluation(bs_args, model, device, "BASELINE",
                              max_scenes=flags.max_scenes,
                              reuse_cache=flags.reuse_cache)
        results['baseline'] = compute_metrics(p, l, "BASELINE")
        results['baseline']['scenes'] = len(p)

    # ── Pass 2: Optimization ──
    if run_optim:
        print(f"\n{'#'*60}")
        print(f"# PASS 2: OPTIMIZATION")
        print(f"{'#'*60}")

        opt_args = make_args(enable_optimization=True)
        utils.args = opt_args
        _decoder_mod.args = opt_args
        _vectornet_mod.args = opt_args
        p, l = run_evaluation(opt_args, model, device, "OPTIMIZATION",
                              max_scenes=flags.max_scenes,
                              reuse_cache=flags.reuse_cache)
        results['optimization'] = compute_metrics(p, l, "OPTIMIZATION")
        results['optimization']['scenes'] = len(p)

        try:
            utils.select_goals_by_optimization(None, None, close=True)
        except Exception:
            pass

    # ── 结果 ──
    print(f"\n{'='*60}")
    print("结果对比")
    print(f"{'='*60}")

    if 'baseline' in results and 'optimization' in results:
        b, o = results['baseline'], results['optimization']
        print(f"\n{'Metric':<12} {'Baseline':>10} {'Optim':>10} {'Δ':>10} {'Δ%':>8}")
        print(f"{'-'*50}")
        for m in ['minADE', 'minFDE', 'MR']:
            bv, ov = b.get(m, -1), o.get(m, -1)
            d = ov - bv
            pct = (d / bv * 100) if bv else 0
            print(f"{m:<12} {bv:>10.4f} {ov:>10.4f} {d:>+10.4f} {pct:>+7.2f}%")
        print(f"\n场景数: {b['scenes']}")
    elif 'baseline' in results:
        r = results['baseline']
        print(f"  minADE={r['minADE']:.4f}  minFDE={r['minFDE']:.4f}  MR={r['MR']*100:.2f}%")
    elif 'optimization' in results:
        r = results['optimization']
        print(f"  minADE={r['minADE']:.4f}  minFDE={r['minFDE']:.4f}  MR={r['MR']*100:.2f}%")

    out = os.path.join(REPO_ROOT, 'outputs', 'eval_output', 'optimization_comparison.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存: {out}")
    print("Done!")


if __name__ == '__main__':
    main()
