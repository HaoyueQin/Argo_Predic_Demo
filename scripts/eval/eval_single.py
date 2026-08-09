#!/usr/bin/env python3
"""Evaluate a single DenseTNT epoch checkpoint.

Usage:
    python scripts/eval/eval_single.py 15            # evaluate model.16.bin
    python scripts/eval/eval_single.py --model-path /path/to/model.10.bin
    python scripts/eval/eval_single.py 15 --data-dir sampled_val_2000/ --no-cache

Notes:
    - `epoch` is 0-based: epoch N corresponds to model.{N+1}.bin
    - Validation cache under <output_dir>/temp_file_val is reused unless
      --no-cache is given
    - Outputs minADE / minFDE / MR
"""

import argparse
import logging
import os
import sys

import numpy as np  # noqa: F401
import torch
from tqdm import tqdm

# 仓库根：本脚本位于 <repo>/scripts/eval/，上溯三级到仓库根
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

import utils  # noqa: E402
from argoverse.evaluation import eval_forecasting  # noqa: E402
from dataset_argoverse_chunked import Dataset as ChunkedDataset  # noqa: E402
from modeling.vectornet import VectorNet  # noqa: E402
from torch.utils.data import SequentialSampler  # noqa: E402

OTHER_PARAMS_DEFAULT = [
    'semantic_lane', 'direction', 'l1_loss', 'goals_2D',
    'enhance_global_graph', 'subdivide', 'goal_scoring',
    'laneGCN', 'point_sub_graph', 'lane_scoring',
    'complete_traj', 'complete_traj-3',
]


def main():
    parser = argparse.ArgumentParser(description='Evaluate a single DenseTNT checkpoint')
    parser.add_argument('epoch', type=int, nargs='?', default=-1,
                        help='0-based epoch (model.{epoch+1}.bin); ignored if --model-path given')
    parser.add_argument('--model-path', default=None,
                        help='direct path to a model .bin file (overrides epoch)')
    parser.add_argument('--data-dir', default='val/data',
                        help='validation data dir (default: val/data)')
    parser.add_argument('--output-dir', default='model_save_full_chunked',
                        help='training output dir (contains model_save/ and temp_file_*)')
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--core-num', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--future-frame-num', type=int, default=30)
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available)')
    parser.add_argument('--no-cache', action='store_true',
                        help='rebuild the validation cache from scratch')
    args_cli = parser.parse_args()

    # --- Resolve model path ---
    if args_cli.model_path:
        model_path = args_cli.model_path
        basename = os.path.basename(model_path)
        try:
            model_idx = int(basename.split('.')[1])
        except (IndexError, ValueError):
            model_idx = 0
    else:
        if args_cli.epoch < 0:
            parser.error('either <epoch> or --model-path is required')
        model_idx = args_cli.epoch + 1
        model_path = os.path.join(args_cli.output_dir, 'model_save', f'model.{model_idx}.bin')

    if not os.path.exists(model_path):
        print(f'[Error] {model_path} not found')
        sys.exit(1)

    # --- Build model args (mirrors eval_all_models.py) ---
    arg_parser = argparse.ArgumentParser()
    utils.add_argument(arg_parser)
    args, _ = arg_parser.parse_known_args([
        '--data_dir', args_cli.data_dir,
        '--data_dir_for_val', args_cli.data_dir,
        '--output_dir', args_cli.output_dir,
        '--temp_file_dir', os.path.join(args_cli.output_dir, 'temp_file_val'),
        '--eval_batch_size', str(args_cli.eval_batch_size),
        '--future_frame_num', str(args_cli.future_frame_num),
        '--hidden_size', str(args_cli.hidden_size),
        '--core_num', str(args_cli.core_num),
        '--use_map', '--use_centerline', '--argoverse',
        '--other_params', *OTHER_PARAMS_DEFAULT,
        '--distributed_training', '0', '--do_eval',
    ])

    logger = logging.getLogger('eval_single')
    logging.basicConfig(level=logging.INFO)
    utils.init(args, logger)

    # 缓存目录规则与 train_v4 / eval_all_models 统一（按数据目录 basename 隔离）；
    # 缓存必须含 origin_labels（do_eval=True 时构建），故不能用 utils.init 的 reuse 逻辑。
    args.temp_file_dir = utils.get_eval_temp_dir(args_cli.output_dir, args_cli.data_dir)
    args.reuse_temp_file = args_cli.no_cache is False
    os.makedirs(args.temp_file_dir, exist_ok=True)

    device = torch.device(args_cli.device if args_cli.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[Eval] Device: {device}')
    print(f'[Eval] Model: {model_path}')

    # --- Model ---
    model = VectorNet(args).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # --- Validation dataset ---
    print('[Eval] Loading validation data...')
    # ChunkedDataset expects data_dir as a list (see train_v4.py do_validate)
    args.data_dir = [args_cli.data_dir]
    eval_dataset = ChunkedDataset(args, args_cli.eval_batch_size, to_screen=True,
                                  force_rebuild=args_cli.no_cache)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args_cli.eval_batch_size,
        sampler=SequentialSampler(eval_dataset),
        collate_fn=utils.batch_list_to_batch_tensors)
    print(f'[Eval] Samples: {len(eval_dataset)}')

    # --- Inference ---
    file2pred, file2labels = {}, {}
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc=f'model.{model_idx} (epoch {model_idx - 1})',
                    unit='batch', dynamic_ncols=True)
        for batch in pbar:
            pred, score, _ = model(batch, device)
            mapping = batch
            for i in range(pred.shape[0]):
                fid = int(os.path.split(mapping[i]['file_name'])[1][:-4])
                file2pred[fid] = pred[i]
                file2labels[fid] = mapping[i]['origin_labels']

    # --- Metrics ---
    mr = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, args_cli.future_frame_num, 2.0)
    ade = mr.get('minADE', float('nan'))
    fde = mr.get('minFDE', float('nan'))
    miss = mr.get('MR', float('nan'))

    print()
    print('=' * 50)
    print(f'  model.{model_idx}')
    print(f'  minADE = {ade:.4f}')
    print(f'  minFDE = {fde:.4f}')
    print(f'  MR     = {miss:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
