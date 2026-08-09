#!/usr/bin/env python3
"""Batch-evaluate all saved DenseTNT epoch checkpoints.

Usage:
    python eval_all_models.py [--data_dir val/data] [--output_dir model_save_full_chunked]
                              [--eval_batch_size 64] [--core_num 4] [--device cuda]

Evaluates every model.{n}.bin under <output_dir>/model_save/ on the given
validation split and writes minADE/minFDE/MR results to
<output_dir>/val_results_all.txt.
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# 仓库根：本脚本位于 <repo>/（tools 下使用时需自行调整）
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    parser = argparse.ArgumentParser(description='Evaluate all DenseTNT checkpoints')
    parser.add_argument('--data_dir', default='val/data', help='validation data dir')
    parser.add_argument('--output_dir', default='model_save_full_chunked',
                        help='training output dir (contains model_save/ and temp_file_val/)')
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--core_num', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available)')
    parser.add_argument('--future_frame_num', type=int, default=30)
    parser.add_argument('--result_file', default=None,
                        help='output result file (default: <output_dir>/val_results_all.txt)')
    args_cli = parser.parse_args()

    model_save_dir = os.path.join(args_cli.output_dir, 'model_save')
    if not os.path.isdir(model_save_dir):
        print(f'[Error] Model dir not found: {model_save_dir}')
        sys.exit(1)

    # --- Build model args (mirrors train_v4 evaluation config) ---
    parser_args = argparse.ArgumentParser()
    utils.add_argument(parser_args)
    args, _ = parser_args.parse_known_args([
        '--data_dir', args_cli.data_dir, '--data_dir_for_val', args_cli.data_dir,
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

    logger = logging.getLogger('eval_all')
    logging.basicConfig(level=logging.INFO)
    utils.init(args, logger)

    # utils.init 会追加 '/temp_file' 到 temp_file_dir 并置 reuse_temp_file=True，
    # 这里按验证用途覆盖：独立缓存目录 + 自动检测（缓存必须含 origin_labels）
    args.temp_file_dir = os.path.join(args_cli.output_dir, 'temp_file_val')
    args.reuse_temp_file = False
    os.makedirs(args.temp_file_dir, exist_ok=True)

    device = torch.device(args_cli.device if args_cli.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[Eval] Device: {device}')

    # --- Model & dataset ---
    model = VectorNet(args).to(device)
    print('[Eval] Building validation cache...')
    # ChunkedDataset expects data_dir as a list (see train_v4.py do_validate)
    args.data_dir = [args_cli.data_dir]
    eval_dataset = ChunkedDataset(args, args_cli.eval_batch_size, to_screen=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args_cli.eval_batch_size,
        sampler=SequentialSampler(eval_dataset),
        collate_fn=utils.batch_list_to_batch_tensors)
    print(f'[Eval] Dataset ready: {len(eval_dataset)} samples')

    # --- Discover checkpoints ---
    model_paths = sorted(glob.glob(os.path.join(model_save_dir, 'model.*.bin')),
                         key=lambda p: int(os.path.basename(p).split('.')[1]))
    if not model_paths:
        print(f'[Error] No model.*.bin found in {model_save_dir}')
        sys.exit(1)

    result_file = args_cli.result_file or os.path.join(args_cli.output_dir, 'val_results_all.txt')
    with open(result_file, 'w') as f:
        f.write('model | epoch | minADE | minFDE | MR\n')

    results = []
    for model_path in model_paths:
        model_idx = int(os.path.basename(model_path).split('.')[1])
        epoch = model_idx - 1
        print(f'\n[Eval] Loading {model_path} ...')
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()

        file2pred, file2labels = {}, {}
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc=f'model.{model_idx} (epoch {epoch})',
                        total=len(eval_loader), unit='batch')
            for batch in pbar:
                pred, score, _ = model(batch, device)
                mapping = batch
                bs = pred.shape[0]
                for i in range(bs):
                    fid = int(os.path.splitext(os.path.basename(mapping[i]['file_name']))[0])
                    file2pred[fid] = pred[i]
                    file2labels[fid] = mapping[i]['origin_labels']

        mr = eval_forecasting.get_displacement_errors_and_miss_rate(
            file2pred, file2labels, 6, args_cli.future_frame_num, 2.0)
        ade = mr.get('minADE', float('nan'))
        fde = mr.get('minFDE', float('nan'))
        miss = mr.get('MR', float('nan'))

        results.append((model_idx, epoch, ade, fde, miss))
        print(f'[Eval] model.{model_idx} epoch={epoch}: minADE={ade:.4f} '
              f'minFDE={fde:.4f} MR={miss:.4f}')
        with open(result_file, 'a') as f:
            f.write(f'model.{model_idx} | {epoch} | {ade:.4f} | {fde:.4f} | {miss:.4f}\n')

    print('\n' + '=' * 60)
    print('SUMMARY (sorted by minFDE)')
    print('=' * 60)
    results.sort(key=lambda x: x[3])
    for model_idx, epoch, ade, fde, miss in results:
        star = ' *** BEST' if fde == results[0][3] else ''
        print(f'model.{model_idx}  epoch={epoch:2d}  minADE={ade:.4f}  '
              f'minFDE={fde:.4f}  MR={miss:.4f}{star}')
    print(f'\nResults: {result_file}')


if __name__ == '__main__':
    main()
