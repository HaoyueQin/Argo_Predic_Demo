#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v4.py — DenseTNT 训练脚本 v4.0
======================================
基于 v3.2 的训练逻辑，epoch 循环在 Python 中完成。

功能：
  - 单次 DDP 启动，跨 epoch 保留 optimizer 状态
  - 每 epoch 之后自动验证（minFDE / minADE / MR）
  - LR 每 5 epoch 衰减为 30%
  - patience=5 early stopping
  - 训练历史写入 training_history.json
  - --resume 从 checkpoint.pt 恢复（intra-epoch checkpoint 仅作归档，
    不参与恢复；恢复请用 --resume）

用法：
  python src/train_v4.py \
    --data_dir train_60k/data \
    --data_dir_for_val val/data \
    --output_dir model_save_full_chunked \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --hidden_size 128 \
    --future_frame_num 30 \
    --num_train_epochs 16 \
    --patience 5 \
    --checkpoint_interval 100 \
    --learning_rate 0.001 \
    --core_num 4 \
    --num_workers 0 \
    --distributed_training 1 \
    --use_map --use_centerline --argoverse \
    --other_params semantic_lane direction l1_loss goals_2D \
                   enhance_global_graph subdivide goal_scoring \
                   laneGCN point_sub_graph lane_scoring \
                   complete_traj complete_traj-3
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---- Compile Cython (path-independent; only when stale or missing) ----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_pyx_files():
    pyx = os.path.join(SRC_DIR, 'utils_cython.pyx')
    c_file = os.path.join(SRC_DIR, 'utils_cython.c')
    # Linux 产物为 .so，Windows 为 .pyd
    so_files = glob.glob(os.path.join(SRC_DIR, 'utils_cython*.so')) + \
        glob.glob(os.path.join(SRC_DIR, 'utils_cython*.pyd'))
    try:
        needs_compile = not so_files or not os.path.exists(c_file) or \
            os.path.getmtime(pyx) > os.path.getmtime(c_file)
        if needs_compile:
            cwd = os.getcwd()
            os.chdir(SRC_DIR)
            os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
            os.chdir(cwd)
    except Exception as e:  # never block training on cython
        print(f'[WARN] Cython compile skipped: {e}')


compile_pyx_files()

import utils, structs
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm_, dynamic_ncols=True)


# =========================================================================
#  Constants
# =========================================================================
LR_DECAY = 0.3    # 每 5 epoch 衰减因子（原论文策略）


def is_main_device(device):
    return isinstance(device, torch.device) or device == 0


# =========================================================================
#  训练一个 epoch（来自 v3.2，保留 intra-epoch checkpoint）
# =========================================================================
def train_one_epoch(model, iter_bar, optimizer, device, args, i_epoch,
                    queue=None, optimizer_2=None):
    li_FDE = []
    utils.other_errors_dict.clear()

    if args.distributed_training:
        assert dist.get_world_size() == args.distributed_training

    checkpoint_interval = getattr(args, 'checkpoint_interval', 0)
    resume_iter = getattr(args, 'resume_iter', 0)

    for step, batch in enumerate(iter_bar):
        if step < resume_iter:
            continue

        loss, DE, _ = model(batch, device)
        loss.backward()

        if is_main_device(device):
            iter_bar.set_description(f'loss={loss.item():.3f}')

        final_idx = batch[0].get('final_idx', -1)
        li_FDE.extend([each for each in DE[:, final_idx]])

        if optimizer_2 is not None:
            optimizer_2.step()
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        # intra-epoch checkpoint
        if checkpoint_interval > 0 and is_main_device(device) and (step + 1) % checkpoint_interval == 0:
            _save_intra_checkpoint(model, optimizer, optimizer_2, i_epoch, step + 1, args.model_save_dir)
            iter_bar.set_postfix_str(f'ckpt@{step + 1}')

    # 保存 epoch 结束的模型
    if not args.debug and is_main_device(device):
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.model_save_dir, f'model.{i_epoch + 1}.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        print(f'[Save] model.{i_epoch + 1}.bin')

        # 保存完整 checkpoint（含 optimizer）用于恢复
        checkpoint = {
            'epoch': i_epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if optimizer_2 is not None:
            checkpoint['optimizer_2_state_dict'] = optimizer_2.state_dict()
        torch.save(checkpoint, os.path.join(args.model_save_dir, 'checkpoint.pt'))

        # 清理 intra-epoch checkpoint
        for f in glob.glob(os.path.join(args.model_save_dir, f'checkpoint_intra_e{i_epoch}_s*.pt')):
            try:
                os.remove(f)
            except OSError:
                pass

    # 收集其他 worker 的 errors
    if args.argoverse:
        if is_main_device(device):
            for _ in range(args.distributed_training - 1):
                other_errors_dict_ = queue.get()
                for key in utils.other_errors_dict:
                    utils.other_errors_dict[key].extend(other_errors_dict_[key])
        else:
            queue.put(utils.other_errors_dict)

    if is_main_device(device):
        miss_rates = (utils.get_miss_rate(li_FDE, dis=2.0),
                      utils.get_miss_rate(li_FDE, dis=4.0),
                      utils.get_miss_rate(li_FDE, dis=6.0))
        utils.logging(f'Train FDE: {np.mean(li_FDE) if len(li_FDE) > 0 else None}',
                      f'MR(2m,4m,6m): {miss_rates}',
                      type='train_loss', to_screen=True)


def _save_intra_checkpoint(model, optimizer, optimizer_2, i_epoch, step, model_save_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'epoch': i_epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if optimizer_2 is not None:
        checkpoint['optimizer_2_state_dict'] = optimizer_2.state_dict()
    torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_intra_e{i_epoch}_s{step}.pt'))
    torch.save(checkpoint, os.path.join(model_save_dir, 'checkpoint_intra_latest.pt'))


# =========================================================================
# =========================================================================
#  验证（inline chunked，独立 temp_file_dir 避免缓存冲突）
# =========================================================================
def do_validate(args, epoch, model_save_dir, model):
    """Inline chunked validation — 用 chunked dataset 在进城内验证，避免 OOM"""
    model_path = os.path.join(model_save_dir, f'model.{epoch + 1}.bin')
    if not os.path.exists(model_path):
        print(f'[Val] Model not found: {model_path}')
        return float('nan'), float('nan'), float('nan')

    print(f'[Val] Epoch {epoch}: inline chunked validation...')

    # 独立 temp_file_dir 避免与训练缓存冲突
    val_args = copy.deepcopy(args)
    val_args.temp_file_dir = utils.get_eval_temp_dir(args.output_dir, args.data_dir_for_val)
    val_args.data_dir = [args.data_dir_for_val]  # ChunkedDataset expects list
    # 验证缓存由主进程预构建（build_validation_cache），此处直接复用。
    # 不能在 mp.spawn 子进程内用 fork Pool 重建缓存——fork 继承的
    # ArgoverseMap 状态会损坏，导致全部样本处理失败（历史上 0 条缓存）。
    val_args.reuse_temp_file = True
    val_args.do_eval = True           # must be True for origin_labels
    val_args.do_train = False         # not training during validation
    os.makedirs(val_args.temp_file_dir, exist_ok=True)
    os.makedirs(os.path.join(val_args.temp_file_dir, 'ex'), exist_ok=True)

    # 解包 DDP wrapper（如果使用分布式训练）
    if args.distributed_training and hasattr(model, 'module'):
        raw_model = model.module
    else:
        raw_model = model

    device = next(raw_model.parameters()).device

    # 创建 chunked eval dataset（用 val_args，独立缓存）
    from dataset_argoverse_chunked import Dataset as ChunkedDataset
    from torch.utils.data import SequentialSampler

    try:
        eval_dataset = ChunkedDataset(val_args, args.eval_batch_size, to_screen=True)
        if len(eval_dataset) == 0:
            raise RuntimeError("ChunkedDataset returned 0 valid entries — Pool workers likely could not access ArgoverseMap")
    except Exception:
        print("[Val] ChunkedDataset init failed, falling back to in-memory dataset_argoverse.Dataset...")
        import traceback
        traceback.print_exc()
        from dataset_argoverse import Dataset as MemDataset
        val_args.reuse_temp_file = True  # try cache first
        val_args.temp_file_dir = os.path.join(args.output_dir, 'temp_file_val_mem')
        os.makedirs(val_args.temp_file_dir, exist_ok=True)
        try:
            eval_dataset = MemDataset(val_args, args.eval_batch_size, to_screen=True)
        except Exception:
            # cache doesn't exist, build from scratch
            val_args.reuse_temp_file = False
            eval_dataset = MemDataset(val_args, args.eval_batch_size, to_screen=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=utils.batch_list_to_batch_tensors)

    # 推理 — must patch args.do_eval/do_train because decoder uses module-level global args
    raw_model.eval()
    file2pred, file2labels = {}, {}

    _saved_do_eval = args.do_eval
    _saved_do_train = args.do_train
    args.do_eval = True
    args.do_train = False

    try:
        with torch.no_grad():
            for batch in eval_dataloader:
                pred_trajectory, pred_score, _ = raw_model(batch, device)
                mapping = batch
                bs = pred_trajectory.shape[0]
                for i in range(bs):
                    fid = int(os.path.split(mapping[i]['file_name'])[1][:-4])
                    file2pred[fid] = pred_trajectory[i]
                    file2labels[fid] = mapping[i]['origin_labels']
    finally:
        args.do_eval = _saved_do_eval
        args.do_train = _saved_do_train
        raw_model.train()

    # 计算指标
    from argoverse.evaluation import eval_forecasting
    mr = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, args.future_frame_num, 2.0)

    minADE = mr.get('minADE', float('nan'))
    minFDE = mr.get('minFDE', float('nan'))
    MR = mr.get('MR', float('nan'))
    print(f'[Val] Epoch {epoch}: minADE={minADE:.4f}, minFDE={minFDE:.4f}, MR={MR:.4f}')
    return minADE, minFDE, MR


# =========================================================================
#  训练历史管理
# =========================================================================
HISTORY_FILE = 'training_history.json'

def load_history(output_dir):
    path = os.path.join(output_dir, HISTORY_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'epochs': [], 'losses': [], 'val_losses': [],
            'min_ade': [], 'min_fde': [], 'mr': [],
            'best_loss': 999999, 'best_epoch': 0, 'no_improve_count': 0}

def save_history(output_dir, history):
    path = os.path.join(output_dir, HISTORY_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, path)


# =========================================================================
#  DDP worker
# =========================================================================
def demo_basic(rank, world_size, kwargs, queue):
    args = kwargs['args']
    history = kwargs['history']
    stop_flag = kwargs['stop_flag']  # multiprocessing.Value for early stopping

    if world_size > 0:
        print(f"Running DDP on rank {rank}.")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        utils.args = args
        model = VectorNet(args).to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = VectorNet(args).to(rank)

    # Optimizer
    if 'complete_traj-3' in args.other_params:
        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters()
             if not n.startswith('module.decoder.complete_traj')],
            lr=args.learning_rate)
        optimizer_2 = torch.optim.Adam(
            [p for n, p in model.named_parameters()
             if n.startswith('module.decoder.complete_traj')],
            lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer_2 = None

    # Resume from checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(args.model_save_dir, 'checkpoint.pt')
    if args.resume and os.path.exists(checkpoint_path):
        print(f'[Resume] Loading checkpoint from {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if world_size > 0:
            model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except (ValueError, RuntimeError) as e:
            print(f'[Resume] Skipping optimizer state: {e}')
        if optimizer_2 is not None and 'optimizer_2_state_dict' in ckpt:
            try:
                optimizer_2.load_state_dict(ckpt['optimizer_2_state_dict'])
            except (ValueError, RuntimeError) as e:
                print(f'[Resume] Skipping optimizer_2 state: {e}')
        start_epoch = ckpt.get('epoch', 0)
        print(f'[Resume] Resuming from epoch {start_epoch}')

    # Sync barrier
    if rank == 0 and world_size > 0:
        receive = queue.get()
        assert receive == True
    if args.distributed_training:
        dist.barrier()
    # Only reuse temp_file cache if it actually exists (new output dir may not have one)
    _cache_path = os.path.join(args.temp_file_dir, utils.get_name('ex_list'))
    args.reuse_temp_file = os.path.exists(_cache_path)

    # Load training data
    from dataset_argoverse_chunked import Dataset as ChunkedDataset
    train_dataset = ChunkedDataset(args, args.train_batch_size, to_screen=(rank == 0))
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.train_batch_size // world_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=utils.batch_list_to_batch_tensors)

    # ===== Epoch loop =====
    for i_epoch in range(start_epoch, int(args.num_train_epochs)):
        # Check early stopping
        if stop_flag.value:
            print(f'[EarlyStop] Stopping at epoch {i_epoch} (flag set)')
            break

        # LR 按 epoch 确定性衰减：lr(epoch) = lr0 * LR_DECAY^(epoch // 5)。
        # 绝对公式保证 --resume 后与从头训练完全一致——旧的累积乘法会在
        # checkpoint 已衰减的 lr 上再乘一次（双重衰减，见 review S1）。
        lr_now = args.learning_rate * (LR_DECAY ** (i_epoch // 5))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now
        if optimizer_2 is not None:
            for pg in optimizer_2.param_groups:
                pg['lr'] = lr_now

        if rank == 0:
            print(f'\n{"="*60}')
            print(f'Epoch: {i_epoch}/{int(args.num_train_epochs)}  '
                  f'LR = {optimizer.param_groups[0]["lr"]:.8f}')
            print(f'{"="*60}')

        train_sampler.set_epoch(i_epoch)
        if rank == 0:
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else:
            iter_bar = train_dataloader

        # Train
        train_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue, optimizer_2)

        # Reset intra-epoch resume
        args.resume_iter = 0

        if args.distributed_training:
            dist.barrier()

        # Validation (rank 0 only)
        if rank == 0:
            print(f'\n[Val] Running validation after epoch {i_epoch}...')
            try:
                minADE, minFDE, MR = do_validate(args, i_epoch, args.model_save_dir, model)
                print(f'[Val] Epoch {i_epoch}: minADE={minADE:.4f}, minFDE={minFDE:.4f}, MR={MR:.4f}')
            except Exception as e:
                import traceback
                print(f'[Val] Validation failed: {e}')
                traceback.print_exc()
                minADE, minFDE, MR = float('nan'), float('nan'), float('nan')
                print(f'[Val] Skipping this epoch for early stopping (NaN)')

            # NaN check FIRST — before touching history
            if math.isnan(minFDE):
                print(f'[Val] Validation returned NaN, skipping this epoch entirely')
            else:
                # Update history (only valid numbers)
                history['epochs'].append(i_epoch)
                history['losses'].append(minFDE)
                history['val_losses'].append(minFDE)
                history.setdefault('min_ade', []).append(minADE)
                history.setdefault('min_fde', []).append(minFDE)
                history.setdefault('mr', []).append(MR)

                if minFDE < history['best_loss']:
                    history['best_loss'] = minFDE
                    history['best_epoch'] = i_epoch
                    history['no_improve_count'] = 0
                    print(f'[Val] *** NEW BEST *** Epoch {i_epoch}, minFDE={minFDE:.4f}')
                else:
                    history['no_improve_count'] += 1
                    print(f'[Val] No improvement ({history["no_improve_count"]}/{args.patience})')

            save_history(args.output_dir, history)

            if history['no_improve_count'] >= args.patience:
                print(f'[EarlyStop] Patience {args.patience} reached. Best: epoch {history["best_epoch"]}, '
                      f'minFDE={history["best_loss"]:.4f}')
                stop_flag.value = True

        # Broadcast stop flag to all workers
        if args.distributed_training:
            stop_tensor = torch.tensor([1 if stop_flag.value else 0], device=rank)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                stop_flag.value = True

    if args.distributed_training:
        dist.destroy_process_group()


# =========================================================================
#  验证缓存预构建（主进程）
# =========================================================================
def build_validation_cache(args):
    """在主进程构建验证集缓存，供 train 期间 inline validation 复用。

    背景：do_validate 在 mp.spawn 子进程内用 fork Pool 重建缓存时，fork 继承
    的 ArgoverseMap 状态会损坏，历史上导致全部验证样本处理失败（0 条缓存）。
    在主进程（无 spawn 污染）构建一次缓存后，所有 epoch 的验证直接复用。
    """
    if not args.data_dir_for_val or not os.path.exists(args.data_dir_for_val):
        print(f'[WARN] Validation dir not found: {args.data_dir_for_val}, '
              f'skip validation cache (inline validation will be unavailable)')
        return
    val_args = copy.deepcopy(args)
    val_args.temp_file_dir = utils.get_eval_temp_dir(args.output_dir, args.data_dir_for_val)
    val_args.data_dir = [args.data_dir_for_val]
    val_args.reuse_temp_file = False  # auto-detect existing cache
    val_args.do_eval = True           # origin_labels required for evaluation
    val_args.do_train = False
    os.makedirs(val_args.temp_file_dir, exist_ok=True)
    os.makedirs(os.path.join(val_args.temp_file_dir, 'ex'), exist_ok=True)
    from dataset_argoverse_chunked import Dataset as ChunkedDataset
    try:
        ds = ChunkedDataset(val_args, args.eval_batch_size, to_screen=False)
        print(f'[Val] Validation cache ready: {len(ds)} samples')
        del ds
    except Exception as e:
        print(f'[WARN] Validation cache build failed: {e}')
        import traceback
        traceback.print_exc()


# =========================================================================
#  主入口
# =========================================================================
def build_train_cache(args):
    """主进程预构建训练数据缓存。

    每个 DDP rank 的 demo_basic 都会构造 ChunkedDataset；若缓存不存在，
    多 rank 会并发重建同一批文件（竞态 + 重复劳动，见 review S4）。
    在主进程（spawn 之前）构建一次，所有 rank 复用。
    """
    cache_path = os.path.join(args.temp_file_dir, utils.get_name('ex_list'))
    if os.path.exists(cache_path):
        return
    from dataset_argoverse_chunked import Dataset as ChunkedDataset
    try:
        ds = ChunkedDataset(args, args.train_batch_size, to_screen=True)
        print(f'[Data] Training cache ready: {len(ds)} samples')
        del ds
    except Exception as e:
        print(f'[WARN] Training cache build failed ({e}); ranks will retry independently')
        import traceback
        traceback.print_exc()


def run(args):
    history = load_history(args.output_dir)
    # 预构建验证缓存必须在主进程完成（详见 build_validation_cache docstring）
    build_validation_cache(args)
    # 训练缓存同样在主进程预构建，避免多 rank 并发重建（见 S4）
    build_train_cache(args)
    ctx = mp.get_context('spawn')
    stop_flag = ctx.Value('i', 0)

    # 如果 history 里已有足够 epochs，检查是否需要恢复
    if history['no_improve_count'] >= args.patience:
        print(f'[Info] Previous run hit patience limit. Resetting no_improve_count.')
        history['no_improve_count'] = 0

    print(f'[Config] batch_size={args.train_batch_size}, lr={args.learning_rate}, '
          f'epochs={args.num_train_epochs}, patience={args.patience}')
    print(f'[Config] LR decay={LR_DECAY}x every 5 epochs (paper strategy)')
    print(f'[Config] Data: {args.data_dir}')
    print(f'[Config] Val:   {args.data_dir_for_val}')

    if args.distributed_training:
        queue = ctx.Manager().Queue()
        kwargs = {'args': args, 'history': history, 'stop_flag': stop_flag}
        spawn_context = mp.spawn(demo_basic,
                                 args=(args.distributed_training, kwargs, queue),
                                 nprocs=args.distributed_training,
                                 join=False)
        # Signal train dataset ready (no need to create full dataset here)
        queue.put(True)
        spawn_context.join()
    else:
        assert False, 'Please set "--distributed_training 1"'


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--checkpoint_interval", default=100, type=int)
    parser.add_argument("--resume_iter", default=0, type=int,
                        help="Skip the first N batches of the first training epoch "
                             "(intra-epoch resume; reset after each epoch)")
    parser.add_argument("--patience", default=5, type=int)
    args = parser.parse_args()
    utils.init(args, logger)

    logger.info(f"device: cuda")

    if args.argoverse:
        if args.do_train:
            run(args)
        else:
            from do_eval import do_eval
            do_eval(args)
    else:
        assert False

    logger.info('Finish.')


if __name__ == "__main__":
    main()
