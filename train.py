#!/usr/bin/env python3
"""Train the matcher (lib/model.py).  The defaults ARE the shipped recipe (v102):

    python train.py --dataset synthetic_fsp5q8k --name <run> --live [--gpu_mem_frac 0.7]

Data: --live generates every epoch's pairs with fullrange_sp5_gen.py; without it, the written split
<data_dir>/<dataset>_train.  Validation always reads <data_dir>/<dataset>_valid.  Resume a run with
--resume output/<dataset>/<run>/checkpoint.pth; warm-start the weights only with --init_from <ckpt>.
"""
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.dataloader import BucketBatchSampler, LiveData, PairData, collate  # noqa: E402
from lib.trainer import Trainer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--name', required=True, help='run name: output/<dataset>/<name>/')
    p.add_argument('--data_dir', default='./dataset')
    p.add_argument('--out_dir', default='output')
    p.add_argument('--resume', help='checkpoint to resume (weights, optimizer, schedule, epoch)')
    p.add_argument('--init_from', help='checkpoint to take the weights from, with a fresh optimizer and schedule')
    p.add_argument('--live', action='store_true', help='generate fresh training pairs every epoch')
    p.add_argument('--live_workers', type=int, default=8, help='generator processes')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--train_epoch_size', type=int, default=32000, help='pairs per epoch (0 = the whole split)')
    p.add_argument('--lr', type=float, default=5e-4, help='cosine from lr to 1e-6 over the run')
    p.add_argument('--iter_size', type=int, default=32, help='samples per optimizer step')
    p.add_argument('--match_w', type=float, default=0.2, help='weight of the matchability loss')
    p.add_argument('--max_lines', type=int, default=6000, help='cap per cloud, keeping every matched line')
    p.add_argument('--batch_budget', type=int, default=16384, help='bucket batch <= budget // n')
    p.add_argument('--max_bucket', type=int, default=32, help='bucket batch <= max_bucket')
    p.add_argument('--mem_budget', type=int, default=8400000, help='bucket batch <= mem_budget // n^2 (0 = off)')
    p.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    p.add_argument('--val_epoch_freq', type=int, default=1)
    p.add_argument('--val_max_iter', type=int, default=-1, help='validation pairs per epoch (-1 = all)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--gpu_mem_frac', type=float, default=0.0, help='cap this process at this share of GPU memory')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)])
    if args.gpu_mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac, args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.live:
        n = args.train_epoch_size or 32000
        logging.info(f'generating the first epoch ({n} pairs, {args.live_workers} workers)')
        train = LiveData(n, workers=args.live_workers, seed=args.seed, max_lines=args.max_lines)
    else:
        train = PairData(os.path.join(args.data_dir, f'{args.dataset}_train'), args.max_lines)
    sampler = BucketBatchSampler(train, args.train_epoch_size, args.batch_budget, args.max_bucket, args.mem_budget,
                                 args.seed)
    # live: workers fork per epoch (not persistent) so they read the fresh buffer copy-on-write
    train_loader = DataLoader(train, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0 and not args.live, collate_fn=collate)
    val = PairData(os.path.join(args.data_dir, f'{args.dataset}_valid'), args.max_lines)
    val_loader = DataLoader(val, batch_size=1, num_workers=2, pin_memory=True, persistent_workers=True,
                            collate_fn=collate)
    logging.info(f'{args.name}: {sampler.epoch_size} pairs/epoch, {len(set(sampler.shapes))} shapes, '
                 f'{args.epochs} epochs, {"live" if args.live else "file-backed"}')
    Trainer(args, train_loader, val_loader, sampler.epoch_size).train()


if __name__ == '__main__':
    main()
