"""Training loop for ScalePluckerNet.

One optimizer step = iter_size SAMPLES (a bucketed batch carries B of them), summed loss, grad-norm clip 1.0, Adam,
cosine LR from lr to 1e-6 over the run
(or --schedule exp: lr x gamma per epoch, PlueckerNet's).  Validation reports the share of GT correspondences among the top-100 entries
of the plan (avg_inlier_ratio), which picks best_val_checkpoint.  It is a training monitor only: rank checkpoints by
registration on the real benchmarks, never by this number.
"""
import hashlib
import json
import logging
import os
import time

import numpy as np
import torch

from lib.loss import plan_bce
from lib.model import HEADS, LAYERS, WIDTH, ScalePluckerNet


class _Writer:
    """Logs one number per (tag, step) to TensorBoard, mirrored to Weights & Biases when WANDB=1."""

    def __init__(self, logdir, config):
        from tensorboardX import SummaryWriter
        self.tb, self.wb, self.steps_per_epoch = SummaryWriter(logdir=logdir), None, None
        if os.environ.get('WANDB', '0') != '1':
            return
        try:
            import wandb
            name = os.path.basename(logdir.rstrip('/'))
            wandb.init(project=os.environ.get('WANDB_PROJECT', 'scalepluckernet'), name=name,
                       id=hashlib.md5(name.encode()).hexdigest()[:8], config=config, resume='allow')
            for k in ('epoch', 'train/*', 'val/*'):
                wandb.define_metric(k, **({} if k == 'epoch' else {'step_metric': 'epoch'}))
            self.wb = wandb
        except Exception as e:
            logging.warning(f'wandb disabled ({type(e).__name__}: {e})')

    def log(self, tag, value, step):
        self.tb.add_scalar(tag, value, step)
        if self.wb is not None:
            ep = step / self.steps_per_epoch + 1.0 if tag.startswith('train/') else float(step)
            try:
                self.wb.log({tag: float(value), 'epoch': ep})
            except Exception:
                pass


class Trainer:
    def __init__(self, args, train_loader, val_loader, epoch_size):
        self.args, self.train_loader, self.val_loader, self.epoch_size = args, train_loader, val_loader, epoch_size
        self.device = torch.device('cuda', args.gpu)
        torch.cuda.set_device(self.device)
        self.model = ScalePluckerNet().to(self.device)
        # compiled handle for forward passes (~1.5x on this launch-overhead-bound model); checkpoints are saved from
        # self.model so their keys carry no _orig_mod. prefix
        self.net = torch.compile(self.model, dynamic=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        if args.schedule == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        else:   # T_0 = epochs + 1: epochs 0..epochs are trained (inclusive), WarmRestarts would restart at T_0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=args.epochs + 1, T_mult=1, eta_min=1e-6)
        self.start_epoch, self.best_val, self.best_val_epoch = 0, -np.inf, -1
        # the architecture keys let load_matcher refuse a checkpoint of any other architecture
        self.config = dict(vars(args), encoder='ordinal_q_wide', net_nchannel=WIDTH, GNN_layers=LAYERS, attn_heads=HEADS)
        self.dir = os.path.join(args.out_dir, args.dataset, args.name)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        if args.resume:
            s = torch.load(args.resume, weights_only=False, map_location='cpu')
            self.model.load_state_dict(s['state_dict'])
            self.optimizer.load_state_dict(s['optimizer'])
            self.scheduler.load_state_dict(s['scheduler'])
            # s['epoch'] FINISHED before it was saved: resume at the next one (v104 re-trained epoch 2 five times)
            self.start_epoch = s['epoch'] + 1
            self.best_val, self.best_val_epoch = s.get('best_val', -np.inf), s.get('best_val_epoch', -1)
            logging.info(f'resumed {args.resume} at epoch {self.start_epoch}')
        elif args.init_from:
            self.model.load_state_dict(torch.load(args.init_from, weights_only=False, map_location='cpu')['state_dict'])
            logging.info(f'weights from {args.init_from} (fresh optimizer and schedule)')
        self.writer = _Writer(self.dir, self.config)

    def train(self):
        v = self.validate()
        self.writer.log('val/avg_inlier_ratio', v, self.start_epoch - 1)          # = after the previous epoch
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            logging.info(f'Epoch {epoch}, LR {self.scheduler.get_last_lr()}')
            self.train_epoch(epoch)
            self.scheduler.step()
            if epoch % self.args.val_epoch_freq == 0:
                v = self.validate()
                self.writer.log('val/avg_inlier_ratio', v, epoch)
                if v > self.best_val:
                    self.best_val, self.best_val_epoch = v, epoch
                    self.save(epoch, 'best_val_checkpoint')
                logging.info(f'best avg_inlier_ratio {self.best_val:.3f} at epoch {self.best_val_epoch}')
            # saved LAST, after the scheduler step and the best-val update: a resume continues with the next epoch's
            # LR and the current best (saved before them, every resume trained one LR step behind and a stale
            # best_val could let a worse epoch overwrite best_val_checkpoint)
            self.save(epoch, 'checkpoint')

    def save(self, epoch, name):
        torch.save(dict(epoch=epoch, state_dict=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(), config=self.config, best_val=self.best_val,
                        best_val_epoch=self.best_val_epoch, best_val_metric='avg_inlier_ratio'),
                   os.path.join(self.dir, f'{name}.pth'))

    def train_epoch(self, epoch):
        self.model.train()
        self.train_loader.batch_sampler.set_epoch(epoch)
        it = iter(self.train_loader)
        n_steps = max(1, self.epoch_size // self.args.iter_size)
        self.writer.steps_per_epoch = n_steps
        t_data = t_all = 0.0
        for step in range(n_steps):
            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss_sum = torch.zeros((), device=self.device)
            prob_sum = torch.zeros((), device=self.device)
            n = 0
            while n < self.args.iter_size:
                t1 = time.perf_counter()
                try:
                    C, p1, p2 = next(it)
                except StopIteration:
                    # a bucketed batch overshoots iter_size by up to B-1 samples, so the draw runs out before n_steps
                    # (~15% early): finish the step with what there is
                    break
                t_data += time.perf_counter() - t1
                C, p1, p2 = (x.to(self.device, non_blocking=True) for x in (C, p1, p2))
                B = C.shape[0]
                P = self.net(p1, p2)
                loss = plan_bce(P, C) * B                                    # x B: sum over samples at any B
                loss.backward()
                loss_sum += loss.detach()
                prob_sum += ((1.0 - 2.0 * C) * P.detach()).sum(dim=(-2, -1)).mean() * B
                n += B
            if n == 0:
                break
            # one host sync per step: a non-finite gradient (a NaN sample) skips the step instead of a per-sample check
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)):
                self.optimizer.step()
            else:
                logging.warning(f'epoch {epoch} step {step}: non-finite gradient, step skipped')
            t_all += time.perf_counter() - t0
            if step % 10 == 0:
                gstep = (epoch - 1) * n_steps + step
                self.writer.log('train/total_loss', loss_sum.item(), gstep)
                self.writer.log('train/prob_loss', prob_sum.item(), gstep)
                k = 1 if step == 0 else 10
                # prob_loss = sum over the step's pairs of 1 - 2 x (plan mass on GT pairs): lower is better
                logging.info(f'epoch {epoch} [{step}/{n_steps}] loss {loss_sum.item():.3e} '
                             f'prob_loss {prob_sum.item():.3f} data {t_data / k:.4f}s train {(t_all - t_data) / k:.4f}s')
                t_data = t_all = 0.0
            if n < self.args.iter_size:
                break

    @torch.no_grad()
    def validate(self):
        """Mean % of GT correspondences among the plan's top-100 entries.  Pairs whose plucker1 has more than 3000
        lines are skipped, as in every recorded run (the check was written for an inlier count and has always read
        the line count), so best_val stays comparable across runs."""
        self.model.eval()
        ratios = []
        for i, (C, p1, p2) in enumerate(self.val_loader):
            if self.args.val_max_iter > 0 and i >= self.args.val_max_iter:
                break
            if not 2 <= C.shape[1] <= 3000:
                continue
            P = self.net(p1.to(self.device), p2.to(self.device)).flatten(1)
            k = min(100, P.shape[1])
            top = P.topk(k, dim=-1).indices[0]
            ratios.append(float(C.flatten(1)[0, top.cpu()].sum()) / k * 100.0 if k > 3 else 0.0)
        v = float(np.mean(ratios)) if ratios else 0.0
        logging.info(f'avg_inlier_ratio {v:.1f}% over {len(ratios)} pairs')
        return v
