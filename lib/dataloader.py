"""Training data for the matcher: one item = (C (N1,N2) GT correspondence matrix, plucker1 (N1,6), plucker2 (N2,6)),
plucker1 = REFERENCE, plucker2 = QUERY, rows [m; d].  No input normalization: the network is Sim(3)-invariant.

    PairData(folder)        a split written by fullrange_sp5_gen.py, <data_dir>/<dataset>_<split>/{matches,plucker1,
                            plucker2}.pkl, served from a packed memory-mapped copy built next to it once
    BucketBatchSampler      batches of same-shape pairs (the generator snaps sizes to a grid)

Live generation (a fresh epoch from the generator while the last one trained) was removed 2026-10-29: too much
overhead (merged pairs 0.17 s each); train on a written split.
"""
import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

VARS = {'matches': 1, 'plucker1': 0, 'plucker2': 0}      # variable -> concatenation axis in the packed copy
PACK = 'packed_v1'

def cap_cloud(pl, matches, row, max_n):
    """Uniformly subsample a cloud to <= max_n lines, as inference does (registration.cap), and keep the GT matches on
    `row` of `matches` (2, n) (0 = plucker1, 1 = plucker2) whose line survived, reindexed.  Uniform on purpose: the
    earlier version kept every matched line, so a capped training pair was cleaner than a capped test map (2026-10-29)."""
    n = len(pl)
    if n <= max_n:
        return pl, matches
    sel = np.sort(np.random.choice(n, max_n, replace=False))
    remap = -np.ones(n, np.int64)
    remap[sel] = np.arange(max_n)
    m = matches.copy()
    m[row] = remap[m[row]]
    return pl[sel], m[:, m[row] >= 0]


class _Ragged:
    """A list of arrays stored as ONE concatenated (memory-mapped) array + offsets; item i is a view along `axis`."""

    def __init__(self, cat, off, axis):
        self.cat, self.off, self.axis = cat, off, axis

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.cat[a:b] if self.axis == 0 else self.cat[:, a:b]

    def lengths(self):
        return np.diff(self.off)


def _pack(folder):
    """<folder>/packed_v1 from the per-variable pickles, one variable at a time """
    out, tmp = os.path.join(folder, PACK), os.path.join(folder, PACK + '.tmp')
    os.makedirs(tmp, exist_ok=True)
    for var, ax in VARS.items():
        with open(os.path.join(folder, f'{var}.pkl'), 'rb') as f:
            L = pickle.load(f, encoding='latin1')
        np.save(os.path.join(tmp, f'{var}_off.npy'), np.r_[0, np.cumsum([x.shape[ax] for x in L])].astype(np.int64))
        np.save(os.path.join(tmp, f'{var}.npy'), np.concatenate(L, axis=ax))
        del L
    os.replace(tmp, out)


class PairData(Dataset):
    def __init__(self, folder, max_lines=6000):
        self.max_lines = max_lines
        pk = os.path.join(folder, PACK)
        if not os.path.isdir(pk):
            logging.info(f'packing {folder} -> {PACK} (once)')
            _pack(folder)
        self.data = {v: _Ragged(np.load(os.path.join(pk, f'{v}.npy'), mmap_mode='r'),
                                np.load(os.path.join(pk, f'{v}_off.npy')), ax) for v, ax in VARS.items()}
        logging.info(f'{folder}: {len(self)} pairs')

    def __len__(self):
        return len(self.data['plucker1'])

    def __getitem__(self, i):
        m = np.asarray(self.data['matches'][i])
        p1, p2 = np.asarray(self.data['plucker1'][i]), np.asarray(self.data['plucker2'][i])
        if self.max_lines:
            p1, m = cap_cloud(p1, m, 0, self.max_lines)
            p2, m = cap_cloud(p2, m, 1, self.max_lines)
        C = np.zeros((len(p1), len(p2)), np.float32)
        C[m[0], m[1]] = 1.0
        return torch.from_numpy(C), torch.from_numpy(p1.astype(np.float32)), torch.from_numpy(p2.astype(np.float32))

    def shapes(self):
        """(n1, n2) per pair, before the cap, for BucketBatchSampler."""
        return list(zip(self.data['plucker1'].lengths().tolist(), self.data['plucker2'].lengths().tolist()))


def collate(batch):
    """Stack same-shape items (BucketBatchSampler guarantees it; the cap maps equal shapes to equal shapes)."""
    return tuple(torch.stack(x) for x in zip(*batch))


class BucketBatchSampler(Sampler):
    """Batches of same-shape (n1, n2) pairs.  Batch size max(1, min(max_b, budget // n, mem_budget // n^2)),
    n = max(n1, n2): attention is O(B n^2). Each epoch draws pairs without replacement and shuffles the order."""

    def __init__(self, dataset, epoch_size, budget=16384, max_b=32, mem_budget=8400000, seed=0):
        self.dataset, self.seed, self.epoch = dataset, seed, 0
        self.budget, self.max_b, self.mem_budget = budget, max_b, mem_budget
        self.shapes = dataset.shapes()
        self.epoch_size = min(epoch_size, len(self.shapes)) if epoch_size > 0 else len(self.shapes)

    def batch_size(self, n1, n2):
        n = max(n1, n2)
        b = min(self.max_b, self.budget // n)
        if self.mem_budget:
            b = min(b, self.mem_budget // (n * n))
        return max(1, b)

    def set_epoch(self, epoch):
        """Call before iter(loader); the epoch seeds that epoch's draw (__iter__ is a pure function of it)."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = {}
        for i in rng.choice(len(self.shapes), self.epoch_size, replace=False):
            groups.setdefault(self.shapes[i], []).append(int(i))
        batches = []
        for sh, ids in groups.items():
            B = self.batch_size(*sh)
            batches += [ids[j:j + B] for j in range(0, len(ids), B)]
        return iter([batches[j] for j in rng.permutation(len(batches))])
