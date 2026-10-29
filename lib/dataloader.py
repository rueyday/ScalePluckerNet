"""Training data for the matcher: one item = (C (N1,N2) GT correspondence matrix, plucker1 (N1,6), plucker2 (N2,6)),
plucker1 = REFERENCE, plucker2 = QUERY, rows [m; d].  No input normalization: the network is Sim(3)-invariant.

    PairData(folder)        a split written by fullrange_sp5_gen.py, <data_dir>/<dataset>_<split>/{matches,plucker1,
                            plucker2}.pkl, served from a packed memory-mapped copy built next to it once
    LiveData(epoch_size)    fresh generator pairs every epoch (train.py --live)
    BucketBatchSampler      batches of same-shape pairs (the generator snaps sizes to a grid)
"""
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

VARS = {'matches': 1, 'plucker1': 0, 'plucker2': 0}      # variable -> concatenation axis in the packed copy
PACK = 'packed_v1'


def cap_cloud(pl, matches, row, max_n):
    """Subsample a cloud to <= max_n lines, keeping every line with a GT match on `row` of `matches` (2, n)
    (0 = plucker1, 1 = plucker2), and reindex the matches."""
    n = len(pl)
    if n <= max_n:
        return pl, matches
    keep = np.unique(matches[row])
    others = np.setdiff1d(np.arange(n), keep)
    if max_n > len(keep):
        sel = np.concatenate([keep, np.random.choice(others, min(len(others), max_n - len(keep)), replace=False)])
    else:
        sel = keep[:max_n]                                   # more matched lines than max_n (rare): truncate
    sel.sort()
    remap = -np.ones(n, np.int64)
    remap[sel] = np.arange(len(sel))
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
    """<folder>/packed_v1 from the per-variable pickles, one variable at a time (peak ~2x the largest one).

    Why (2026-10-26, v103): a list of 120k small arrays is millions of Python objects; every forked DataLoader worker
    touches their refcounts, copy-on-write duplicates the pages, and 7 GB of data OOM-killed the trainer 6x under a
    20 GB cap.  A memory map is shared page cache: no copies, reclaimable."""
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
    def __init__(self, folder=None, max_lines=6000):
        self.max_lines = max_lines
        if folder is None:                                   # LiveData fills self.data itself
            return
        pk = os.path.join(folder, PACK)
        if not os.path.isdir(pk):
            print(f'packing {folder} -> {PACK} (once)', flush=True)
            _pack(folder)
        self.data = {v: _Ragged(np.load(os.path.join(pk, f'{v}.npy'), mmap_mode='r'),
                                np.load(os.path.join(pk, f'{v}_off.npy')), ax) for v, ax in VARS.items()}
        print(f'{folder}: {len(self)} pairs', flush=True)

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
        p1, p2 = self.data['plucker1'], self.data['plucker2']
        if isinstance(p1, _Ragged):
            return list(zip(p1.lengths().tolist(), p2.lengths().tolist()))
        return [(len(a), len(b)) for a, b in zip(p1, p2)]


def collate(batch):
    """Stack same-shape items (BucketBatchSampler guarantees it; the cap maps equal shapes to equal shapes)."""
    return tuple(torch.stack(x) for x in zip(*batch))


# ---- live generation ----
# A fixed set is re-seen 25x (300k pairs) to 154x (50k) over a 200-epoch run, and v63 measured the cost (the 50k subset
# trailed the 300k parent by ~2 points, widening).  Live pairs never repeat and need no dataset in RAM.  One epoch is
# buffered (the sampler needs every shape up front) while the next one is generated in the background.

def _seed_worker(seed):
    np.random.seed((seed * 7919 + os.getpid()) % (2 ** 32))


def _generate(_):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from fullrange_sp5_gen import generate_pair
    d = generate_pair()
    return d['matches'], d['plucker1'], d['plucker2']


class LiveData(PairData):
    def __init__(self, epoch_size, workers=8, seed=0, max_lines=6000):
        super().__init__(None, max_lines)
        import multiprocessing as mp
        self.epoch_size = int(epoch_size)
        self._pool = mp.get_context('spawn').Pool(workers, initializer=_seed_worker, initargs=(seed,))
        # imap, not map: map materialises the whole 32000-pair result in the parent next to the buffer
        self._fill(list(self._pool.imap_unordered(_generate, range(self.epoch_size), chunksize=64)))
        self._next = None

    def _fill(self, rows):
        self.data = dict(zip(('matches', 'plucker1', 'plucker2'), map(list, zip(*rows))))

    def next_epoch(self):
        """Swap in the prefetched epoch (the first epoch was made by the constructor) and start the next one.
        map_async, not a Python thread: a thread collecting imap results holds the GIL and does not overlap training
        (measured 14.7 vs 9.6 min/epoch)."""
        if self._next is not None:
            self._fill(self._next.get())
        self._next = self._pool.map_async(_generate, range(self.epoch_size), chunksize=64)


class BucketBatchSampler(Sampler):
    """Batches of same-shape (n1, n2) pairs.  Batch size max(1, min(max_b, budget // n, mem_budget // n^2)),
    n = max(n1, n2): attention is O(B n^2), so the quadratic cap keeps large clouds alone (measured on the RTX 5090:
    n=512 gains +52% going B 8 -> 32, n=1094 gains nothing, n>=2460 would OOM).  Each epoch draws epoch_size pairs
    without replacement and shuffles the batch order."""

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

    def start_epoch(self, epoch):
        """Call ONCE per epoch, BEFORE iter(loader).  The live buffer swap must not happen in __iter__: DataLoader
        calls iter(sampler) twice per iter(loader), once before forking its workers and once after, so a swap there
        reached the parent only and the workers served the OLD buffer at indices bucketed by the NEW shapes (the v83
        OOM, 2026-09-29).  __iter__ is a pure function of (shapes, epoch)."""
        self.epoch = int(epoch)
        if isinstance(self.dataset, LiveData):
            self.dataset.next_epoch()
            self.shapes = self.dataset.shapes()

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
