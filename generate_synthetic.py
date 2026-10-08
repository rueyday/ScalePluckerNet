"""generate_synthetic.py -- ONE GENERATOR: FULLRANGE (Ruey 2026-10-07,
"make v84's the only dataset the generator make and clean up the generating
script too").

Every training pair now comes from fullrange_gen.generate_pair_fullrange(): the
dimensionless planar-scene generator written for the exactly Sim(3)-invariant
ordinal encoder. Pose is a no-op for that encoder, so the whole sampling budget
goes to the axes it must actually learn -- sizes, overlap, structure,
population, noise, fragmentation. See fullrange_gen.py for the ranges and the
measurements behind them.

WHAT WAS REMOVED (the previous 1848-line file is kept verbatim at
attic/generate_synthetic_pre_fullrange_2026-10-07.py):
  * the 16-scenario _SPEC table and _sample_scenario
  * the pose machinery -- scale laws, rotation mixtures, street calibrations,
    drift knots -- all dead weight for a Sim(3)-invariant encoder
  * flags FOUND / INDOOR / MONO / DEALIAS / SOUND / SIMPLE / LOCAL / XMODAL /
    BROAD / KITTI_MONO / FULL / NOPOSE / ROT_CAL and every *_P mixture knob
  * REAL / REAL_P   (real ScanNet++ mesh creases) -- TESTED NEGATIVE 2026-10-06:
    ScanNet++ strict 228 vs the control's 246 on the 500-scene cell
  * KITTI_SUB / KITTI_SUB_P (real held-out KITTI submaps) -- TESTED NEGATIVE
    2026-10-07: on the 8 genuinely held-out submap pairs it scored 0 rot<5 /
    1 rot<10 vs v84's 3/3, i.e. it memorised the training sequences
Both real-data arms failed for the same reason: too few distinct real scenes.

WHAT SURVIVES, and why each is still here:
  _physical_noise     used by fullrange_gen (Rayleigh direction + perpendicular
                      offset, preserves m.d = 0)
  _apply_traj_drift   used by fullrange_gen (the one non-rigid effect, which is
                      also the only thing that breaks exact invariance)
  _quantize_cloud     snaps cloud sizes onto _SIZE_GRID so same-shape pairs
                      batch (--bucket_batch depends on it)
  _MIN_INLIERS        the minimum trainable correspondence count
  _generate_pair      the entry point lib/dataloader.py imports for --live
  generate_split      the file-backed dataset builder (CLI below)
All extracted VERBATIM from the previous file -- not retyped.
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent
_MIN_INLIERS      = 20
_SIZE_GRID = np.array([64, 96, 128, 192, 256, 384, 512, 768,
                       1024, 1536, 2048, 3072, 4096])


def _physical_noise(L: np.ndarray, dir_sigma_deg: float, perp_sigma_m: float,
                    dir_cap_deg: float = 90.0) -> np.ndarray:
    """Perturb lines physically: Rayleigh rotation of the direction plus a
    Rayleigh perpendicular offset of the foot point, then recompute m.
    Preserves m·d = 0 (unlike additive Gaussian noise on m)."""
    if len(L) == 0:
        return L
    m, d = L[:, :3], L[:, 3:]
    n = len(L)
    p0 = np.cross(d, m)                                   # foot of perpendicular
    ax = np.cross(d, np.random.randn(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ang = np.radians(np.minimum(np.random.rayleigh(dir_sigma_deg, n), dir_cap_deg))
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d_new = ca * d + sa * np.cross(ax, d) + (1 - ca) * (ax * d).sum(1, keepdims=True) * ax
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-9
    off = np.random.randn(n, 3)
    off -= (off * d_new).sum(1, keepdims=True) * d_new
    off *= (np.random.rayleigh(perp_sigma_m, n)
            / (np.linalg.norm(off, axis=1) + 1e-9))[:, None]
    m_new = np.cross(p0 + off, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _apply_traj_drift(lines: np.ndarray) -> np.ndarray:
    """
    FOUND drift model: smooth trajectory-correlated Sim(3) warp, matching the
    drift measured on real KITTI mono maps (build_kitti_gt_corr_v2: warp
    median 2-13 m over 100-400 m extents = 1-5% of extent, local scale
    p10/p90 ~ 0.95/1.09, slowly varying along the drive).

    Lines are parameterized by the projection tau of their perpendicular foot
    point onto a random axis (a proxy for "position along the trajectory").
    Piecewise-linear control knots define smooth rotation / scale /
    translation fields over tau; each line is warped by its local Sim(3)
    about the moving centre. Preserves m . d = 0 exactly.
    """
    n = len(lines)
    if n < 4:
        return lines
    m, d = lines[:, :3].astype(np.float64), lines[:, 3:].astype(np.float64)
    p0 = np.cross(d, m)                       # perpendicular foot points
    ctr = p0.mean(0)
    ext = float(np.abs(p0 - ctr).max()) + 1e-9
    u = np.random.randn(3); u /= np.linalg.norm(u) + 1e-12
    tau = (p0 - ctr) @ u
    tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-9)

    K = np.random.randint(4, 9)               # control knots
    kt = np.linspace(0.0, 1.0, K)
    # random-walk knot values (drift accumulates), zeroed at the start
    def walk(scale, dim):
        w = np.cumsum(np.random.randn(K, dim), axis=0) * scale
        return w - w[0]
    rot_amp   = np.radians(np.random.uniform(0.5, 4.0))
    scale_amp = np.random.uniform(0.02, 0.12)
    trans_amp = np.random.uniform(0.01, 0.05) * ext
    k_rot   = walk(rot_amp / np.sqrt(K), 3)         # axis-angle per knot
    k_scale = walk(scale_amp / np.sqrt(K), 1)[:, 0]  # log-scale per knot
    k_tr    = walk(trans_amp / np.sqrt(K), 3)

    w_rot = np.stack([np.interp(tau, kt, k_rot[:, i]) for i in range(3)], 1)
    sig   = np.exp(np.interp(tau, kt, k_scale))
    delta = np.stack([np.interp(tau, kt, k_tr[:, i]) for i in range(3)], 1)

    th = np.linalg.norm(w_rot, axis=1, keepdims=True) + 1e-12
    ax = w_rot / th
    ca, sa = np.cos(th), np.sin(th)
    def rot(v):                                # Rodrigues, rowwise
        return (ca * v + sa * np.cross(ax, v)
                + (1 - ca) * (ax * v).sum(1, keepdims=True) * ax)
    p_new = sig[:, None] * rot(p0 - ctr) + ctr + delta
    d_new = rot(d)
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-12
    m_new = np.cross(p_new, d_new)
    return np.concatenate([m_new, d_new], 1).astype(np.float32)


def _quantize_cloud(pl: np.ndarray, matches: np.ndarray, row: int):
    """FOUND: uniformly subsample a cloud to the largest _SIZE_GRID value <= n
    (cap 4096) so same-shape pairs batch at train time. Uniform (not
    inlier-preserving) so the inlier fraction stays honest; dropped matches
    are removed."""
    n = len(pl)
    g = int(_SIZE_GRID[_SIZE_GRID <= max(n, _SIZE_GRID[0])].max())
    g = min(g, n)
    if n == g:
        return pl, matches
    sel = np.sort(np.random.choice(n, g, replace=False))
    remap = -np.ones(n, np.int64); remap[sel] = np.arange(g)
    mi = matches.copy()
    if mi.shape[1] > 0:
        mi[row] = remap[mi[row]]
        mi = mi[:, mi[row] >= 0]
    return pl[sel], mi


def _generate_pair() -> dict:
    """THE one pair generator. No flags, no mixture, no scenarios."""
    from fullrange_gen import generate_pair_fullrange
    return generate_pair_fullrange()


def _worker(args):
    n, seed = args
    np.random.seed(seed)
    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    chunk = {k: [] for k in keys}
    for _ in range(n):
        pair = _generate_pair()
        for k in keys:
            chunk[k].append(pair[k])
    return chunk


def generate_split(n_pairs: int, workers: int, chunk_size: int,
                   seed: int, label: str) -> dict:
    n_chunks  = max(1, (n_pairs + chunk_size - 1) // chunk_size)
    remaining = n_pairs
    tasks     = []
    for i in range(n_chunks):
        n = min(chunk_size, remaining)
        tasks.append((n, seed + i * 7919))
        remaining -= n
        if remaining <= 0:
            break

    keys     = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    combined = {k: [] for k in keys}
    t0, done = time.time(), 0

    if workers == 1:
        for task in tasks:
            chunk = _worker(task)
            for k in keys:
                combined[k].extend(chunk[k])
            done += task[0]
            elapsed = time.time() - t0
            print(f'  {label}: {done}/{n_pairs}  ({done/elapsed:.0f} pairs/s)', end='\r')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker, tasks):
                for k in keys:
                    combined[k].extend(chunk[k])
                done += chunk_size
                elapsed = time.time() - t0
                print(f'  {label}: {min(done, n_pairs)}/{n_pairs}  '
                      f'({done/elapsed:.0f} pairs/s)', end='\r')

    print()
    return combined


def main():
    ap = argparse.ArgumentParser(description='FULLRANGE synthetic Sim(3) line-map pairs')
    ap.add_argument('--n_train', type=int, default=200_000)
    ap.add_argument('--n_valid', type=int, default=2_000)
    ap.add_argument('--out_dir', default=str(_ROOT / 'dataset'))
    ap.add_argument('--name', default='synthetic_fullrange')
    ap.add_argument('--chunk_size', type=int, default=2_000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    np.random.seed(a.seed)
    for split, n in (('train', a.n_train), ('valid', a.n_valid)):
        out = os.path.join(a.out_dir, f'{a.name}_{split}')
        print(f'[{time.strftime("%T")}] {split}: {n} pairs -> {out}', flush=True)
        generate_split(n, a.workers, a.chunk_size, out)


if __name__ == '__main__':
    main()
