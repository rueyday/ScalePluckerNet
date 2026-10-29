"""fullrange_sp5_gen.py -- synthetic training pairs for the matcher (fsp5): ScanNet++-like indoor + spacecraft.

    python fullrange_sp5_gen.py --name synthetic_fsp5 --n_train 300000 --n_valid 3000 --workers 8
    (train.py --live calls generate_pair() below instead of reading a written split)

Two domains, 70% indoor / 30% space; every axis is log-uniform over a box bracketing that domain's real
p10-p90 (measured on 810 ScanNet++ + 184 space benchmark pairs).

  REFERENCE = creases of a scene (+ spurious on-face lines).  Indoor: room shell + boxes/prisms.  Space: built
              from the primitives real spacecraft are made of (CAD-labelled inventory of the real pairs: thin
              sheets 32%, boxes+prisms 27%, faceted cylinders + dishes 23%, booms + trusses 5%, detail 6%):
              body, groups of identical tiled oblique sheets, faceted cylinders, dishes/cones, trusses and booms,
              a few boxes/prisms, with REPETITION (mirror pairs, arrays of 2-6) as a placement rule.
  QUERY     = noisy duplicates of a visible subset of the creases + on-surface texture + off-surface junk.
              The visible subset is drawn with EDGE-VISIBILITY odds fitted on real pairs (long / isolated / hull
              edges are observed more often; coefficients VIS below, sharpened by VIS_POW).  Duplicates of one
              edge share a correlated error plus small jitter (space: tight bundles of near-identical copies).
  LABELS    = identity UNION the benchmark rule (< 10 deg, both endpoints within perp, overlap > 0.3),
              many-to-many.  perp is tied to the reference's parallel spacing (spacing/perp in [0.9, 3]).
  POSE      = identity (s=1, R=I, t=0): the ordinal encoder is exactly Sim(3)-invariant.

Output: plucker1 = REFERENCE, plucker2 = QUERY, matches = [ref_idx; qry_idx]; clouds subsampled to a size grid so
same-shape pairs batch (train.py --bucket_batch).  Lineage fsp1 -> fsp2 (relations) -> fsp4 (edge visibility) ->
fsp5 (spacecraft primitives); the earlier generators live in the SCALAR repo's attic/generators_2026-10-23/.
"""
import argparse
import glob
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_MIN_INLIERS = 20                         # pairs with fewer labelled matches are redrawn
_SIZE_GRID = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 4608, 5120, 6144, 7168, 8000,
                       9216, 10000, 10240, 12288, 14336, 16384])
# QUERY size limit (2026-10-24, Ruey: train with up to 8000 query lines).  6000 = fsp5 as v101 saw it (query clouds
# then snapped to <= 4096); above 6000 the query snaps to a finer grid up to 8000 (4608 ... 8000).  The REFERENCE is always <= 4096
# (8000 x 8000 does not fit a 32 GB training step; synthetic references never exceed ~2000 lines).
QMAX = int(os.environ.get("FSP_QMAX", 6000))
_GMAX_Q, _GMAX_R = (16384 if QMAX > 8000 else 8000 if QMAX > 6000 else 4096), 4096
# LARGE SCENES (2026-10-27, Ruey: "bigger map").  Real Spires maps reach 17k query / 13k reference lines after merging
# (median 4.3k / 6.6k), while synthetic references never exceed ~2000 lines.  With probability FSP_BIG a pair draws its
# reference size from FSP_BIG_NR (log-uniform) and the scene builders scale up with it; FSP_RMAX caps the reference
# (default 4096 = unchanged).  A training step at 16384 x 16384 needs 15 GB once attention runs on the memory-efficient
# kernel (lib/model.py).  FSP_BIG=0 (default) reproduces the generator exactly.
BIG = float(os.environ.get("FSP_BIG", 0.0))
BIG_NR = tuple(float(x) for x in os.environ.get("FSP_BIG_NR", "4000,13000").split(","))
_GMAX_R = int(os.environ.get("FSP_RMAX", _GMAX_R))
# MERGED-PROTOCOL knobs (2026-10-25, Ruey: make synthetic space as large as real).  The benchmark merges near-duplicate
# lines before the matcher (lib.utils, merge first), and measured AFTER merging real space is far bigger and
# denser than fsp5 (tools/fullrange_sp/merged_prior.py: query 2300 vs 377 lines, ref 621 vs 378, 3.3 vs 1.4 query
# lines per matched ref edge).  Defaults reproduce fsp5 exactly.
#   FSP_MERGE=1           merge both maps (same settings as the benchmark), labels carried through the merge
#   FSP_SPACE_KEXP        exponent of the space part-count multiplier (fsp5 0.5); FSP_SPACE_CLIP scales its ceilings
#   FSP_SPACE_FR          space shared-edge fraction range "lo,hi" (fsp5 0.015,0.10)
#   FSP_SPACE_JIT         space duplicate jitter relative to the shared noise (fsp5 0.2: copies nearly identical)
MERGE = os.environ.get("FSP_MERGE", "0") == "1"
SPACE_KEXP = float(os.environ.get("FSP_SPACE_KEXP", 0.5))
SPACE_CLIP = float(os.environ.get("FSP_SPACE_CLIP", 1.0))
SPACE_JIT = float(os.environ.get("FSP_SPACE_JIT", 0.2))
_FR = os.environ.get("FSP_SPACE_FR", "")
_FQ = os.environ.get("FSP_SPACE_FQ", "")              # space query inlier-fraction range (fsp5 0.08,0.50)
_NR = os.environ.get("FSP_SPACE_NR", "")              # space reference size range (fsp5 200,2000)
_rng = lambda s_, d: tuple(float(x) for x in s_.split(",")) if s_ else d
MIX = {"indoor": 0.7, "space": 0.3}
# OUTDOOR (2026-10-28, opt-in; 0 = fsp5 unchanged, bit-identical): share of pairs drawn from the outdoor domain, a
# college-quad site (Oxford Spires-like) whose facades carry irregular window grids, recessed double frames (the
# TLS "parallel doubles"), string courses and pilasters -- the parallel-line structure behind the measured Spires
# aliasing.  Ranges from the per-domain census (docs/primitive_dataset.md section 3, outdoor column).
OUTDOOR = float(os.environ.get("FSP_OUTDOOR", 0.0))
# FSP_OD=2 (2026-10-28, after v105 = noise on Spires): the v1 site was LESS learnable than real Spires (far-bin context
# AUC 0.524 vs real 0.645).  Measured on the MERGED reference (what the matcher sees): real keeps 32-45% of its lines
# through the benchmark merge, 62-71% of them singletons, 35-56% OBLIQUE; v1 keeps 2-10%, 4-13% oblique -- perfectly
# regular Manhattan facades collapse into a few hundred infinite lines.  v2: irregular facades (per-window jitter,
# varied sizes, sills/cornices, arched window heads, fewer pilasters), reference-only oblique clutter (trees, carved
# detail), less query junk, and outdoor pairs MERGED in the generator (training input = test input for this domain).
OD = int(os.environ.get("FSP_OD", 1))
# FSP_OD_POOL=<dir>: outdoor pairs are READ from a pre-generated pool (tools/fullrange_sp/make_od_pool.py) instead of
# generated live -- a v2 pair takes ~1.4 s (merge of ~10k lines) vs ~0.1 s indoor/space, which made a 30%-outdoor
# epoch data-bound (~42 min vs ~15 GPU).  Indoor/space stay live; return_info calls still generate.
OD_POOL = os.environ.get("FSP_OD_POOL", "")
_POOL_FILES, _POOL_DRAWS = [], [0]
P = {
    "indoor": dict(n_r=(150, 1500), f_q=(0.10, 0.60), f_r=(0.08, 0.60), lam=(1.5, 3.2), cap=8, cover=(0.1, 0.5),
                   sdir=(3.0, 12.0), soff=(0.3, 1.0), thr_E=(0.025, 0.08), junk=(0.0, 0.05), spur=(0.15, 0.45),
                   alpha=(0.15, 0.7), qext=(0.75, 1.2), p_drift=0.2),
    "space": dict(n_r=_rng(_NR, (200, 2000)), f_q=_rng(_FQ, (0.08, 0.50)), f_r=_rng(_FR, (0.015, 0.10)), lam=(4.0, 14.0), cap=30, cover=(0.85, 1.0),
                  sdir=(1.5, 6.0), soff=(0.4, 1.0), thr_E=(0.03, 0.06), junk=(0.05, 0.20), spur=(0.0, 0.12),
                  alpha=(0.2, 0.7), qext=(1.0, 2.5), p_drift=0.2),
    # outdoor: partner fractions f_q .011-.080 / f_r .007-.043 measured (sampled ranges set ~0.6x lower: the rule
    # labels add partners, generated f_q/f_r came out 1.6x/2.4x the draw at the census ranges), fragments 1.2-2.0, direction error median 4.3 deg,
    # offset/spacing .48-2.7, reference clutter .04, query shared-region locality .12-.59 (census p10-p90)
    "outdoor": dict(n_r=(1500, 10000), f_q=(0.007, 0.05), f_r=(0.004, 0.025), lam=(1.2, 2.0), cap=8, cover=(0.4, 0.9),
                    sdir=(1.5, 8.0), soff=(0.5, 1.5), thr_E=(0.004, 0.012), junk=(0.15, 0.45), spur=(0.02, 0.10),
                    alpha=(0.6, 0.95), qext=(1.0, 1.0), p_drift=0.3, local=(0.12, 0.59)),
}
REF_SDIR, REF_SOFF = (0.3, 2.0), (0.02, 0.15)
# edge-visibility logistic coefficients [log len/E, rad/E, hull/E, log1p dens(0.1E)], measured 2026-10-21
VIS = {"space": (1.068, 0.043, -0.942, -0.739), "indoor": (0.574, -0.769, 2.44, -0.329)}
VIS["outdoor"] = VIS["indoor"]      # NOT fitted on outdoor pairs; locality (P["outdoor"]["local"]) does most of the work
# SHARPENING exponent: rule-added matches (texture on/near edges) and sampling without replacement dilute the
# selection, so the odds are raised to VIS_POW so the MEASURED quintile curves match real (calibrated 2026-10-21)
VIS_POW = float(os.environ.get("FSP_VIS_POW", 2.0))
KEYS = ("matches", "plucker1", "plucker2", "R_gt", "t_gt", "s_gt")


# ── small geometry helpers ──────────────────────────────────────────────────────────────────────
def _logu(lo, hi):
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


def _rand_rot():
    q = np.random.randn(4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _yaw(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _box(center, half, R):
    """12 edge segments (12,2,3) and 6 faces [(centre, u*hu, v*hv)] of a box."""
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    c = c @ R.T + center
    e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    segs = np.stack([np.stack([c[a], c[b]]) for a, b in e])
    faces = []
    for ax in range(3):
        for sgn in (-1, 1):
            n = R[:, ax] * sgn; u, v = R[:, (ax + 1) % 3], R[:, (ax + 2) % 3]
            faces.append((center + n * half[ax], u * half[(ax + 1) % 3], v * half[(ax + 2) % 3]))
    return segs, faces


def _quantize_cloud(pl, matches, row, gmax=4096):
    """Uniformly subsample a cloud to the largest _SIZE_GRID value <= min(n, gmax) (so same-shape pairs batch).
    Uniform, not inlier-preserving, so the inlier fraction stays honest; dropped matches are removed."""
    n = len(pl)
    grid = _SIZE_GRID[_SIZE_GRID <= gmax]
    g = min(int(grid[grid <= max(n, grid[0])].max()), n)
    if n == g:
        return pl, matches
    sel = np.sort(np.random.choice(n, g, replace=False))
    remap = -np.ones(n, np.int64); remap[sel] = np.arange(g)
    mi = matches.copy()
    if mi.shape[1] > 0:
        mi[row] = remap[mi[row]]
        mi = mi[:, mi[row] >= 0]
    return pl[sel], mi


# ── scenes: segments (n,2,3) + faces [(centre, u*hu, v*hv)] ─────────────────────────────────────
def _scene_indoor(alpha, n_boxes):
    segs, faces = [], []
    grow = max(1.0, n_boxes / 15.0) ** (1 / 3)
    ext = np.array([_logu(3, 10) * grow, _logu(3, 10) * grow, np.random.uniform(2.3, 4.0) * min(grow, 2.0)])
    if np.random.random() < 0.85:
        s, f = _box(np.zeros(3), ext / 2, np.eye(3)); segs.append(s); faces += f
    for _ in range(n_boxes):
        half = np.minimum(np.array([_logu(0.12, 1.5), _logu(0.12, 1.5), _logu(0.12, 1.5)]), ext / 2 * 0.9)
        c = np.random.uniform(-ext / 2 + half, ext / 2 - half)
        R = np.eye(3) if np.random.random() < alpha else (_yaw(np.random.uniform(0, 2 * np.pi)) if np.random.random() < 0.8 else _rand_rot())
        s, f = _part(c, half, R, p_prism=0.6, sides=(3, 4, 5, 6, 8)); segs.append(s); faces += f
    return np.concatenate(segs), faces


def _facade(c, u, v, nrm):
    """Facade detail on a vertical wall (centre c, half-vectors u horizontal / v vertical, outward normal nrm), metres:
    storeys x bays of windows with jittered (irregular) spacing and dropouts, a recessed second frame per window (TLS
    sees both reveal edges ~0.1-0.35 m apart), mullions/transoms, string courses (some doubled), pilasters, one door."""
    W, H = 2 * np.linalg.norm(u), 2 * np.linalg.norm(v); eu, ev = _unit(u), _unit(v)
    if W < 3 or H < 3:
        return []
    out = []
    seg = lambda p, q: out.append(np.stack([p, q]))
    rect = lambda ctr, a, b: [seg(ctr + s1 * a * eu + s2 * b * ev, ctr + t1 * a * eu + t2 * b * ev)
                              for (s1, s2, t1, t2) in ((-1, -1, 1, -1), (1, -1, 1, 1), (1, 1, -1, 1), (-1, 1, -1, -1))]
    v2 = OD >= 2
    fh = _logu(3.0, 4.8); nrow = max(1, int(H / fh)); fh = H / nrow
    bay = _logu(2.2, 6.0); ncol = max(1, int(W / bay)); bay = W / ncol
    ww, wh = bay * np.random.uniform(0.25, 0.6) / 2, fh * np.random.uniform(0.35, 0.7) / 2
    recess = np.random.uniform(0.1, 0.35) if np.random.random() < 0.7 else 0.0
    p_mull, p_tran, p_drop = np.random.uniform(0, 0.3 if v2 else 0.6), np.random.uniform(0, 0.5), np.random.uniform(0.0, 0.25)
    xj = np.random.uniform(-0.15, 0.15, ncol) * bay                      # per-column offsets: irregular gaps
    if v2:                                                                 # per-column window size, per-row height
        cw = ww * np.random.uniform(0.7, 1.3, ncol); rh = wh * np.random.uniform(0.75, 1.25, nrow)
        p_sill = np.random.uniform(0.3, 0.9)
        arch, pointed = np.random.random() < 0.55, np.random.random() < 0.6
    door = np.random.randint(ncol) if np.random.random() < 0.6 else -1
    for r in range(nrow):
        zr = -H / 2 + (r + 0.5) * fh + np.random.uniform(-0.05, 0.05) * fh
        for k in range(ncol):
            x = -W / 2 + (k + 0.5) * bay + xj[k]
            if r == 0 and k == door:                                       # unique element: a taller, wider door
                ctr = c + x * eu + (-H / 2 + fh * 0.45) * ev; rect(ctr, ww * 1.4, fh * 0.42); continue
            if np.random.random() < p_drop:
                continue
            if v2:                                                         # old facades: nothing exactly collinear
                ww, wh = cw[k], rh[r]
                ctr = c + (x + np.random.uniform(-0.04, 0.04) * bay) * eu + (zr + np.random.uniform(-0.06, 0.06) * fh) * ev \
                    + nrm * np.random.uniform(-0.08, 0.08)
            else:
                ctr = c + x * eu + zr * ev
            rect(ctr, ww, wh)
            if v2 and arch:                                                # pointed / round arch head: oblique chords
                kk = np.random.randint(2, 5); rise = ww * np.random.uniform(0.6, 1.4)
                ph = np.linspace(0, np.pi, 2 * kk + 1)
                pts = [ctr + ww * np.cos(t) * eu + (wh + rise * np.sin(t) ** (0.6 if pointed else 1.0)) * ev for t in ph]
                for j in range(len(pts) - 1):
                    seg(pts[j], pts[j + 1])
            if recess > 0:
                sh = np.random.uniform(0.02, 0.08); rect(ctr - nrm * recess, max(ww - sh, 0.1), max(wh - sh, 0.1))
            if v2 and np.random.random() < p_sill:                       # projecting sill: two horizontals
                o = nrm * np.random.uniform(0.05, 0.15); z0 = -wh - np.random.uniform(0.05, 0.15)
                seg(ctr - 1.15 * ww * eu + z0 * ev + o, ctr + 1.15 * ww * eu + z0 * ev + o)
                seg(ctr - 1.15 * ww * eu + (z0 - 0.08) * ev + o, ctr + 1.15 * ww * eu + (z0 - 0.08) * ev + o)
            if np.random.random() < p_mull:
                seg(ctr - wh * ev, ctr + wh * ev)
            if np.random.random() < p_tran:
                z0 = np.random.uniform(-0.3, 0.5) * wh; seg(ctr - ww * eu + z0 * ev, ctr + ww * eu + z0 * ev)
    for r in range(1, nrow):                                               # string courses at storey lines
        if np.random.random() < 0.5:
            z = -H / 2 + r * fh; seg(c - W / 2 * eu + z * ev, c + W / 2 * eu + z * ev)
            if np.random.random() < 0.5:
                o = nrm * np.random.uniform(0.08, 0.3); seg(c - W / 2 * eu + z * ev + o, c + W / 2 * eu + z * ev + o)
    if v2:                                                                 # cornice: 2-3 stacked horizontals at the top
        for j in range(np.random.randint(1, 4)):
            z = H / 2 - j * np.random.uniform(0.1, 0.3); o = nrm * np.random.uniform(0.05, 0.4)
            seg(c - W / 2 * eu + z * ev + o, c + W / 2 * eu + z * ev + o)
    if np.random.random() < (0.12 if v2 else 0.35):                        # pilasters / buttresses between bays
        wp = np.random.uniform(0.15, 0.4)
        for k in range(1, ncol):
            x = -W / 2 + k * bay
            for dx in (-wp, wp):
                seg(c + (x + dx) * eu - H / 2 * ev + nrm * 0.25, c + (x + dx) * eu + H / 2 * ev + nrm * 0.25)
    return out


def _scene_outdoor(alpha, n_target):
    """A college quad: buildings (boxes, facades with _facade detail, some gables / towers with spires) around a
    courtyard, mostly aligned to the site axes, plus kerb lines on the ground; buildings are added until the site
    has >= n_target lines (at most 48).  Metres; normalised by the caller."""
    segs, faces = [], []
    site = np.array([_logu(30, 90), _logu(30, 90)])
    Rs = _yaw(np.random.uniform(0, 2 * np.pi))
    for i in range(48):
        if sum(len(x) for x in segs) >= n_target:
            break
        side = np.random.randint(4); ax = side % 2; sg = 1 if side < 2 else -1
        L = _logu(10, 45); D = _logu(7, 16); Hh = _logu(7, 22)
        half = np.array([L / 2, D / 2, Hh / 2]) if ax == 1 else np.array([D / 2, L / 2, Hh / 2])
        pos = np.zeros(3); pos[ax] = sg * (site[ax] / 2 + half[ax]); pos[1 - ax] = np.random.uniform(-0.5, 0.5) * site[1 - ax]
        pos[2] = Hh / 2
        R = Rs if np.random.random() < alpha else Rs @ _yaw(np.random.uniform(-0.6, 0.6))
        c = R @ pos
        s, f = _box(c, half, R); segs.append(s); faces += f
        for fc, fu, fv in f:                                               # vertical faces get facade detail
            zu, zv = abs(_unit(fu)[2]), abs(_unit(fv)[2])
            if max(zu, zv) < 0.9:
                continue
            hu, vv = (fv, fu) if zu > zv else (fu, fv)
            nrm = _unit(fc - c); d = _facade(fc, hu, vv, nrm)
            if d:
                segs.append(np.stack(d))
        top = c + R[:, 2] * Hh / 2
        if np.random.random() < 0.4:                                       # gable roof: ridge + 4 rafters
            rise = np.random.uniform(0.2, 0.6) * min(half[0], half[1]); la = 0 if half[0] > half[1] else 1
            e = R[:, la] * half[la]; o = R[:, 1 - la] * half[1 - la]; rr = top + R[:, 2] * rise
            segs.append(np.stack([np.stack([rr - e, rr + e])] +
                                 [np.stack([rr + sx * e, top + sx * e + so * o]) for sx in (-1, 1) for so in (-1, 1)]))
        if np.random.random() < 0.25:                                      # tower + spire
            tw = np.random.uniform(2.5, 5.0); th = np.random.uniform(5, 18)
            tc = top + R[:, 0] * np.random.uniform(-0.6, 0.6) * half[0] + R[:, 1] * np.random.uniform(-0.6, 0.6) * half[1] + R[:, 2] * th / 2
            s2, f2 = _box(tc, np.array([tw / 2, tw / 2, th / 2]), R); segs.append(s2); faces += f2
            tt = tc + R[:, 2] * th / 2; apex = tt + R[:, 2] * np.random.uniform(6, 25)
            segs.append(np.stack([np.stack([tt + R @ np.array([sx * tw / 2, sy * tw / 2, 0]), apex]) for sx in (-1, 1) for sy in (-1, 1)]))
    for _ in range(np.random.randint(2, 9)):                               # kerbs / path edges on the ground
        ax = np.random.randint(2); off = np.random.uniform(-0.5, 0.5) * site[1 - ax]; ln = np.random.uniform(0.3, 1.0) * site[ax]
        a = np.zeros(3); b = np.zeros(3); a[ax], b[ax] = -ln / 2, ln / 2; a[1 - ax] = b[1 - ax] = off
        segs.append(np.stack([np.stack([Rs @ a, Rs @ b])]))
    faces.append((np.zeros(3), Rs[:, 0] * site[0] / 2, Rs[:, 1] * site[1] / 2))   # the courtyard ground plane
    return np.concatenate(segs), faces


def _prism(center, radius, half_h, nside, R):
    """n-gon prism: 2 rings of n edges + n vertical edges; faces = n side quads + 2 caps (as quads)."""
    th = np.arange(nside) * 2 * np.pi / nside + np.random.uniform(0, 2 * np.pi)
    ring = np.stack([np.cos(th), np.sin(th), np.zeros(nside)], 1) * radius
    top, bot = ring + [0, 0, half_h], ring - [0, 0, half_h]
    nx = np.roll(np.arange(nside), -1)
    segs = np.concatenate([np.stack([top, top[nx]], 1), np.stack([bot, bot[nx]], 1), np.stack([top, bot], 1)])
    segs = segs @ R.T + center
    faces = []
    for i in range(nside):
        m = (ring[i] + ring[nx[i]]) / 2; t = (ring[nx[i]] - ring[i]) / 2
        faces.append((R @ m + center, R @ t, R @ np.array([0, 0, half_h])))
    for z in (half_h, -half_h):
        faces.append((R @ np.array([0, 0, z]) + center, R @ np.array([radius * 0.7, 0, 0]), R @ np.array([0, radius * 0.7, 0])))
    return segs, faces


def _part(center, size, R, p_prism, sides=(3, 5, 6, 8, 12, 16)):
    """A box, or with probability p_prism an n-gon prism of comparable size (oblique ring edges)."""
    if np.random.random() < p_prism:
        return _prism(center, float(np.mean(size[:2])), float(size[2]), int(np.random.choice(sides)), R)
    return _box(center, size, R)


def _grid_on_face(c, u, v, alpha):
    """Irregular cell grid on a panel face (solar arrays): gaps exponential-like (CV ~0.8)."""
    out = []
    for a, b in ((u, v), (v, u)):
        n = np.random.randint(3, 12)
        g = np.cumsum(np.random.exponential(1.0, n + 1)); g = g[:-1] / g[-1] * 2 - 1
        for t in g:
            p = c + a * t
            out.append(np.stack([p - b, p + b]))
    return out


def _cyl(center, axis, radius, half_len, n, keep_long=0.15):
    """Faceted cylinder: two n-gon rings + a fraction of the longitudinal facet edges; faces = the n facets."""
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
    th = np.arange(n) * 2 * np.pi / n + np.random.uniform(0, 2 * np.pi)
    ring = (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v) * radius
    top, bot = center + ring + axis * half_len, center + ring - axis * half_len
    nx = np.roll(np.arange(n), -1)
    segs = [np.stack([top, top[nx]], 1), np.stack([bot, bot[nx]], 1)]
    kl = np.random.random(n) < keep_long
    if kl.any():
        segs.append(np.stack([top[kl], bot[kl]], 1))
    faces = [(center + (ring[i] + ring[nx[i]]) / 2, (ring[nx[i]] - ring[i]) / 2, axis * half_len) for i in range(n)]
    lab = ["cylinder"] * (2 * n + int(kl.sum()))
    return np.concatenate(segs), faces, lab


def _dish(center, axis, radius, depth_ratio, n, n_rings, n_spokes):
    """Dish / cone: concentric polygonal rings (radius r_k at depth along the axis) + radial spokes."""
    axis = axis / np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
    th = np.arange(n) * 2 * np.pi / n; nx = np.roll(np.arange(n), -1)
    segs, rings = [], []
    for k in range(n_rings):
        f = 1.0 - k / n_rings
        rk = radius * f; zk = depth_ratio * 2 * radius * (1 - f * f)
        P = center + (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v) * rk + axis * zk
        rings.append(P); segs.append(np.stack([P, P[nx]], 1))
    apex = center + axis * depth_ratio * 2 * radius
    for j in np.linspace(0, n, n_spokes, endpoint=False).astype(int):
        segs.append(np.stack([rings[0][j], apex])[None])
    faces = [(center, u * radius * 0.7, v * radius * 0.7)]
    S = np.concatenate(segs)
    return S, faces, ["dish"] * len(S)


def _sheet(center, R, a, b, pitch, both):
    """Thin flat sheet (panel / radiator / blanket) as a single face: 4 rim edges + coplanar seam tiles."""
    ex, ey = R[:, 0], R[:, 1]
    C = [center + sx * a * ex + sy * b * ey for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    segs = [np.stack([C[i], C[(i + 1) % 4]]) for i in range(4)]
    lab = ["sheet_rim"] * 4
    for t in np.arange(-a + pitch, a - pitch * 0.5, pitch)[:14]:
        segs.append(np.stack([center + t * ex - b * ey, center + t * ex + b * ey])); lab.append("sheet_seam")
    if both:
        for t in np.arange(-b + pitch, b - pitch * 0.5, pitch)[:14]:
            segs.append(np.stack([center - a * ex + t * ey, center + a * ex + t * ey])); lab.append("sheet_seam")
    return np.stack(segs), [(center, ex * a, ey * b)], lab


def _truss(p0, direction, length, width, n_bays):
    """Truss: two parallel longerons + oblique zig-zag diagonals between them."""
    d = direction / np.linalg.norm(direction)
    w = np.cross(d, np.random.randn(3)); w /= np.linalg.norm(w)
    a0, b0 = p0, p0 + w * width
    segs = [np.stack([a0, a0 + d * length]), np.stack([b0, b0 + d * length])]
    for k in range(n_bays):
        t0, t1 = k * length / n_bays, (k + 1) * length / n_bays
        segs.append(np.stack([a0 + d * t0, b0 + d * t1]) if k % 2 == 0 else np.stack([b0 + d * t0, a0 + d * t1]))
    return np.stack(segs), [], ["truss"] * len(segs)


def _repeat(builder, count, mode, step):
    """REPETITION as a placement rule: `count` copies of builder(offset) as a linear ARRAY along `step`, or a MIRROR
    pair (offset and its reflection through the body centre plane given by `step`)."""
    out_s, out_f, out_l = [], [], []
    if mode == "mirror":
        n = step / np.linalg.norm(step)
        for sgn in (1, -1):
            s_, f_, l_ = builder(sgn)
            if sgn < 0:                                     # reflect through the plane x.n = 0
                Rf = np.eye(3) - 2 * np.outer(n, n)
                s_ = s_ @ Rf.T; f_ = [(c @ Rf.T, u @ Rf.T, v @ Rf.T) for c, u, v in f_]
            out_s.append(s_); out_f += f_; out_l += l_
    else:
        for k in range(count):
            s_, f_, l_ = builder(0)
            off = step * k
            out_s.append(s_ + off); out_f += [(c + off, u, v) for c, u, v in f_]; out_l += l_
    return np.concatenate(out_s), out_f, out_l


_LAST_LABELS = []


def _scene_space(alpha, n_app):
    """Spacecraft composed of the measured real primitives (see module docstring).  `n_app` scales part counts."""
    global _LAST_LABELS
    segs, faces, labs = [], [], []
    def add(s_, f_, l_):
        segs.append(s_); faces.extend(f_); labs.extend(l_)
    body = np.array([_logu(0.6, 2.0), _logu(0.6, 2.0), _logu(0.6, 2.5)]); B = body.max()
    s_, f_ = _part(np.zeros(3), body, np.eye(3), p_prism=0.6, sides=(6, 8)); add(s_, f_, ["body"] * len(s_))
    k = max(1.0, n_app / 25.0)                                          # part-count multiplier with reference size
    def mount_point():
        ax = np.random.randint(0, 3); sg = np.random.choice([-1, 1])
        c = np.random.uniform(-0.8, 0.8, 3) * body; c[ax] = sg * body[ax]
        nrm = np.zeros(3); nrm[ax] = sg
        return c, nrm
    def orient():
        return np.eye(3) if np.random.random() < alpha * 0.5 else _rand_rot()
    # 1. SHEETS: 2-4 groups of identical tiled oblique sheets (4-12 sheets total), mirror pairs or arrays
    for _ in range(int(np.clip(np.random.randint(3, 7) * k ** SPACE_KEXP, 3, 12 * SPACE_CLIP))):
        a, b = _logu(0.3, 1.6) * B, _logu(0.15, 0.8) * B
        pitch = min(a, b) * _logu(0.2, 0.6); both = np.random.random() < 0.5; R_ = orient()
        c0, nrm = mount_point(); c0 = c0 + nrm * (a + 0.1 * B)
        cnt = np.random.randint(1, 5); mode = "mirror" if np.random.random() < 0.5 else "array"
        step = nrm if mode == "mirror" else R_[:, 0] * 2.1 * a
        s_, f_, l_ = _repeat(lambda sg: _sheet(c0 * (1 if sg >= 0 else 1), R_, a, b, pitch, both), cnt, mode, step); add(s_, f_, l_)
    # 2. CYLINDERS (tanks, thrusters): groups of 1-4 identical faceted cylinders, mounted along the body normal
    for _ in range(int(np.clip(np.random.randint(1, 3) * k ** SPACE_KEXP, 1, 4 * SPACE_CLIP))):
        r = _logu(0.05, 0.4) * B; hl = r * _logu(0.5, 2.0); n = np.random.randint(12, 21)
        c0, nrm = mount_point(); axis = nrm if np.random.random() < 0.6 else _unit(np.random.randn(1, 3))[0]
        c0 = c0 + nrm * hl; cnt = np.random.randint(1, 4); mode = "mirror" if np.random.random() < 0.4 else "array"
        tang = _unit(np.cross(nrm, np.random.randn(3))[None])[0]
        step = nrm if mode == "mirror" else tang * 2.4 * r
        s_, f_, l_ = _repeat(lambda sg: _cyl(c0, axis, r, hl, n), cnt, mode, step); add(s_, f_, l_)
    # 3. DISHES / cones (antennas): 0-2
    for _ in range(np.random.randint(0, 3)):
        r = _logu(0.1, 0.8) * B; c0, nrm = mount_point()
        s_, f_, l_ = _dish(c0 + nrm * 0.2 * B, nrm if np.random.random() < 0.5 else _unit(np.random.randn(1, 3))[0], r,
                           np.random.uniform(0.15, 0.35), np.random.randint(12, 21), np.random.randint(2, 4), np.random.randint(4, 13))
        add(s_, f_, l_)
    # 4. TRUSSES + BOOMS: 0-2 trusses, 0-3 booms
    for _ in range(np.random.randint(1, 4)):
        c0, nrm = mount_point(); L = _logu(1.0, 3.0) * B
        s_, f_, l_ = _truss(c0, nrm + 0.4 * np.random.randn(3), L, _logu(0.08, 0.25) * B, np.random.randint(4, 13)); add(s_, f_, l_)
    nb = np.random.randint(1, 5)
    if nb:
        a0 = np.stack([mount_point()[0] for _ in range(nb)]); b0 = a0 + _unit(np.random.randn(nb, 3)) * _logu(1.0, 3.0) * B
        add(np.stack([a0, b0], 1), [], ["boom"] * nb)
    # 5. a FEW boxes / prisms (fsp4 had 39% of lines here; real 27% incl. the body) and a LITTLE detail
    for _ in range(int(np.clip(np.random.randint(4, 11) * k ** SPACE_KEXP, 4, 18 * SPACE_CLIP))):
        half = np.array([_logu(0.1, 0.6), _logu(0.1, 0.6), _logu(0.1, 0.6)]) * B
        c0, nrm = mount_point(); c0 = c0 + nrm * half.max() * 0.8
        s_, f_ = _part(c0, half, orient(), p_prism=0.4, sides=(5, 6, 8)); add(s_, f_, ["box"] * len(s_))
    for _ in range(np.random.randint(0, 2)):
        anchor, _n = mount_point(); rad = B * _logu(0.08, 0.25)
        for _ in range(np.random.randint(3, 7)):
            half = np.array([_logu(0.02, 0.1), _logu(0.02, 0.1), _logu(0.02, 0.1)]) * B
            s_, f_ = _part(anchor + np.random.uniform(-1, 1, 3) * rad, half, orient(), p_prism=0.5, sides=(5, 6, 8))
            add(s_, f_, ["detail"] * len(s_))
    _LAST_LABELS = labs
    return np.concatenate(segs), faces


# ── segment helpers ─────────────────────────────────────────────────────────────────────────────
def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _perp_dir(d):
    a = np.where(np.abs(d[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    p = _unit(np.cross(d, a)); q = np.cross(d, p)
    th = np.random.uniform(0, 2 * np.pi, (len(d), 1))
    return p * np.cos(th) + q * np.sin(th)


def _noise(S, sdir_deg, soff):
    """Rayleigh direction tilt about the midpoint + Rayleigh perpendicular offset (cap 25 deg)."""
    if len(S) == 0:
        return S
    m = S.mean(1); d = S[:, 1] - S[:, 0]; L = np.linalg.norm(d, axis=1, keepdims=True); d = d / (L + 1e-12)
    th = np.minimum(np.random.rayleigh(np.radians(sdir_deg), len(S)), np.radians(25))[:, None]
    d = _unit(d * np.cos(th) + _perp_dir(d) * np.sin(th))
    m = m + _perp_dir(d) * np.random.rayleigh(soff, (len(S), 1))
    return np.stack([m - d * L / 2, m + d * L / 2], 1)


def _cut(S, cover, overshoot=0.0):
    """A sub-segment covering a fraction ~U(cover) of each segment, at a random position."""
    a, b = S[:, 0], S[:, 1]; f = np.random.uniform(*cover, (len(S), 1))
    t0 = np.random.uniform(0, 1, (len(S), 1)) * (1 - f)
    t0 = t0 - overshoot * np.random.uniform(0, 1, (len(S), 1)); t1 = t0 + f + overshoot * np.random.uniform(0, 1, (len(S), 1))
    return np.stack([a + (b - a) * t0, a + (b - a) * t1], 1)


def _on_faces(faces, n, alpha, length):
    if n <= 0 or not faces:
        return np.zeros((0, 2, 3))
    area = np.array([np.linalg.norm(f[1]) * np.linalg.norm(f[2]) for f in faces]) + 1e-12
    fi = np.random.choice(len(faces), n, p=area / area.sum())          # texture lives on large surfaces
    C = np.stack([faces[i][0] for i in fi]); U = np.stack([faces[i][1] for i in fi]); V = np.stack([faces[i][2] for i in fi])
    p = C + U * np.random.uniform(-0.85, 0.85, (n, 1)) + V * np.random.uniform(-0.85, 0.85, (n, 1))   # off the face borders
    ang = np.where(np.random.random(n) < alpha, np.random.randint(0, 2, n) * np.pi / 2, np.random.uniform(0, np.pi, n))
    d = _unit(U) * np.cos(ang)[:, None] + _unit(V) * np.sin(ang)[:, None]
    L = np.array([length() for _ in range(n)])[:, None]
    L = np.minimum(L, 2.2 * np.maximum(np.linalg.norm(U, axis=1), np.linalg.norm(V, axis=1))[:, None])
    return np.stack([p - d * L / 2, p + d * L / 2], 1)


def _motifs(faces, n, alpha, len_q):
    """Indoor photometric texture as LOCAL motifs: small frames (4 joined edges, corners) and clusters
    of 2-8 short lines around a point on a face (real mono lines are local and joined)."""
    out, k = [], 0
    area = np.array([np.linalg.norm(f[1]) * np.linalg.norm(f[2]) for f in faces]) + 1e-12
    pw = area ** 0.25; pw /= pw.sum()          # bias toward furniture-sized faces (real mono lines cluster on objects)
    while k < n:
        c, u, v = faces[np.random.choice(len(faces), p=pw)]
        un, vn = _unit(u), _unit(v)
        hu, hv = np.linalg.norm(u), np.linalg.norm(v)
        ctr = c + u * np.random.uniform(-0.8, 0.8) + v * np.random.uniform(-0.8, 0.8)
        if np.random.random() < 0.4:                                   # frame: 4 joined edges
            a = min(hu, len_q * _logu(0.5, 2.0)); b = min(hv, len_q * _logu(0.5, 2.0))
            ang = 0.0 if np.random.random() < alpha else np.random.uniform(0, np.pi)
            e1 = un * np.cos(ang) + vn * np.sin(ang); e2 = -un * np.sin(ang) + vn * np.cos(ang)
            P4 = [ctr + e1 * a + e2 * b, ctr - e1 * a + e2 * b, ctr - e1 * a - e2 * b, ctr + e1 * a - e2 * b]
            out += [np.stack([P4[i], P4[(i + 1) % 4]]) for i in range(4)]; k += 4
        else:                                                          # cluster of short lines
            m = np.random.randint(2, 9); rad = len_q * _logu(0.3, 1.2)
            for _ in range(m):
                p0 = ctr + un * np.random.uniform(-rad, rad) + vn * np.random.uniform(-rad, rad)
                ang = np.random.randint(0, 2) * np.pi / 2 if np.random.random() < alpha else np.random.uniform(0, np.pi)
                d = un * np.cos(ang) + vn * np.sin(ang); L = len_q * _logu(0.4, 2.0)
                out.append(np.stack([p0 - d * L / 2, p0 + d * L / 2]))
            k += m
    return np.stack(out)[:n] if out else np.zeros((0, 2, 3))


def _junk(n, centre, r_in, r_out, length):
    if n <= 0:
        return np.zeros((0, 2, 3))
    u = _unit(np.random.randn(n, 3)); r = np.random.uniform(r_in, r_out, (n, 1))
    p = centre + u * r; d = _unit(np.random.randn(n, 3)); L = np.array([length() for _ in range(n)])[:, None]
    return np.stack([p - d * L / 2, p + d * L / 2], 1)


def _radius(S, c=None):
    m = S.mean(1); c = np.median(m, 0) if c is None else c
    return float(np.percentile(np.linalg.norm(m - c, axis=1), 90)) + 1e-9, c


def _par_spacing(S):
    m = S.mean(1); d = _unit(S[:, 1] - S[:, 0])
    if len(S) > 1200:
        i = np.random.choice(len(S), 1200, replace=False); m, d, S = m[i], d[i], S[i]
    par = np.abs(d @ d.T) > np.cos(np.radians(10)); np.fill_diagonal(par, False)
    dist = np.linalg.norm(np.cross(m[:, None] - S[None, :, 0], d[None]), axis=2)
    dist = np.where(par, dist, np.inf).min(1)
    return float(np.median(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else 1.0


def rule_pairs(Q, L, perp, ang=10.0, ov=0.3, block=1000):
    """The benchmark rule (space_common.gt_corr) WITHOUT the argmax: all valid (q, r).  Evaluated in blocks of query
    lines (identical output, same order): on real-size space maps (12k x 3k, many near-parallel lines) the candidate
    arrays of one call ran to several GB per worker (2026-10-25)."""
    if len(Q) > block:
        parts = [rule_pairs(Q[i:i + block], L, perp, ang, ov, block) for i in range(0, len(Q), block)]
        return (np.concatenate([q + i for (q, _), i in zip(parts, range(0, len(Q), block))]),
                np.concatenate([r for _, r in parts]))
    a, b = Q[:, 0], Q[:, 1]
    dm = _unit(b - a); c = L[:, 0]; dl = L[:, 1] - c; ll = np.linalg.norm(dl, axis=1); dq = dl / (ll[:, None] + 1e-12)
    ok = np.abs(dm @ dq.T) > np.cos(np.radians(ang))
    qi, ri = np.nonzero(ok)
    if len(qi) == 0:
        return qi, ri
    da = np.linalg.norm(np.cross(a[qi] - c[ri], dq[ri]), axis=1); db = np.linalg.norm(np.cross(b[qi] - c[ri], dq[ri]), axis=1)
    keep = np.maximum(da, db) < perp
    qi, ri = qi[keep], ri[keep]
    ta = ((a[qi] - c[ri]) * dq[ri]).sum(1); tb = ((b[qi] - c[ri]) * dq[ri]).sum(1)
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
    keep = (np.minimum(hi, ll[ri]) - np.maximum(lo, 0)) > ov * np.minimum(hi - lo, ll[ri])
    return qi[keep], ri[keep]


def _drift(S, amp):
    """Smooth low-frequency displacement field (trajectory drift), amplitude `amp`."""
    k = np.random.randn(3, 3) * 0.3; ph = np.random.uniform(0, 2 * np.pi, 3)
    f = lambda X: X + amp * np.sin(X @ k + ph)
    return np.stack([f(S[:, 0]), f(S[:, 1])], 1)


def _vis_weights(S, domain):
    """Observation odds of each segment under the measured edge-visibility model (see VIS)."""
    from scipy.spatial import ConvexHull, cKDTree
    a, b = S[:, 0], S[:, 1]; m = (a + b) / 2
    c = np.median(m, 0); E = np.percentile(np.linalg.norm(m - c, axis=1), 90) + 1e-9
    try:
        eq = ConvexHull(np.vstack([a, b])).equations
        hull = np.min(np.abs(m @ eq[:, :3].T + eq[:, 3]), 1) / E
    except Exception:
        hull = np.zeros(len(S))
    dens = np.array([len(x) - 1 for x in cKDTree(m).query_ball_point(m, 0.1 * E)], float)
    x = np.c_[np.log(np.linalg.norm(b - a, axis=1) / E + 1e-4), np.linalg.norm(m - c, axis=1) / E, hull, np.log1p(dens)]
    z = VIS_POW * (x @ np.array(VIS[domain]))
    return np.exp(z - z.max())


def _plucker(S):
    a, b = S[:, 0], S[:, 1]; d = _unit(b - a)
    return np.concatenate([np.cross((a + b) / 2, d), d], 1).astype(np.float32)


# ── the pair ────────────────────────────────────────────────────────────────────────────────────
def generate_pair_fsp(domain=None, return_info=False):
    if domain is None and OUTDOOR > 0 and np.random.random() < OUTDOOR:
        domain = "outdoor"
    domain = domain or ("indoor" if np.random.random() < MIX["indoor"] else "space")
    if domain == "outdoor" and OD_POOL and not return_info:
        if not _POOL_FILES or _POOL_DRAWS[0] % 500 == 0:          # re-scan: the pool may still be growing
            _POOL_FILES[:] = sorted(f_ for f_ in glob.glob(os.path.join(OD_POOL, "*.npz")) if not f_.endswith(".tmp.npz"))
        _POOL_DRAWS[0] += 1
        z = np.load(_POOL_FILES[np.random.randint(len(_POOL_FILES))])
        return {k: z[k] for k in KEYS}
    p = P[domain]
    alpha = np.random.uniform(*p["alpha"])
    big = BIG > 0 and np.random.random() < BIG
    n_r = int(_logu(*(BIG_NR if big else p["n_r"])))
    if domain == "indoor":
        creases, faces = _scene_indoor(alpha, int(np.clip(n_r / 14 * np.random.uniform(0.7, 1.3), 1, 1200 if big else 400)))
    elif domain == "outdoor":
        creases, faces = _scene_outdoor(alpha, int(n_r * 1.1))
    else:
        creases, faces = _scene_space(alpha, int(np.clip(n_r / 20 * np.random.uniform(0.7, 1.3), 2, 800 if big else 120)))
    # normalise: reference extent E = 1 about the crease median
    E0, c0 = _radius(creases); creases = (creases - c0) / E0
    faces = [((fc - c0) / E0, u / E0, v / E0) for fc, u, v in faces]
    # reference = creases (subsampled to n_r if larger) + spurious lines on faces
    n_cre = min(len(creases), n_r)
    _perm = np.random.permutation(len(creases))[:n_cre]
    ref_c = creases[_perm]
    _piece = (np.array(_LAST_LABELS, object)[_perm] if (domain == 'space' and len(_LAST_LABELS) == len(creases))
              else np.array(['crease'] * n_cre, object))
    sp = _par_spacing(ref_c)
    perp = float(np.clip(sp / np.random.uniform(0.9, 3.0), p["thr_E"][0], p["thr_E"][1]))
    ref_c = _noise(ref_c, _logu(*REF_SDIR), _logu(*REF_SOFF) * perp)
    len_r = np.median(np.linalg.norm(ref_c[:, 1] - ref_c[:, 0], axis=1))
    n_spur = int(np.random.uniform(*p["spur"]) * n_cre)
    # space: on-surface lines rarely align with the panel axes, else they coincide with the grid creases
    # lying in the same face plane (measured: they took ~70% of all rule matches and flattened the degree)
    a_tex = 0.4 if domain == "space" else alpha
    spur = _on_faces(faces, n_spur, 0.3, lambda: len_r * _logu(0.3, 1.5))
    if domain == "outdoor" and OD >= 2:
        # reference-only oblique population (trees, carved detail, crease-extractor noise on stone): real merged TLS
        # references are ~45% oblique singletons; the query never sees these lines
        n_cl = int(np.random.uniform(0.4, 1.0) * n_cre)
        n_tree = int(n_cl * np.random.uniform(0.3, 0.7)); cl = []
        if n_tree:
            nt = np.random.randint(3, 15); tc = ref_c.reshape(-1, 3)[np.random.choice(2 * n_cre, nt)]
            tc[:, 2] = np.random.uniform(-0.3, 0.1, nt)
            cl.append(np.concatenate([_junk(n_tree // nt + 1, c_, 0.0, np.random.uniform(0.03, 0.08), lambda: len_r * _logu(0.1, 0.6)) for c_ in tc])[:n_tree])
        cl.append(_on_faces(faces, n_cl - n_tree, 0.0, lambda: len_r * _logu(0.1, 0.8)))
        spur = np.concatenate([spur] + cl) if len(spur) else np.concatenate(cl)
    ref = np.concatenate([ref_c, spur]) if len(spur) else ref_c
    # shared creases and their query duplicates (identity labels)
    k = int(np.clip(_logu(*p["f_r"]) * n_cre, _MIN_INLIERS // 2, n_cre))
    w = _vis_weights(ref, domain)[:n_cre]                    # features on the full reference, creases are its first rows
    if domain == "outdoor":
        # a walk quarter sees one part of the site: shared creases lie in a ball (radius = measured locality, in E)
        mid = ref_c.mean(1); ctr = mid[np.random.choice(n_cre, p=w / w.sum())]
        r_loc = np.random.uniform(*p["local"]); dist = np.linalg.norm(mid - ctr, axis=1)
        while (dist < r_loc).sum() < max(k, _MIN_INLIERS):
            r_loc *= 1.25
        w = np.where(dist < r_loc, w, 0.0)
    shared = np.random.choice(n_cre, k, replace=False, p=w / w.sum()) if (w > 0).sum() >= k else np.random.choice(n_cre, k, replace=False)
    lam = _logu(*p["lam"])
    ndup = np.clip(1 + np.random.poisson(lam - 1.0, k), 1, p["cap"])
    src = np.repeat(shared, ndup)
    over = 0.08 if domain == "space" else 0.0
    sdir, soff = _logu(*p["sdir"]), _logu(*p["soff"]) * perp
    # CORRELATED noise: repeated observations of one edge share its error (the local map distortion),
    # plus a small independent jitter -- with independent noise the duplicates scatter onto
    # neighbouring parallel creases and the per-ref degree collapses (measured 1.8 vs real 4.3 space)
    common = _noise(ref_c[shared], sdir, soff)
    jit = SPACE_JIT if domain == "space" else 0.35    # space: bundles of near-identical copies
    inl = _noise(_cut(common[np.repeat(np.arange(k), ndup)], p["cover"], over), jit * sdir, jit * soff)
    # the rest of the query: on-surface texture (+ grid) lines and off-surface junk
    f_q = _logu(*((0.003, 0.02) if domain == "outdoor" and OD >= 2 else p["f_q"]))   # v2: real merged queries 2.6-9.3k lines
    n_q = int(np.clip(len(inl) / f_q, len(inl) + 10, QMAX))
    n_rest = n_q - len(inl)
    n_junk = int(np.random.uniform(*((0.03, 0.2) if domain == "outdoor" and OD >= 2 else p["junk"])) * n_rest)
    len_q = np.median(np.linalg.norm(inl[:, 1] - inl[:, 0], axis=1))
    tex_len = (lambda: _logu(0.25, 1.2)) if domain == "space" else (lambda: len_q * _logu(0.4, 2.0))   # real space query lines are long (0.53 E)
    n_tex = n_rest - n_junk
    if domain == "space":
        # SfM bundles: each on-surface edge observed several times (copies with tight jitter)
        mt = _logu(2.0, 8.0); n_base = max(1, int(n_tex / mt))
        base = _noise(_on_faces(faces, n_base, a_tex, tex_len), sdir, soff)
        rep_ = np.clip(1 + np.random.poisson(mt - 1, n_base), 1, 30)
        tex = _noise(_cut(base[np.repeat(np.arange(n_base), rep_)], (0.85, 1.0), 0.08), 0.2 * sdir, 0.2 * soff)[:n_tex]
    elif domain == "outdoor":
        # texture and clutter (trees, far facades, signage) live where the walk was: within ~1.5 r_loc of its centre
        near = [f_ for f_ in faces if np.linalg.norm(f_[0] - ctr) < 1.5 * r_loc + max(np.linalg.norm(f_[1]), np.linalg.norm(f_[2]))]
        # v2: real merged query lines are directionally messy (11-19% Manhattan-horizontal vs 38-75% in v1)
        tex = _motifs(near or faces, 3 * n_tex + 8, np.random.uniform(0.05, 0.3) if OD >= 2 else alpha, len_q)
        tex = tex[np.linalg.norm(tex.mean(1) - ctr, axis=1) < 1.5 * r_loc][:n_tex]
        if OD >= 2:
            # near-misses: the camera also sees local reference structure it does not reproduce within the label gate
            # (real: 29% of query lines within ~1 m of a TLS line vs 15% in v1) -- shared context, no labels
            loc = np.where(np.linalg.norm(ref_c.mean(1) - ctr, axis=1) < 1.5 * r_loc)[0]
            n_nm = min(int(np.random.uniform(0.15, 0.4) * len(tex)), 3 * len(loc))
            if n_nm and len(loc):
                nm = _cut(ref_c[np.random.choice(loc, n_nm)], p["cover"])
                nm = _noise(nm, sdir, 0.0); off = _perp_dir(_unit(nm[:, 1] - nm[:, 0])) * (perp * np.random.uniform(1.5, 4.0, (n_nm, 1)))
                tex = np.concatenate([nm + off[:, None], tex[:len(tex) - n_nm]])
        n_junk = n_rest - len(tex)
    else:
        tex = _motifs(faces, n_tex, alpha, len_q)
    q_out = p["qext"][1] if domain == "space" else 1.0
    junk = _junk(n_junk, np.zeros(3), 1.0 if domain == "space" else 0.0, max(1.05, _logu(*p["qext"]) * q_out / p["qext"][1] * 2.5)
                 if domain == "space" else 1.0, lambda: len_q * _logu(0.5, 2.0))
    if domain == "outdoor":
        junk = _junk(n_junk, ctr, 0.0, 1.5 * r_loc, lambda: len_q * _logu(0.3, 1.5))
    rest = _noise(np.concatenate([tex, junk]) if len(tex) + len(junk) else np.zeros((0, 2, 3)), sdir, soff)
    qry = np.concatenate([inl, rest])
    if np.random.random() < p["p_drift"]:
        qry = _drift(qry, np.random.uniform(0.1, 0.5) * perp)
    # labels: identity U rule (many-to-many)
    ident = set(zip(np.arange(len(inl)).tolist(), src.tolist()))
    rq, rr = rule_pairs(qry, ref, perp, block=400 if domain == "outdoor" else 1000)
    pairs = np.array(sorted(ident | set(zip(rq.tolist(), rr.tolist()))), np.int64).reshape(-1, 2)
    # shuffle, Plucker, size grid
    i_q, i_r = np.random.permutation(len(qry)), np.random.permutation(len(ref))
    inv_q, inv_r = np.argsort(i_q), np.argsort(i_r)
    matches = np.stack([inv_r[pairs[:, 1]], inv_q[pairs[:, 0]]], 0).astype(np.int64)
    ref_p, qry_p = _plucker(ref[i_r]), _plucker(qry[i_q])
    if MERGE or (domain == "outdoor" and OD >= 2):  # what the matcher sees at test time (merge first)
        from lib.utils import merge_lines
        ref_p, lr = merge_lines(ref_p.astype(np.float64))
        qry_p, lq = merge_lines(qry_p.astype(np.float64))
        ref_p, qry_p = ref_p.astype(np.float32), qry_p.astype(np.float32)
        matches = np.unique(np.stack([lr[matches[0]], lq[matches[1]]], 0), axis=1).astype(np.int64)
    ref_p, matches = _quantize_cloud(ref_p, matches, 0, _GMAX_R)
    qry_p, matches = _quantize_cloud(qry_p, matches, 1, _GMAX_Q)
    out = dict(plucker1=ref_p, plucker2=qry_p, matches=matches.astype(np.int32), R_gt=np.eye(3, dtype=np.float32),
               t_gt=np.zeros((3, 1), np.float32), s_gt=np.float32(1.0))
    if return_info:
        out["info"] = dict(domain=domain, perp=perp, spacing=sp, n_ident=len(ident), n_rule=len(rq),
                           n_rule_only=len(pairs) - len(ident), qry_seg=qry[i_q], ref_seg=ref[i_r],
                           qry_is_dup=i_q < len(inl), ref_is_crease=i_r < len(ref_c), faces=faces,
                           ref_piece=np.r_[_piece, np.array(['spur'] * (len(ref) - n_cre), object)][i_r])
    return out


def generate_pair():
    """One training pair (>= _MIN_INLIERS labelled matches), as train.py --live consumes it."""
    while True:
        pr = generate_pair_fsp()
        if pr["matches"].shape[1] >= _MIN_INLIERS:
            return pr


# ── dataset writer (same layout as generate_synthetic / prim_gen) ───────────────────────────────
def _worker(args):
    n, seed = args
    np.random.seed(seed)
    chunk = {k: [] for k in KEYS}
    while len(chunk["matches"]) < n:
        pr = generate_pair_fsp()
        if pr["matches"].shape[1] < _MIN_INLIERS:
            continue
        for k in KEYS:
            chunk[k].append(pr[k])
    return chunk


def write_split(out_dir, n_pairs, workers, chunk_size, seed):
    tasks = [(min(chunk_size, n_pairs - i), seed + 7919 * (i // chunk_size)) for i in range(0, n_pairs, chunk_size)]
    data = {k: [] for k in KEYS}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(_worker, tasks):
            for k in KEYS:
                data[k].extend(chunk[k])
            n = len(data["matches"])
            print(f"  {os.path.basename(out_dir)}: {n}/{n_pairs} ({n / (time.time() - t0):.0f} pairs/s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    for k in KEYS:
        with open(os.path.join(out_dir, f"{k}.pkl"), "wb") as f:
            pickle.dump(data[k], f, protocol=4)


def main():
    ap = argparse.ArgumentParser(description="full-range indoor+space pairs with GT-rule labels")
    ap.add_argument("--name", default="synthetic_fsp5")
    ap.add_argument("--n_train", type=int, default=300_000)
    ap.add_argument("--n_valid", type=int, default=3_000)
    ap.add_argument("--out_dir", default=str(_ROOT / "dataset"))
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    for split, n, seed in (("valid", a.n_valid, a.seed + 1), ("train", a.n_train, a.seed)):
        if n > 0:
            out = os.path.join(a.out_dir, f"{a.name}_{split}")
            print(f"[{time.strftime('%T')}] {split}: {n} pairs -> {out}", flush=True)
            write_split(out, n, a.workers, a.chunk_size, seed)


if __name__ == "__main__":
    main()
