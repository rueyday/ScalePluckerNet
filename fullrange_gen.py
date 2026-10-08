"""fullrange_gen.py -- FULLRANGE=1: the dimensionless training-pair generator for
the exactly Sim(3)-invariant (ordinal) matcher, v82 (2026-09-28).

WHY A NEW GENERATOR. The ordinal encoder's features are squared cosines pooled
in rank bins, so a global similarity of either map changes NOTHING (measured on
the trained v81: identical inlier ratio under any pose incl. scale x200). Every
pose law in generate_synthetic (scale distributions, rotation mixtures,
translation ranges, street calibrations) is therefore dead weight for that
matcher, and the whole sampling budget goes to the axes the encoder still has
to learn from data. Each axis is DIMENSIONLESS and sampled log-uniform over a
range bracketing the measured real cells (ScanNet++ / KITTI full / KITTI
submap, tools p10-p90, 2026-09-28) by ~3x on each side -- coverage with
margin, no calibration:

  sizes        query 30-6000, reference = query x ratio, ratio 0.05-50
                                                  (real ratio 0.18-19)
  overlap      query partner fraction 1-90% (real 3-17%), reference partner
               fraction 0.3-90% (real 0-30%); CONTIGUOUS (ball window = submap)
               half the time, random subset otherwise
  structure    a PLANAR SCENE: room shell + clutter boxes + oblique elements
               (stairs / roof); Manhattan-ness alpha in [0,1] = fraction of
               boxes axis-aligned (pair-angle degeneracy spans random -> full
               Manhattan; real 0.18-0.49)
  population   the CROSS-MODAL gap, modelled by mechanism not by number: the
               reference is the scene's CREASE edges (what a laser/LiDAR crease
               extractor sees); the query is a subset of creases PLUS lines that
               lie ON the planar faces without being creases (photometric
               texture edges: measured 63% of real mono lines are on a surface
               but on no crease) PLUS off-surface junk. The reference also gets
               a small spurious population.
  noise        direction 1-15 deg (real 5-7); perpendicular offset as a
               FRACTION of the reference's nearest-neighbour spacing 0.02-1.0
               (dimensionless: it decides whether ranks scramble)
  fragments    1-4 query fragments per shared crease, independent noise
               (real 1.2-2.3); trajectory drift on 35% of queries (the one
               non-rigid effect that breaks exact invariance)

Pose: s=1, R=I, t=0 (NOPOSE). Output format identical to generate_synthetic:
plucker1 = REFERENCE, plucker2 = QUERY, matches = [ref_idx; qry_idx].
"""
import numpy as np

from generate_synthetic import (_physical_noise, _apply_traj_drift,
                                _quantize_cloud, _MIN_INLIERS)

RANGES = dict(
    n_q=(30, 6000), ratio=(0.05, 50.0), n_cap=(30, 8000),
    f_q=(0.01, 0.9), f_r=(0.003, 0.9),
    n_boxes=(1, 800), dir_deg=(1.0, 15.0), perp_frac=(0.02, 1.0),
    ref_dir_deg=(0.3, 3.0), ref_perp_frac=(0.005, 0.1),
    window=(0.08, 0.6), texture=(0.1, 0.8), ref_spurious=(0.0, 0.3),
)
P_CONTIGUOUS, P_ROOM, P_OBLIQUE, P_DRIFT = 0.5, 0.75, 0.35, 0.35


def _logu(lo, hi):
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


def _rand_rot():
    q = np.random.randn(4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


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


def _scene(alpha, n_boxes, room, oblique):
    segs, faces = [], []
    grow = max(1.0, n_boxes / 10.0) ** (1.0 / 3.0)          # keep clutter density plausible as boxes grow
    ext = np.array([_logu(3, 14) * grow, _logu(3, 14) * grow, np.random.uniform(2.3, 4.5) * min(grow, 2.0)])
    if room:
        s, f = _box(np.zeros(3), ext / 2, np.eye(3)); segs.append(s); faces += f
    for _ in range(n_boxes):
        half = np.array([_logu(0.05, 1.5), _logu(0.05, 1.5), _logu(0.05, 1.5)])
        centre = np.random.uniform(-ext / 2 + half, ext / 2 - half) if room else np.random.uniform(-ext / 2, ext / 2)
        if np.random.random() < alpha:
            R = np.eye(3)
        else:
            R = _rand_rot() if np.random.random() < 0.5 else _yaw(np.random.uniform(0, 2 * np.pi))
        s, f = _box(centre, half, R); segs.append(s); faces += f
    if oblique:
        # staircase: k parallel step edges along a tilted run; roof: two tilted panels
        k = np.random.randint(4, 14); rise, run = np.random.uniform(0.12, 0.25), np.random.uniform(0.25, 0.4)
        base = np.random.uniform(-ext / 2, ext / 2); R = _yaw(np.random.uniform(0, 2 * np.pi)); w = np.random.uniform(0.6, 2.0)
        st = []
        for i in range(k):
            p = base + R @ np.array([i * run, 0, i * rise]); st.append(np.stack([p - R[:, 1] * w / 2, p + R[:, 1] * w / 2]))
            p2 = p + R @ np.array([0, 0, rise]); st.append(np.stack([p2 - R[:, 1] * w / 2, p2 + R[:, 1] * w / 2]))
        segs.append(np.stack(st))
        if np.random.random() < 0.5:
            tilt = np.random.uniform(0.3, 1.0); Rr = _yaw(np.random.uniform(0, 2 * np.pi)) @ _rotx(tilt)
            s, f = _box(np.random.uniform(-ext / 2, ext / 2), np.array([_logu(0.5, 4), _logu(0.5, 4), 0.02]), Rr)
            segs.append(s); faces += f
    return np.concatenate(segs, 0), faces, ext


def _yaw(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rotx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _plucker(segs):
    a, b = segs[:, 0], segs[:, 1]
    d = b - a; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return np.concatenate([np.cross((a + b) / 2, d), d], 1).astype(np.float32)


def _texture_lines(faces, n, alpha, ext):
    """n lines lying ON faces (photometric edges): in-plane direction, aligned
    with the face axes with probability alpha (tiles, boards) else random."""
    if n <= 0 or not faces:
        return np.zeros((0, 6), np.float32)
    fi = np.random.randint(0, len(faces), n)
    C = np.stack([faces[i][0] for i in fi]); U = np.stack([faces[i][1] for i in fi]); V = np.stack([faces[i][2] for i in fi])
    p = C + U * np.random.uniform(-1, 1, (n, 1)) + V * np.random.uniform(-1, 1, (n, 1))
    ang = np.where(np.random.random(n) < alpha, np.random.randint(0, 2, n) * np.pi / 2, np.random.uniform(0, np.pi, n))
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12); Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    d = Un * np.cos(ang)[:, None] + Vn * np.sin(ang)[:, None]
    return np.concatenate([np.cross(p, d), d], 1).astype(np.float32)


def _junk_lines(n, ext):
    if n <= 0:
        return np.zeros((0, 6), np.float32)
    p = np.random.uniform(-ext / 2, ext / 2, (n, 3)); d = np.random.randn(n, 3); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([np.cross(p, d), d], 1).astype(np.float32)


def _spacing(L):
    p0 = np.cross(L[:, 3:], L[:, :3])
    if len(p0) < 3:
        return 1.0
    from scipy.spatial import cKDTree
    return float(np.median(cKDTree(p0).query(p0, 2)[0][:, 1])) + 1e-6


def generate_pair_fullrange() -> dict:
    # ── structure ──────────────────────────────────────────────────────────
    alpha = np.random.uniform(0, 1)
    # ── sizes first: the reference size decides how much structure to build ──
    n_q = int(_logu(*RANGES['n_q']))
    n_r = int(np.clip(n_q * _logu(*RANGES['ratio']), *RANGES['n_cap']))
    n_boxes = int(np.clip(n_r / 12.0 * np.random.uniform(0.7, 1.3), *RANGES['n_boxes']))
    segs, faces, ext = _scene(alpha, n_boxes, np.random.random() < P_ROOM, np.random.random() < P_OBLIQUE)
    creases = _plucker(segs)
    f_q, f_r = _logu(*RANGES['f_q']), _logu(*RANGES['f_r'])
    k = int(np.clip(min(f_q * n_q, f_r * n_r), _MIN_INLIERS, len(creases)))
    # shared creases: contiguous window (submap) or random subset
    foot = np.cross(creases[:, 3:], creases[:, :3])
    if np.random.random() < P_CONTIGUOUS:
        anchor = foot[np.random.randint(len(foot))]
        order = np.argsort(np.linalg.norm(foot - anchor, axis=1))
        shared = order[:k]; window = (anchor, np.linalg.norm(foot[order[k - 1]] - anchor) * np.random.uniform(1.0, 1.6))
    else:
        shared = np.random.choice(len(creases), k, replace=False); window = None
    # ── reference: all creases (subsampled to n_r) + spurious population ──
    n_spur_r = int(np.random.uniform(*RANGES['ref_spurious']) * n_r)
    n_cre_r = max(k, n_r - n_spur_r)
    keep = np.concatenate([shared, np.random.permutation(np.setdiff1d(np.arange(len(creases)), shared))])[:n_cre_r]
    ref = creases[keep]
    ref_shared_rows = np.arange(k)                               # shared creases are the first k rows of ref
    spur_r = np.concatenate([_texture_lines(faces, n_spur_r // 2, alpha, ext), _junk_lines(n_spur_r - n_spur_r // 2, ext)])
    sp = _spacing(ref)
    ref = _physical_noise(ref, _logu(*RANGES['ref_dir_deg']), _logu(*RANGES['ref_perp_frac']) * sp, 25.0)
    ref_all = np.concatenate([ref, spur_r]) if len(spur_r) else ref
    # ── query: fragments of the shared creases + texture + junk ────────────
    lam = _logu(1.0, 3.0)                                        # mean fragments per shared crease (real 1.0-2.3)
    n_frag = np.clip(1 + np.random.poisson(lam - 1.0, size=k), 1, 4)
    src = np.repeat(np.arange(k), n_frag)                        # fragment -> shared crease row
    dir_deg, perp = _logu(*RANGES['dir_deg']), _logu(*RANGES['perp_frac']) * sp
    inl_q = _physical_noise(creases[shared][src], dir_deg, perp, 25.0)
    n_rest = max(0, n_q - len(inl_q))
    n_tex = int(np.random.uniform(*RANGES['texture']) * n_rest)
    tex = _texture_lines(faces, n_tex, alpha, ext); junk = _junk_lines(n_rest - n_tex, ext)
    if window is not None:                                       # a submap only sees its window
        for arr in (tex, junk):
            if len(arr):
                f0 = np.cross(arr[:, 3:], arr[:, :3]); far = np.linalg.norm(f0 - window[0], axis=1) > window[1]
                if far.any():
                    keep_n = (~far).sum(); arr[far] = _texture_lines(faces, far.sum(), alpha, ext) if arr is tex else _junk_lines(far.sum(), ext)
    rest = np.concatenate([tex, junk]) if (len(tex) + len(junk)) else np.zeros((0, 6), np.float32)
    rest = _physical_noise(rest, dir_deg, perp, 25.0)
    qry_all = np.concatenate([inl_q, rest])
    if np.random.random() < P_DRIFT:
        qry_all = _apply_traj_drift(qry_all)
    # ── shuffle, matches, grid ─────────────────────────────────────────────
    i_q, i_r = np.random.permutation(len(qry_all)), np.random.permutation(len(ref_all))
    qry_all, ref_all = qry_all[i_q], ref_all[i_r]
    q_pos, r_pos = np.argsort(i_q)[:len(inl_q)], np.argsort(i_r)[:k]
    matches = np.stack([r_pos[src], q_pos], 0).astype(np.int32)
    ref_all, matches = _quantize_cloud(ref_all, matches, 0)
    qry_all, matches = _quantize_cloud(qry_all, matches, 1)
    return dict(plucker1=ref_all.astype(np.float32), plucker2=qry_all.astype(np.float32),
                matches=matches.astype(np.int32), R_gt=np.eye(3, dtype=np.float32),
                t_gt=np.zeros((3, 1), np.float32), s_gt=np.float32(1.0))
