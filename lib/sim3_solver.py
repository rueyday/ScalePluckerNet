"""
Sim(3) registration of two Plücker line maps from matcher correspondences.

    from lib.sim3_solver import solve_sim3
    R, t, s = solve_sim3(p_q, p_r, prob)   # p_q (Nq,6), p_r (Nr,6) as [m; d]; prob (Nr, Nq)

The result maps query lines onto reference lines:  d_r = R d_q,  m_r = s R m_q + t x d_r.
"""
import numpy as np

def _dir_foot(L):
    """Plücker [m; d] (..., 6) -> unit direction d and foot point f = d x m """
    d = L[..., 3:] / np.linalg.norm(L[..., 3:], axis=-1, keepdims=True)
    return d, np.cross(d, L[..., :3])

def _transform(d, f, R, t, s):
    """Apply x -> s R x + t to lines (d, f) (K, 3) under H poses -> (H, K, 3) each."""
    Rt = np.swapaxes(R, 1, 2)
    d2 = d @ Rt
    p = s[:, None, None] * (f @ Rt) + t[:, None, :]
    return d2, p - d2 * (d2 * p).sum(-1, keepdims=True)

def _basis(d, f):
    """A line's point on G(2,4): the orthonormal 4x2 basis of span{[d; 0], [f; 1]} """
    n = np.sqrt(1.0 + (f * f).sum(-1, keepdims=True))
    z = np.zeros_like(n)
    return np.stack([np.concatenate([d, z], -1), np.concatenate([f, 1.0 + z], -1) / n], -1)

def _overlap(R, t, s, dq, fq, Yr):
    """||Yq^T Yr||_F^2 for H poses x K correspondences -> (H, K). It is 2 when the
    transformed query line coincides with its reference line; the projection distance is
    ||Pq - Pr||_F^2 = 4 - 2 * overlap."""
    Yq = _basis(*_transform(dq, fq, R, t, s))                  # (H, K, 4, 2)
    return ((np.swapaxes(Yq, -1, -2) @ Yr) ** 2).sum((-2, -1))

def _procrustes(Dq, Dr):
    """Rotations best mapping directions Dq onto Dr (Kabsch): (NS, L, 3) -> (NS, 3, 3)."""
    U, _, Vt = np.linalg.svd(np.einsum('nli,nlj->nij', Dr, Dq))
    D = np.tile(np.eye(3), (len(U), 1, 1))
    D[:, 2, 2] = np.linalg.det(U @ Vt)
    return U @ D @ Vt

def _skew(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack([np.stack([o, -z, y], -1),
                     np.stack([z, o, -x], -1),
                     np.stack([-y, x, o], -1)], -2)

def _solve_ts(dq, mq, dr, mr, R):
    """(t, s) given R, by least squares on m_r = s R m_q + t x (R d_q) """
    v = np.einsum('nij,nlj->nli', R, dq)
    w = np.einsum('nij,nlj->nli', R, mq)
    sg = np.where((v * dr).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    v, w = v * sg, w * sg
    n, L = dq.shape[:2]
    A = np.concatenate([-_skew(v), w[..., None]], -1).reshape(n, 3 * L, 4)  # t x v = -[v]_x t
    b = mr.reshape(n, 3 * L, 1)
    At = np.swapaxes(A, 1, 2)
    x = np.linalg.solve(At @ A + 1e-9 * np.eye(4), At @ b)[..., 0]
    return x[:, :3], np.clip(x[:, 3], 1e-3, 1e3)

def _observable(d, T, skew_min):
    """Keep line pairs that are not near-parallel: |d1 x d2| = sin(angle) > skew_min."""
    return np.linalg.norm(np.cross(d[T[:, 0]], d[T[:, 1]]), axis=-1) > skew_min

def _alternation_step(dq, fq, dr, fr, R, t, s, sx_max=np.inf):
    """One step of closed-form alternating descent on the Grassmann projection cost"""
    eta = 1.0 / np.sqrt(1.0 + (fr * fr).sum(-1))
    y = eta[:, None] * fr                                  # reference [f; 1]/n, spatial part
    u = dq @ R.T                                           # transformed query direction
    q = s * fq @ R.T + t                                   # transformed query foot
    perp = lambda a: a - u * (u * a).sum(-1, keepdims=True)
    # block 1: latents
    lam = (u * dr).sum(-1) / s
    gam = ((perp(q) * perp(y)).sum(-1) + eta) / ((perp(q) ** 2).sum(-1) + 1.0)
    alp = (u * (y - gam[:, None] * q)).sum(-1) / s
    x_dir = dq * lam[:, None]
    x_aff = dq * alp[:, None] + gam[:, None] * fq
    # block 2: similarity Procrustes, with the affine points centred under weights gam
    H = (gam * gam).sum()
    if H < 1e-12:
        return None
    mux = (gam[:, None] * x_aff).sum(0) / H
    muy = (gam[:, None] * y).sum(0) / H
    xa = x_aff - gam[:, None] * mux
    ya = y - gam[:, None] * muy
    U, S, Vt = np.linalg.svd(dr.T @ x_dir + ya.T @ xa)
    det = np.linalg.det(U @ Vt)
    Rn = U @ np.diag([1.0, 1.0, det]) @ Vt
    Sx = (x_dir ** 2).sum() + (xa ** 2).sum()
    if Sx / len(dq) > sx_max:     # degenerate: Sx grows like 1/s^2 as the scale collapses
        return None
    sn = np.clip((S[0] + S[1] + det * S[2]) / Sx, 1e-3, 1e3)
    tn = muy - sn * Rn @ mux
    if not (np.isfinite(Rn).all() and np.isfinite(tn).all() and np.isfinite(sn)):
        return None
    return Rn, tn, sn

def _best_hypothesis(R, t, s, dq, fq, Yr, thr, chunk=256, block=50):
    """Index of the hypothesis with the most inliers (overlap > thr); the first on ties. """
    K = len(Yr)
    best_n, best_i = -1, 0
    for lo in range(0, len(R), chunk):
        idx = np.arange(lo, min(lo + chunk, len(R)))
        n = np.zeros(len(idx), int)
        for k in range(0, K, block):
            ks = slice(k, k + block)
            n += (_overlap(R[idx], t[idx], s[idx], dq[ks], fq[ks], Yr[ks]) > thr).sum(1)
            alive = n + max(K - k - block, 0) > best_n
            idx, n = idx[alive], n[alive]
            if len(idx) == 0:
                break
        if len(idx) and n.max() > best_n:
            best_n, best_i = n.max(), idx[n.argmax()]
    return best_i

def _refit(dq, fq, dr, fr, Yr, R, t, s, thr, iters, sx_max):
    """Re-solve the winning pose on ALL its inliers with `iters` alternating steps."""
    inl = _overlap(R[None], t[None], np.array([s]), dq, fq, Yr)[0] > thr

    for _ in range(iters):
        step = _alternation_step(dq[inl], fq[inl], dr[inl], fr[inl], R, t, s, sx_max)
        if step is None:
            break
        R, t, s = step
    return R, t, float(s)

def solve_sim3(p_q, p_r, prob, topk=300, nsamp=3000, proj_thresh=1.984,
               skew_min=0.3, seed=0, refit_iters=3, sx_max=1e3):
    """Sim(3) estimate (R, t, s) from a matcher probability matrix, in the caller's frame."""
    # candidate correspondences = the top-k scores, sorted by index
    k = min(topk, prob.size)
    flat = np.sort(np.argpartition(prob.ravel(), -k)[-k:])
    ir, iq = np.divmod(flat, prob.shape[1])
    Lq, Lr = p_q[iq].astype(float), p_r[ir].astype(float)
    dq, fq = _dir_foot(Lq)
    dr, fr = _dir_foot(Lr)
    mq, mr = Lq[:, :3], Lr[:, :3]

    # random 2-line samples, near-parallel pairs dropped on both sides
    T = np.random.default_rng(seed).integers(0, k, (nsamp, 2))
    T = T[T[:, 0] != T[:, 1]]
    if skew_min > 0:
        T = T[_observable(dq, T, skew_min) & _observable(dr, T, skew_min)]
    if len(T) == 0:
        return np.eye(3), np.zeros(3), 1.0

    # two rotation hypotheses per sample, then (t, s) for each
    R0 = _procrustes(dq[T], dr[T])
    a = dr[T[:, 0]]
    R = np.concatenate([R0, (2 * a[:, :, None] * a[:, None, :] - np.eye(3)) @ R0])
    TT = np.concatenate([T, T])
    t, s = _solve_ts(dq[TT], mq[TT], dr[TT], mr[TT], R)
    ok = np.isfinite(t).all(1) & np.isfinite(s)
    R, t, s = R[ok], t[ok], s[ok]
    if len(R) == 0:
        return np.eye(3), np.zeros(3), 1.0

    #  most G(2,4) inliers wins;  5. refit it on its inliers
    Yr = _basis(dr, fr)
    b = _best_hypothesis(R, t, s, dq, fq, Yr, proj_thresh)
    return _refit(dq, fq, dr, fr, Yr, R[b], t[b], float(s[b]), proj_thresh, refit_iters, sx_max)
