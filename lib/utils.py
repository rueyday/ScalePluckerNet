"""Line-set utilities shared by registration, training and the tools -- numpy only.

Plücker convention [m; d] (moment first): d unit direction, m = p x d for any point p on the line.

    segments_to_plucker(p1, p2)    endpoints (N,3),(N,3) -> (N,6) Plücker, m from the midpoint
    merge_lines(P)                 collapse near-identical infinite lines (fragments of one edge, duplicate bundles)
                                   into their average -> (P_merged, labels)
    prepare_lines(P)               THE line-cloud preparation: merge_lines at the default 5 deg / 2% of the extent

PROTOCOL (Ruey 2026-10-22/23): merge the full raw map first, then cap the MERGED lines (uniform) only above the
matcher's size.  Measured with v95, 11 seeds: ScanNet++ 379 -> 390 strict (rot<10 469 -> 495), space query
2988 -> 1558 lines at equal accuracy.
"""
import numpy as np


def segments_to_plucker(p1, p2):
    d = p2 - p1
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


def feet(P):
    """Closest point of each line to the origin, d x m (invariant to the Plücker sign)."""
    P = np.asarray(P, float)
    return np.cross(P[:, 3:6], P[:, :3])


def extent(P):
    """p90 radius of the foot points about their median: the map's scale, for dimensionless thresholds."""
    f = feet(P)
    return float(np.percentile(np.linalg.norm(f - np.median(f, 0), axis=1), 90)) + 1e-12


def merge_lines(P, ang_deg=5.0, dist_frac=0.02, block=2048):
    """Collapse near-identical INFINITE lines into one averaged line.

    Two lines are near-identical when the angle between their directions is < ang_deg AND the perpendicular
    separation (foot-point difference projected off the direction, exact for parallel lines) is < dist_frac x
    extent(P).  Clusters are grown GREEDILY from the line with the most near-identical partners, taking only lines
    not yet assigned that are near-identical to the SEED (so a slowly rotating family cannot chain into one cluster).
    A cluster's line: the sign-aligned mean direction through the mean of the members' closest points to the
    cluster's mean foot point.  Candidate pairs come from a direction tree (only lines within ang_deg can merge), so
    the cost grows with the number of near-parallel pairs, not N^2.

    Returns (P_merged (K,6) float32, labels (N,) int: original line -> merged row).
    """
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    n = len(P)
    if n == 0:
        return P.astype(np.float32), np.zeros(0, int)
    d = P[:, 3:6] / (np.linalg.norm(P[:, 3:6], axis=1, keepdims=True) + 1e-12)
    f = np.cross(d, P[:, :3])
    tol = dist_frac * extent(P)
    # candidates: directions within ang_deg of +-d (chord radius, slightly inflated), then the exact tests
    tree = cKDTree(np.r_[d, -d])
    rad = 2 * np.sin(np.radians(ang_deg) / 2) * 1.000001
    ii, jj = [], []
    for i0 in range(0, n, block):
        for k, l in enumerate(tree.query_ball_point(d[i0:i0 + block], rad)):
            c = np.unique(np.asarray(l) % n)
            ii.append(np.full(len(c), i0 + k)); jj.append(c)
    ii, jj = np.concatenate(ii), np.concatenate(jj)
    ok = np.abs((d[ii] * d[jj]).sum(1)) > np.cos(np.radians(ang_deg))
    ii, jj = ii[ok], jj[ok]
    df = f[jj] - f[ii]
    pj = (df * d[ii]).sum(1)
    keep = np.sqrt(np.maximum((df * df).sum(1) - pj ** 2, 0.0)) < tol
    ii, jj = ii[keep], jj[keep]
    nbr = np.split(jj, np.searchsorted(ii, np.arange(1, n)))        # sorted neighbours of every line (incl. itself)

    labels = -np.ones(n, int)
    out = []
    for i in np.argsort([-len(x) for x in nbr], kind="stable"):
        if labels[i] >= 0:
            continue
        mem = nbr[i][labels[nbr[i]] < 0]
        if len(mem) <= 1:                                           # singleton: the arithmetic below, inlined
            labels[i] = len(out)
            dm = d[i] / (np.linalg.norm(d[i]) + 1e-12)
            out.append(np.r_[np.cross(f[i], dm), dm])
            continue
        if i not in mem:
            mem = np.r_[i, mem]
        labels[mem] = len(out)
        sg = np.sign(d[mem] @ d[i]); sg[sg == 0] = 1
        dm = (d[mem] * sg[:, None]).mean(0); dm /= np.linalg.norm(dm) + 1e-12
        c = f[mem].mean(0)
        # closest point on each member line to the cluster centre, averaged -> a point on the merged line
        p = (f[mem] + ((c - f[mem]) * d[mem]).sum(1, keepdims=True) * d[mem]).mean(0)
        out.append(np.r_[np.cross(p, dm), dm])
    return np.asarray(out, np.float32), labels


def prepare_lines(P):
    """THE line-cloud preparation: Plücker (N,6) -> merged Plücker (K,6)."""
    return merge_lines(np.asarray(P, np.float32))[0]
