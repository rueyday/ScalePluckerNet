"""
Register a query line map onto a reference line map: Sim(3) (R, t, s) from line geometry alone.

    python registration.py query.npy reference.npy --ckpt output/<dataset>/<run>/best_val_checkpoint.pth

Each map is (N, 6) segment endpoints [p1 p2] (.npy, .npz with the array first, or whitespace text).
The result maps query onto reference:  x_ref = s R x_query + t.

Pipeline: segments -> Plücker [m; d] -> merge near-identical lines (lib.utils.prepare_lines)
-> uniform sample of 6000 merged lines -> matcher (lib/model.py) -> Sim(3) solver.

The solver is the C++ port (cpp/, built into lib/_sim3_solver_cpp*.so) when it is built, else the
numpy reference lib/sim3_solver.py; both give the same pose. Build the C++ one with
    PY=$(which python)
    cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$PY -Dpybind11_DIR=$($PY -m pybind11 --cmakedir) && cmake --build cpp/build
"""
import argparse
import numpy as np
import torch

from lib.model import load_matcher
from lib.utils import segments_to_plucker, prepare_lines
from lib import sim3_solver

try:
    from lib import _sim3_solver_cpp
except ImportError:
    _sim3_solver_cpp = None

MAX_LINES = 6000      # merged lines per map, uniformly sampled above it (also the training cap)


def solve_sim3(p_q, p_r, prob, topk=300, nsamp=3000, proj_thresh=1.984,
               skew_min=0.3, seed=0, refit_iters=3, sx_max=1e3):
    """lib.sim3_solver.solve_sim3, run in C++ when the extension is built. The samples are drawn here
    with numpy exactly as the numpy solver draws them, so both backends return the same pose."""
    if _sim3_solver_cpp is None:
        return sim3_solver.solve_sim3(p_q, p_r, prob, topk, nsamp, proj_thresh, skew_min, seed, refit_iters, sx_max)
    prob = np.ascontiguousarray(prob)
    if prob.dtype not in (np.float32, np.float64):
        prob = prob.astype(np.float64)
    T = np.random.default_rng(seed).integers(0, min(topk, prob.size), (nsamp, 2))
    return _sim3_solver_cpp.solve_sim3(p_q, p_r, prob, topk=topk, proj_thresh=proj_thresh, skew_min=skew_min,
                                       refit_iters=refit_iters, sx_max=sx_max, samples=T)


def prepare(segments, seed=0):
    """(N, 6) endpoints -> the Plücker lines the matcher and the solver see."""
    P = prepare_lines(segments_to_plucker(segments[:, :3], segments[:, 3:6]))
    if len(P) > MAX_LINES:
        P = P[np.sort(np.random.default_rng(seed).choice(len(P), MAX_LINES, replace=False))]
    return P


@torch.no_grad()
def match(model, p_q, p_r, device):
    """Matcher probabilities (len(p_r), len(p_q)).  No input normalization: the matcher is Sim(3)-invariant."""
    t = lambda p: torch.from_numpy(np.asarray(p, np.float32))[None].to(device)
    return model(t(p_r), t(p_q))[0][0].float().cpu().numpy()


def register(query_segments, reference_segments, model, device, seed=0):
    """Sim(3) (R, t, s) with x_ref = s R x_query + t."""
    p_q, p_r = prepare(query_segments, seed), prepare(reference_segments, seed)
    return solve_sim3(p_q, p_r, match(model, p_q, p_r, device), seed=seed)


def _load(path):
    if path.endswith('.npy'):
        a = np.load(path)
    elif path.endswith('.npz'):
        z = np.load(path)
        a = z[z.files[0]]
    else:
        a = np.loadtxt(path)
    return np.asarray(a, float).reshape(-1, 6)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('query'); ap.add_argument('reference')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', help='save R, t, s to this .npz')
    a = ap.parse_args()
    dev = torch.device(a.device)
    R, t, s = register(_load(a.query), _load(a.reference), load_matcher(a.ckpt, dev), dev, a.seed)
    np.set_printoptions(precision=6, suppress=True)
    print(f'solver: {"C++" if _sim3_solver_cpp else "numpy"}\nR =\n{np.asarray(R)}\nt = {np.asarray(t)}\ns = {float(s):.6f}')
    if a.out:
        np.savez(a.out, R=R, t=t, s=s)
