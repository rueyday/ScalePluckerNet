"""
Register a query line map onto a reference line map: Sim(3) (R, t, s) from line geometry alone.

    python registration.py query.npy reference.npy --ckpt output/<dataset>/<run>/best_val_checkpoint.pth

Each map is (N, 6) segment endpoints [p1 p2] (.npy, .npz with the array first, or whitespace text).
The result maps query onto reference:  x_ref = s R x_query + t.

Pipeline: segments -> Plücker [m; d] -> merge near-identical lines (lib.line_utils.prepare_lines)
-> uniform cap at 4000 merged lines -> matcher (lib/model.py) -> Sim(3) solver.

The solver is the C++ port (cpp/, built into lib/_sim3_solver_cpp*.so) when it is built, else the
numpy reference lib/sim3_solver.py; both give the same pose. Build the C++ one with
    PY=$(which python)
    cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$PY -Dpybind11_DIR=$($PY -m pybind11 --cmakedir) && cmake --build cpp/build
"""
import argparse
import numpy as np
import torch
from easydict import EasyDict

from lib.line_utils import segments_to_plucker, prepare_lines
from lib.utils import load_model
from lib import sim3_solver

try:
    from lib import _sim3_solver_cpp
except ImportError:
    _sim3_solver_cpp = None

MAX_LINES = 4000      # the training size


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


def load_matcher(ckpt, device):
    """The shipped matcher with a checkpoint's weights. Only SIZE and the encoder are read from the
    checkpoint config; a weight the checkpoint cannot fill is an error, never a silent random init."""
    state = torch.load(ckpt, weights_only=False, map_location='cpu')
    ck = dict(state.get('config', {}) or {})
    layers = list(ck.get('GNN_layers') or ['self', 'cross'] * 6)
    cfg = EasyDict(dict(model='KNNContextNormNet', normalize_n_lines=200, normalize_n_inliers=60,
                        net_depth=len(layers), net_nchannel=int(ck.get('net_nchannel', 128)), GNN_layers=layers,
                        net_gcnorm=True, net_batchnorm=True, net_topK=2000, net_lambda=0.1, net_maxiter=30,
                        net_knn=10, in_channel=6, attn_heads=int(ck.get('attn_heads', 4)),
                        encoder=ck.get('encoder', 'graff'), ordinal_nums=ck.get('ordinal_nums'),
                        use_gpu=device.type == 'cuda', gpu_inds=0))
    model = load_model('PluckerNetKnn')(cfg).to(device)
    missing, _ = model.load_state_dict(state.get('state_dict', state), strict=False)
    if missing:
        raise RuntimeError(f'{ckpt}: {len(missing)} weights missing from the checkpoint, e.g. {list(missing)[:3]}')
    return model.eval()


def prepare(segments, seed=0):
    """(N, 6) endpoints -> the Plücker lines the matcher and the solver see."""
    P = prepare_lines(segments_to_plucker(segments[:, :3], segments[:, 3:6]))
    if len(P) > MAX_LINES:
        P = P[np.sort(np.random.default_rng(seed).choice(len(P), MAX_LINES, replace=False))]
    return P


@torch.no_grad()
def match(model, p_q, p_r, device):
    """Matcher probabilities (len(p_r), len(p_q)). Input normalization: both clouds' moments divided by
    the query moment std (unit-free, keeps the query/reference scale ratio)."""
    std = float(p_q[:, :3].std()) + 1e-6
    w = lambda p: torch.from_numpy(np.concatenate([p[:, :3] / std, p[:, 3:]], 1).astype(np.float32))[None].to(device)
    return model(w(p_r), w(p_q))[0][0].float().cpu().numpy()


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
