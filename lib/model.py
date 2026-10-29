"""PluckerNetKnn: the Plücker line-cloud matcher (architecture final, 2026-10-29).

    lines [m; d] -> ordinal shape-context encoder -> 12-layer self/cross attention GNN
                 -> L2 descriptors + matchability marginals -> balanced Sinkhorn OT -> P (N1 x N2)

The encoder sees the geometry only through Sim(3)-invariant quantities (a rank ordering and squared cosines), so the
whole network is invariant to the similarity transform and to the Plücker sign of every line, and needs no input
normalization.  Earlier encoders (graff, absolute-rank ordinal, order*, the bins ablation) live in git history only.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 128                      # channels
LAYERS = ['self', 'cross'] * 6   # GNN layers
HEADS = 4
MU = 0.1                         # Sinkhorn entropy
SINKHORN_ITERS = 30              # already past convergence


def MLP(channels: list, do_gn=True):
    n, layers = len(channels), []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < n - 1:
            if do_gn:
                layers.append(nn.GroupNorm(4, channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def pairwiseL2Dist(x1, x2):
    """Batched pairwise L2. (B,N,D),(B,M,D) -> (B,N,M)."""
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    return torch.baddbmm(
        x2_norm2.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()


def line_offsets(raw):
    """Unit directions d and perpendicular offsets q (both (B,3,N)) of every line from the cloud's
    Sim(3)-EQUIVARIANT centre c = argmin_x sum_i dist(x, line_i)^2, i.e. A c = b with A = sum (I - d d^T) and
    b = sum (I - d d^T) p0 = sum p0 (the foot point p0 = d x m is perpendicular to d).  Under (s,R,t):
    c -> sRc + t and q -> sRq.  Foot points are sign-invariant, so q is too.  raw: (B,6,N) = [m; d]."""
    d = F.normalize(raw[:, 3:], dim=1)
    p0 = torch.cross(d, raw[:, :3], dim=1)
    eye = torch.eye(3, device=d.device, dtype=d.dtype)
    A = (d.shape[-1] + 1e-6) * eye - d @ d.transpose(1, 2)
    c = torch.linalg.solve(A, p0.sum(-1, keepdim=True))            # (B,3,1)
    dp = p0 - c
    return d, dp - d * (d * dp).sum(1, keepdim=True)


def graff_order(d, q):
    """(B,N,N-1): every other line ranked by its AFFINE-GRASSMANNIAN Graff(1,3) distance to line i, nearest first.

    Each line is the 2-plane span{[q/sigma; 1], [d; 0]} in R^4 (sigma = median |q|, which makes the ordering
    Sim(3)-invariant).  The 10 unique entries of its projector P = Y Y^T (sqrt(2) on the off-diagonals) make Euclidean
    distance the chordal Grassmann distance (GraffMatch Eq. 3-4, Lusk et al. RA-L 2022).  Sign-invariant:
    [m;d] -> [-m;-d] leaves the subspace unchanged."""
    sig = q.norm(dim=1).median(dim=1).values.clamp_min(1e-9)
    q = q / sig.view(-1, 1, 1)
    one = torch.ones_like(q[:, :1])
    c0 = F.normalize(torch.cat([q, one], dim=1), dim=1)            # (B,4,N)
    c1 = torch.cat([d, torch.zeros_like(one)], dim=1)
    c1 = F.normalize(c1 - c0 * (c1 * c0).sum(1, keepdim=True), dim=1)
    a, b = torch.triu_indices(4, 4, device=d.device)
    w = torch.where(a == b, 1.0, 2.0 ** 0.5).to(d.dtype)[:, None]
    v = (w * (c0[:, a] * c0[:, b] + c1[:, a] * c1[:, b])).transpose(1, 2)      # (B,N,10)
    return torch.cdist(v, v).argsort(dim=-1)[..., 1:]               # self (distance 0) dropped


class conv_in_ordinal(nn.Module):
    """ORDINAL SHAPE CONTEXT over the whole map: the input encoder.

    For line i, every other line j is ranked by Graff(1,3) distance (graff_order; only the ORDER is used, never a
    distance value).  The ranking is cut into 8 QUANTILE shells -- ranks up to 1/512 of the map, then 1/512-1/128,
    1/128-1/32, 1/32-1/8, 1/8-1/4, 1/4-1/2, 1/2-3/4, 3/4-all (adjacent shells share their boundary rank).  In each
    shell the mean and std of three angle statistics, cos^2(d_i,d_j), cos^2(d_i,u_ij), cos^2(d_j,u_ij)
    (u_ij = unit direction of q_j - q_i), give a 48-D feature -> MLP -> C.  Invariant to Sim(3) and to any monotone
    reparametrisation of distance; sign-invariant (squared cosines); quantile shells cover the same share of the map
    for any N, so a 150-line query and a 4000-line reference describe comparable supports.
    """

    # Shell edges as EXACT INTEGER fractions NUMS/DEN.  Integers on purpose: torch.compile (dynamic=True) makes N a
    # SymInt and float arithmetic on it hangs dynamo.
    DEN = 512
    NUMS = (1, 4, 16, 64, 128, 256, 384, 512)

    def __init__(self, out_channel: int):
        super().__init__()
        self.mlp_node = MLP([6 * len(self.NUMS), out_channel, out_channel, out_channel])

    @staticmethod
    @(torch._dynamo.disable if hasattr(torch, '_dynamo') else (lambda f: f))
    def _shells(N):
        """Inclusive (lo, hi) rank ranges per shell, outside the compiled graph: the chained edges otherwise build a
        nested min/max/FloorDiv tree that sympy never finishes (measured 2026-10-03, >19 min stall).  A shell that
        is empty on a tiny cloud falls back to the two farthest ranks."""
        M = max(int(N) - 1, 1)                                      # neighbours available
        out, prev = [], 1
        for num in conv_in_ordinal.NUMS:                            # strictly increasing, width >= 1
            hi = min(M, max(prev + 1, (num * M) // conv_in_ordinal.DEN))
            out.append((prev, hi) if hi > prev else (max(M - 1, 1), M))
            prev = hi
        return out

    def forward(self, raw):
        d, q = line_offsets(raw)                                    # (B,3,N)
        B, _, N = q.shape
        order = graff_order(d, q)                                   # (B,N,N-1)
        dT, qT = d.transpose(1, 2), q.transpose(1, 2)               # (B,N,3)
        base = torch.arange(B, device=q.device).view(-1, 1, 1) * N
        dF, qF = dT.reshape(B * N, 3), qT.reshape(B * N, 3)
        di, qi = dT.unsqueeze(2), qT.unsqueeze(2)
        feats = []
        for lo, hi in self._shells(N):
            # flat gather: dT[b, idx] advanced indexing trips an inductor assertion under dynamic=True (torch 2.6)
            flat = (order[..., lo - 1:hi] + base).reshape(-1)
            dj, qj = dF[flat].view(B, N, -1, 3), qF[flat].view(B, N, -1, 3)     # (B,N,w,3)
            u = F.normalize(qj - qi, dim=-1)
            a = torch.stack([(di * dj).sum(-1) ** 2, (di * u).sum(-1) ** 2, (dj * u).sum(-1) ** 2], 1)   # (B,3,N,w)
            feats += [a.mean(-1), a.std(-1, unbiased=False)]
        return self.mlp_node(torch.cat(feats, 1))                   # (B,48,N) -> (B,C,N)


def attention(query, key, value):
    """Scaled dot-product attention, memory-efficient kernel. (B, d, H, N) -> (B, d, H, N)

    .contiguous() is load-bearing (2026-10-27): the permuted views are not contiguous in the head dim, so SDPA fell
    back to the MATH kernel, which materialises every N x N attention matrix for backward -- 14 of the 16 GB of a
    4096 x 8000 training step.  Contiguous inputs select the memory-efficient kernel: that step needs 3.1 GB and runs
    3x faster, 16384 x 16384 fits in 15 GB; outputs equal to float precision (max |dP| 4e-7, argmax 99.8%)."""
    q, k, v = (x.permute(0, 2, 3, 1).contiguous() for x in (query, key, value))     # (B, H, N, d)
    return F.scaled_dot_product_attention(q, k, v).permute(0, 3, 1, 2)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        B = query.size(0)
        query, key, value = [l(x).view(B, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x = attention(query, key, value)
        return self.merge(x.contiguous().view(B, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        return self.mlp(torch.cat([x, self.attn(x, source, source)], dim=1))


class SpatialAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, num_heads)
                                     for _ in range(len(layer_names))])
        self.names = layer_names
        self.mlp = MLP([feature_dim * 3, feature_dim * 2, feature_dim * 2, feature_dim])

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            src0, src1 = (desc1, desc0) if name == 'cross' else (desc0, desc1)
            desc0 = desc0 + layer(desc0, src0)
            desc1 = desc1 + layer(desc1, src1)

        # Per-line matchability prior: each side gets global (mean+max) context from the other
        N0, N1 = desc0.size(-1), desc1.size(-1)
        g0 = torch.cat([desc0.mean(-1, keepdim=True), desc0.max(-1, keepdim=True)[0]], dim=1)
        g1 = torch.cat([desc1.mean(-1, keepdim=True), desc1.max(-1, keepdim=True)[0]], dim=1)
        desc0_reg = self.mlp(torch.cat([desc0, g1.expand(-1, -1, N0)], dim=1))
        desc1_reg = self.mlp(torch.cat([desc1, g0.expand(-1, -1, N1)], dim=1))
        return desc0, desc1, desc0_reg, desc1_reg


# EAGER on purpose (2026-10-27): compiled, inductor saved an N1 x N2 buffer per iteration for backward -- 16.8 vs
# 4.1 GB at 8000 x 8000, OOM at 12288 x 16384.  Eager it keeps one K; the rest of the network stays compiled.
@(torch._dynamo.disable if hasattr(torch, "_dynamo") else (lambda f: f))
def sinkhorn(M, r, c, mu=MU, iterations=SINKHORN_ITERS, eps=1e-12):
    """Entropic OT plan for cost M (B,N1,N2) with learned marginals r (B,N1), c (B,N2): balanced transport.
    A SuperGlue dustbin was tested and rejected (2026-08-30): calibrated scores that never converted into
    registration accuracy."""
    K = (-M / mu).exp()
    K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(eps)
    u = r.unsqueeze(-1)
    c = c.unsqueeze(-1)
    r = u.clone()
    v = torch.ones_like(c)
    for _ in range(iterations):                  # fixed count: no per-iteration CUDA sync
        v = c / K.transpose(-2, -1).matmul(u).clamp_min_(eps)
        u = r / K.matmul(v).clamp_min_(eps)
    return (u * K) * v.transpose(-2, -1)


class FeatureExtractorGraph(nn.Module):
    """Ordinal encoder -> self/cross attention GNN -> descriptors + matchability."""

    def __init__(self):
        super().__init__()
        self.conv_in = conv_in_ordinal(WIDTH)
        self.gnn = SpatialAttentionalGNN(WIDTH, LAYERS, HEADS)
        self.final_proj = nn.Conv1d(WIDTH, WIDTH, kernel_size=1, bias=True)
        self.regress = nn.Conv1d(WIDTH, 1, kernel_size=1, bias=True)

    def forward(self, x, y):
        desc0, desc1, x_prob, y_prob = self.gnn(self.conv_in(x), self.conv_in(y))
        return (self.final_proj(desc0), self.final_proj(desc1),
                self.regress(x_prob).softmax(dim=-1), self.regress(y_prob).softmax(dim=-1))


class PluckerNetKnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.FeatureExtractor = FeatureExtractorGraph()

    def forward(self, plucker1, plucker2):
        """plucker1 (B,N1,6), plucker2 (B,N2,6), rows [m; d] -> (P (B,N1,N2), r (B,N1), c (B,N2))."""
        f1, f2, p1, p2 = self.FeatureExtractor(plucker1.transpose(-2, -1), plucker2.transpose(-2, -1))
        f1 = F.normalize(f1.transpose(-2, -1), p=2, dim=-1)
        f2 = F.normalize(f2.transpose(-2, -1), p=2, dim=-1)
        r, c = p1.squeeze(1), p2.squeeze(1)
        return sinkhorn(pairwiseL2Dist(f1, f2), r, c), r, c


def load_matcher(ckpt, device='cpu'):
    """PluckerNetKnn with a checkpoint's weights, in eval mode.  Refuses a checkpoint of any other architecture
    (missing or unexpected weights) instead of running it with randomly initialised layers."""
    state = torch.load(ckpt, weights_only=False, map_location='cpu')
    cfg = dict(state.get('config') or {})
    if cfg.get('encoder', 'ordinal_q') != 'ordinal_q' or cfg.get('ordinal_nums') \
            or int(cfg.get('net_nchannel', WIDTH)) != WIDTH or list(cfg.get('GNN_layers') or LAYERS) != LAYERS \
            or int(cfg.get('attn_heads', HEADS)) != HEADS:
        raise ValueError(f'{ckpt}: not the shipped architecture (config {cfg})')
    model = PluckerNetKnn()
    model.load_state_dict(state.get('state_dict', state))       # strict
    return model.to(device).eval()
