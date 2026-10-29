"""Training loss: balanced BCE on the Sinkhorn plan + match_w x matchability cross-entropy on its marginals."""
import torch


def plan_bce(P, C):
    """Balanced BCE of the plan P against the GT correspondence matrix C (both (B,N1,N2)): positives and
    negatives each weigh 1/2 whatever their counts.  Mean over the batch."""
    n_pos = (C.sum(dim=(-2, -1)) - 1.0).relu() + 1.0
    n_neg = ((1.0 - C).sum(dim=(-2, -1)) - 1.0).relu() + 1.0
    ll = ((P + 1e-20).log() * C).sum(dim=(-2, -1)) * 0.5 / n_pos \
        + ((1.0 - P + 1e-20).log() * (1.0 - C)).sum(dim=(-2, -1)) * 0.5 / n_neg
    return -ll.mean()


def matchability_ce(r, c, C, eps=1e-12):
    """Cross-entropy pushing the marginal heads r (B,N1) and c (B,N2), softmax distributions over lines, onto the
    normalized indicator of lines with at least one GT correspondence, so balanced OT stops spending mass on
    matchless lines.  A sample without GT matches contributes 0.  Mean over the batch."""
    out = 0.0
    for p, m in ((r, (C.sum(-1) > 0).float()), (c, (C.sum(-2) > 0).float())):
        s = m.sum(-1, keepdim=True)
        out = out - ((m / s.clamp_min(1.0)) * (p + eps).log()).sum(-1) * (s.squeeze(-1) > 0)
    return out.mean()


def total_loss(P, C, r, c, match_w):
    loss = plan_bce(P, C)
    return loss + match_w * matchability_ce(r, c, C) if match_w > 0 else loss
