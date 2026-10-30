"""Training loss: balanced BCE on the Sinkhorn plan (PlueckerNet's loss) """
import torch

def plan_bce(P, C):
    """Balanced BCE of the plan P against the GT correspondence matrix C (both (B,N1,N2)): positives and
    negatives each weigh 1/2 whatever their counts.  Mean over the batch."""
    n_pos = (C.sum(dim=(-2, -1)) - 1.0).relu() + 1.0
    n_neg = ((1.0 - C).sum(dim=(-2, -1)) - 1.0).relu() + 1.0
    ll = ((P + 1e-20).log() * C).sum(dim=(-2, -1)) * 0.5 / n_pos \
        + ((1.0 - P + 1e-20).log() * (1.0 - C)).sum(dim=(-2, -1)) * 0.5 / n_neg
    return -ll.mean()
