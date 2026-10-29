# ScalePluckerNet — SCALAR

**Sim(3) registration of 3D line maps across sensor modalities.** ScalePluckerNet
is a scale-aware Plücker line matcher trained on fully synthetic, SLAM-calibrated
Sim(3) data; paired with the line-based Sim(3) estimator in `lib/sim3_solver.py` (C++ port in `cpp/`)
it registers a scale-ambiguous monocular SLAM map onto a metric RGB-D or LiDAR
line map — recovering scale, rotation and translation from line geometry alone.

**Explanation, theory, and full results live on the
[project page](https://rueyday.github.io/scale-aware-cross-modal-registration/).**
It lives in the parent
[SCALAR repo](https://github.com/rueyday/scale-aware-cross-modal-registration) with the benchmarks and
evaluation tools; this README covers only the codebase.

---

## Architecture (final; `lib/model.py`, no flags)

```
lines [m; d] -> ordinal shape-context encoder -> 12 self/cross attention layers (128 ch, 4 heads)
             -> L2 descriptors + matchability marginals -> balanced Sinkhorn OT (mu 0.1, 30 iterations) -> P
```

**Encoder (ordinal shape context).** Every line gets its unit direction `d` and its perpendicular offset `q`
from the cloud's least-squares centre (the point closest to all lines, which moves with the map under Sim(3)).
For line `i`, every other line is ranked by affine-Grassmannian Graff(1,3) chordal distance (offsets divided by
the median `|q|`). The ranking is cut into 8 quantile shells (nearest 1/512 of the map ... the farthest quarter),
and in each shell the mean and std of `cos^2(d_i, d_j)`, `cos^2(d_i, u_ij)`, `cos^2(d_j, u_ij)` (`u_ij` = unit
direction of `q_j - q_i`) give a 48-D feature, then an MLP. Only an ORDER and squared cosines reach the network,
so it is exactly invariant to Sim(3) and to each line's Plücker sign, needs no input normalization, and quantile
shells describe the same share of a 150-line query and a 4000-line reference.

**Matching.** SuperGlue-style residual attentional propagation; the matchability head sees each line plus the
other cloud's global mean/max and outputs the Sinkhorn marginals. Loss: balanced BCE on the plan + 0.2 x
cross-entropy of the marginals against the has-a-GT-match indicator.

Earlier encoders (graff k-NN, absolute-rank ordinal, the order* family, bin ablations), the dustbin, sign-even
embeddings, geometric attention and graph wavelets were all measured and removed; the checkpoints that used them
do not load (`load_matcher` refuses any other architecture rather than running random weights).

## Setup

```bash
conda activate torch5090        # Python 3.11, PyTorch 2.6, CUDA
pip install easydict scipy tensorboardX   # + numpy, torch
```

`registration.py` + `lib/` register two maps with no external service or dataset, on CPU or GPU. The C++
solver additionally needs Eigen, CMake and pybind11 (optional).

## Layout

```
registration.py        END-TO-END: two line maps -> matcher -> Sim(3) (R, t, s)
lib/model.py           the matcher (one architecture) + load_matcher
lib/sim3_solver.py     the Sim(3) estimator, numpy reference (frozen)
cpp/                   the same estimator in C++ (header-only, Eigen) + its pybind11 module
lib/utils.py           Plücker conversion + the line merge (prepare_lines); numpy only
lib/{dataloader,loss,trainer}.py   training
train.py               training entry point (defaults = the shipped recipe)
fullrange_sp5_gen.py   synthetic training-pair generator (indoor + spacecraft + outdoor), work in progress
output/                checkpoints
```

## Registering two maps

```bash
python registration.py query.npy reference.npy --ckpt output/<dataset>/<run>/best_val_checkpoint.pth
```

Each map is `(N, 6)` segment endpoints `[p1 p2]`; the result maps the query onto the reference,
`x_ref = s R x_query + t`. The pipeline: segments -> Plücker -> merge near-identical lines
(`lib.utils.prepare_lines`: < 5 deg apart and < 2% of the map extent) -> uniform sample of 6000 merged lines (the inference limit: ~100 ms per pair on the RTX 5090) ->
matcher -> solver.

```python
from registration import load_matcher, register, solve_sim3
model = load_matcher(ckpt, device)
R, t, s = register(query_segments, reference_segments, model, device)
R, t, s = solve_sim3(p_q, p_r, prob)        # the solver alone: Plücker (N,6) [m; d], prob (Nr, Nq)
```

## The Sim(3) estimator (frozen 2026-10-29 -- only its defaults may still be tuned)

`solve_sim3(p_q, p_r, prob, topk=300, nsamp=3000, proj_thresh=1.984, skew_min=0.3, seed=0,
refit_iters=3, sx_max=1e3)`:

1. candidate correspondences = the top-`topk` matcher scores;
2. `nsamp` random 2-line samples; drop near-parallel pairs, `|d1 x d2| > skew_min` on both maps;
3. per sample, Kabsch rotation plus its 180-degree flip about reference line 1 (the other relative
   line sign), then `(t, s)` by least squares on `m_r = s R m_q + t x (R d_q)`;
4. the hypothesis with the most G(2,4) inliers wins (`||Yq^T Yr||_F^2 > proj_thresh`), counted with an
   exact early exit;
5. `refit_iters` closed-form alternating steps on the winner's inliers; a step stops if the scale is
   degenerate (`Sx / L > sx_max`, which grows like `1/s^2` as the scale collapses).

No normalization, no scale prior, no clamp beyond `s in [1e-3, 1e3]`.

**Two backends, one result.** `lib/sim3_solver.py` is the readable numpy reference. `cpp/` is the
optimized port, which `registration.solve_sim3` uses whenever the extension is built. It draws the
samples with numpy exactly as the reference does, so the pose matches: same pose on all 810 ScanNet++
pairs, **1.2 ms vs 110 ms per pair** single-threaded (~90x). It is the same algorithm: every speed-up
either skips work that provably cannot change the winner (exact early exit, a direction-only upper bound
on the inlier count) or scores in float32 with a rounding guard that falls back to double whenever float
could flip a comparison -- so the selected hypothesis is the double-precision one. The argument is in the
header comment of `cpp/sim3_solver.hpp`.
Build:

```bash
PY=$(which python)
cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$PY -Dpybind11_DIR=$($PY -m pybind11 --cmakedir)
cmake --build cpp/build          # -> lib/_sim3_solver_cpp*.so
```

From C++ directly: `#include "cpp/sim3_solver.hpp"`, `sim3::solve_sim3(p_q, nq, p_r, nr, prob, opts)`.
Without a `samples` array it uses its own generator, so the same seed gives different samples than
numpy.

## Training

```bash
python train.py --dataset synthetic_fsp5q8k --name <run> --live [--gpu_mem_frac 0.7]
python train.py ... --resume output/<dataset>/<run>/checkpoint.pth     # weights, optimizer, schedule, epoch
python train.py ... --init_from <ckpt>                                  # weights only
```

The defaults are the shipped (v102) recipe: 200 epochs x 32000 pairs, Adam 5e-4 with a cosine anneal to 1e-6,
32 samples per step, matchability weight 0.2, clouds capped at 6000 lines (every matched line kept; v102 itself trained at 8000). Batches are
same-shape buckets (the generator snaps sizes to a grid) of `min(32, 16384 // n, 8400000 // n^2)` pairs.
`--live` generates a fresh epoch with `fullrange_sp5_gen.py` in the background while the current one trains;
without it the trainer reads `dataset/<dataset>_train/` (served from a packed, memory-mapped copy it builds
once). Validation always reads `dataset/<dataset>_valid/`.

TensorBoard goes to `output/<dataset>/<run>/`; `WANDB=1` mirrors it to Weights & Biases (run id = md5 of the
run name, so restarts and the parent repo's `tools/wandb_backfill.py` update the same run).

**Trainer validation (top-100 precision) is a monitor, not a selection rule.** It has disagreed with the real
benchmarks many times; rank checkpoints by registration on the parent repo's cells
(`tools/eval_matchers_real.py`).

## Plücker conventions

`[m; d]` everywhere (moment first) -- upstream PlueckerNet's data and weights too; only its variable names say
`[d; m]`. Sim(3) law: `d' = R d`, `m' = s R m + t x d'`. Lines are sign-ambiguous (`[m; d] ~ [-m; -d]`). The
**matcher** is invariant by construction (above). The **solver** never canonicalizes: foot points `d x m` and
the G(2,4) projector are sign-invariant, the `(t, s)` solve sign-aligns each line to its reference
(`sign((R d_q).d_r)`), and the two relative signs of a 2-line sample are both proposed (Kabsch + its
180-degree flip about reference line 1), letting the G(2,4) inlier count arbitrate.
