# ScalePluckerNet — SCALAR

**Sim(3) registration of 3D line maps across sensor modalities.** ScalePluckerNet
is a scale-aware Plücker line matcher trained on fully synthetic, SLAM-calibrated
Sim(3) data; paired with the line-based Sim(3) estimator in `lib/sim3_solver.py` (C++ port in `cpp/`)
it registers a scale-ambiguous monocular SLAM map onto a metric RGB-D or LiDAR
line map — recovering scale, rotation and translation from line geometry alone.

**Explanation, theory, and full results live on the
[project page](https://rueyday.github.io/scale-aware-cross-modal-registration/).**
A runnable [Colab demo notebook](https://github.com/rueyday/scale-aware-cross-modal-registration/blob/main/docs/scalar_demo.ipynb)
registers a real bundled 7-Scenes map pair end-to-end on CPU. Both live in the
parent [SCALAR repo](https://github.com/rueyday/scale-aware-cross-modal-registration);
this README covers only the codebase.

---

## Architecture (one configuration, no flags)

The encoder is a **single affine-Grassmannian branch**. Every line becomes a
2-plane in R^4; `vec(P)` is the 10 unique entries of the projector `P = Y Yᵀ`
with √2 on the off-diagonals, so Euclidean distance on `vec(P)` **is** the
chordal Grassmann distance (GraffMatch Eq. 3-4, Lusk et al. RA-L 2022).

```
per line i, over its k=10 nearest neighbours j in Graff(1,3):
    edge_ij = cat(vec(P)_j − vec(P)_i, vec(P)_i)        20-D
    node_i  = MLP( mean_j Conv2d(edge_ij) )             20→32→128
then 12 alternating self/cross attention layers (128 ch, 4 heads)
then L2-normalised descriptors → pairwise L2 → Sinkhorn (balanced, μ=0.1, 30 it)
```

**There is no sign handling in the matcher, and none is needed.** Flipping
`[m;d] → [−m;−d]` is the *same line*: it leaves `p0 = d × m` untouched and only
sends `c1 → −c1`, so the subspace — and hence `P` — is unchanged. Measured
exactly `0.000e+00` under random per-line flips. The k-NN graph and the
equivariant origin are built from foot points, which are sign-invariant too.
(The ambiguity is real: 47.3% of true cross-modal matches carry opposite Plücker
sign. A max-|component| hemisphere canon reduced that to a 7% residual seam; the
Grassmannian removes it by construction.)

**There is no σ normalisation.** Per-cloud σ is a statistic that assumes both
clouds cover comparable content — measured 21.5% error at full overlap, 35.8% at
1/8 crop. The matcher's only normalisation is a moment whitening by the query
moment std, which is invariant to the choice of world units while preserving the
query/reference scale ratio.

### Tested and removed — do not re-add
`sign_inv` (sign-even embedding), `dual_knn`, `geo_edge`/`geo_knn`/`graff_knn`,
graph wavelets, geometric attention bias, 6 GNN layers, max/min pooling,
separate node+edge branches, explicit principal-angle edge channels, the
SuperGlue dustbin, and a size-adaptive k. Each was measured neutral or worse;
the verdicts and numbers are in `lib/model.py` and the parent repo's CLAUDE.md.

## Setup

```bash
conda activate torch5090        # Python 3.11, PyTorch 2.6, CUDA
pip install easydict msgpack    # + numpy, torch, matplotlib
```

Self-contained: `registration.py` + `lib/` register two maps with no external service or
dataset. The C++ solver additionally needs Eigen, CMake and pybind11 (optional).

## Layout

```
registration.py        END-TO-END: two line maps -> matcher -> Sim(3) (R, t, s)
lib/model.py           the matcher -- ONE encoder, no architecture flags
lib/sim3_solver.py     THE Sim(3) estimator, numpy reference (frozen)
cpp/                   the same estimator in C++ (header-only, Eigen) + its pybind11 module
lib/line_utils.py      Plücker conversion + the line-merge preparation (numpy only)
lib/{trainer,loss,dataloader,utils}.py   training machinery
generate_synthetic.py  calibrated Sim(3) pair generator
train.py               training entry point
output/                checkpoints
```

## Registering two maps

```bash
python registration.py query.npy reference.npy --ckpt output/<dataset>/<run>/best_val_checkpoint.pth
```

Each map is `(N, 6)` segment endpoints `[p1 p2]`; the result maps the query onto the reference,
`x_ref = s R x_query + t`. The pipeline: segments -> Plücker -> merge near-identical lines
(`lib.line_utils.prepare_lines`) -> uniform cap at 4000 lines -> matcher -> solver. It runs on CPU.

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
# 1) generate data -- FLAGS MATTER, see ../../docs/scalepluckernet_dataset_flags.md
FOUND=1 INDOOR=1 MONO=1 DEALIAS=1 python generate_synthetic.py \
    --name synthetic_found8 --n_train 300000 --n_valid 3000 --workers 12

# 2) train (no architecture flags -- there is only one configuration)
python train.py --dataset synthetic_found8 --bucket_batch --match_w 0.2 \
    --batch 1 --iter_size 32 --lr 5e-4 --gamma 0.99 \
    --train_epoch_size 32000 --workers 4 --name my_run

# resume (restores optimizer + schedule):
python train.py ... --resume output/synthetic_found8/my_run/checkpoint.pth

# optional: generate pairs on the fly instead of loading a .pkl dataset
python train.py ... --live --live_workers 6 --workers 0
```

`--live` works but was measured to buy **nothing**: infinite unique pairs climb
at the same rate as the fixed 300k set (+0.0115 vs +0.0128 val/epoch) and score
the same on the real benchmark, at ~48% slower epochs. 300k is past the point
where repetition matters.

### Generator flags

The scenario mix is a declarative table (`_SPEC` in `generate_synthetic.py`),
not a chain of hand-written branches: each entry names a STRUCTURE (which line
pool) and a REGIME (sizes, overlap, noise). The street constants are fitted to
measured KITTI statistics -- do not "tidy" them.

`SOUND=1` (opt-in; `SOUND=0` reproduces `synthetic_found8` exactly, same RNG
stream) fixes two measured coverage defects. found8 put only **8.0%** of its
pairs in the 7-Scenes operating regime because uniform-in-angle rotation has
median 90 deg while the benchmarks sit at 1-29 deg, and because the indoor
scenarios drew scale log-uniform over 6 octaves. `SOUND` replaces the rotation
law with a MIXTURE (half uniform-angle, half log-uniform) -- **not a cap**;
v19 capped at 90 deg and got 4.9% inlier retention at 180 deg vs v41's 91.2%.
Support is unchanged: still full SO(3), still 8.8% of pairs above 150 deg.

    regime                              found8   SOUND
    7-Scenes (s 1.19-3.06, rot<=58)       8.0%   15.5%
    KITTI mono_best (s 8-42, rot<=12)    10.4%   14.7%
    median rotation                       80.0    32.3 deg

`LOCAL=1` (opt-in) makes SPATIAL LOCALITY a regime axis. Measured: only
`street_submap` concentrated its correspondences (0.28 of the reference
radius); the generic `submap` scenario was 0.98, i.e. scattered -- it modelled
a submap as fewer query lines, not as a localized region of a bigger cloud.
LOCAL lets any pool be drawn windowed (`LOCAL_P`=0.25) with the reference
forced 2.5-8x the query, taking submap-regime pairs from 7.8% to 28.8% of the
mix across all structure types rather than streets only.

The generator COVERS the deployment regimes; it is never CALIBRATED to them.
The claim is zero-shot, so measuring a target dataset's regime is a legitimate
check that the cell is populated, but must not become a loop that reshapes the
bands to fit it. See `../../docs/scalepluckernet_dataset_flags.md`.

## Monitoring

TensorBoard is always written to `output/<dataset>/<run>/`. **wandb needs a
separate mirror process** -- `train.py`'s own `WANDB=1` path is a no-op here
because wandb is not installed in the `torch5090` env and this directory's
`wandb/` shadows the import anyway:

```bash
# live: holds ONE run open, run stays "running" until the trainer exits
python tools/wandb_live.py <run-name> --match "[t]rain\.py --dataset <dataset>"
# finished runs: one-shot snapshot, updates the same run id in place
python tools/wandb_backfill.py <run-name>
```

Both live in the PARENT repo's `tools/` and must run under **base** python from
the parent repo root, never from this directory. Run ids are deterministic
(md5 of the name) so re-running updates in place; a `.wandb_id` file in the run
directory overrides that, which is the only recovery if a run was ever deleted
(wandb permanently burns a deleted id).

**`--train_epoch_size` is NOT saved in the checkpoint config.** Its absence
proves nothing. Recover the real epoch size from the TensorBoard step range:
`train/*` starts at `-(iters_per_epoch)`. Runs here use 1000 iters/epoch
(32000 pairs); omitting the flag silently gives 9375 and changes both
epoch-for-epoch comparability and the per-epoch LR decay.

**Checkpoint selection: never use trainer val.** It has disagreed with the real
benchmark five times in this project — including one arm 4.3 points *behind* on
val that tied on the benchmark, and a +0.66 val gain that scored slightly worse.
Select on the benchmarks below.

## Evaluation (benchmark data lives in the parent repo)

```bash
# cache the matcher forward once, then A/B solvers on the same cache
python tools/cache_matcher.py <ckpt> cache_7s.pkl --dataset 7scenes
python tools/full_results.py                      # the main results table
python tools/judge_ckpt.py <ckpt>                 # real @5 + alias audit
```

Matcher forward passes are GPU-numerics-sensitive and do not reproduce
bit-exactly across sessions — cache the probabilities once and compare solvers
on the same cache. RANSAC seed noise is ±2 pairs on 7-Scenes, so report
mean±std over ≥5 seeds; 3-seed "bests" have collapsed under re-measurement four
separate times.

## Plücker conventions

All code here uses `[m, d]` order (moment first); the original PlueckerNet uses
`[d, m]`. Sim(3) law: `d' = R d`, `m' = s·R·m + t × d'`. Lines are sign-ambiguous
(`[m,d] ~ [−m,−d]`). The **matcher** is invariant by construction (above). The
**solver** never canonicalizes: foot points `d x m` and the G(2,4) projector are sign-invariant,
the `(t, s)` solve sign-aligns each line to its reference (`sign((R d_q).d_r)`), and the two
relative signs of a 2-line sample are both proposed (Kabsch + its 180-degree flip about
reference line 1), letting the G(2,4) inlier count arbitrate.
