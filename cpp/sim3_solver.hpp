// C++ port of lib/sim3_solver.py::solve_sim3 -- same algorithm, same defaults.
//
//   #include "sim3_solver.hpp"
//   sim3::Pose P = sim3::solve_sim3(p_q, nq, p_r, nr, prob);   // p_* (n,6) row-major [m; d]; prob (nr, nq)
//
// The result maps query lines onto reference lines:  d_r = R d_q,  m_r = s R m_q + t x d_r.
// Header-only, needs Eigen. Single-threaded: parallelise over problems, not inside one.
//
// Parity with the numpy lib: pass the numpy samples (`default_rng(seed).integers(0, k, (nsamp, 2))`)
// through `samples` -- registration.py::solve_sim3 does this, so it is a drop-in replacement.
// Without them, samples come from the built-in generator and differ from numpy for the same seed.
//
// SAME ALGORITHM AS THE NUMPY CODE: every stage computes the same mathematical quantity, so the two
// differ only by floating-point rounding. Where the computation is reorganized, and why it is equivalent:
//  * Procrustes in closed form (two planes + the in-plane angle). It is the unique Kabsch maximizer whenever
//    the two directions are not parallel, which the skew gate guarantees; the SVD is kept as the fallback.
//  * (t, s) by a Schur complement of the same 4x4 normal equations (same 1e-9 ridge, same clip).
//  * Inlier test: overlap > thr evaluated as  num > den * (thr - ...)  (den = 1 + |p_f|^2 > 0).
//  * Selection returns exactly the hypothesis the double count picks (most inliers, first on ties), but
//    avoids most of the work, with two EXACT shortcuts:
//      - early exit: stop counting once count + remaining <= best (the hypothesis can at most tie, and a
//        tie loses to the earlier index);
//      - direction bound: the [d; 0] column of Yq contributes (v.dr)^2 + (v.fr)^2 ir^2 and the other column
//        at most 1, so an inlier needs the first > thr - 1; that count bounds the inlier count from above;
//    and the count itself runs in float32 with a rounding guard: a lane whose float margin is within the
//    error bound is UNDECIDED, so the double count lies in [decided, decided + undecided], and the double
//    count is computed only when that interval does not settle the comparison with the best.
//  So the winner -- and hence the pose -- is the double-precision one. Verified: same pose as the numpy lib
//  on all 810 ScanNet++ pairs, and 0 interval violations in ~2e8 lane tests at scene scales 0.01 to 1e4.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace sim3 {

struct Options {
    int topk = 300;
    int nsamp = 3000;
    double proj_thresh = 1.984;
    double skew_min = 0.3;
    uint64_t seed = 0;
    int refit_iters = 3;
    double sx_max = 1e3;
};

struct Pose {
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // row-major
    double t[3] = {0, 0, 0};
    double s = 1.0;
};

namespace detail {

using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;
using std::vector;

inline void cross3(const double* a, const double* b, double* c) {
    c[0] = a[1] * b[2] - a[2] * b[1]; c[1] = a[2] * b[0] - a[0] * b[2]; c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double norm_cross(const double* a, const double* b) { double c[3]; cross3(a, b, c); return std::sqrt(dot3(c, c)); }

// sorted flat indices of the k largest entries. A strided subsample gives a threshold that keeps a
// few more than k candidates, so the full array is scanned once instead of partially sorted.
template <typename P>
void topk_flat(const P* x, size_t N, int k, vector<int64_t>& out, vector<P>& sub, vector<int64_t>& cand) {
    auto greater = [&](int64_t a, int64_t b) { return x[a] > x[b]; };
    out.clear();
    if (N > size_t(64) * k) {
        size_t step = N / (16 * size_t(k));
        sub.clear();
        for (size_t i = 0; i < N; i += step) sub.push_back(x[i]);
        long long ls = sub.size();
        long long j = std::max<long long>(ls - (2LL * k * ls) / (long long)N - 8, 0);
        std::nth_element(sub.begin(), sub.begin() + j, sub.end());
        P lo = sub[j];
        cand.clear();
        constexpr size_t W = 64;                         // vectorized count per block; scalar only on blocks with a hit
        size_t i = 0;
        for (; i + W <= N; i += W) {
            int hit = 0;
#pragma omp simd reduction(+ : hit)
            for (size_t j = 0; j < W; ++j) hit += x[i + j] >= lo;
            if (hit) for (size_t j = i; j < i + W; ++j) if (x[j] >= lo) cand.push_back(j);
        }
        for (; i < N; ++i) if (x[i] >= lo) cand.push_back(i);
        if ((int)cand.size() >= k) {
            std::nth_element(cand.begin(), cand.begin() + k, cand.end(), greater);
            out.assign(cand.begin(), cand.begin() + k);
            std::sort(out.begin(), out.end());
            return;
        }
    }
    cand.resize(N);
    for (size_t i = 0; i < N; ++i) cand[i] = i;
    std::nth_element(cand.begin(), cand.begin() + k, cand.end(), greater);
    out.assign(cand.begin(), cand.begin() + k);
    std::sort(out.begin(), out.end());
}

// Kabsch for two direction pairs, in closed form: the rotation is fixed by the two planes and the
// in-plane angle. Falls back to the SVD when a pair is (near) parallel.
inline M3 procrustes2(const double* q0, const double* q1, const double* r0, const double* r1) {
    auto frame = [](const double* a, const double* b, double* e1, double* e2, double* e3,
                    double& al, double& be, double& ga) {
        al = std::sqrt(dot3(a, a));
        for (int i = 0; i < 3; ++i) e1[i] = a[i] / al;
        be = dot3(e1, b);
        double v[3] = {b[0] - e1[0] * be, b[1] - e1[1] * be, b[2] - e1[2] * be};
        ga = std::sqrt(dot3(v, v));
        for (int i = 0; i < 3; ++i) e2[i] = v[i] / ga;
        cross3(e1, e2, e3);
    };
    double e1q[3], e2q[3], e3q[3], e1r[3], e2r[3], e3r[3], aq, bq, gq, ar, br, gr;
    frame(q0, q1, e1q, e2q, e3q, aq, bq, gq);
    frame(r0, r1, e1r, e2r, e3r, ar, br, gr);
    double p = ar * aq + br * bq + gr * gq, q = br * gq - gr * bq;
    double inr = 1.0 / std::sqrt(p * p + q * q); p *= inr; q *= inr;
    double f1[3], f2[3];
    for (int i = 0; i < 3; ++i) { f1[i] = p * e1q[i] + q * e2q[i]; f2[i] = p * e2q[i] - q * e1q[i]; }
    M3 R;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R(i, j) = e1r[i] * f1[j] + e2r[i] * f2[j] + e3r[i] * e3q[j];
    if (R.allFinite() && gq > 1e-12 * aq && gr > 1e-12 * ar) return R;
    M3 M;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) M(i, j) = r0[i] * q0[j] + r1[i] * q1[j];
    Eigen::JacobiSVD<M3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    M3 D = M3::Identity(); D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    return svd.matrixU() * D * svd.matrixV().transpose();
}

// (t, s) given R, least squares on m_r = s R m_q + t x (R d_q) over two lines (port of _solve_ts)
inline void solve_ts(const double* const dq[2], const double* const mq[2], const double* const dr[2],
                     const double* const mr[2], const M3& R, double* t, double& s) {
    double vv = 0, Pxx = 0, Pyy = 0, Pzz = 0, Pxy = 0, Pxz = 0, Pyz = 0;
    double g[3] = {0, 0, 0}, r[3] = {0, 0, 0}, h = 0, rho = 0;
    for (int l = 0; l < 2; ++l) {
        V3 v = R * Eigen::Map<const V3>(dq[l]), w = R * Eigen::Map<const V3>(mq[l]);
        if (v.dot(Eigen::Map<const V3>(dr[l])) < 0) { v = -v; w = -w; }   // sign-align to the reference
        vv += v.squaredNorm();
        Pxx += v[0] * v[0]; Pyy += v[1] * v[1]; Pzz += v[2] * v[2];
        Pxy += v[0] * v[1]; Pxz += v[0] * v[2]; Pyz += v[1] * v[2];
        double vw[3], vb[3]; cross3(v.data(), w.data(), vw); cross3(v.data(), mr[l], vb);
        for (int i = 0; i < 3; ++i) { g[i] += vw[i]; r[i] += vb[i]; }
        h += w.squaredNorm();
        rho += w.dot(Eigen::Map<const V3>(mr[l]));
    }
    // normal equations [[Ptt, g], [g^T, h]] [t; s] = [r; rho], ridge 1e-9, Ptt = sum |v|^2 I - v v^T
    double d = vv + 1e-9;
    double p00 = d - Pxx, p11 = d - Pyy, p22 = d - Pzz, p01 = -Pxy, p02 = -Pxz, p12 = -Pyz;
    h += 1e-9;
    double c00 = p11 * p22 - p12 * p12, c01 = p02 * p12 - p01 * p22, c02 = p01 * p12 - p02 * p11;
    double c11 = p00 * p22 - p02 * p02, c12 = p01 * p02 - p00 * p12, c22 = p00 * p11 - p01 * p01;
    double idet = 1.0 / (p00 * c00 + p01 * c01 + p02 * c02);
    double y[3] = {(c00 * g[0] + c01 * g[1] + c02 * g[2]) * idet, (c01 * g[0] + c11 * g[1] + c12 * g[2]) * idet,
                   (c02 * g[0] + c12 * g[1] + c22 * g[2]) * idet};   // Ptt^-1 g
    double z[3] = {(c00 * r[0] + c01 * r[1] + c02 * r[2]) * idet, (c01 * r[0] + c11 * r[1] + c12 * r[2]) * idet,
                   (c02 * r[0] + c12 * r[1] + c22 * r[2]) * idet};   // Ptt^-1 r
    s = (rho - dot3(g, z)) / (h - dot3(g, y));
    for (int i = 0; i < 3; ++i) t[i] = z[i] - s * y[i];
    if (std::isfinite(s)) s = std::min(std::max(s, 1e-3), 1e3);
}

// the correspondences, structure-of-arrays, plus the per-line constants the overlap needs
template <typename F>
struct Corr {
    int K = 0;
    vector<F> dq[3], fq[3], dr[3], fr[3], fq2, fqn, ir;   // fqn = |fq|, ir = 1 / sqrt(1 + |fr|^2)
    void resize(int k) {
        K = k;
        for (int i = 0; i < 3; ++i) { dq[i].resize(k); fq[i].resize(k); dr[i].resize(k); fr[i].resize(k); }
        fq2.resize(k); fqn.resize(k); ir.resize(k);
    }
    template <typename G> void copy_from(const Corr<G>& o) {
        resize(o.K);
        for (int i = 0; i < 3; ++i) for (int k = 0; k < K; ++k) {
            dq[i][k] = F(o.dq[i][k]); fq[i][k] = F(o.fq[i][k]); dr[i][k] = F(o.dr[i][k]); fr[i][k] = F(o.fr[i][k]);
        }
        for (int k = 0; k < K; ++k) { fq2[k] = F(o.fq2[k]); fqn[k] = F(o.fqn[k]); ir[k] = F(o.ir[k]); }
    }
};

// Inlier count of one pose: #(overlap > thr), overlap = ||Yq^T Yr||_F^2 (see _overlap), expanded:
//   v = R dq, w = R fq, transformed foot p_f = s w + t - v (v.t)   (dq . fq = 0),
//   overlap = (v.dr)^2 + (v.fr)^2 ir^2 + ((p_f.dr)^2 + (p_f.fr + 1)^2 ir^2) / (1 + |p_f|^2).
// Returns -1 as soon as the count cannot exceed `best` (exact: a hypothesis that can only tie
// never wins, since the first maximum does).
template <typename F>
int count_inliers(const Corr<F>& C, const M3& Rd, const V3& td, double sd, double thr, int best) {
    const F R0 = Rd(0, 0), R1 = Rd(0, 1), R2 = Rd(0, 2), R3 = Rd(1, 0), R4 = Rd(1, 1), R5 = Rd(1, 2),
            R6 = Rd(2, 0), R7 = Rd(2, 1), R8 = Rd(2, 2);
    const F t0 = td[0], t1 = td[1], t2 = td[2], s = sd, th = thr;
    const F c = F(1) + t0 * t0 + t1 * t1 + t2 * t2;
    const F *dqx = C.dq[0].data(), *dqy = C.dq[1].data(), *dqz = C.dq[2].data();
    const F *fqx = C.fq[0].data(), *fqy = C.fq[1].data(), *fqz = C.fq[2].data();
    const F *drx = C.dr[0].data(), *dry = C.dr[1].data(), *drz = C.dr[2].data();
    const F *frx = C.fr[0].data(), *fry = C.fr[1].data(), *frz = C.fr[2].data();
    const F *fq2 = C.fq2.data(), *ir = C.ir.data();
    constexpr int BLK = 64;
    int cnt = 0;
    for (int k0 = 0; k0 < C.K; k0 += BLK) {
        if (cnt + (C.K - k0) <= best) return -1;
        const int k1 = std::min(C.K, k0 + BLK);
#pragma omp simd reduction(+ : cnt)
        for (int k = k0; k < k1; ++k) {
            F vx = R0 * dqx[k] + R1 * dqy[k] + R2 * dqz[k];
            F vy = R3 * dqx[k] + R4 * dqy[k] + R5 * dqz[k];
            F vz = R6 * dqx[k] + R7 * dqy[k] + R8 * dqz[k];
            F wx = R0 * fqx[k] + R1 * fqy[k] + R2 * fqz[k];
            F wy = R3 * fqx[k] + R4 * fqy[k] + R5 * fqz[k];
            F wz = R6 * fqx[k] + R7 * fqy[k] + R8 * fqz[k];
            F a = vx * t0 + vy * t1 + vz * t2;                                     // v.t
            F dd = vx * drx[k] + vy * dry[k] + vz * drz[k];                        // v.dr
            F df = vx * frx[k] + vy * fry[k] + vz * frz[k];                        // v.fr
            F pf = s * (wx * frx[k] + wy * fry[k] + wz * frz[k]) + (t0 * frx[k] + t1 * fry[k] + t2 * frz[k]) - a * df;
            F pd = s * (wx * drx[k] + wy * dry[k] + wz * drz[k]) + (t0 * drx[k] + t1 * dry[k] + t2 * drz[k]) - a * dd;
            F np2 = c + s * s * fq2[k] + F(2) * s * (wx * t0 + wy * t1 + wz * t2) - a * a;   // 1 + |p_f|^2
            F pfi = (pf + F(1)) * ir[k], dfi = df * ir[k];
            cnt += (pfi * pfi + pd * pd > np2 * (th - dfi * dfi - dd * dd)) ? 1 : 0;
        }
    }
    return cnt;
}


// Necessary condition for an inlier: the [d; 0] column of Yq contributes (v.dr)^2 + (v.fr)^2 ir^2 and the
// other column at most 1, so overlap > thr needs (v.dr)^2 + (v.fr)^2 ir^2 > thr - 1. Counting that (a third of
// the arithmetic) bounds the inlier count from above; a pose whose bound cannot beat `best` is skipped
// exactly. The margin keeps rounding from ever excluding a true inlier.
template <typename F>
int count_direction_bound(const Corr<F>& C, const M3& Rd, double thr, int best) {
    const F R0 = Rd(0, 0), R1 = Rd(0, 1), R2 = Rd(0, 2), R3 = Rd(1, 0), R4 = Rd(1, 1), R5 = Rd(1, 2),
            R6 = Rd(2, 0), R7 = Rd(2, 1), R8 = Rd(2, 2);
    const F th = F(thr - 1.0 - (std::is_same_v<F, float> ? 1e-5 : 1e-9));
    const F *dqx = C.dq[0].data(), *dqy = C.dq[1].data(), *dqz = C.dq[2].data();
    const F *drx = C.dr[0].data(), *dry = C.dr[1].data(), *drz = C.dr[2].data();
    const F *frx = C.fr[0].data(), *fry = C.fr[1].data(), *frz = C.fr[2].data();
    const F* ir = C.ir.data();
    constexpr int BLK = 64;
    int cnt = 0;
    for (int k0 = 0; k0 < C.K; k0 += BLK) {
        if (cnt + (C.K - k0) <= best) return -1;
        const int k1 = std::min(C.K, k0 + BLK);
#pragma omp simd reduction(+ : cnt)
        for (int k = k0; k < k1; ++k) {
            F vx = R0 * dqx[k] + R1 * dqy[k] + R2 * dqz[k];
            F vy = R3 * dqx[k] + R4 * dqy[k] + R5 * dqz[k];
            F vz = R6 * dqx[k] + R7 * dqy[k] + R8 * dqz[k];
            F dd = vx * drx[k] + vy * dry[k] + vz * drz[k];
            F dfi = (vx * frx[k] + vy * fry[k] + vz * frz[k]) * ir[k];
            cnt += (dd * dd + dfi * dfi > th) ? 1 : 0;
        }
    }
    return cnt;
}

// float32 count with a rounding guard: a lane whose float margin |lhs - rhs| is within the error bound
// 256 u P^2, P = s |fq| + |t| + 1 (every product in lhs and rhs is at most a few P^2), is UNDECIDED.
// Returns the decided inlier count c and the undecided count u, so the double count lies in [c, c + u].
// Stops early (returns c = -1) once c + u + remaining <= best.
struct Bounds { int c, u; };
inline Bounds count_float(const Corr<float>& C, const M3& Rd, const V3& td, double sd, double thr, int best) {
    using F = float;
    const F R0 = Rd(0, 0), R1 = Rd(0, 1), R2 = Rd(0, 2), R3 = Rd(1, 0), R4 = Rd(1, 1), R5 = Rd(1, 2),
            R6 = Rd(2, 0), R7 = Rd(2, 1), R8 = Rd(2, 2);
    const F t0 = td[0], t1 = td[1], t2 = td[2], s = sd, th = thr;
    const F c = F(1) + t0 * t0 + t1 * t1 + t2 * t2, tn1 = F(td.norm() + 1.0), eps = F(1.0 / 16384);
    const F *dqx = C.dq[0].data(), *dqy = C.dq[1].data(), *dqz = C.dq[2].data();
    const F *fqx = C.fq[0].data(), *fqy = C.fq[1].data(), *fqz = C.fq[2].data();
    const F *drx = C.dr[0].data(), *dry = C.dr[1].data(), *drz = C.dr[2].data();
    const F *frx = C.fr[0].data(), *fry = C.fr[1].data(), *frz = C.fr[2].data();
    const F *fq2 = C.fq2.data(), *fqn = C.fqn.data(), *ir = C.ir.data();
    constexpr int BLK = 64;
    int cnt = 0, unc = 0;
    for (int k0 = 0; k0 < C.K; k0 += BLK) {
        if (cnt + unc + (C.K - k0) <= best) return {-1, 0};
        const int k1 = std::min(C.K, k0 + BLK);
#pragma omp simd reduction(+ : cnt, unc)
        for (int k = k0; k < k1; ++k) {
            F vx = R0 * dqx[k] + R1 * dqy[k] + R2 * dqz[k];
            F vy = R3 * dqx[k] + R4 * dqy[k] + R5 * dqz[k];
            F vz = R6 * dqx[k] + R7 * dqy[k] + R8 * dqz[k];
            F wx = R0 * fqx[k] + R1 * fqy[k] + R2 * fqz[k];
            F wy = R3 * fqx[k] + R4 * fqy[k] + R5 * fqz[k];
            F wz = R6 * fqx[k] + R7 * fqy[k] + R8 * fqz[k];
            F a = vx * t0 + vy * t1 + vz * t2;
            F dd = vx * drx[k] + vy * dry[k] + vz * drz[k];
            F df = vx * frx[k] + vy * fry[k] + vz * frz[k];
            F pf = s * (wx * frx[k] + wy * fry[k] + wz * frz[k]) + (t0 * frx[k] + t1 * fry[k] + t2 * frz[k]) - a * df;
            F pd = s * (wx * drx[k] + wy * dry[k] + wz * drz[k]) + (t0 * drx[k] + t1 * dry[k] + t2 * drz[k]) - a * dd;
            F np2 = c + s * s * fq2[k] + F(2) * s * (wx * t0 + wy * t1 + wz * t2) - a * a;
            F pfi = (pf + F(1)) * ir[k], dfi = df * ir[k];
            F m = pfi * pfi + pd * pd - np2 * (th - dfi * dfi - dd * dd);
            F P = s * fqn[k] + tn1;
            bool sure = std::abs(m) > eps * P * P;
            cnt += (sure && m > 0) ? 1 : 0;
            unc += sure ? 0 : 1;
        }
    }
    return {cnt, unc};
}

// one step of closed-form alternating descent on the Grassmann projection cost (port of
// _alternation_step); false = degenerate (no update)
inline bool alternation_step(const vector<V3>& dq, const vector<V3>& fq, const vector<V3>& dr,
                             const vector<V3>& fr, M3& R, V3& t, double& s, double sx_max) {
    const size_t L = dq.size();
    vector<double> gam(L), eta(L);
    vector<V3> y(L), xdir(L), xaff(L);
    double H = 0;
    V3 mux = V3::Zero(), muy = V3::Zero();
    for (size_t l = 0; l < L; ++l) {
        eta[l] = 1.0 / std::sqrt(1.0 + fr[l].squaredNorm());
        y[l] = eta[l] * fr[l];                                   // reference [f; 1]/n, spatial part
        V3 u = R * dq[l], q = s * (R * fq[l]) + t;               // transformed query direction, foot
        V3 qp = q - u * u.dot(q), yp = y[l] - u * u.dot(y[l]);
        // block 1: latents
        double lam = u.dot(dr[l]) / s;
        double g = (qp.dot(yp) + eta[l]) / (qp.squaredNorm() + 1.0);
        double alp = u.dot(y[l] - g * q) / s;
        xdir[l] = dq[l] * lam;
        xaff[l] = dq[l] * alp + g * fq[l];
        gam[l] = g;
        H += g * g;
        mux += g * xaff[l];
        muy += g * y[l];
    }
    // block 2: similarity Procrustes, with the affine points centred under weights gam
    if (H < 1e-12) return false;
    mux /= H; muy /= H;
    M3 M = M3::Zero();
    double Sx = 0;
    for (size_t l = 0; l < L; ++l) {
        V3 xa = xaff[l] - gam[l] * mux, ya = y[l] - gam[l] * muy;
        M += dr[l] * xdir[l].transpose() + ya * xa.transpose();
        Sx += xdir[l].squaredNorm() + xa.squaredNorm();
    }
    if (Sx / L > sx_max) return false;                           // Sx grows like 1/s^2 as the scale collapses
    Eigen::JacobiSVD<M3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const M3 &U = svd.matrixU(), &V = svd.matrixV();
    const V3& S = svd.singularValues();
    double det = (U * V.transpose()).determinant();
    M3 D = M3::Identity(); D(2, 2) = det;
    M3 Rn = U * D * V.transpose();
    double sn = (S[0] + S[1] + det * S[2]) / Sx;
    if (!std::isfinite(sn)) return false;
    sn = std::min(std::max(sn, 1e-3), 1e3);
    V3 tn = muy - sn * (Rn * mux);
    if (!(Rn.allFinite() && tn.allFinite())) return false;
    R = Rn; t = tn; s = sn;
    return true;
}

// the built-in sampler (splitmix64 + Lemire's bounded draw); NOT numpy's stream
struct Rng {
    uint64_t x;
    uint64_t next() {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int64_t below(uint64_t n) {
        for (;;) {
            unsigned __int128 m = (unsigned __int128)next() * n;
            uint64_t lo = (uint64_t)m;
            if (lo >= n || lo >= (0 - n) % n) return (int64_t)(m >> 64);
        }
    }
};

// Index of the hypothesis with the most inliers, the first on ties -- the winner the double count picks.
// Each hypothesis is screened by the direction bound, then counted in float32; it is recounted in double
// only when its float interval [c, c + u] leaves the comparison with the best undecided.
inline int best_hypothesis(const Corr<double>& C, const Corr<float>& Cf, const vector<M3>& R,
                           const vector<V3>& t, const vector<double>& s, double thr) {
    int bi = 0, best = -1;
    for (size_t h = 0; h < R.size(); ++h) {
        if (count_direction_bound<float>(Cf, R[h], thr, best) <= best) continue;
        Bounds b = count_float(Cf, R[h], t[h], s[h], thr, best);
        if (b.c < 0 || b.c + b.u <= best) continue;
        int c = b.u == 0 ? b.c : count_inliers<double>(C, R[h], t[h], s[h], thr, best);
        if (c > best) { best = c; bi = (int)h; }
    }
    return bi;
}

}  // namespace detail

// Sim(3) estimate from a matcher probability matrix `prob` (nr x nq, row-major, float or double).
// `samples` = nsamp (i, j) index pairs into the top-k list, row-major; nullptr draws them.
template <typename P>
Pose solve_sim3(const double* p_q, int nq, const double* p_r, int nr, const P* prob,
                const Options& o = Options(), const int64_t* samples = nullptr) {
    using namespace detail;
    Pose out;
    const size_t N = size_t(nr) * nq;
    const int K = (int)std::min<size_t>(o.topk, N);
    if (K <= 0) return out;

    // candidate correspondences = the top-k scores, sorted by index
    vector<int64_t> flat, cand; vector<P> sub;
    topk_flat(prob, N, K, flat, sub, cand);
    Corr<double> C; C.resize(K);
    vector<V3> mq(K), mr(K), dqv(K), drv(K);
    for (int k = 0; k < K; ++k) {
        const double* Lq = p_q + 6 * (flat[k] % nq);
        const double* Lr = p_r + 6 * (flat[k] / nq);
        double nq_ = std::sqrt(dot3(Lq + 3, Lq + 3)), nr_ = std::sqrt(dot3(Lr + 3, Lr + 3));
        double dq[3] = {Lq[3] / nq_, Lq[4] / nq_, Lq[5] / nq_}, dr[3] = {Lr[3] / nr_, Lr[4] / nr_, Lr[5] / nr_};
        double fq[3], fr[3]; cross3(dq, Lq, fq); cross3(dr, Lr, fr);   // foot f = d x m
        for (int i = 0; i < 3; ++i) { C.dq[i][k] = dq[i]; C.dr[i][k] = dr[i]; C.fq[i][k] = fq[i]; C.fr[i][k] = fr[i]; }
        C.fq2[k] = dot3(fq, fq); C.fqn[k] = std::sqrt(C.fq2[k]); C.ir[k] = 1.0 / std::sqrt(1.0 + dot3(fr, fr));
        dqv[k] = V3(dq[0], dq[1], dq[2]); drv[k] = V3(dr[0], dr[1], dr[2]);
        mq[k] = V3(Lq[0], Lq[1], Lq[2]); mr[k] = V3(Lr[0], Lr[1], Lr[2]);
    }

    // random 2-line samples, near-parallel pairs dropped on both sides
    vector<int> T0, T1;
    T0.reserve(o.nsamp); T1.reserve(o.nsamp);
    Rng rng{o.seed};
    for (int n = 0; n < o.nsamp; ++n) {
        int a, b;
        if (samples) { a = (int)samples[2 * n]; b = (int)samples[2 * n + 1]; }
        else { a = (int)rng.below(K); b = (int)rng.below(K); }
        if (a == b) continue;
        if (o.skew_min > 0 && !(norm_cross(dqv[a].data(), dqv[b].data()) > o.skew_min &&
                                norm_cross(drv[a].data(), drv[b].data()) > o.skew_min)) continue;
        T0.push_back(a); T1.push_back(b);
    }
    const int NS = (int)T0.size();
    if (NS == 0) return out;

    // two rotation hypotheses per sample (all Procrustes first, then all flips, as in the lib),
    // then (t, s) for each; non-finite ones dropped
    vector<M3> Rh; vector<V3> th; vector<double> sh;
    Rh.reserve(2 * NS); th.reserve(2 * NS); sh.reserve(2 * NS);
    vector<M3> R0(NS);
    for (int n = 0; n < NS; ++n) R0[n] = procrustes2(dqv[T0[n]].data(), dqv[T1[n]].data(), drv[T0[n]].data(), drv[T1[n]].data());
    for (int pass = 0; pass < 2; ++pass)
        for (int n = 0; n < NS; ++n) {
            const int a = T0[n], b = T1[n];
            M3 R = R0[n];
            if (pass == 1) {                                     // 180 deg about reference line 1: (2 a a^T - I) R0
                const V3& e = drv[a];
                R = (2.0 * e * e.transpose() - M3::Identity()) * R;
            }
            const double* dq[2] = {dqv[a].data(), dqv[b].data()};
            const double* mqp[2] = {mq[a].data(), mq[b].data()};
            const double* dr[2] = {drv[a].data(), drv[b].data()};
            const double* mrp[2] = {mr[a].data(), mr[b].data()};
            double t[3], s;
            solve_ts(dq, mqp, dr, mrp, R, t, s);
            if (!(std::isfinite(t[0]) && std::isfinite(t[1]) && std::isfinite(t[2]) && std::isfinite(s))) continue;
            Rh.push_back(R); th.emplace_back(t[0], t[1], t[2]); sh.push_back(s);
        }
    if (Rh.empty()) return out;

    // most G(2,4) inliers wins
    Corr<float> Cf; Cf.copy_from(C);
    const int b = best_hypothesis(C, Cf, Rh, th, sh, o.proj_thresh);
    M3 R = Rh[b]; V3 t = th[b]; double s = sh[b];

    // refit it on its inliers (overlap > thr, in double, the same expansion as the count)
    vector<V3> idq, ifq, idr, ifr;
    for (int k = 0; k < K; ++k) {
        V3 dq(C.dq[0][k], C.dq[1][k], C.dq[2][k]), fq(C.fq[0][k], C.fq[1][k], C.fq[2][k]);
        V3 dr(C.dr[0][k], C.dr[1][k], C.dr[2][k]), fr(C.fr[0][k], C.fr[1][k], C.fr[2][k]);
        V3 v = R * dq, p = s * (R * fq) + t;
        p -= v * v.dot(p);
        double ir = C.ir[k], np2 = 1.0 + p.squaredNorm();
        double ov = v.dot(dr) * v.dot(dr) + v.dot(fr) * v.dot(fr) * ir * ir +
                    (p.dot(dr) * p.dot(dr) + (p.dot(fr) + 1.0) * (p.dot(fr) + 1.0) * ir * ir) / np2;
        if (ov > o.proj_thresh) { idq.push_back(dq); ifq.push_back(fq); idr.push_back(dr); ifr.push_back(fr); }
    }
    for (int it = 0; it < o.refit_iters; ++it)
        if (!alternation_step(idq, ifq, idr, ifr, R, t, s, o.sx_max)) break;

    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) out.R[3 * i + j] = R(i, j);
    for (int i = 0; i < 3; ++i) out.t[i] = t[i];
    out.s = s;
    return out;
}

}  // namespace sim3
