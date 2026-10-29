// pybind11 module `_sim3_solver_cpp` -- use it through registration.py::solve_sim3.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <stdexcept>
#include "sim3_solver.hpp"

namespace py = pybind11;
using Lines = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Samples = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

template <typename Prob>
static py::tuple solve(Lines p_q, Lines p_r, py::array_t<Prob, py::array::c_style> prob, int topk, int nsamp,
                       double proj_thresh, double skew_min, uint64_t seed, int refit_iters, double sx_max,
                       py::object samples) {
    if (p_q.ndim() != 2 || p_q.shape(1) != 6 || p_r.ndim() != 2 || p_r.shape(1) != 6)
        throw std::invalid_argument("p_q, p_r must be (n, 6) [m; d]");
    if (prob.ndim() != 2 || prob.shape(0) != p_r.shape(0) || prob.shape(1) != p_q.shape(0))
        throw std::invalid_argument("prob must be (len(p_r), len(p_q))");
    sim3::Options o;
    o.topk = topk; o.nsamp = nsamp; o.proj_thresh = proj_thresh; o.skew_min = skew_min; o.seed = seed;
    o.refit_iters = refit_iters; o.sx_max = sx_max;
    Samples T;
    const int64_t* Tp = nullptr;
    if (!samples.is_none()) {
        T = Samples::ensure(samples);
        if (!T || T.ndim() != 2 || T.shape(1) != 2) throw std::invalid_argument("samples must be (nsamp, 2)");
        o.nsamp = (int)T.shape(0); Tp = T.data();
    }
    sim3::Pose P;
    {
        py::gil_scoped_release nogil;
        P = sim3::solve_sim3(p_q.data(), (int)p_q.shape(0), p_r.data(), (int)p_r.shape(0), prob.data(), o, Tp);
    }
    py::array_t<double> R({3, 3}), t(3);
    std::memcpy(R.mutable_data(), P.R, sizeof P.R);
    std::memcpy(t.mutable_data(), P.t, sizeof P.t);
    return py::make_tuple(R, t, P.s);
}

#define SIM3_ARGS py::arg("p_q"), py::arg("p_r"), py::arg("prob"), py::arg("topk") = 300, \
    py::arg("nsamp") = 3000, py::arg("proj_thresh") = 1.984, py::arg("skew_min") = 0.3, py::arg("seed") = 0, \
    py::arg("refit_iters") = 3, py::arg("sx_max") = 1e3, py::arg("samples") = py::none()

PYBIND11_MODULE(_sim3_solver_cpp, m) {
    m.doc() = "C++ port of lib/sim3_solver.py::solve_sim3";
    m.def("solve_sim3", &solve<float>, SIM3_ARGS);    // prob float32 (the matcher's output) is read in place
    m.def("solve_sim3", &solve<double>, SIM3_ARGS);
}
