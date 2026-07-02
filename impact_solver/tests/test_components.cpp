// Component tests for solver building blocks:
//   1. projectComplementarity closed-form (Y,Z)-update vs per-pair brute force.
//   2. GaussNewtonSolver saddle backend vs classical normal equations on a
//      least-squares problem with a cost block + a rho-scaled penalty block.

#include <casadi/casadi.hpp>
#include <cmath>
#include <iostream>
#include <random>

#include "impact/complementarity_projection.h"
#include "impact/gauss_newton_solver.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
    std::cout << (ok ? "[ PASS ] " : "[ FAIL ] ") << msg << std::endl;
    if (!ok) ++g_failures;
}

// Complementarity projection vs brute force.
void testProjection() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    const int n = 200;
    Eigen::VectorXd G(n), H(n), kG(n), kH(n);
    for (int i = 0; i < n; ++i) {
        G(i) = u(rng);
        H(i) = u(rng);
        kG(i) = u(rng);
        kH(i) = u(rng);
    }
    const double rho = 3.7, scale = 0.6;
    Eigen::VectorXd sG, sH;
    impact::projectComplementarity(G, H, kG, kH, rho, scale, sG, sH);

    // Per-pair augmented penalty cost the projection minimises:
    //   rho * [ (scale*(G-sG) + kG/rho)^2 + (scale*(H-sH) + kH/rho)^2 ]
    auto cost = [&](int i, double y, double z) {
        const double a = scale * (G(i) - y) + kG(i) / rho;
        const double b = scale * (H(i) - z) + kH(i) / rho;
        return a * a + b * b;
    };

    double max_excess = 0.0, max_comp = 0.0;
    for (int i = 0; i < n; ++i) {
        const double c_proj = cost(i, sG(i), sH(i));
        // Brute-force over the two legs (z=0 sweep y>=0, y=0 sweep z>=0).
        double best = 1e300;
        for (int k = 0; k <= 4000; ++k) {
            const double t = 4.0 * k / 4000.0;  // candidate magnitude in [0,4]
            best = std::min(best, cost(i, t, 0.0));
            best = std::min(best, cost(i, 0.0, t));
        }
        max_excess = std::max(max_excess, c_proj - best);    // proj must be <= grid best (+eps)
        max_comp = std::max(max_comp, std::abs(sG(i) * sH(i)));
        if (sG(i) < -1e-12 || sH(i) < -1e-12) max_excess = 1e9;  // must stay nonnegative
    }
    check(max_excess < 1e-6, "projection achieves the per-pair minimum (excess=" +
                                 std::to_string(max_excess) + ")");
    check(max_comp < 1e-12, "projection is exactly complementary (max|sG*sH|=" +
                                std::to_string(max_comp) + ")");
}

// Saddle vs normal-equations Gauss-Newton.
// r(z; p) = [ z - a ; sqrt(rho) * (z - b) ],  p = [a(n); b(n); rho(1)].
// argmin ||r||^2 = (a + rho*b) / (1 + rho), componentwise.
double solveLS(bool use_saddle, const Eigen::VectorXd& a, const Eigen::VectorXd& b, double rho) {
    using casadi::SX;
    const int n = static_cast<int>(a.size());
    SX z = SX::sym("z", n);
    SX pa = SX::sym("pa", n), pb = SX::sym("pb", n), prho = SX::sym("prho", 1);
    SX p = SX::vertcat({pa, pb, prho});
    SX r = SX::vertcat({z - pa, SX::sqrt(prho) * (z - pb)});
    casadi::Function res("res", {z, p}, {r});
    casadi::Function jac("jacobian_fn", {z, p}, {SX::jacobian(r, z)});

    impact::GaussNewtonConfig cfg;
    cfg.use_saddle = use_saddle;
    cfg.max_iter = 50;
    cfg.grad_tol = 1e-12;

    impact::SaddleLayout layout;
    layout.n_z = n;
    layout.n_cost = n;                       // first n rows are the cost block
    layout.n_dual = n;                       // next n rows are the rho penalty block
    layout.blocks.push_back({n, n, 2 * n});  // row_start=n, count=n, rho at offset 2n

    impact::GaussNewtonSolver gn;
    gn.init(res, jac, cfg, use_saddle ? &layout : nullptr);

    std::vector<double> pv;
    pv.insert(pv.end(), a.data(), a.data() + n);
    pv.insert(pv.end(), b.data(), b.data() + n);
    pv.push_back(rho);
    casadi::DM params(pv);

    Eigen::VectorXd z0 = Eigen::VectorXd::Zero(n);
    gn.minimize(params, z0);

    Eigen::VectorXd z_star = (a + rho * b) / (1.0 + rho);
    return (z0 - z_star).lpNorm<Eigen::Infinity>();
}

void testGaussNewton() {
    Eigen::VectorXd a(4), b(4);
    a << 1.0, -2.0, 0.5, 3.0;
    b << 0.0, 1.0, -1.0, 2.0;
    for (double rho : {1.0, 100.0, 1e4}) {
        const double err_normal = solveLS(false, a, b, rho);
        const double err_saddle = solveLS(true, a, b, rho);
        check(err_normal < 1e-8, "normal-equations GN solves LS (rho=" + std::to_string(rho) +
                                     ", err=" + std::to_string(err_normal) + ")");
        check(err_saddle < 1e-8, "saddle GN solves LS (rho=" + std::to_string(rho) +
                                     ", err=" + std::to_string(err_saddle) + ")");
    }
}

}  // namespace

int main() {
    std::cout << "=== impact component tests ===" << std::endl;
    testProjection();
    testGaussNewton();
    std::cout << (g_failures == 0 ? "ALL PASS" : (std::to_string(g_failures) + " FAILED"))
              << std::endl;
    return g_failures == 0 ? 0 : 1;
}
