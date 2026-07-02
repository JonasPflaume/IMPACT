/**
 * @file toy_mpcc.cpp
 * @brief Minimal 4D MPCC solved through buildMPCC().
 *
 *     minimize    ||x - [0.8, 0.2, 0.25, 0.75]||^2
 *     subject to  0 <= x1  ⊥  x2 >= 0
 *                 0 <= x3  ⊥  x4 >= 0
 *
 * This intentionally uses two separate complementarity groups to exercise the
 * direct MPCC API's multi-CompBlock support. The closest feasible point is
 * [0.8, 0, 0, 0.75], with objective 0.2^2 + 0.25^2 = 0.1025.
 *
 * This is NOT a trajectory optimization problem: there is no horizon, no
 * shooting. We build an MPCCDescription by hand and pass it to buildMPCC().
 *
 * Usage: toy_mpcc [x1_0 x2_0 x3_0 x4_0]   (optional initial guess; default 0 0 0 0)
 */

#include <casadi/casadi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "impact/bcd_aula_config.h"
#include "impact/bcd_aula_solver.h"
#include "impact/mpcc_subproblem.h"

int main(int argc, char* argv[]) {
    using casadi::SX;

    Eigen::Vector4d z0(0.0, 0.0, 0.0, 0.0);  // start from biactive points
    if (argc >= 5) {
        z0(0) = std::atof(argv[1]);
        z0(1) = std::atof(argv[2]);
        z0(2) = std::atof(argv[3]);
        z0(3) = std::atof(argv[4]);
    }

    std::cout << "=== Toy 4D MPCC (two complementarity blocks) ===\n"
              << "  min ||x - [0.8, 0.2, 0.25, 0.75]||^2\n"
              << "  s.t. 0 <= x1 perp x2 >= 0,  0 <= x3 perp x4 >= 0\n"
              << "  initial guess: [" << z0.transpose() << "]\n\n";

    // Build the MPCC description (z = [x1, x2, x3, x4]).
    SX z = SX::sym("z", 4);
    SX target = casadi::DM(std::vector<double>{0.8, 0.2, 0.25, 0.75});
    impact::MPCCDescription desc;
    desc.z = z;
    desc.p = SX::sym("p", 0);  // no runtime parameters
    desc.cost = z - target;
    desc.cost_is_linear = true;
    desc.addComplementarityBlock("axis_a", z(0), z(1), {/*scale=*/1.0, /*rho_init=*/1.0,
                                                         /*tol=*/1e-8});
    desc.addComplementarityBlock("axis_b", z(2), z(3), {/*scale=*/0.75, /*rho_init=*/2.0,
                                                         /*tol=*/1e-8});

    impact::MPCCSubproblem mpcc = impact::buildMPCC(desc);

    // Solver configuration.
    impact::BCDAULAConfig config;
    config.max_outer_iters = 300;
    config.max_inner_iters = 50;
    config.rho_scale = 1.5;
    config.rho_max = 1e3;
    config.outer_tol_h = 1e-7;
    config.outer_tol_comp = 1e-7;
    config.newton_max_iter = 50;
    config.use_saddle = true;
    config.print_level = 1;  // print the per-outer trace

    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(*mpcc.sub, config, Eigen::VectorXd(z0));

    const double comp0 = std::abs(r.z(0) * r.z(1));
    const double comp1 = std::abs(r.z(2) * r.z(3));
    std::cout << "\n=== Result ===\n"
              << "  converged:           " << (r.converged ? "YES" : "NO") << "\n"
              << "  x*                   = [" << r.z.transpose() << "]\n"
              << "  objective            = " << r.objective_value << "  (expected 0.1025)\n"
              << "  complementarity x1*x2 = " << comp0 << "\n"
              << "  complementarity x3*x4 = " << comp1 << "\n"
              << "  outer / inner iters  = " << r.outer_iterations << " / "
              << r.total_inner_iterations << "\n";
    return 0;
}
