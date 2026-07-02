// Multiple-shooting smoke test: run box / push_t / cart_transporter through the
// impact::MultipleShootingSolver and check each converges.
//
// (The exact same-binary equivalence against the legacy aula::MPCCBCDAULASolver
// was verified during migration; once the legacy solver was retired this became a
// standalone convergence check.)
//
//   parity_ms <box|push_t|cart> <x0...> <goal...>

#include <iostream>
#include <memory>
#include <string>

#include "box_pushing.h"
#include "cart_transporter.h"
#include "impact/multiple_shooting.h"
#include "push_t.h"

static impact::BCDAULAConfig boxConfig(const Eigen::VectorXd& x0, const Eigen::VectorXd& goal) {
    impact::BCDAULAConfig c;
    c.horizon = 50;
    c.x_0 = x0;
    c.x_goal = goal;
    c.stage_cost_weight = 0.001;
    c.final_cost_weight = 100.0;
    c.rho_max = 100.0;
    c.rho_scale = 1.1;
    const double s = 25.0;
    c.fix_point_scale = s;
    c.dynamics_scale = s;
    c.eq_scale = s;
    c.ineq_scale = s;
    c.comp_scale = 1.0;
    c.max_outer_iters = 500;
    c.outer_tol_h = 1e-5;
    c.outer_tol_comp = 1e-5;
    c.max_inner_iters = 50;
    c.inner_tol_init = 1e-2;
    c.inner_tol_final = 1e-3;
    c.newton_max_iter = 50;
    c.newton_tol = 1e-6;
    c.newton_regularization = 2e-5;
    c.use_saddle = true;
    c.print_level = 0;
    return c;
}

static impact::BCDAULAConfig pushtConfig(const Eigen::VectorXd& x0, const Eigen::VectorXd& goal) {
    impact::BCDAULAConfig c;
    c.horizon = 50;
    c.x_0 = x0;
    c.x_goal = goal;
    c.stage_cost_weight = 0.01;
    c.final_cost_weight = 100.0;
    c.rho_max = 500.0;
    c.rho_scale = 1.05;
    const double s = 25.0;
    c.fix_point_scale = s;
    c.dynamics_scale = s;
    c.eq_scale = s;
    c.ineq_scale = s;
    c.comp_scale = 1.0;
    c.max_outer_iters = 1000;
    c.outer_tol_h = 1e-5;
    c.outer_tol_comp = 1e-5;
    c.outer_tol_g = 1e-5;
    c.max_inner_iters = 50;
    c.inner_tol_init = 5e-3;
    c.inner_tol_final = 1e-3;
    c.newton_max_iter = 200;
    c.newton_tol = 1e-6;
    c.newton_regularization = 5e-5;
    c.use_saddle = true;
    c.print_level = 0;
    return c;
}

static impact::BCDAULAConfig cartConfig(const Eigen::VectorXd& x0, const Eigen::VectorXd& goal) {
    impact::BCDAULAConfig c;
    c.horizon = 300;
    c.x_0 = x0;
    c.x_goal = goal;
    c.stage_cost_weight = 1e-6;
    c.final_cost_weight = 5000.0;
    c.rho_max = 100000.0;
    c.rho_scale = 1.5;
    c.fix_point_scale = 1.0;
    c.dynamics_scale = 1.0;
    c.eq_scale = 1.0;
    c.ineq_scale = 1.0;
    c.comp_scale = 0.002;
    c.max_outer_iters = 1000;
    c.outer_tol_h = 1e-5;
    c.outer_tol_comp = 1e-5;
    c.outer_tol_g = 1e-5;
    c.max_inner_iters = 10;
    c.inner_tol_init = 1e-2;
    c.inner_tol_final = 1e-3;
    c.newton_max_iter = 100;
    c.newton_tol = 1e-6;
    c.newton_regularization = 1e-5;
    c.use_saddle = true;
    c.use_constant_state_init = true;
    c.print_level = 0;
    return c;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: parity_ms <box|push_t|cart> <x0...> <goal...>\n";
        return 1;
    }
    std::string task = argv[1];
    const int dim = (task == "cart") ? 4 : 3;
    if (argc < 2 + 2 * dim) {
        std::cerr << "Need " << 2 * dim << " state values for task " << task << "\n";
        return 1;
    }
    Eigen::VectorXd x0(dim), goal(dim);
    for (int i = 0; i < dim; ++i) x0(i) = std::atof(argv[2 + i]);
    for (int i = 0; i < dim; ++i) goal(i) = std::atof(argv[2 + dim + i]);

    std::shared_ptr<impact::MPCCProblem> problem;
    impact::BCDAULAConfig config;
    if (task == "box") {
        problem = std::make_shared<box_pushing::BoxPushing>();
        config = boxConfig(x0, goal);
    } else if (task == "push_t") {
        problem = std::make_shared<push_t::PushT>();
        config = pushtConfig(x0, goal);
    } else if (task == "cart") {
        problem = std::make_shared<cart_transporter::CartTransporter>();
        config = cartConfig(x0, goal);
    } else {
        std::cerr << "Unknown task: " << task << "\n";
        return 1;
    }

    impact::MultipleShootingSolver solver(problem);
    impact::MultipleShootingSolution s = solver.solve(config);
    std::cout << "[" << task << "] converged=" << (s.converged ? "Y" : "N")
              << " obj=" << s.objective_value << " outer=" << s.outer_iterations
              << " inner=" << s.total_inner_iterations << " time=" << s.solve_time
              << "s comp=" << s.complementarity_violation << "\n";
    std::cout << "  final=[" << s.state_trajectory.col(config.horizon).transpose() << "] goal=["
              << config.x_goal.transpose() << "]\n";
    return s.converged ? 0 : 1;
}
