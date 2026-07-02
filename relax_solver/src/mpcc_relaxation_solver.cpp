#include "relaxation_solver/mpcc_relaxation_solver.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace aula {

MPCCScholtesRelaxationSolver::MPCCScholtesRelaxationSolver(std::shared_ptr<MPCCProblem> problem)
    : problem_(problem) {}

casadi::Dict MPCCScholtesRelaxationSolver::createIPOPTOptions(
    const ScholtesRelaxationConfig& config) {
    casadi::Dict opts;
    opts["ipopt.print_level"] = config.ipopt_print_level;
    opts["ipopt.max_iter"] = config.ipopt_max_iter;
    opts["ipopt.tol"] = config.ipopt_tol;
    opts["ipopt.acceptable_tol"] = config.ipopt_acceptable_tol;
    opts["print_time"] = config.print_time;
    opts["ipopt.sb"] = "yes";  // Suppress IPOPT banner

    // Warm start options
    if (config.warm_start) {
        opts["ipopt.warm_start_init_point"] = "yes";
        opts["ipopt.warm_start_bound_push"] = 1e-9;
        opts["ipopt.warm_start_bound_frac"] = 1e-9;
        opts["ipopt.warm_start_slack_bound_frac"] = 1e-9;
        opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
        opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
    }

    return opts;
}

void MPCCScholtesRelaxationSolver::buildNLP(const ScholtesRelaxationConfig& config,
                                            casadi::SX& opt_vars, casadi::SX& obj, casadi::SX& g,
                                            int& nx_total, int& nu_total, int& nc) {
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    nc = problem_->getComplementarityDim();
    int neq = problem_->getEqualityConstraintDim();
    int nineq = problem_->getInequalityConstraintDim();
    int horizon = config.horizon;
    double dt = problem_->getTimeStep();

    // Get CasADi functions
    casadi::Function dyna_func = problem_->getDynamicsFunction();
    casadi::Function G_func = problem_->getGFunction();
    casadi::Function H_func = problem_->getHFunction();
    casadi::Function E_func = problem_->getEqualityConstraintFunction();
    casadi::Function I_func = problem_->getInequalityConstraintFunction();

    // Decision variables
    casadi::SX X = casadi::SX::sym("X", nx, horizon + 1);
    casadi::SX U = casadi::SX::sym("U", nu, horizon);

    nx_total = nx * (horizon + 1);
    nu_total = nu * horizon;

    // Objective function
    obj = 0;

    // Goal state
    casadi::SX x_goal_sx = casadi::SX(config.x_goal);

    // Constraint vector
    std::vector<casadi::SX> g_vec;

    // Stage cost and complementarity constraints
    for (int i = 0; i < horizon; ++i) {
        casadi::SX x_curr = X(casadi::Slice(), i);
        casadi::SX u_curr = U(casadi::Slice(), i);

        std::vector<casadi::SX> func_in = {x_curr, u_curr};
        casadi::SX G_value = G_func(func_in)[0];
        casadi::SX H_value = H_func(func_in)[0];

        // Stage cost (control effort)
        obj += config.stage_cost_weight * casadi::SX::sumsqr(u_curr);

        // Stage state tracking cost
        if (config.stage_state_cost_weight > 0) {
            obj += config.stage_state_cost_weight * casadi::SX::sumsqr(x_curr - x_goal_sx);
        }

        // Scholtes constraints: G >= 0
        g_vec.push_back(G_value);

        // Scholtes constraints: H >= 0
        g_vec.push_back(H_value);

        // Scholtes constraints: G*H <= t (element-wise product)
        g_vec.push_back(G_value * H_value);

        // Additional equality constraints: E = 0
        if (neq > 0 && !E_func.is_null()) {
            casadi::SX E_value = E_func(func_in)[0];
            g_vec.push_back(E_value);
        }

        // Additional inequality constraints: g <= 0
        if (nineq > 0 && !I_func.is_null()) {
            casadi::SX I_value = I_func(func_in)[0];
            g_vec.push_back(I_value);
        }
    }

    // Terminal cost
    casadi::SX x_final = X(casadi::Slice(), horizon);
    obj += config.final_cost_weight * casadi::SX::sumsqr(x_final - x_goal_sx);

    // Dynamics constraints
    for (int i = 0; i < horizon; ++i) {
        casadi::SX x_curr = X(casadi::Slice(), i);
        casadi::SX x_next = X(casadi::Slice(), i + 1);
        casadi::SX u_curr = U(casadi::Slice(), i);

        std::vector<casadi::SX> f_in = {x_curr, u_curr};
        casadi::SX f_value = dyna_func(f_in)[0];
        casadi::SX x_next_pred = x_curr + dt * f_value;

        g_vec.push_back(x_next - x_next_pred);
    }

    // Initial condition constraint
    casadi::SX x_0_sx = casadi::SX(config.x_0);
    g_vec.push_back(X(casadi::Slice(), 0) - x_0_sx);

    // Concatenate constraints
    g = casadi::SX::vertcat(g_vec);

    // Flatten decision variables
    opt_vars = casadi::SX::vertcat({casadi::SX::reshape(X, -1, 1), casadi::SX::reshape(U, -1, 1)});
}

void MPCCScholtesRelaxationSolver::buildVariableBounds(const ScholtesRelaxationConfig& config,
                                                       std::vector<double>& lbx,
                                                       std::vector<double>& ubx) {
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int horizon = config.horizon;

    std::vector<double> x_lb = problem_->getStateLowerBounds();
    std::vector<double> x_ub = problem_->getStateUpperBounds();
    std::vector<double> u_lb = problem_->getControlLowerBounds();
    std::vector<double> u_ub = problem_->getControlUpperBounds();

    lbx.clear();
    ubx.clear();

    for (int i = 0; i <= horizon; ++i) {
        lbx.insert(lbx.end(), x_lb.begin(), x_lb.end());
        ubx.insert(ubx.end(), x_ub.begin(), x_ub.end());
    }

    for (int i = 0; i < horizon; ++i) {
        lbx.insert(lbx.end(), u_lb.begin(), u_lb.end());
        ubx.insert(ubx.end(), u_ub.begin(), u_ub.end());
    }
}

void MPCCScholtesRelaxationSolver::buildConstraintBounds(const ScholtesRelaxationConfig& config,
                                                         double t, std::vector<double>& lbg,
                                                         std::vector<double>& ubg) {
    int nx = problem_->getStateDim();
    int nc = problem_->getComplementarityDim();
    int neq = problem_->getEqualityConstraintDim();
    int nineq = problem_->getInequalityConstraintDim();
    int horizon = config.horizon;

    lbg.clear();
    ubg.clear();

    // Per stage constraints: G >= 0, H >= 0, G*H <= t, plus E=0, g<=0
    for (int i = 0; i < horizon; ++i) {
        // G >= 0 (nc constraints)
        for (int j = 0; j < nc; ++j) {
            lbg.push_back(0.0);
            ubg.push_back(casadi::inf);
        }

        // H >= 0 (nc constraints)
        for (int j = 0; j < nc; ++j) {
            lbg.push_back(0.0);
            ubg.push_back(casadi::inf);
        }

        // G*H <= t (nc constraints)
        for (int j = 0; j < nc; ++j) {
            lbg.push_back(-casadi::inf);
            ubg.push_back(t);
        }

        // E = 0 (neq constraints)
        for (int j = 0; j < neq; ++j) {
            lbg.push_back(0.0);
            ubg.push_back(0.0);
        }

        // g <= 0 (nineq constraints)
        for (int j = 0; j < nineq; ++j) {
            lbg.push_back(-casadi::inf);
            ubg.push_back(0.0);
        }
    }

    // Dynamics constraints (nx * horizon)
    for (int i = 0; i < nx * horizon; ++i) {
        lbg.push_back(0.0);
        ubg.push_back(0.0);
    }

    // Initial condition constraints (nx)
    for (int i = 0; i < nx; ++i) {
        lbg.push_back(0.0);
        ubg.push_back(0.0);
    }
}

double MPCCScholtesRelaxationSolver::computeMaxComplementarity(
    const casadi::DM& x_sol, const ScholtesRelaxationConfig& config) {
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int horizon = config.horizon;

    casadi::Function G_func = problem_->getGFunction();
    casadi::Function H_func = problem_->getHFunction();

    int nx_total = nx * (horizon + 1);
    int nu_total = nu * horizon;

    // Reshape solution
    casadi::DM X_sol = casadi::DM::reshape(x_sol(casadi::Slice(0, nx_total)), nx, horizon + 1);
    casadi::DM U_sol =
        casadi::DM::reshape(x_sol(casadi::Slice(nx_total, nx_total + nu_total)), nu, horizon);

    double max_comp_viol = 0.0;
    for (int i = 0; i < horizon; ++i) {
        casadi::DM x_i = X_sol(casadi::Slice(), i);
        casadi::DM u_i = U_sol(casadi::Slice(), i);

        std::vector<casadi::DM> func_in = {x_i, u_i};
        casadi::DM G_val = G_func(func_in)[0];
        casadi::DM H_val = H_func(func_in)[0];

        // Compute element-wise product G * H
        casadi::DM comp = G_val * H_val;

        // Get maximum absolute value
        for (casadi_int j = 0; j < comp.numel(); ++j) {
            double val = std::abs(static_cast<double>(comp.nz(j)));
            if (val > max_comp_viol) {
                max_comp_viol = val;
            }
        }
    }

    return max_comp_viol;
}

ScholtesRelaxationSolution MPCCScholtesRelaxationSolver::extractSolution(
    const std::map<std::string, casadi::DM>& sol_map, const ScholtesRelaxationConfig& config,
    int nx_total, int nu_total) {
    ScholtesRelaxationSolution solution;

    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int horizon = config.horizon;

    // Extract solution variables
    casadi::DM x_sol = sol_map.at("x");

    // Reshape to get state and control trajectories
    solution.state_trajectory =
        casadi::DM::reshape(x_sol(casadi::Slice(0, nx_total)), nx, horizon + 1);
    solution.control_trajectory =
        casadi::DM::reshape(x_sol(casadi::Slice(nx_total, nx_total + nu_total)), nu, horizon);

    // Extract objective value
    solution.objective_value = static_cast<double>(sol_map.at("f"));

    return solution;
}

ScholtesRelaxationSolution MPCCScholtesRelaxationSolver::solve(
    const ScholtesRelaxationConfig& config) {
    auto total_start_time = std::chrono::high_resolution_clock::now();

    ScholtesRelaxationSolution solution;
    solution.converged = false;
    solution.outer_iterations = 0;
    solution.total_ipopt_iterations = 0;

    int nx_total, nu_total, nc;
    casadi::SX opt_vars, obj, g;

    // Build NLP (symbolic, done once)
    buildNLP(config, opt_vars, obj, g, nx_total, nu_total, nc);

    // Build variable bounds (fixed across iterations)
    std::vector<double> lbx, ubx;
    buildVariableBounds(config, lbx, ubx);

    // Create NLP structure
    casadi::SXDict nlp = {{"x", opt_vars}, {"f", obj}, {"g", g}};

    // Create IPOPT solver (once, reuse for warm starting)
    casadi::Dict opts = createIPOPTOptions(config);
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // Print header
    if (config.print_level >= 1) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Scholtes Relaxation: t_init=" << config.t_init
                  << ", t_final=" << config.t_final << ", gamma=" << config.gamma << std::endl;
        std::cout << "Stopping criterion: max|G*H| < " << std::scientific << std::setprecision(2)
                  << config.comp_tol << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

    // Scholtes iteration: progressively tighten relaxation parameter t
    double t = config.t_init;
    casadi::DM x0_warm;      // Primal warm start
    casadi::DM lam_g0_warm;  // Dual warm start for constraints
    casadi::DM lam_x0_warm;  // Dual warm start for bounds
    bool has_warm_start = false;

    for (int outer_iter = 0; outer_iter < config.max_outer_iters; ++outer_iter) {
        solution.outer_iterations = outer_iter + 1;

        // Build constraint bounds for current t
        std::vector<double> lbg, ubg;
        buildConstraintBounds(config, t, lbg, ubg);

        // Prepare solver arguments
        std::map<std::string, casadi::DM> arg;
        arg["lbx"] = lbx;
        arg["ubx"] = ubx;
        arg["lbg"] = lbg;
        arg["ubg"] = ubg;

        if (has_warm_start && config.warm_start) {
            arg["x0"] = x0_warm;
            arg["lam_g0"] = lam_g0_warm;
            arg["lam_x0"] = lam_x0_warm;
        }

        // Solve NLP
        std::map<std::string, casadi::DM> sol = solver(arg);

        // Get solver statistics
        casadi::Dict stats = solver.stats();
        bool success = stats.count("success") ? static_cast<bool>(stats.at("success")) : true;
        int iter_count = stats.count("iter_count") ? static_cast<int>(stats.at("iter_count")) : 0;
        solution.total_ipopt_iterations += iter_count;

        // Compute max complementarity violation
        casadi::DM x_sol = sol.at("x");
        double max_comp_viol = computeMaxComplementarity(x_sol, config);

        // Print iteration info
        if (config.print_level >= 1) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Iter " << std::setw(2) << (outer_iter + 1) << ": t = " << std::scientific
                      << std::setprecision(2) << t
                      << ", success = " << (success ? "true " : "false")
                      << ", max(G*H) = " << std::scientific << std::setprecision(2) << max_comp_viol
                      << ", obj = " << std::fixed << std::setprecision(4)
                      << static_cast<double>(sol.at("f")) << ", iters = " << iter_count
                      << std::endl;
        }

        // Update warm start
        x0_warm = sol.at("x");
        lam_g0_warm = sol.at("lam_g");
        lam_x0_warm = sol.at("lam_x");
        has_warm_start = true;

        // Update solution (preserve total_ipopt_iterations before extractSolution overwrites)
        int saved_total_iters = solution.total_ipopt_iterations;
        solution = extractSolution(sol, config, nx_total, nu_total);
        solution.total_ipopt_iterations = saved_total_iters;
        solution.max_complementarity = max_comp_viol;
        solution.final_relaxation_t = t;
        solution.outer_iterations = outer_iter + 1;

        // Check convergence: complementarity tolerance satisfied
        if (max_comp_viol < config.comp_tol) {
            solution.converged = true;
            solution.status_message = "Converged: max|G*H| < comp_tol";
            if (config.print_level >= 1) {
                std::cout << "\nScholtes converged: max|G*H| = " << std::scientific
                          << std::setprecision(2) << max_comp_viol
                          << " < comp_tol = " << config.comp_tol << std::endl;
            }
            break;
        }

        // Check if relaxation parameter reached minimum
        if (t <= config.t_final) {
            solution.status_message = "Reached t_final without full convergence";
            if (config.print_level >= 1) {
                std::cout << "\nScholtes reached t_final: t = " << std::scientific
                          << std::setprecision(2) << t << " <= t_final = " << config.t_final
                          << std::endl;
                std::cout << "  (max|G*H| = " << max_comp_viol << ", comp_tol = " << config.comp_tol
                          << ")" << std::endl;
            }
            break;
        }

        // Decrease relaxation parameter
        t = std::max(t * config.gamma, config.t_final);
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    solution.solve_time = std::chrono::duration<double>(total_end_time - total_start_time).count();

    if (config.print_level >= 1) {
        std::cout << "\nTotal solve time: " << std::fixed << std::setprecision(3)
                  << solution.solve_time << " s" << std::endl;
        std::cout << "Total IPOPT iterations: " << solution.total_ipopt_iterations << std::endl;
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }

    return solution;
}

ScholtesRelaxationSolution MPCCScholtesRelaxationSolver::solveWithInitialGuess(
    const ScholtesRelaxationConfig& config, const casadi::DM& x_init, const casadi::DM& u_init) {
    auto total_start_time = std::chrono::high_resolution_clock::now();

    ScholtesRelaxationSolution solution;
    solution.converged = false;
    solution.outer_iterations = 0;
    solution.total_ipopt_iterations = 0;

    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int horizon = config.horizon;
    int nx_total, nu_total, nc;
    casadi::SX opt_vars, obj, g;

    // Build NLP (symbolic, done once)
    buildNLP(config, opt_vars, obj, g, nx_total, nu_total, nc);

    // Build variable bounds (fixed across iterations)
    std::vector<double> lbx, ubx;
    buildVariableBounds(config, lbx, ubx);

    // Create NLP structure
    casadi::SXDict nlp = {{"x", opt_vars}, {"f", obj}, {"g", g}};

    // Create IPOPT solver
    casadi::Dict opts = createIPOPTOptions(config);
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // Print header
    if (config.print_level >= 1) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Scholtes Relaxation (with initial guess): t_init=" << config.t_init
                  << ", t_final=" << config.t_final << ", gamma=" << config.gamma << std::endl;
        std::cout << "Stopping criterion: max|G*H| < " << std::scientific << std::setprecision(2)
                  << config.comp_tol << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

    // Build initial guess from provided trajectories
    casadi::DM x0_init = casadi::DM::vertcat(
        {casadi::DM::reshape(x_init, -1, 1), casadi::DM::reshape(u_init, -1, 1)});

    // Scholtes iteration
    double t = config.t_init;
    casadi::DM x0_warm = x0_init;
    casadi::DM lam_g0_warm;
    casadi::DM lam_x0_warm;
    bool has_full_warm_start = false;

    for (int outer_iter = 0; outer_iter < config.max_outer_iters; ++outer_iter) {
        solution.outer_iterations = outer_iter + 1;

        // Build constraint bounds for current t
        std::vector<double> lbg, ubg;
        buildConstraintBounds(config, t, lbg, ubg);

        // Prepare solver arguments
        std::map<std::string, casadi::DM> arg;
        arg["lbx"] = lbx;
        arg["ubx"] = ubx;
        arg["lbg"] = lbg;
        arg["ubg"] = ubg;
        arg["x0"] = x0_warm;

        if (has_full_warm_start && config.warm_start) {
            arg["lam_g0"] = lam_g0_warm;
            arg["lam_x0"] = lam_x0_warm;
        }

        // Solve NLP
        std::map<std::string, casadi::DM> sol = solver(arg);

        // Get solver statistics
        casadi::Dict stats = solver.stats();
        bool success = stats.count("success") ? static_cast<bool>(stats.at("success")) : true;
        int iter_count = stats.count("iter_count") ? static_cast<int>(stats.at("iter_count")) : 0;
        solution.total_ipopt_iterations += iter_count;

        // Compute max complementarity violation
        casadi::DM x_sol = sol.at("x");
        double max_comp_viol = computeMaxComplementarity(x_sol, config);

        // Print iteration info
        if (config.print_level >= 1) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Iter " << std::setw(2) << (outer_iter + 1) << ": t = " << std::scientific
                      << std::setprecision(2) << t
                      << ", success = " << (success ? "true " : "false")
                      << ", max(G*H) = " << std::scientific << std::setprecision(2) << max_comp_viol
                      << ", obj = " << std::fixed << std::setprecision(4)
                      << static_cast<double>(sol.at("f")) << ", iters = " << iter_count
                      << std::endl;
        }

        // Update warm start
        x0_warm = sol.at("x");
        lam_g0_warm = sol.at("lam_g");
        lam_x0_warm = sol.at("lam_x");
        has_full_warm_start = true;

        // Update solution (preserve total_ipopt_iterations before extractSolution overwrites)
        int saved_total_iters = solution.total_ipopt_iterations;
        solution = extractSolution(sol, config, nx_total, nu_total);
        solution.total_ipopt_iterations = saved_total_iters;
        solution.max_complementarity = max_comp_viol;
        solution.final_relaxation_t = t;
        solution.outer_iterations = outer_iter + 1;

        // Check convergence
        if (max_comp_viol < config.comp_tol) {
            solution.converged = true;
            solution.status_message = "Converged: max|G*H| < comp_tol";
            if (config.print_level >= 1) {
                std::cout << "\nScholtes converged: max|G*H| = " << std::scientific
                          << std::setprecision(2) << max_comp_viol
                          << " < comp_tol = " << config.comp_tol << std::endl;
            }
            break;
        }

        // Check if relaxation parameter reached minimum
        if (t <= config.t_final) {
            solution.status_message = "Reached t_final without full convergence";
            if (config.print_level >= 1) {
                std::cout << "\nScholtes reached t_final: t = " << std::scientific
                          << std::setprecision(2) << t << " <= t_final = " << config.t_final
                          << std::endl;
            }
            break;
        }

        // Decrease relaxation parameter
        t = std::max(t * config.gamma, config.t_final);
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    solution.solve_time = std::chrono::duration<double>(total_end_time - total_start_time).count();

    if (config.print_level >= 1) {
        std::cout << "\nTotal solve time: " << std::fixed << std::setprecision(3)
                  << solution.solve_time << " s" << std::endl;
        std::cout << "Total IPOPT iterations: " << solution.total_ipopt_iterations << std::endl;
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }

    return solution;
}

}  // namespace aula
