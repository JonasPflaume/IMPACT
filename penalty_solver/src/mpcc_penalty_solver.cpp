#include "penalty_solver/mpcc_penalty_solver.h"

#include <chrono>
#include <iostream>

namespace aula {

MPCCPenaltySolver::MPCCPenaltySolver(std::shared_ptr<MPCCProblem> problem) : problem_(problem) {}

casadi::Dict MPCCPenaltySolver::createIPOPTOptions(const MPCCPenaltyConfig& config) {
    casadi::Dict opts;
    opts["ipopt.print_level"] = config.ipopt_print_level;
    opts["ipopt.max_iter"] = config.ipopt_max_iter;
    opts["ipopt.tol"] = config.ipopt_tol;
    opts["ipopt.acceptable_tol"] = config.ipopt_acceptable_tol;
    opts["print_time"] = config.print_time;
    opts["ipopt.sb"] = "yes";  // Suppress banner
    return opts;
}

void MPCCPenaltySolver::buildNLP(const MPCCPenaltyConfig& config, casadi::SX& opt_vars,
                                 std::vector<double>& lbx, std::vector<double>& ubx,
                                 std::vector<double>& lbg, std::vector<double>& ubg, int& nx_total,
                                 int& nu_total) {
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int nc = problem_->getComplementarityDim();
    int horizon = config.horizon;

    // Bounds on decision variables
    nx_total = nx * (horizon + 1);
    nu_total = nu * horizon;

    // State bounds
    std::vector<double> x_lb = problem_->getStateLowerBounds();
    std::vector<double> x_ub = problem_->getStateUpperBounds();

    lbx.reserve(nx_total + nu_total);
    ubx.reserve(nx_total + nu_total);

    for (int i = 0; i <= horizon; ++i) {
        lbx.insert(lbx.end(), x_lb.begin(), x_lb.end());
        ubx.insert(ubx.end(), x_ub.begin(), x_ub.end());
    }

    // Control bounds
    std::vector<double> u_lb = problem_->getControlLowerBounds();
    std::vector<double> u_ub = problem_->getControlUpperBounds();

    for (int i = 0; i < horizon; ++i) {
        lbx.insert(lbx.end(), u_lb.begin(), u_lb.end());
        ubx.insert(ubx.end(), u_ub.begin(), u_ub.end());
    }

    // Bounds on constraints
    // Inequality constraints (G >= 0, H >= 0): nc * 2 * horizon constraints
    int num_ineq = nc * 2 * horizon;
    lbg.insert(lbg.end(), num_ineq, 0.0);
    ubg.insert(ubg.end(), num_ineq, casadi::inf);

    // Equality constraints (dynamics): nx * horizon constraints
    int num_dyn = nx * horizon;
    lbg.insert(lbg.end(), num_dyn, 0.0);
    ubg.insert(ubg.end(), num_dyn, 0.0);

    // Equality constraints (initial condition): nx constraints
    lbg.insert(lbg.end(), nx, 0.0);
    ubg.insert(ubg.end(), nx, 0.0);
}

MPCCSolution MPCCPenaltySolver::extractSolution(const std::map<std::string, casadi::DM>& sol_map,
                                                const MPCCPenaltyConfig& config, int nx_total,
                                                int nu_total) {
    MPCCSolution solution;

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

    // Solver status is filled from solver.stats() after the solve.
    solution.success = false;
    solution.solver_stats.clear();

    // Initialize iterations (will be set by solver stats if available)
    solution.iterations = 0;

    return solution;
}
MPCCSolution MPCCPenaltySolver::solve(const MPCCPenaltyConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get problem dimensions
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int nc = problem_->getComplementarityDim();
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

    // Objective function
    casadi::SX obj = 0;

    // Goal state for stage tracking
    casadi::SX x_goal_sx = casadi::SX(config.x_goal);

    // Constraint vector and bounds
    std::vector<casadi::SX> g_vec;
    std::vector<double> lbg_vec, ubg_vec;

    // Stage cost and complementarity penalty
    for (int i = 0; i < horizon; ++i) {
        casadi::SX x_curr = X(casadi::Slice(), i);
        casadi::SX u_curr = U(casadi::Slice(), i);

        // Get G and H values
        std::vector<casadi::SX> G_in = {x_curr, u_curr};
        casadi::SX G_value = G_func(G_in)[0];
        casadi::SX H_value = H_func(G_in)[0];

        // Stage cost (control effort)
        obj += config.stage_cost_weight * casadi::SX::sumsqr(u_curr);

        // Stage state tracking cost (toward goal)
        if (config.stage_state_cost_weight > 0) {
            obj += config.stage_state_cost_weight * casadi::SX::sumsqr(x_curr - x_goal_sx);
        }

        // Complementarity penalty
        obj += config.complementarity_penalty_weight * casadi::SX::sumsqr(G_value * H_value);

        // Inequality constraints: G >= 0
        g_vec.push_back(G_value);
        for (int j = 0; j < nc; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(casadi::inf);
        }

        // Inequality constraints: H >= 0
        g_vec.push_back(H_value);
        for (int j = 0; j < nc; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(casadi::inf);
        }

        // Additional equality constraints: E = 0
        if (neq > 0 && !E_func.is_null()) {
            casadi::SX E_value = E_func(G_in)[0];
            g_vec.push_back(E_value);
            for (int j = 0; j < neq; ++j) {
                lbg_vec.push_back(0.0);
                ubg_vec.push_back(0.0);
            }
        }

        // Additional inequality constraints: g <= 0
        if (nineq > 0 && !I_func.is_null()) {
            casadi::SX g_value = I_func(G_in)[0];
            g_vec.push_back(g_value);
            for (int j = 0; j < nineq; ++j) {
                lbg_vec.push_back(-casadi::inf);
                ubg_vec.push_back(0.0);
            }
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
        for (int j = 0; j < nx; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(0.0);
        }
    }

    // Initial condition constraint
    casadi::SX x_0_sx = casadi::SX(config.x_0);
    g_vec.push_back(X(casadi::Slice(), 0) - x_0_sx);
    for (int j = 0; j < nx; ++j) {
        lbg_vec.push_back(0.0);
        ubg_vec.push_back(0.0);
    }

    // Concatenate constraints
    casadi::SX g = casadi::SX::vertcat(g_vec);

    // Variable bounds
    std::vector<double> lbx, ubx;
    std::vector<double> x_lb = problem_->getStateLowerBounds();
    std::vector<double> x_ub = problem_->getStateUpperBounds();
    std::vector<double> u_lb = problem_->getControlLowerBounds();
    std::vector<double> u_ub = problem_->getControlUpperBounds();

    for (int i = 0; i <= horizon; ++i) {
        lbx.insert(lbx.end(), x_lb.begin(), x_lb.end());
        ubx.insert(ubx.end(), x_ub.begin(), x_ub.end());
    }
    for (int i = 0; i < horizon; ++i) {
        lbx.insert(lbx.end(), u_lb.begin(), u_lb.end());
        ubx.insert(ubx.end(), u_ub.begin(), u_ub.end());
    }

    int nx_total = nx * (horizon + 1);
    int nu_total = nu * horizon;

    // Flatten decision variables
    casadi::SX opt_vars_flat =
        casadi::SX::vertcat({casadi::SX::reshape(X, -1, 1), casadi::SX::reshape(U, -1, 1)});

    // Create NLP
    casadi::SXDict nlp = {{"x", opt_vars_flat}, {"f", obj}, {"g", g}};

    // Create IPOPT solver
    casadi::Dict opts = createIPOPTOptions(config);
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // Solve
    std::map<std::string, casadi::DM> arg;
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg_vec;
    arg["ubg"] = ubg_vec;

    std::map<std::string, casadi::DM> sol = solver(arg);

    // Extract solution
    MPCCSolution solution = extractSolution(sol, config, nx_total, nu_total);

    // Get IPOPT iterations from solver stats
    casadi::Dict stats = solver.stats();
    solution.success =
        stats.find("success") != stats.end() && static_cast<bool>(stats.at("success"));
    if (stats.find("return_status") != stats.end()) {
        solution.solver_stats = stats.at("return_status").to_string();
    }
    if (stats.find("iter_count") != stats.end()) {
        solution.iterations = static_cast<int>(stats.at("iter_count"));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    solution.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    if (config.print_time) {
        std::cout << "Solution time: " << solution.solve_time << " seconds" << std::endl;
    }

    return solution;
}

MPCCSolution MPCCPenaltySolver::solveWithInitialGuess(const MPCCPenaltyConfig& config,
                                                      const casadi::DM& x_init,
                                                      const casadi::DM& u_init) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get problem dimensions
    int nx = problem_->getStateDim();
    int nu = problem_->getControlDim();
    int nc = problem_->getComplementarityDim();
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

    // Objective function
    casadi::SX obj = 0;

    // Goal state for stage tracking
    casadi::SX x_goal_sx = casadi::SX(config.x_goal);

    // Constraint vector and bounds
    std::vector<casadi::SX> g_vec;
    std::vector<double> lbg_vec, ubg_vec;

    // Stage cost and complementarity penalty
    for (int i = 0; i < horizon; ++i) {
        casadi::SX x_curr = X(casadi::Slice(), i);
        casadi::SX u_curr = U(casadi::Slice(), i);

        // Get G and H values
        std::vector<casadi::SX> G_in = {x_curr, u_curr};
        casadi::SX G_value = G_func(G_in)[0];
        casadi::SX H_value = H_func(G_in)[0];

        // Stage cost (control effort)
        obj += config.stage_cost_weight * casadi::SX::sumsqr(u_curr);

        // Stage state tracking cost (toward goal)
        if (config.stage_state_cost_weight > 0) {
            obj += config.stage_state_cost_weight * casadi::SX::sumsqr(x_curr - x_goal_sx);
        }

        // Complementarity penalty
        obj += config.complementarity_penalty_weight * casadi::SX::sumsqr(G_value * H_value);

        // Inequality constraints: G >= 0
        g_vec.push_back(G_value);
        for (int j = 0; j < nc; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(casadi::inf);
        }

        // Inequality constraints: H >= 0
        g_vec.push_back(H_value);
        for (int j = 0; j < nc; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(casadi::inf);
        }

        // Additional equality constraints: E = 0
        if (neq > 0 && !E_func.is_null()) {
            casadi::SX E_value = E_func(G_in)[0];
            g_vec.push_back(E_value);
            for (int j = 0; j < neq; ++j) {
                lbg_vec.push_back(0.0);
                ubg_vec.push_back(0.0);
            }
        }

        // Additional inequality constraints: g <= 0
        if (nineq > 0 && !I_func.is_null()) {
            casadi::SX g_value = I_func(G_in)[0];
            g_vec.push_back(g_value);
            for (int j = 0; j < nineq; ++j) {
                lbg_vec.push_back(-casadi::inf);
                ubg_vec.push_back(0.0);
            }
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
        for (int j = 0; j < nx; ++j) {
            lbg_vec.push_back(0.0);
            ubg_vec.push_back(0.0);
        }
    }

    // Initial condition constraint
    casadi::SX x_0_sx = casadi::SX(config.x_0);
    g_vec.push_back(X(casadi::Slice(), 0) - x_0_sx);
    for (int j = 0; j < nx; ++j) {
        lbg_vec.push_back(0.0);
        ubg_vec.push_back(0.0);
    }

    // Concatenate constraints
    casadi::SX g = casadi::SX::vertcat(g_vec);

    // Variable bounds
    std::vector<double> lbx, ubx;
    std::vector<double> x_lb = problem_->getStateLowerBounds();
    std::vector<double> x_ub = problem_->getStateUpperBounds();
    std::vector<double> u_lb = problem_->getControlLowerBounds();
    std::vector<double> u_ub = problem_->getControlUpperBounds();

    for (int i = 0; i <= horizon; ++i) {
        lbx.insert(lbx.end(), x_lb.begin(), x_lb.end());
        ubx.insert(ubx.end(), x_ub.begin(), x_ub.end());
    }
    for (int i = 0; i < horizon; ++i) {
        lbx.insert(lbx.end(), u_lb.begin(), u_lb.end());
        ubx.insert(ubx.end(), u_ub.begin(), u_ub.end());
    }

    int nx_total = nx * (horizon + 1);
    int nu_total = nu * horizon;

    // Flatten decision variables
    casadi::SX opt_vars_flat =
        casadi::SX::vertcat({casadi::SX::reshape(X, -1, 1), casadi::SX::reshape(U, -1, 1)});

    // Create NLP
    casadi::SXDict nlp = {{"x", opt_vars_flat}, {"f", obj}, {"g", g}};

    // Create IPOPT solver
    casadi::Dict opts = createIPOPTOptions(config);
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // Prepare initial guess
    casadi::DM x0 = casadi::DM::vertcat(
        {casadi::DM::reshape(x_init, -1, 1), casadi::DM::reshape(u_init, -1, 1)});

    // Solve
    std::map<std::string, casadi::DM> arg;
    arg["x0"] = x0;
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg_vec;
    arg["ubg"] = ubg_vec;

    std::map<std::string, casadi::DM> sol = solver(arg);

    // Extract solution
    MPCCSolution solution = extractSolution(sol, config, nx_total, nu_total);

    // Get IPOPT iterations from solver stats
    casadi::Dict stats = solver.stats();
    solution.success =
        stats.find("success") != stats.end() && static_cast<bool>(stats.at("success"));
    if (stats.find("return_status") != stats.end()) {
        solution.solver_stats = stats.at("return_status").to_string();
    }
    if (stats.find("iter_count") != stats.end()) {
        solution.iterations = static_cast<int>(stats.at("iter_count"));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    solution.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    if (config.print_time) {
        std::cout << "Solution time: " << solution.solve_time << " seconds" << std::endl;
    }

    return solution;
}

}  // namespace aula
