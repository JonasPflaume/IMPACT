#include "impact/multiple_shooting.h"

#include <algorithm>
#include <cmath>

#include "impact/mpcc_stage.h"
#include "impact/mpcc_subproblem.h"

namespace impact {

using casadi::Slice;
using casadi::SX;

// Multiple shooting keeps the state trajectory free. Dynamics become defect
// constraints, the initial condition is a separate equality block, and the whole
// stacked problem is handed to buildMPCC().
MultipleShootingLayout buildMultipleShooting(const StageProblem& stage,
                                             const BCDAULAConfig& config) {
    const int nx = stage.stateDim();
    const int nu = stage.controlDim();
    const int nc = stage.compDim();
    const int neq_s = stage.eqDim();
    const int nineq_s = stage.ineqDim();
    const int ndynres_s = stage.dynamicsResidualDim();
    const int np = stage.runtimeParamDim();
    const int horizon = config.horizon;

    const int n_opt = nx * (horizon + 1) + nu * horizon;

    const double fix_point_scale = config.fix_point_scale;
    const double dynamics_scale = config.dynamics_scale;
    const double eq_scale = config.eq_scale;
    const double ineq_scale = config.ineq_scale;
    const double comp_scale = config.comp_scale;

    // Symbolic decision variables: z = [vec(X); vec(U)].
    SX X = SX::sym("X", nx, horizon + 1);
    SX U = SX::sym("U", nu, horizon);
    SX z = SX::vertcat({SX::reshape(X, -1, 1), SX::reshape(U, -1, 1)});
    std::vector<SX> Xlist(horizon + 1), Ulist(horizon);
    for (int t = 0; t <= horizon; ++t) Xlist[t] = X(Slice(), t);
    for (int t = 0; t < horizon; ++t) Ulist[t] = U(Slice(), t);

    // Runtime parameter (x_goal) and fixed initial condition x_0.
    SX p_rt = SX::sym("p_rt", std::max(np, 1));
    SX x_0 = SX::sym("x_0", nx);

    // Stack stage quantities over the horizon.
    std::vector<SX> G_list, H_list, h_list, dynres_list, eq_list, ineq_list;
    for (int t = 0; t < horizon; ++t) {
        const SX& xt = Xlist[t];
        const SX& ut = Ulist[t];
        G_list.push_back(stage.G(xt, ut, p_rt));
        H_list.push_back(stage.H(xt, ut, p_rt));
        h_list.push_back(Xlist[t + 1] - stage.step(xt, ut, p_rt));  // dynamics defect
        if (ndynres_s > 0) dynres_list.push_back(stage.dynamicsResidual(xt, ut, p_rt));
        if (neq_s > 0) eq_list.push_back(stage.eq(xt, ut, p_rt));
        if (nineq_s > 0) ineq_list.push_back(stage.ineq(xt, ut, p_rt));
    }

    // Assemble the generic MPCC in multiple-shooting block order.
    MPCCDescription desc;
    desc.z = z;
    desc.p = SX::vertcat({p_rt, x_0});  // x_goal then the initial condition x_0
    desc.cost = stage.costResidual(Xlist, Ulist, p_rt);
    desc.cost_is_linear = stage.costIsLinear();
    desc.check_stationarity = config.check_stationarity;
    desc.conditioned_complementarity = config.conditioned_complementarity;
    desc.stationarity_tol = config.stationarity_tol;
    desc.max_stagnation_restarts = config.max_stagnation_restarts;

    if (nc > 0)
        desc.addComplementarityBlock(
            "comp", SX::vertcat(G_list), SX::vertcat(H_list),
            {comp_scale, config.rho_comp_init, config.outer_tol_comp});
    desc.addEqualityBlock("dynamics", SX::vertcat(h_list),
                          {dynamics_scale, config.rho_dynamics_init, config.outer_tol_h});
    if (ndynres_s > 0)
        desc.addEqualityBlock("physics", SX::vertcat(dynres_list),
                              {dynamics_scale, config.rho_dynamics_init, config.outer_tol_h});
    if (neq_s > 0)
        desc.addEqualityBlock("equality", SX::vertcat(eq_list),
                              {eq_scale, config.rho_eq_init, config.outer_tol_h});
    if (nineq_s > 0)
        desc.addInequalityBlock("inequality", SX::vertcat(ineq_list),
                                {ineq_scale, config.rho_ineq_init, config.outer_tol_g});
    desc.addEqualityBlock("fix_point", Xlist[0] - x_0,
                          {fix_point_scale, config.rho_fix_point_init, config.outer_tol_h});

    desc.jit = config.jit;
    MPCCSubproblem m = buildMPCC(desc);
    MultipleShootingLayout out;
    out.sub = std::move(m.sub);
    out.off_p = m.off_p;                     // x_goal at the start of the combined parameter
    out.off_x0 = m.off_p + std::max(np, 1);  // x_0 follows x_goal
    (void)n_opt;
    return out;
}

// Front-end wrapper.

MultipleShootingSolution MultipleShootingSolver::solve(const BCDAULAConfig& config) {
    const int nx = problem_->getStateDim();
    const int nu = problem_->getControlDim();
    Eigen::MatrixXd x_init(nx, config.horizon + 1);
    Eigen::MatrixXd u_init = Eigen::MatrixXd::Zero(nu, config.horizon);
    for (int k = 0; k <= config.horizon; ++k) {
        if (config.use_constant_state_init) {
            x_init.col(k) = config.x_0;
        } else {
            const double a = static_cast<double>(k) / config.horizon;
            x_init.col(k) = (1.0 - a) * config.x_0 + a * config.x_goal;
        }
    }
    return solveWithInitialGuess(config, x_init, u_init);
}

MultipleShootingSolution MultipleShootingSolver::solveWithInitialGuess(
    const BCDAULAConfig& config, const Eigen::MatrixXd& x_init, const Eigen::MatrixXd& u_init) {
    const int nx = problem_->getStateDim();
    const int nu = problem_->getControlDim();
    const int horizon = config.horizon;
    const int nx_total = nx * (horizon + 1);

    MPCCStage stage(problem_, config);
    MultipleShootingLayout layout = buildMultipleShooting(stage, config);
    AulaSubproblem& sub = *layout.sub;

    // Fill the runtime parameter (x_goal) and the initial condition x_0.
    sub.setParamValue(layout.off_p, config.x_goal);
    sub.setParamValue(layout.off_x0, config.x_0);

    // Flatten initial guess z = [vec(X); vec(U)].
    Eigen::VectorXd z(sub.numOpt());
    for (int k = 0; k <= horizon; ++k) z.segment(k * nx, nx) = x_init.col(k);
    for (int k = 0; k < horizon; ++k) z.segment(nx_total + k * nu, nu) = u_init.col(k);

    BCDAULASolver solver;
    BCDAULAResult r = solver.solve(sub, config, z);

    MultipleShootingSolution sol;
    sol.state_trajectory.resize(nx, horizon + 1);
    sol.control_trajectory.resize(nu, horizon);
    for (int k = 0; k <= horizon; ++k) sol.state_trajectory.col(k) = r.z.segment(k * nx, nx);
    for (int k = 0; k < horizon; ++k)
        sol.control_trajectory.col(k) = r.z.segment(nx_total + k * nu, nu);

    sol.objective_value = r.objective_value;
    sol.dynamics_violation = r.dynamics_violation;
    sol.equality_violation = r.equality_violation;
    sol.inequality_violation = r.inequality_violation;
    sol.complementarity_violation = r.complementarity_violation;
    sol.converged = r.converged;
    sol.outer_iterations = r.outer_iterations;
    sol.total_inner_iterations = r.total_inner_iterations;
    sol.solve_time = r.solve_time;
    sol.status_message = r.status_message;
    return sol;
}

}  // namespace impact
