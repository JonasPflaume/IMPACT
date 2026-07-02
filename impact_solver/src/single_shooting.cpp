#include "impact/single_shooting.h"

#include <algorithm>
#include <cmath>

#include "impact/mpcc_subproblem.h"

namespace impact {

using casadi::Slice;
using casadi::SX;

// Single shooting optimizes only the controls. The state is rolled out from x_0
// with step(), then cost and constraints are stacked over the horizon and passed
// to buildMPCC().
SingleShootingLayout buildSingleShooting(const StageProblem& stage, const BCDAULAConfig& config) {
    const int horizon = config.horizon;
    const int nx = stage.stateDim();
    const int nu = stage.controlDim();
    const int nc = stage.compDim();
    const int ndynres_s = stage.dynamicsResidualDim();
    const int nineq_s = stage.ineqDim();
    const int neq_s = stage.eqDim();
    const int np = std::max(stage.runtimeParamDim(), 1);

    const int n_opt = nu * horizon;
    const int n_comp = nc * horizon;

    // Symbolic decision variables z = [vec(U)]; state is rolled out from x_0.
    SX z = SX::sym("z", n_opt);
    std::vector<SX> Ulist(horizon);
    for (int k = 0; k < horizon; ++k) Ulist[k] = z(Slice(k * nu, (k + 1) * nu));

    SX x_0 = SX::sym("x_0", nx);
    SX p_rt = SX::sym("p_rt", np);

    std::vector<SX> Xlist(horizon + 1);
    Xlist[0] = x_0;
    for (int k = 0; k < horizon; ++k) Xlist[k + 1] = stage.step(Xlist[k], Ulist[k], p_rt);

    // Stage quantities stacked over the horizon; each uses the current rolled state.
    std::vector<SX> G_list, H_list, dynres_list, ineq_list, eq_list;
    for (int k = 0; k < horizon; ++k) {
        G_list.push_back(stage.G(Xlist[k], Ulist[k], p_rt));
        H_list.push_back(stage.H(Xlist[k], Ulist[k], p_rt));
        if (ndynres_s > 0) dynres_list.push_back(stage.dynamicsResidual(Xlist[k], Ulist[k], p_rt));
        if (nineq_s > 0) ineq_list.push_back(stage.ineq(Xlist[k], Ulist[k], p_rt));
        if (neq_s > 0) eq_list.push_back(stage.eq(Xlist[k], Ulist[k], p_rt));
    }

    // Assemble the generic MPCC in single-shooting block order.
    MPCCDescription desc;
    desc.z = z;
    desc.p = SX::vertcat({p_rt, x_0});  // p_rt (targets/phi/J) then the rollout start x_0
    desc.cost = stage.costResidual(Xlist, Ulist, p_rt);
    desc.cost_is_linear = false;  // rolled-out state makes the cost nonlinear

    if (ndynres_s > 0)
        desc.addEqualityBlock("dynamics", SX::vertcat(dynres_list),
                              {config.dynamics_scale, config.rho_dynamics_init,
                               config.outer_tol_h});
    if (nc > 0)
        desc.addComplementarityBlock(
            "comp", SX::vertcat(G_list), SX::vertcat(H_list),
            {config.comp_scale, config.rho_comp_init, config.outer_tol_comp});
    if (nineq_s > 0)
        desc.addInequalityBlock("inequality", SX::vertcat(ineq_list),
                                {config.ineq_scale, config.rho_ineq_init, config.outer_tol_g});
    if (neq_s > 0)
        desc.addEqualityBlock("equality", SX::vertcat(eq_list),
                              {config.eq_scale, config.rho_eq_init, config.outer_tol_h});

    desc.jit = config.jit;
    MPCCSubproblem m = buildMPCC(desc);
    SingleShootingLayout out;
    out.sub = std::move(m.sub);
    out.off_p = m.off_p;        // p_rt at the start of the combined runtime parameter
    out.off_x0 = m.off_p + np;  // x_0 (q0) follows p_rt
    (void)n_comp;
    return out;
}

// Front-end wrapper.

void SingleShootingSolver::ensureBuilt(const BCDAULAConfig& config) {
    if (horizon_built_ == config.horizon) return;
    n_qpos_ = problem_->getConfigDim();
    n_qvel_ = problem_->getVelocityDim();
    n_cmd_ = problem_->getCommandDim();
    n_lam_ = problem_->getMaxContacts() * 4;
    h_ = problem_->getTimeStep();
    vars_per_step_ = n_cmd_ + n_lam_ + n_qvel_;
    stage_ = std::make_unique<LCPStage>(problem_, config);
    layout_ = buildSingleShooting(*stage_, config);
    horizon_built_ = config.horizon;
}

SingleShootingSolution SingleShootingSolver::solve(const BCDAULAConfig& config,
                                                   const Eigen::VectorXd& q0,
                                                   const Eigen::VectorXd& phi_vec,
                                                   const Eigen::MatrixXd& jac_mat,
                                                   const Eigen::Vector3d& target_p,
                                                   const Eigen::Vector4d& target_q) {
    const int H = config.horizon;
    Eigen::MatrixXd zero_cmd = Eigen::MatrixXd::Zero(problem_->getCommandDim(), H);
    Eigen::MatrixXd zero_lam = Eigen::MatrixXd::Zero(problem_->getMaxContacts() * 4, H);
    Eigen::MatrixXd zero_vel = Eigen::MatrixXd::Zero(problem_->getVelocityDim(), H);
    return solveWithInitialGuess(config, q0, phi_vec, jac_mat, target_p, target_q, zero_cmd,
                                 zero_lam, zero_vel);
}

SingleShootingSolution SingleShootingSolver::solveWithInitialGuess(
    const BCDAULAConfig& config, const Eigen::VectorXd& q0, const Eigen::VectorXd& phi_vec,
    const Eigen::MatrixXd& jac_mat, const Eigen::Vector3d& target_p,
    const Eigen::Vector4d& target_q, const Eigen::MatrixXd& cmd_init,
    const Eigen::MatrixXd& lam_init, const Eigen::MatrixXd& vel_init) {
    ensureBuilt(config);
    AulaSubproblem& sub = *layout_.sub;
    const int H = config.horizon;

    // Runtime parameter p = [target_p, target_q, phi, vec(J)] and initial q0.
    Eigen::VectorXd p(3 + 4 + n_lam_ + n_lam_ * n_qvel_);
    p.segment(0, 3) = target_p;
    p.segment(3, 4) = target_q;
    p.segment(7, n_lam_) = phi_vec;
    p.segment(7 + n_lam_, n_lam_ * n_qvel_) =
        Eigen::Map<const Eigen::VectorXd>(jac_mat.data(), jac_mat.size());
    sub.setParamValue(layout_.off_p, p);
    sub.setParamValue(layout_.off_x0, q0);

    // Reset AuLa state for this solve.
    // Penalties reset to each block's build-time seed, so every block kind gets its
    // own configured rho back (not just the ones this front-end knows by name).
    for (DualBlock& b : sub.dualBlocks()) {
        b.kappa.setZero();
        b.rho = b.rho_init;
    }
    for (CompBlock& comp : sub.compBlocks()) {
        comp.kappaG.setZero();
        comp.kappaH.setZero();
        comp.sG.setZero();
        comp.sH.setZero();
        comp.rho = comp.rho_init;
    }
    sub.syncParams();

    // Warm start z = [cmd_0, lam_0, vel_0, ...].
    Eigen::VectorXd z(sub.numOpt());
    for (int k = 0; k < H; ++k) {
        int o = k * vars_per_step_;
        z.segment(o, n_cmd_) = cmd_init.col(k);
        z.segment(o + n_cmd_, n_lam_) = lam_init.col(k);
        z.segment(o + n_cmd_ + n_lam_, n_qvel_) = vel_init.col(k);
    }

    BCDAULAResult r = solver_.solve(sub, config, z);

    // Extract trajectories from the raw decision vector.
    SingleShootingSolution sol;
    sol.command_trajectory.resize(n_cmd_, H);
    sol.lambda_trajectory.resize(n_lam_, H);
    sol.velocity_trajectory.resize(n_qvel_, H);
    for (int k = 0; k < H; ++k) {
        int o = k * vars_per_step_;
        sol.command_trajectory.col(k) = r.z.segment(o, n_cmd_);
        sol.lambda_trajectory.col(k) = r.z.segment(o + n_cmd_, n_lam_);
        sol.velocity_trajectory.col(k) = r.z.segment(o + n_cmd_ + n_lam_, n_qvel_);
    }
    sol.config_trajectory.resize(n_qpos_, H + 1);
    sol.config_trajectory.col(0) = q0;
    for (int k = 0; k < H; ++k)
        sol.config_trajectory.col(k + 1) = problem_->integrateState(
            sol.config_trajectory.col(k), sol.velocity_trajectory.col(k), h_);
    sol.first_command = sol.command_trajectory.col(0);

    sol.objective_value = r.objective_value;
    sol.dynamics_violation = r.dynamics_violation;
    sol.complementarity_violation = r.complementarity_violation;
    sol.converged = r.converged;
    sol.outer_iterations = r.outer_iterations;
    sol.total_inner_iterations = r.total_inner_iterations;
    sol.solve_time = r.solve_time;
    sol.status_message = r.status_message;
    return sol;
}

}  // namespace impact
