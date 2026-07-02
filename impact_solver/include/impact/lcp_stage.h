#pragma once

#include <Eigen/Dense>
#include <memory>

#include "impact/bcd_aula_config.h"
#include "impact/lcp_problem.h"
#include "impact/stage_problem.h"

namespace impact {

/**
 * @brief Adapter from an LCP contact problem to StageProblem.
 *
 * The mapping used here is a DAE-style MPCC:
 *
 *   x = q                              (configuration)
 *   u = [cmd, lam, vel]                (controls + always-free contact aux)
 *   step(x,u) = integrate(q, vel)      (quaternion-aware kinematics)
 *   dynamicsResidual = Q*vel - b/h - Jᵀλ/h = 0   (LCP force balance, algebraic)
 *   G = lam,  H = J*vel + phi/h        (contact complementarity)
 *   ineq = [cmd_lower - cmd; cmd - cmd_upper] <= 0   (control bounds)
 *   cost = manipulation cost (control / contact-FK / grasp / velocity / terminal)
 *
 * runtime parameter p = [target_p(3), target_q(4), phi(n_lam), vec(J)(n_lam*n_qvel)],
 * which changes every MPC step. The initial configuration q0 is supplied by the
 * single-shooting front-end as the initial state.
 */
class LCPStage : public StageProblem {
   public:
    LCPStage(std::shared_ptr<LCPProblem> problem, const BCDAULAConfig& config);

    int stateDim() const override { return n_qpos_; }
    int controlDim() const override { return n_cmd_ + n_lam_ + n_qvel_; }
    int compDim() const override { return n_lam_; }
    int ineqDim() const override { return use_cmd_ ? 2 * n_cmd_ : 0; }
    int dynamicsResidualDim() const override { return n_qvel_; }
    int runtimeParamDim() const override { return 3 + 4 + n_lam_ + n_lam_ * n_qvel_; }
    double timeStep() const override { return h_; }

    casadi::SX step(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX G(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX H(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX dynamicsResidual(const casadi::SX& x, const casadi::SX& u,
                                const casadi::SX& p) const override;
    casadi::SX ineq(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX costResidual(const std::vector<casadi::SX>& X, const std::vector<casadi::SX>& U,
                            const casadi::SX& p) const override;

    // Runtime-param layout helpers shared with the front-end.
    int n_qpos() const { return n_qpos_; }
    int n_qvel() const { return n_qvel_; }
    int n_cmd() const { return n_cmd_; }
    int n_lam() const { return n_lam_; }

   private:
    // u = [cmd(n_cmd), lam(n_lam), vel(n_qvel)]; p = [tp(3), tq(4), phi(n_lam), J(n_lam*n_qvel)].
    casadi::SX cmdOf(const casadi::SX& u) const;
    casadi::SX lamOf(const casadi::SX& u) const;
    casadi::SX velOf(const casadi::SX& u) const;
    casadi::SX targetPOf(const casadi::SX& p) const;
    casadi::SX targetQOf(const casadi::SX& p) const;
    casadi::SX phiOf(const casadi::SX& p) const;
    casadi::SX jacOf(const casadi::SX& p) const;  // reshaped n_lam x n_qvel

    std::shared_ptr<LCPProblem> problem_;
    int n_qpos_, n_qvel_, n_cmd_, n_lam_, n_ftp_;
    double h_;
    double control_w_, contact_w_, grasp_w_, vel_w_, final_mult_, final_pos_w_, final_ori_w_;
    Eigen::MatrixXd Q_, robot_stiff_;
    Eigen::Vector3d b_o_;
    Eigen::VectorXd cmd_lower_, cmd_upper_;
    bool use_cmd_;
};

}  // namespace impact
