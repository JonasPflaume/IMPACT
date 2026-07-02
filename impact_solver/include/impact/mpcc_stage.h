#pragma once

#include <memory>

#include "impact/bcd_aula_config.h"
#include "impact/problem.h"
#include "impact/stage_problem.h"

namespace impact {

/**
 * @brief Encodes an MPCCProblem (explicit ODE f(x,u), generic quadratic cost) as
 *        a StageProblem.
 *
 *   step(x,u)   = x + dt * f(x,u)        (explicit Euler)
 *   G,H,eq,ineq = the problem's functions
 *   cost        = the config-driven quadratic (stage ||u||^2, optional ||x-goal||^2,
 *                 optional control rate, terminal ||x_T - goal||^2)
 *
 * runtime parameter p = x_goal. This adapter lets the box / push_t / cart tasks,
 * which already implement MPCCProblem for the IPOPT/Scholtes baselines, use the
 * same shooting builders.
 */
class MPCCStage : public StageProblem {
   public:
    MPCCStage(std::shared_ptr<MPCCProblem> problem, const BCDAULAConfig& config);

    int stateDim() const override { return nx_; }
    int controlDim() const override { return nu_; }
    int compDim() const override { return nc_; }
    int eqDim() const override { return neq_; }
    int ineqDim() const override { return nineq_; }
    int runtimeParamDim() const override { return nx_; }  // x_goal
    double timeStep() const override { return dt_; }
    bool costIsLinear() const override { return true; }  // quadratic objective

    casadi::SX step(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX G(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX H(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX eq(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX ineq(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const override;
    casadi::SX costResidual(const std::vector<casadi::SX>& X, const std::vector<casadi::SX>& U,
                            const casadi::SX& p) const override;

   private:
    std::shared_ptr<MPCCProblem> problem_;
    int nx_, nu_, nc_, neq_, nineq_;
    double dt_;
    double stage_cost_weight_, stage_state_cost_weight_, control_rate_weight_, final_cost_weight_;
    casadi::Function dyn_, G_, H_, eq_, ineq_;
};

}  // namespace impact
