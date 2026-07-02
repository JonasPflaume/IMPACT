#include "impact/mpcc_stage.h"

#include <cmath>

namespace impact {

using casadi::SX;

MPCCStage::MPCCStage(std::shared_ptr<MPCCProblem> problem, const BCDAULAConfig& config)
    : problem_(std::move(problem)) {
    nx_ = problem_->getStateDim();
    nu_ = problem_->getControlDim();
    nc_ = problem_->getComplementarityDim();
    neq_ = problem_->getEqualityConstraintDim();
    nineq_ = problem_->getInequalityConstraintDim();
    dt_ = problem_->getTimeStep();
    stage_cost_weight_ = config.stage_cost_weight;
    stage_state_cost_weight_ = config.stage_state_cost_weight;
    control_rate_weight_ = config.control_rate_weight;
    final_cost_weight_ = config.final_cost_weight;
    dyn_ = problem_->getDynamicsFunction();
    G_ = problem_->getGFunction();
    H_ = problem_->getHFunction();
    if (neq_ > 0) eq_ = problem_->getEqualityConstraintFunction();
    if (nineq_ > 0) ineq_ = problem_->getInequalityConstraintFunction();
}

casadi::SX MPCCStage::step(const SX& x, const SX& u, const SX&) const {
    return x + dt_ * dyn_(std::vector<SX>{x, u})[0];
}
casadi::SX MPCCStage::G(const SX& x, const SX& u, const SX&) const {
    return G_(std::vector<SX>{x, u})[0];
}
casadi::SX MPCCStage::H(const SX& x, const SX& u, const SX&) const {
    return H_(std::vector<SX>{x, u})[0];
}
casadi::SX MPCCStage::eq(const SX& x, const SX& u, const SX&) const {
    return eq_(std::vector<SX>{x, u})[0];
}
casadi::SX MPCCStage::ineq(const SX& x, const SX& u, const SX&) const {
    return ineq_(std::vector<SX>{x, u})[0];
}

casadi::SX MPCCStage::costResidual(const std::vector<SX>& X, const std::vector<SX>& U,
                                   const SX& p) const {
    // p == x_goal. Row order reproduces the legacy MPCC residual exactly:
    // per stage [sqrt(stage)*u, optional sqrt(state)*(x-goal)], optional control
    // rate rows, then the terminal sqrt(final)*(x_T - goal).
    const int H = static_cast<int>(U.size());
    const SX& x_goal = p;
    std::vector<SX> rows;
    for (int t = 0; t < H; ++t) {
        rows.push_back(std::sqrt(stage_cost_weight_) * U[t]);
        if (stage_state_cost_weight_ > 0.0)
            rows.push_back(std::sqrt(stage_state_cost_weight_) * (X[t] - x_goal));
    }
    if (control_rate_weight_ > 0.0)
        for (int t = 0; t < H - 1; ++t)
            rows.push_back(std::sqrt(control_rate_weight_) * (U[t + 1] - U[t]));
    rows.push_back(std::sqrt(final_cost_weight_) * (X[H] - x_goal));
    return SX::vertcat(rows);
}

}  // namespace impact
