#pragma once

#include <Eigen/Core>
#include <casadi/casadi.hpp>
#include <functional>
#include <vector>

#include "impact/dual_block.h"
#include "impact/saddle_layout.h"

namespace impact {

/**
 * @brief Data container passed to BCDAULASolver.
 *
 * Built by the shooting builders or by buildMPCC(). It owns the least-squares
 * residual r(z; p), its Jacobian, the persistent parameter buffer p, the AuLa
 * dual state, and a few evaluators used by the solver. The solver does not need
 * to know whether the subproblem came from single shooting, multiple shooting, or
 * a direct MPCC.
 *
 * The builder defines the parameter layout: it sets each dual block's
 * kappa_offset / rho_offset and the comp block's offsets so syncParams() scatters
 * the AuLa state into p in the exact order the symbolic p expects.
 */
class AulaSubproblem {
   public:
    AulaSubproblem() = default;

    // Builder-facing setup.
    void setFunctions(casadi::Function residual, casadi::Function jacobian, casadi::Function gh,
                      int n_opt, int n_params);
    void setSaddleLayout(SaddleLayout layout) { layout_ = std::move(layout); }
    void setObjective(std::function<double(const Eigen::VectorXd&)> obj) { obj_ = std::move(obj); }
    void setObjectiveGradient(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad) {
        obj_grad_ = std::move(grad);
    }
    void setStationarityEvaluator(std::function<double(const Eigen::VectorXd&)> eval) {
        stationarity_eval_ = std::move(eval);
    }
    void setTerminationOptions(bool check_stationarity, bool conditioned_complementarity,
                               double stationarity_tol, int max_stagnation_restarts) {
        check_stationarity_ = check_stationarity;
        conditioned_complementarity_ = conditioned_complementarity;
        stationarity_tol_ = stationarity_tol;
        max_stagnation_restarts_ = max_stagnation_restarts;
    }
    // Write a fixed (non-AuLa) parameter sub-vector once, e.g. targets / runtime data.
    void setParamValue(int offset, const Eigen::VectorXd& v);
    std::vector<DualBlock>& dualBlocks() { return dual_blocks_; }
    std::vector<CompBlock>& compBlocks() { return comp_blocks_; }
    const std::vector<CompBlock>& compBlocks() const { return comp_blocks_; }

    // Backward-compatible access for code that still expects exactly one
    // complementarity channel. Builders that support multiple groups should use
    // compBlocks() directly.
    CompBlock& compBlock();

    // Solver-facing accessors.
    const casadi::Function& residualFunction() const { return residual_func_; }
    const casadi::Function& jacobianFunction() const { return jacobian_func_; }
    const SaddleLayout& saddleLayout() const { return layout_; }
    int numOpt() const { return n_opt_; }
    const casadi::DM& params() const { return params_; }
    const std::vector<DualBlock>& dualBlocks() const { return dual_blocks_; }
    const CompBlock& compBlock() const;

    // Scatter the current dual-block + comp AuLa state into the parameter buffer.
    void syncParams();

    // Evaluate the complementarity functions G(z), H(z).
    void evalGH(const Eigen::VectorXd& z, Eigen::VectorXd& G, Eigen::VectorXd& H) const;

    // ||r(z; params_)||^2, used for inner-BCD stagnation.
    double evalAugmentedObjective(const Eigen::VectorXd& z) const;

    // Task objective for reporting (builder-supplied closure).
    double evalTaskObjective(const Eigen::VectorXd& z) const;
    Eigen::VectorXd evalTaskGradient(const Eigen::VectorXd& z) const;
    double evalAugmentedGradientInf(const Eigen::VectorXd& z) const;
    bool checkStationarity() const { return check_stationarity_; }
    bool conditionedComplementarity() const { return conditioned_complementarity_; }
    double stationarityTol() const { return stationarity_tol_; }
    int maxStagnationRestarts() const { return max_stagnation_restarts_; }

   private:
    casadi::Function residual_func_;
    casadi::Function jacobian_func_;
    casadi::Function gh_func_;  // z -> {G_all, H_all}
    SaddleLayout layout_;
    int n_opt_ = 0;

    casadi::DM params_;
    mutable casadi::DM z_dm_;

    std::vector<DualBlock> dual_blocks_;
    std::vector<CompBlock> comp_blocks_;
    CompBlock empty_comp_;
    std::function<double(const Eigen::VectorXd&)> obj_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> obj_grad_;
    std::function<double(const Eigen::VectorXd&)> stationarity_eval_;
    bool check_stationarity_ = false;
    bool conditioned_complementarity_ = false;
    double stationarity_tol_ = 1e-5;
    int max_stagnation_restarts_ = 0;

    void writeBlock(int offset, const Eigen::VectorXd& v);
    const casadi::DM& loadZ(const Eigen::VectorXd& z) const;
};

}  // namespace impact
