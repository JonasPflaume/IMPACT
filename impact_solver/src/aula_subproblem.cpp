#include "impact/aula_subproblem.h"

#include <algorithm>

namespace impact {

void AulaSubproblem::setFunctions(casadi::Function residual, casadi::Function jacobian,
                                  casadi::Function gh, int n_opt, int n_params) {
    residual_func_ = std::move(residual);
    jacobian_func_ = std::move(jacobian);
    gh_func_ = std::move(gh);
    n_opt_ = n_opt;
    params_ = casadi::DM::zeros(n_params);
    z_dm_ = casadi::DM::zeros(n_opt);
}

void AulaSubproblem::writeBlock(int offset, const Eigen::VectorXd& v) {
    std::copy(v.data(), v.data() + v.size(), params_.nonzeros().begin() + offset);
}

void AulaSubproblem::setParamValue(int offset, const Eigen::VectorXd& v) { writeBlock(offset, v); }

CompBlock& AulaSubproblem::compBlock() {
    if (comp_blocks_.empty()) comp_blocks_.push_back(CompBlock{});
    return comp_blocks_.front();
}

const CompBlock& AulaSubproblem::compBlock() const {
    return comp_blocks_.empty() ? empty_comp_ : comp_blocks_.front();
}

void AulaSubproblem::syncParams() {
    std::vector<double>& p = params_.nonzeros();
    for (const DualBlock& b : dual_blocks_) {
        if (b.dim > 0) writeBlock(b.kappa_offset, b.kappa);
        p[b.rho_offset] = b.rho;
    }
    for (const CompBlock& comp : comp_blocks_) {
        if (comp.dim <= 0) continue;
        writeBlock(comp.kappaG_offset, comp.kappaG);
        writeBlock(comp.kappaH_offset, comp.kappaH);
        writeBlock(comp.sG_offset, comp.sG);
        writeBlock(comp.sH_offset, comp.sH);
        p[comp.rho_offset] = comp.rho;
    }
}

const casadi::DM& AulaSubproblem::loadZ(const Eigen::VectorXd& z) const {
    std::copy(z.data(), z.data() + z.size(), z_dm_.nonzeros().begin());
    return z_dm_;
}

void AulaSubproblem::evalGH(const Eigen::VectorXd& z, Eigen::VectorXd& G, Eigen::VectorXd& H) const {
    // gh_func takes (z, p): single shooting reads the runtime phi/J from p; multiple
    // shooting ignores p. Passing params_ keeps the call site uniform.
    std::vector<casadi::DM> out = gh_func_(std::vector<casadi::DM>{loadZ(z), params_});
    G = Eigen::Map<const Eigen::VectorXd>(out[0].ptr(), out[0].numel());
    H = Eigen::Map<const Eigen::VectorXd>(out[1].ptr(), out[1].numel());
}

double AulaSubproblem::evalAugmentedObjective(const Eigen::VectorXd& z) const {
    casadi::DM out = residual_func_(std::vector<casadi::DM>{loadZ(z), params_})[0];
    return Eigen::Map<const Eigen::VectorXd>(out.ptr(), out.numel()).squaredNorm();
}

double AulaSubproblem::evalTaskObjective(const Eigen::VectorXd& z) const {
    return obj_ ? obj_(z) : 0.0;
}

Eigen::VectorXd AulaSubproblem::evalTaskGradient(const Eigen::VectorXd& z) const {
    return obj_grad_ ? obj_grad_(z) : Eigen::VectorXd::Zero(n_opt_);
}

double AulaSubproblem::evalAugmentedGradientInf(const Eigen::VectorXd& z) const {
    return stationarity_eval_ ? stationarity_eval_(z) : 0.0;
}

}  // namespace impact
