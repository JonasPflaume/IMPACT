#pragma once

#include <casadi/casadi.hpp>
#include <memory>
#include <string>
#include <vector>

#include "impact/aula_subproblem.h"

namespace impact {

/**
 * @brief One constraint group of a generic MPCC.
 *
 * Equality   : 0 = c(z, p)            -> AuLa block  sqrt(rho)*(scale*c + kappa/rho)
 * Inequality : 0 >= c(z, p)           -> AuLa block  sqrt(rho)*max(scale*c + mu/rho, 0)
 * Complementarity : 0 <= G ⊥ H >= 0   -> closed-form (Y,Z) projection on (G,H)
 *
 * `scale` is a conditioning factor. `rho_init` and `tol` initialize the
 * safeguarded penalty update for this group.
 */
struct MPCCConstraint {
    enum Kind { Equality, Inequality, Complementarity } kind = Equality;
    std::string name;
    casadi::SX c;        // Equality / Inequality residual (function of z, p)
    casadi::SX G, H;     // Complementarity legs (Kind == Complementarity)
    double scale = 1.0;
    double rho_init = 1.0;
    double tol = 1e-5;
};

/**
 * @brief Per-constraint-block AuLa tuning.
 *
 * `scale` is the residual conditioning/weight for this block, `rho_init` is the
 * initial augmented-Lagrangian penalty, and `tol` is the unscaled convergence
 * tolerance for this block. Add one large vertcat'ed block for shared tuning, or
 * add multiple blocks when different groups need different tuning.
 */
struct MPCCBlockOptions {
    double scale = 1.0;
    double rho_init = 1.0;
    double tol = 1e-5;
};

/**
 * @brief A generic MPCC, independent of any trajectory structure:
 *
 *     minimize    || cost(z, p) ||^2
 *     subject to  the ordered list of MPCCConstraints.
 *
 * `z` is the decision vector, `p` the (per-solve) runtime parameters. The
 * Constraint rows are emitted in `constraints` order after the cost rows, so the
 * caller controls the residual layout. Complementarity constraints can be one
 * stacked group or several groups; each group gets its own slack, multiplier,
 * rho, scale and tolerance.
 */
struct MPCCDescription {
    casadi::SX z;
    casadi::SX p;  // runtime parameters (size 0 if none)
    casadi::SX cost;
    bool cost_is_linear = false;  // fold cost into the constant saddle block if true
    std::vector<MPCCConstraint> constraints;

    // Compile the residual/Jacobian CasADi functions through a C compiler instead
    // of the interpreter. This costs extra at build time, but helps when the same
    // problem is solved many times, as in receding-horizon MPC.
    bool jit = false;

    void addEqualityBlock(const std::string& name, const casadi::SX& c,
                          MPCCBlockOptions opts = {}) {
        constraints.push_back(
            {MPCCConstraint::Equality, name, c, casadi::SX(), casadi::SX(), opts.scale,
             opts.rho_init, opts.tol});
    }

    void addEqualityBlock(const std::string& name, const std::vector<casadi::SX>& c,
                          MPCCBlockOptions opts = {}) {
        addEqualityBlock(name, casadi::SX::vertcat(c), opts);
    }

    void addInequalityBlock(const std::string& name, const casadi::SX& c,
                            MPCCBlockOptions opts = {}) {
        constraints.push_back(
            {MPCCConstraint::Inequality, name, c, casadi::SX(), casadi::SX(), opts.scale,
             opts.rho_init, opts.tol});
    }

    void addInequalityBlock(const std::string& name, const std::vector<casadi::SX>& c,
                            MPCCBlockOptions opts = {}) {
        addInequalityBlock(name, casadi::SX::vertcat(c), opts);
    }

    void addComplementarityBlock(const std::string& name, const casadi::SX& G,
                                 const casadi::SX& H, MPCCBlockOptions opts = {}) {
        constraints.push_back(
            {MPCCConstraint::Complementarity, name, casadi::SX(), G, H, opts.scale,
             opts.rho_init, opts.tol});
    }

    void addComplementarityBlock(const std::string& name, const std::vector<casadi::SX>& G,
                                 const std::vector<casadi::SX>& H,
                                 MPCCBlockOptions opts = {}) {
        addComplementarityBlock(name, casadi::SX::vertcat(G), casadi::SX::vertcat(H), opts);
    }
};

/**
 * @brief Assembled MPCC plus the offset of the user runtime parameter p.
 *
 * Write p at `off_p` in the subproblem parameter buffer before solving.
 */
struct MPCCSubproblem {
    std::unique_ptr<AulaSubproblem> sub;
    int off_p = 0;
};

/**
 * @brief Assemble a generic MPCC into an AulaSubproblem.
 *
 * Both shooting builders eventually call this. Direct users can also call it for
 * MPCCs that are not trajectory problems. It builds the augmented-Lagrangian
 * residual/Jacobian, dual blocks, complementarity blocks and saddle layout.
 */
MPCCSubproblem buildMPCC(const MPCCDescription& desc);

}  // namespace impact
