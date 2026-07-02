#pragma once

#include <casadi/casadi.hpp>
#include <vector>

namespace impact {

/**
 * @brief Per-stage description used by the shooting builders.
 *
 * Every trajectory task is reduced to this interface before it reaches the solver.
 * LCP contact problems use the same template: the state update is kinematic, and
 * the force balance appears as an algebraic dynamics residual (see LCPStage).
 *
 * The shooting transcription is chosen outside the task:
 *   - multiple shooting makes the state X free and adds the defect
 *     X_{k+1} - step(X_k, U_k) as an equality block;
 *   - single shooting rolls X out via step() so only the controls are free.
 * Both builders emit an AulaSubproblem for BCDAULASolver.
 *
 * Symbolic methods are evaluated when the subproblem is built, not during the
 * solver loop. The runtime parameter vector p is for data that changes between
 * solves, such as targets or contact data (J/phi). Tasks without such data return
 * runtimeParamDim() == 0.
 */
class StageProblem {
   public:
    virtual ~StageProblem() = default;

    // Stage dimensions.
    virtual int stateDim() const = 0;        // nx  (state x, subject to the shooting choice)
    virtual int controlDim() const = 0;      // nu  (controls + always-free aux, e.g. lam/vel)
    virtual int compDim() const = 0;         // complementarity pairs per stage
    virtual int eqDim() const { return 0; }  // extra per-stage equalities
    virtual int ineqDim() const { return 0; }              // per-stage inequalities g <= 0
    virtual int dynamicsResidualDim() const { return 0; }  // algebraic dynamics (LCP force balance)
    virtual int runtimeParamDim() const { return 0; }      // per-solve parameter p
    virtual double timeStep() const = 0;

    /// True when the multiple-shooting cost residual is linear in the decision
    /// variables, which makes the objective quadratic. In that case the builder
    /// can put the cost in the constant saddle block. Nonlinear costs stay as
    /// rho=1 penalty rows. Single shooting rolls out the state, so this flag only
    /// affects the multiple-shooting path.
    virtual bool costIsLinear() const { return false; }

    // Symbolic stage model (x: nx, u: nu, p: runtimeParamDim).

    /// State evolution x_{k+1} = step(x_k, u_k, p) (used for rollout or defect).
    virtual casadi::SX step(const casadi::SX& x, const casadi::SX& u,
                            const casadi::SX& p) const = 0;

    /// Complementarity 0 <= G(x,u,p) ⊥ H(x,u,p) >= 0.
    virtual casadi::SX G(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const = 0;
    virtual casadi::SX H(const casadi::SX& x, const casadi::SX& u, const casadi::SX& p) const = 0;

    /// Algebraic dynamics residual r_dyn(x,u,p) = 0 (LCP force balance). Empty when
    /// dynamicsResidualDim() == 0 (pure-ODE tasks).
    virtual casadi::SX dynamicsResidual(const casadi::SX&, const casadi::SX&,
                                        const casadi::SX&) const {
        return casadi::SX();
    }

    /// Extra equalities eq(x,u,p) = 0. Empty when eqDim() == 0.
    virtual casadi::SX eq(const casadi::SX&, const casadi::SX&, const casadi::SX&) const {
        return casadi::SX();
    }

    /// Inequalities ineq(x,u,p) <= 0. Empty when ineqDim() == 0.
    virtual casadi::SX ineq(const casadi::SX&, const casadi::SX&, const casadi::SX&) const {
        return casadi::SX();
    }

    /// Full nonlinear-least-squares cost residual (objective = ||costResidual||^2),
    /// given the symbolic per-stage states X (size horizon+1) and controls U (size
    /// horizon). The task owns its cost and emits the rows in its own order.
    virtual casadi::SX costResidual(const std::vector<casadi::SX>& X,
                                    const std::vector<casadi::SX>& U,
                                    const casadi::SX& p) const = 0;
};

}  // namespace impact
