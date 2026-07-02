#pragma once

#include <casadi/casadi.hpp>
#include <memory>
#include <vector>

namespace impact {

/**
 * @brief Per-stage formulation interface for a multiple-shooting MPCC task.
 *
 * A task is defined entirely by these symbolic per-stage pieces; the solver core
 * and the multiple-shooting builder do not depend on the concrete task class.
 * The state trajectory X and controls U are decision variables; dynamics are
 * enforced as an AuLa equality block (see multiple_shooting.h).
 *
 * Derived classes implement the symbolic functions for a specific problem.
 */
class MPCCProblem {
   public:
    virtual ~MPCCProblem() = default;

    virtual int getStateDim() const = 0;
    virtual int getControlDim() const = 0;
    virtual int getComplementarityDim() const = 0;
    virtual double getTimeStep() const = 0;

    /// dx/dt = f(x, u)
    virtual casadi::DM dynamics(const casadi::DM& x, const casadi::DM& u) const = 0;
    virtual casadi::Function getDynamicsFunction() const = 0;

    /// Complementarity 0 <= G(x,u) ⊥ H(x,u) >= 0
    virtual casadi::DM getG(const casadi::DM& x, const casadi::DM& u) const = 0;
    virtual casadi::DM getH(const casadi::DM& x, const casadi::DM& u) const = 0;
    virtual casadi::Function getGFunction() const = 0;
    virtual casadi::Function getHFunction() const = 0;

    virtual std::vector<double> getStateLowerBounds() const {
        return std::vector<double>(getStateDim(), -casadi::inf);
    }
    virtual std::vector<double> getStateUpperBounds() const {
        return std::vector<double>(getStateDim(), casadi::inf);
    }
    virtual std::vector<double> getControlLowerBounds() const {
        return std::vector<double>(getControlDim(), -casadi::inf);
    }
    virtual std::vector<double> getControlUpperBounds() const {
        return std::vector<double>(getControlDim(), casadi::inf);
    }

    /// Additional per-stage equality constraints eq(x,u) = 0 (0 if none).
    virtual int getEqualityConstraintDim() const { return 0; }
    /// Additional per-stage inequality constraints ineq(x,u) <= 0 (0 if none).
    virtual int getInequalityConstraintDim() const { return 0; }
    virtual casadi::Function getEqualityConstraintFunction() const { return casadi::Function(); }
    virtual casadi::Function getInequalityConstraintFunction() const { return casadi::Function(); }
};

}  // namespace impact
