#pragma once

#include <casadi/casadi.hpp>
#include <cmath>

#include "impact/problem.h"

namespace cart_transporter {

/**
 * @brief Cargo-on-cart transport task written as an MPCCProblem.
 *
 * The cargo (m1) sits on the cart (m2). The optimizer moves both while enforcing
 * Coulomb friction and keeping the cargo within the cart gap.
 *
 * The cargo can slide on the cart with Coulomb friction:
 *   - When cargo and cart move together: |f| <= mu * m1 * g
 *   - When cargo slides forward: f = -mu * m1 * g (v > 0)
 *   - When cargo slides backward: f = mu * m1 * g (w > 0)
 *
 * State: [x1, x2, x1_dot, x2_dot] - positions and velocities of cargo and cart
 *   - x1: cargo position
 *   - x2: cart position
 *   - x1_dot: cargo velocity
 *   - x2_dot: cart velocity
 *
 * Control: [f, u, v, w] - contact force, control input, and slack variables
 *   - f: friction force between cargo and cart
 *   - u: external control force on cart
 *   - v, w: slack variables for |x1_dot - x2_dot| = v - w, with v,w >= 0
 *
 * Complementarity constraints (3 pairs):
 *   1. v ⊥ w : can't slide in both directions
 *   2. w ⊥ (mu*m1*g - f) : if sliding backward, friction at upper limit
 *   3. v ⊥ (f + mu*m1*g) : if sliding forward, friction at lower limit
 *
 * Equality constraints (1 per timestep):
 *   - x1_dot - x2_dot - v + w = 0 : relative velocity definition
 *
 * Inequality constraints (4 per timestep):
 *   - mu*m1*g - f >= 0  (friction upper bound)
 *   - f + mu*m1*g >= 0  (friction lower bound)
 *   - x1 - x2 + l >= 0  (gap constraint: cargo stays on cart)
 *   - l - (x1 - x2) >= 0  (gap constraint: cargo stays on cart)
 */
class CartTransporter : public impact::MPCCProblem {
   public:
    CartTransporter();

    // MPCCProblem interface.
    int getStateDim() const override { return 4; }
    int getControlDim() const override { return 4; }  // [f, u, v, w]
    int getComplementarityDim() const override { return 3; }
    double getTimeStep() const override { return dt_; }

    // Extra per-stage constraints.
    int getEqualityConstraintDim() const override { return 1; }
    int getInequalityConstraintDim() const override { return 4; }

    casadi::DM dynamics(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::Function getDynamicsFunction() const override { return dyna_func_; }

    casadi::DM getG(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::DM getH(const casadi::DM& x, const casadi::DM& u) const override;

    casadi::Function getGFunction() const override { return G_func_; }
    casadi::Function getHFunction() const override { return H_func_; }

    // Equality and inequality functions.
    casadi::Function getEqualityConstraintFunction() const override { return E_func_; }
    casadi::Function getInequalityConstraintFunction() const override { return I_func_; }

    // Control bounds.
    std::vector<double> getControlLowerBounds() const override;
    std::vector<double> getControlUpperBounds() const override;

    // Model parameters used by the experiments.
    double getCargoMass() const { return m1_; }
    double getCartMass() const { return m2_; }
    double getFriction() const { return mu_; }
    double getGravity() const { return g_; }
    double getGapLength() const { return l_; }

   private:
    // Physical parameters.
    double m1_;  // mass of cargo (kg)
    double m2_;  // mass of cart (kg)
    double mu_;  // friction coefficient
    double g_;   // gravitational acceleration (m/s^2)
    double l_;   // half gap length (cargo must stay within |x1 - x2| <= l)
    double dt_;  // time step (s)

    // CasADi symbols for the state.
    casadi::SX x1_;      // cargo position
    casadi::SX x2_;      // cart position
    casadi::SX x1_dot_;  // cargo velocity
    casadi::SX x2_dot_;  // cart velocity

    // CasADi symbols for the control.
    casadi::SX f_;  // friction force
    casadi::SX u_;  // control force on cart
    casadi::SX v_;  // slack for positive relative velocity
    casadi::SX w_;  // slack for negative relative velocity

    casadi::SX state_;    // state vector [x1, x2, x1_dot, x2_dot]
    casadi::SX control_;  // control vector [f, u, v, w]

    // Cached CasADi functions.
    casadi::Function dyna_func_;  // dynamics function
    casadi::Function G_func_;     // G complementarity function
    casadi::Function H_func_;     // H complementarity function
    casadi::Function E_func_;     // equality constraints function
    casadi::Function I_func_;     // inequality constraints function

    void buildDynamics();
    void buildComplementarityConstraints();
    void buildEqualityConstraints();
    void buildInequalityConstraints();
};

}  // namespace cart_transporter
