#pragma once

#include <casadi/casadi.hpp>
#include <cmath>

#include "impact/problem.h"

namespace push_t {

/**
 * @brief Planar Push-T manipulation task written as an MPCCProblem.
 *
 * The contact model uses 8 friction force components with complementarity pairs
 * for contact distance, force direction, and absolute-value slacks.
 *
 * State: [px, py, theta] - position (x, y) and orientation of the T-block
 *
 * Control: [cx, cy, lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7,
 *           v0, v1, v2, v3, v4, v5, v6,
 *           w0, w1, w2, w3, w4, w5, w6]
 *   - cx, cy: contact point in body frame (2D)
 *   - lam[8]: friction force magnitudes
 *   - v[7], w[7]: slack variables for absolute value modeling
 *
 * Complementarity constraints: 43 pairs
 *   - v ⊥ w (7 pairs): absolute value constraints
 *   - lam_mag ⊥ gap (8 pairs): contact-distance complementarity
 *   - lam mutual exclusion (28 pairs): directional force exclusion
 *
 * Equality constraints: 7 (v - w = expression)
 * Inequality constraints: 4 (contact point bounds)
 */
class PushT : public impact::MPCCProblem {
   public:
    PushT();

    // MPCCProblem interface.
    int getStateDim() const override { return 3; }
    int getControlDim() const override { return 24; }
    int getComplementarityDim() const override { return 43; }
    double getTimeStep() const override { return dt_; }

    // Extra per-stage constraints.
    int getEqualityConstraintDim() const override { return 7; }
    int getInequalityConstraintDim() const override { return 4; }  // 4 contact bounds

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
    double getMass() const { return m_; }
    double getGravity() const { return g_; }
    double getFriction() const { return mu_; }
    double getSideLength() const { return l_; }

   private:
    // Physical parameters.
    double m_;   // mass of the T-block (kg)
    double g_;   // gravitational acceleration (m/s^2)
    double mu_;  // friction coefficient
    double c_;   // limit surface parameter (integration constant)
    double l_;   // characteristic length (half side of square)
    double dc_;  // limit surface rotational parameter
    double r_;   // contact radius constraint
    double dt_;  // time step (s)

    // CasADi symbols for the state.
    casadi::SX px_;     // x position
    casadi::SX py_;     // y position
    casadi::SX theta_;  // orientation

    // CasADi symbols for the control.
    casadi::SX cx_;   // contact point x in body frame
    casadi::SX cy_;   // contact point y in body frame
    casadi::SX lam_;  // friction forces (8 components)
    casadi::SX v_;    // absolute value slack (7 components)
    casadi::SX w_;    // absolute value slack (7 components)

    casadi::SX state_;    // state vector [px, py, theta]
    casadi::SX control_;  // control vector (24 components)

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

}  // namespace push_t
