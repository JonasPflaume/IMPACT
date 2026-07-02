#pragma once

#include <casadi/casadi.hpp>
#include <cmath>

#include "impact/problem.h"

namespace box_pushing {

/**
 * @brief Planar box-pushing task written as an MPCCProblem.
 *
 * State: [px, py, theta] is the box pose. Control stores the body-frame contact
 * point followed by the side force variables:
 * [cx, cy, ld1y, ld2x, ld3y, ld4x].
 *
 * Complementarity constraints select the active side/contact force combination.
 */
class BoxPushing : public impact::MPCCProblem {
   public:
    BoxPushing();

    // MPCCProblem interface.
    int getStateDim() const override { return 3; }
    int getControlDim() const override { return 6; }
    int getComplementarityDim() const override { return 10; }
    double getTimeStep() const override { return dt_; }

    casadi::DM dynamics(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::Function getDynamicsFunction() const override { return dyna_func_; }

    casadi::DM getG(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::DM getH(const casadi::DM& x, const casadi::DM& u) const override;

    casadi::Function getGFunction() const override { return G_func_; }
    casadi::Function getHFunction() const override { return H_func_; }

    // Model parameters used by the experiments.
    double getMass() const { return m_; }
    double getGravity() const { return g_; }
    double getFriction() const { return mu_; }
    double getWidth() const { return a_; }
    double getHeight() const { return b_; }

   private:
    // Physical parameters.
    double m_;   // mass of the box (kg)
    double g_;   // gravitational acceleration (m/s^2)
    double mu_;  // friction coefficient
    double c_;   // integration constant
    double a_;   // width of the box (m)
    double b_;   // height of the box (m)
    double r_;   // characteristic distance: sqrt(a^2 + b^2)
    double dt_;  // time step (s)

    // CasADi symbols for the state.
    casadi::SX px_;     // x position
    casadi::SX py_;     // y position
    casadi::SX theta_;  // orientation

    // CasADi symbols for the control.
    casadi::SX cx_;    // contact point x in body frame
    casadi::SX cy_;    // contact point y in body frame
    casadi::SX ld1y_;  // contact force at left side (y direction)
    casadi::SX ld2x_;  // contact force at right side (x direction)
    casadi::SX ld3y_;  // contact force at bottom side (y direction)
    casadi::SX ld4x_;  // contact force at top side (x direction)

    casadi::SX state_;    // state vector [px, py, theta]
    casadi::SX control_;  // control vector [cx, cy, ld1y, ld2x, ld3y, ld4x]

    // Cached CasADi functions.
    casadi::Function dyna_func_;  // dynamics function
    casadi::Function G_func_;     // G complementarity function
    casadi::Function H_func_;     // H complementarity function

    void buildDynamics();
    void buildComplementarityConstraints();
};

}  // namespace box_pushing
