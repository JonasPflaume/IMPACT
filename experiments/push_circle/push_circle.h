#pragma once

#include <casadi/casadi.hpp>
#include <cmath>

#include "impact/problem.h"

namespace push_circle {

/**
 * @brief Planar single-point pushing of a rigid disk, written as an MPCCProblem.
 *
 * Quasi-static pusher-slider with pusher/slider friction:
 *
 *   State   x = [qx, qy, sx, sy]      disk center q, pusher point s
 *   Control u = [f_n, f_t, vx, vy]    normal force, tangential (friction) force,
 *                                     pusher velocity
 *
 * The disk is symmetric, so its orientation is decoupled from the signed-distance
 * field, the contact geometry and the goal; we therefore do not track it. The
 * tangential friction force still steers the disk translationally through the
 * isotropic limit surface, so contact friction is fully modelled for the task.
 *
 * Contact geometry (world frame). Let d = s - q, r = ||d|| (distance from the
 * disk center to the pusher). The smooth signed-distance function of the disk is
 *
 *   phi(x) = r - R.
 *
 * The pusher pushes the disk along the inward normal n_in = -d / r and may apply
 * a tangential friction force along t = (-dy, dx) / r. The net contact force on
 * the disk is F = f_n * n_in + f_t * t, and the quasi-static limit surface maps
 * it to the disk velocity q_dot = c_trans * F with c_trans = 1 / (mu_s m g).
 * The pusher moves kinematically, s_dot = v.
 *
 * Complementarity  0 <= f_n  ⊥  phi >= 0  activates the normal force only in
 * contact and, through phi >= 0, keeps the pusher outside the disk (so it has to
 * travel around it). The friction cone |f_t| <= mu_c f_n is a stage inequality.
 */
class PushCircle : public impact::MPCCProblem {
   public:
    PushCircle();

    // MPCCProblem interface.
    int getStateDim() const override { return 4; }
    int getControlDim() const override { return 4; }
    int getComplementarityDim() const override { return 1; }
    int getEqualityConstraintDim() const override { return 0; }
    int getInequalityConstraintDim() const override { return 2; }  // friction cone
    double getTimeStep() const override { return dt_; }

    casadi::DM dynamics(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::Function getDynamicsFunction() const override { return dyna_func_; }

    casadi::DM getG(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::DM getH(const casadi::DM& x, const casadi::DM& u) const override;
    casadi::Function getGFunction() const override { return G_func_; }
    casadi::Function getHFunction() const override { return H_func_; }

    casadi::Function getInequalityConstraintFunction() const override { return I_func_; }

    // Model parameters used by the experiments / visualizer.
    double getRadius() const { return R_; }
    double getPusherFriction() const { return mu_c_; }
    double getGroundFriction() const { return mu_s_; }

   private:
    // Physical parameters.
    double m_;     // mass of the disk (kg)
    double g_;     // gravitational acceleration (m/s^2)
    double mu_s_;  // slider/ground friction (limit surface)
    double mu_c_;  // pusher/slider friction (contact friction cone)
    double R_;     // disk radius (m)
    double dt_;    // time step (s)

    // CasADi state symbols.
    casadi::SX qx_, qy_;  // disk center
    casadi::SX sx_, sy_;  // pusher point

    // CasADi control symbols.
    casadi::SX fn_;  // normal contact force (>= 0)
    casadi::SX ft_;  // tangential friction force (signed)
    casadi::SX vx_, vy_;  // pusher velocity

    casadi::SX state_;    // [qx, qy, sx, sy]
    casadi::SX control_;  // [fn, ft, vx, vy]

    // Cached CasADi functions.
    casadi::Function dyna_func_;
    casadi::Function G_func_;
    casadi::Function H_func_;
    casadi::Function I_func_;

    void buildDynamics();
    void buildComplementarityConstraints();
    void buildInequalityConstraints();
};

}  // namespace push_circle
