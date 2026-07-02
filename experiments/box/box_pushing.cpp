#include "box_pushing.h"

namespace box_pushing {

BoxPushing::BoxPushing()
    : m_(0.1),   // mass of the box
      g_(9.81),  // gravitational acceleration
      mu_(0.5),  // friction coefficient
      c_(0.5),   // integration constant
      a_(0.3),   // width of the box
      b_(0.4),   // height of the box
      dt_(0.05)  // time step
{
    r_ = std::sqrt(a_ * a_ + b_ * b_);  // characteristic contact distance

    // Symbolic state variables.
    px_ = casadi::SX::sym("px");
    py_ = casadi::SX::sym("py");
    theta_ = casadi::SX::sym("theta");

    // Symbolic control variables.
    cx_ = casadi::SX::sym("cx");
    cy_ = casadi::SX::sym("cy");
    ld1y_ = casadi::SX::sym("ld1y");
    ld2x_ = casadi::SX::sym("ld2x");
    ld3y_ = casadi::SX::sym("ld3y");
    ld4x_ = casadi::SX::sym("ld4x");

    // State and control vectors used by CasADi functions.
    state_ = casadi::SX::vertcat({px_, py_, theta_});
    control_ = casadi::SX::vertcat({cx_, cy_, ld1y_, ld2x_, ld3y_, ld4x_});

    // Build the symbolic functions once.
    buildDynamics();
    buildComplementarityConstraints();
}

void BoxPushing::buildDynamics() {
    // Planar quasi-static dynamics: dx/dt = f(x, u).
    double constant = 1.0 / (mu_ * m_ * g_);

    // dpx/dt
    casadi::SX dpx = constant * ((ld2x_ + ld4x_) * casadi::SX::cos(theta_) -
                                 (ld1y_ + ld3y_) * casadi::SX::sin(theta_));

    // dpy/dt
    casadi::SX dpy = constant * ((ld2x_ + ld4x_) * casadi::SX::sin(theta_) +
                                 (ld1y_ + ld3y_) * casadi::SX::cos(theta_));

    // dtheta/dt
    casadi::SX dtheta = constant / (c_ * r_) * (-cy_ * (ld2x_ + ld4x_) + cx_ * (ld1y_ + ld3y_));

    casadi::SX f = casadi::SX::vertcat({dpx, dpy, dtheta});

    // Dynamics function used by the solver adapters.
    dyna_func_ = casadi::Function("dynamics", {state_, control_}, {f}, {"state", "control"}, {"f"});
}

void BoxPushing::buildComplementarityConstraints() {
    // Complementarity terms: 0 <= G(x,u) ⊥ H(x,u) >= 0.

    casadi::SX G_term = casadi::SX::vertcat({ld1y_,   // force at left side >= 0
                                             ld2x_,   // force at right side >= 0
                                             -ld3y_,  // force at bottom side >= 0 (negated)
                                             -ld4x_,  // force at top side >= 0 (negated)
                                             ld1y_,   // force coupling constraints
                                             ld1y_, ld1y_, ld2x_, ld2x_, -ld3y_});

    casadi::SX H_term = casadi::SX::vertcat({
        cy_ + b_,  // contact point within box (left)
        cx_ + a_,  // contact point within box (right)
        b_ - cy_,  // contact point within box (bottom)
        a_ - cx_,  // contact point within box (top)
        ld2x_,     // force coupling (if left active, right compatible)
        -ld3y_,    // force coupling (if left active, bottom compatible)
        -ld4x_,    // force coupling (if left active, top compatible)
        -ld3y_,    // force coupling (if right active, bottom compatible)
        -ld4x_,    // force coupling (if right active, top compatible)
        -ld4x_     // force coupling (if bottom active, top compatible)
    });

    // G and H functions used by MPCCProblem.
    G_func_ =
        casadi::Function("G_matrix", {state_, control_}, {G_term}, {"state", "control"}, {"G"});

    H_func_ =
        casadi::Function("H_matrix", {state_, control_}, {H_term}, {"state", "control"}, {"H"});
}

casadi::DM BoxPushing::dynamics(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = dyna_func_(arg);
    return res[0];
}

casadi::DM BoxPushing::getG(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = G_func_(arg);
    return res[0];
}

casadi::DM BoxPushing::getH(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = H_func_(arg);
    return res[0];
}

}  // namespace box_pushing
