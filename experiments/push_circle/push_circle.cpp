#include "push_circle.h"

namespace push_circle {

PushCircle::PushCircle()
    : m_(1.0),     // mass of the disk
      g_(9.81),    // gravitational acceleration
      mu_s_(0.1),  // slider/ground friction (limit surface)
      mu_c_(0.1),  // pusher/slider friction (contact friction cone)
      R_(0.3),     // disk radius
      dt_(0.05)    // time step
{
    // Symbolic state variables.
    qx_ = casadi::SX::sym("qx");
    qy_ = casadi::SX::sym("qy");
    sx_ = casadi::SX::sym("sx");
    sy_ = casadi::SX::sym("sy");

    // Symbolic control variables.
    fn_ = casadi::SX::sym("fn");
    ft_ = casadi::SX::sym("ft");
    vx_ = casadi::SX::sym("vx");
    vy_ = casadi::SX::sym("vy");

    state_ = casadi::SX::vertcat({qx_, qy_, sx_, sy_});
    control_ = casadi::SX::vertcat({fn_, ft_, vx_, vy_});

    buildDynamics();
    buildComplementarityConstraints();
    buildInequalityConstraints();
}

void PushCircle::buildDynamics() {
    // Quasi-static limit surface: q_dot = c_trans * F, with the contact force
    // F = f_n * n_in + f_t * t expressed directly in the world frame.
    double c_trans = 1.0 / (mu_s_ * m_ * g_);

    casadi::SX dx = sx_ - qx_;  // vector from disk center to pusher
    casadi::SX dy = sy_ - qy_;
    // Smooth signed-distance norm; the +eps only matters if the pusher reaches the
    // disk center, which non-penetration prevents, so it never distorts contact.
    casadi::SX nrm = casadi::SX::sqrt(dx * dx + dy * dy + 1e-9);

    // Net contact force on the disk (world frame):
    //   normal     f_n * (-d / nrm)
    //   tangential f_t * (-dy, dx) / nrm
    casadi::SX Fx = (-fn_ * dx - ft_ * dy) / nrm;
    casadi::SX Fy = (-fn_ * dy + ft_ * dx) / nrm;

    casadi::SX dqx = c_trans * Fx;
    casadi::SX dqy = c_trans * Fy;
    // The pusher moves kinematically under its velocity command.
    casadi::SX dsx = vx_;
    casadi::SX dsy = vy_;

    casadi::SX f = casadi::SX::vertcat({dqx, dqy, dsx, dsy});
    dyna_func_ = casadi::Function("dynamics", {state_, control_}, {f}, {"state", "control"}, {"f"});
}

void PushCircle::buildComplementarityConstraints() {
    // 0 <= f_n  ⊥  phi >= 0, phi = ||s - q|| - R (disk signed-distance function).
    // Activates the normal force only on contact and keeps the pusher outside the
    // disk, so a valid trajectory has to route the pusher around it.
    casadi::SX dx = sx_ - qx_;
    casadi::SX dy = sy_ - qy_;
    casadi::SX nrm = casadi::SX::sqrt(dx * dx + dy * dy + 1e-9);
    casadi::SX phi = nrm - R_;

    casadi::SX G_term = casadi::SX::vertcat({fn_});
    casadi::SX H_term = casadi::SX::vertcat({phi});

    G_func_ =
        casadi::Function("G_matrix", {state_, control_}, {G_term}, {"state", "control"}, {"G"});
    H_func_ =
        casadi::Function("H_matrix", {state_, control_}, {H_term}, {"state", "control"}, {"H"});
}

void PushCircle::buildInequalityConstraints() {
    // Coulomb friction cone at the contact, g <= 0 convention:
    //   |f_t| <= mu_c f_n   =>   f_t - mu_c f_n <= 0,  -f_t - mu_c f_n <= 0.
    // When f_n = 0 (no contact) this collapses to f_t = 0.
    std::vector<casadi::SX> g_terms;
    g_terms.push_back(ft_ - mu_c_ * fn_);
    g_terms.push_back(-ft_ - mu_c_ * fn_);
    casadi::SX g_vec = casadi::SX::vertcat(g_terms);

    I_func_ =
        casadi::Function("I_constraints", {state_, control_}, {g_vec}, {"state", "control"}, {"I"});
}

casadi::DM PushCircle::dynamics(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    return dyna_func_(arg)[0];
}

casadi::DM PushCircle::getG(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    return G_func_(arg)[0];
}

casadi::DM PushCircle::getH(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    return H_func_(arg)[0];
}

}  // namespace push_circle
