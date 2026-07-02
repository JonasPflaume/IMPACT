#include "push_t.h"

namespace push_t {

PushT::PushT()
    : m_(0.1),      // mass of the T-block
      g_(9.8),      // gravitational acceleration
      mu_(0.4),     // friction coefficient
      c_(0.4),      // limit surface parameter
      l_(0.05),     // characteristic length (half side)
      dc_(2.6429),  // limit surface rotational parameter
      dt_(0.05)     // time step
{
    r_ = 2.8 * l_;  // contact radius constraint
    // r_ = 0.2 * l_;  // tuned for real robot experiments

    // Define symbolic state variables
    px_ = casadi::SX::sym("px");
    py_ = casadi::SX::sym("py");
    theta_ = casadi::SX::sym("theta");

    // Define symbolic control variables
    cx_ = casadi::SX::sym("cx");
    cy_ = casadi::SX::sym("cy");
    lam_ = casadi::SX::sym("lam", 8);  // 8 friction force components (signed)
    v_ = casadi::SX::sym("v", 7);      // absolute value slack
    w_ = casadi::SX::sym("w", 7);      // absolute value slack

    // Build state and control vectors
    state_ = casadi::SX::vertcat({px_, py_, theta_});
    control_ = casadi::SX::vertcat({cx_, cy_, lam_, v_, w_});

    // Build the dynamics and constraints
    buildDynamics();
    buildComplementarityConstraints();
    buildEqualityConstraints();
    buildInequalityConstraints();
}

void PushT::buildDynamics() {
    // Build the dynamics model matching Python exactly:
    // sum_x = lam2 + lam4 + lam6 + lam8 (1-based indices 2,4,6,8 = 0-based 1,3,5,7)
    // sum_y = lam1 + lam3 + lam5 + lam7 (1-based indices 1,3,5,7 = 0-based 0,2,4,6)
    //
    // dpx = constant * (sum_x * cos(theta) - sum_y * sin(theta))
    // dpy = constant * (sum_x * sin(theta) + sum_y * cos(theta))
    // dtheta = constant / (c * r) * (-cy * sum_x + cx * sum_y)

    double constant = 1.0 / (mu_ * m_ * g_);

    // lam[0]=lam1, lam[1]=lam2, ..., lam[7]=lam8
    casadi::SX sum_x = lam_(1) + lam_(3) + lam_(5) + lam_(7);  // lam2 + lam4 + lam6 + lam8
    casadi::SX sum_y = lam_(0) + lam_(2) + lam_(4) + lam_(6);  // lam1 + lam3 + lam5 + lam7

    casadi::SX cos_th = casadi::SX::cos(theta_);
    casadi::SX sin_th = casadi::SX::sin(theta_);

    casadi::SX dpx = constant * (sum_x * cos_th - sum_y * sin_th);
    casadi::SX dpy = constant * (sum_x * sin_th + sum_y * cos_th);
    casadi::SX dtheta = constant / (c_ * r_) * (-cy_ * sum_x + cx_ * sum_y);

    casadi::SX f = casadi::SX::vertcat({dpx, dpy, dtheta});

    // Create the dynamics function
    dyna_func_ = casadi::Function("dynamics", {state_, control_}, {f}, {"state", "control"}, {"f"});
}

void PushT::buildComplementarityConstraints() {
    // Build the complementarity constraints matching Python exactly
    //
    // Python code:
    // lam_mag = cs.vertcat(-lam1, -lam2, lam3, -lam4, lam5, lam6, lam7, lam8)  # all >=0
    //
    // abs1 = v[0] + w[0], ..., abs7 = v[6] + w[6]
    //
    // gap = cs.vertcat(
    //     (4-dc)*l - cy,
    //     abs1 + abs2 + abs3 - 1.0*l,
    //     abs1 + abs3 + abs4 - 1.5*l,
    //     abs3 + abs4 + abs5 - 3.0*l,
    //     abs4 + abs5 + abs6 - 1.0*l,
    //     abs3 + abs5 + abs6 - 3.0*l,
    //     abs3 + abs6 + abs7 - 1.5*l,
    //     abs2 + abs3 + abs7 - 1.0*l
    // )
    //
    // G_abs = v, H_abs = w (7 pairs)
    // G_contact = lam_mag, H_contact = gap (8 pairs)
    // G_pairs, H_pairs = mutual exclusion (28 pairs)

    // Compute absolute values: abs[i] = v[i] + w[i]
    std::vector<casadi::SX> abs_vals(7);
    for (int i = 0; i < 7; ++i) {
        abs_vals[i] = v_(i) + w_(i);
    }

    // Gap functions (8 components) - matching Python exactly
    casadi::SX gap0 = (4 - dc_) * l_ - cy_;
    casadi::SX gap1 = abs_vals[0] + abs_vals[1] + abs_vals[2] - 1.0 * l_;
    casadi::SX gap2 = abs_vals[0] + abs_vals[2] + abs_vals[3] - 1.5 * l_;
    casadi::SX gap3 = abs_vals[2] + abs_vals[3] + abs_vals[4] - 3.0 * l_;
    casadi::SX gap4 = abs_vals[3] + abs_vals[4] + abs_vals[5] - 1.0 * l_;
    casadi::SX gap5 = abs_vals[2] + abs_vals[4] + abs_vals[5] - 3.0 * l_;
    casadi::SX gap6 = abs_vals[2] + abs_vals[5] + abs_vals[6] - 1.5 * l_;
    casadi::SX gap7 = abs_vals[1] + abs_vals[2] + abs_vals[6] - 1.0 * l_;

    casadi::SX gap = casadi::SX::vertcat({gap0, gap1, gap2, gap3, gap4, gap5, gap6, gap7});

    // lam_mag: force magnitudes (all should be >= 0)
    // Python: lam_mag = cs.vertcat(-lam1, -lam2, lam3, -lam4, lam5, lam6, lam7, lam8)
    // In 0-based: -lam[0], -lam[1], lam[2], -lam[3], lam[4], lam[5], lam[6], lam[7]
    casadi::SX lam_mag = casadi::SX::vertcat(
        {-lam_(0), -lam_(1), lam_(2), -lam_(3), lam_(4), lam_(5), lam_(6), lam_(7)});

    std::vector<casadi::SX> G_terms;
    std::vector<casadi::SX> H_terms;

    // (A) v perp w (7 pairs)
    for (int i = 0; i < 7; ++i) {
        G_terms.push_back(v_(i));
        H_terms.push_back(w_(i));
    }

    // (B) lam_mag perp gap (8 pairs)
    for (int i = 0; i < 8; ++i) {
        G_terms.push_back(lam_mag(i));
        H_terms.push_back(gap(i));
    }

    // (C) Mutual exclusion: lam_mag[i] perp lam_mag[j] for i < j (28 pairs)
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            G_terms.push_back(lam_mag(i));
            H_terms.push_back(lam_mag(j));
        }
    }

    casadi::SX G_vec = casadi::SX::vertcat(G_terms);
    casadi::SX H_vec = casadi::SX::vertcat(H_terms);

    // Create the G and H functions
    G_func_ =
        casadi::Function("G_matrix", {state_, control_}, {G_vec}, {"state", "control"}, {"G"});
    H_func_ =
        casadi::Function("H_matrix", {state_, control_}, {H_vec}, {"state", "control"}, {"H"});
}

void PushT::buildEqualityConstraints() {
    // Build equality constraints matching Python exactly:
    // E = cs.vertcat(
    //     (v[0] - w[0]) - (cx - 2*l),
    //     (v[1] - w[1]) - (cy - (4-dc)*l),
    //     (v[2] - w[2]) - (cy - (3-dc)*l),
    //     (v[3] - w[3]) - (cx - 0.5*l),
    //     (v[4] - w[4]) - (cy + dc*l),
    //     (v[5] - w[5]) - (cx + 0.5*l),
    //     (v[6] - w[6]) - (cx + 2*l),
    // )

    std::vector<casadi::SX> E_terms;

    E_terms.push_back((v_(0) - w_(0)) - (cx_ - 2 * l_));
    E_terms.push_back((v_(1) - w_(1)) - (cy_ - (4 - dc_) * l_));
    E_terms.push_back((v_(2) - w_(2)) - (cy_ - (3 - dc_) * l_));
    E_terms.push_back((v_(3) - w_(3)) - (cx_ - 0.5 * l_));
    E_terms.push_back((v_(4) - w_(4)) - (cy_ + dc_ * l_));
    E_terms.push_back((v_(5) - w_(5)) - (cx_ + 0.5 * l_));
    E_terms.push_back((v_(6) - w_(6)) - (cx_ + 2 * l_));

    casadi::SX E_vec = casadi::SX::vertcat(E_terms);

    // Create the equality constraint function
    E_func_ =
        casadi::Function("E_constraints", {state_, control_}, {E_vec}, {"state", "control"}, {"E"});
}

void PushT::buildInequalityConstraints() {
    // Build inequality constraints in g <= 0 convention
    // Original formulation was I >= 0, we negate to get g <= 0:
    //   cx >= -2*l       =>  -2*l - cx <= 0
    //   cx <= 2*l        =>  cx - 2*l <= 0
    //   cy >= -dc*l      =>  -dc*l - cy <= 0
    //   cy <= (4-dc)*l   =>  cy - (4-dc)*l <= 0

    std::vector<casadi::SX> g_terms;

    // Contact point bounds (4 constraints)
    g_terms.push_back(-2 * l_ - cx_);         // cx >= -2*l  =>  -2*l - cx <= 0
    g_terms.push_back(cx_ - 2 * l_);          // cx <= 2*l   =>  cx - 2*l <= 0
    g_terms.push_back(-dc_ * l_ - cy_);       // cy >= -dc*l =>  -dc*l - cy <= 0
    g_terms.push_back(cy_ - (4 - dc_) * l_);  // cy <= (4-dc)*l => cy - (4-dc)*l <= 0

    casadi::SX g_vec = casadi::SX::vertcat(g_terms);

    // Create the inequality constraint function (g <= 0 convention)
    I_func_ =
        casadi::Function("I_constraints", {state_, control_}, {g_vec}, {"state", "control"}, {"I"});
}

std::vector<double> PushT::getControlLowerBounds() const {
    // No explicit bounds - constraints are handled via I_func
    std::vector<double> lb(24, -casadi::inf);
    return lb;
}

std::vector<double> PushT::getControlUpperBounds() const {
    // No explicit bounds - constraints are handled via I_func
    std::vector<double> ub(24, casadi::inf);
    return ub;
}

casadi::DM PushT::dynamics(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = dyna_func_(arg);
    return res[0];
}

casadi::DM PushT::getG(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = G_func_(arg);
    return res[0];
}

casadi::DM PushT::getH(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = H_func_(arg);
    return res[0];
}

}  // namespace push_t
