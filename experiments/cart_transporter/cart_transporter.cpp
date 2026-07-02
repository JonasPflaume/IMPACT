#include "cart_transporter.h"

namespace cart_transporter {

CartTransporter::CartTransporter()
    : m1_(0.1),  // cargo mass
      m2_(0.2),  // cart mass
      mu_(0.2),  // friction coefficient
      g_(9.81),  // gravity
      l_(1.0),   // half gap length
      dt_(0.02)  // time step (matching original problem)
{
    // Define symbolic state variables
    x1_ = casadi::SX::sym("x1");          // cargo position
    x2_ = casadi::SX::sym("x2");          // cart position
    x1_dot_ = casadi::SX::sym("x1_dot");  // cargo velocity
    x2_dot_ = casadi::SX::sym("x2_dot");  // cart velocity

    // Define symbolic control variables
    f_ = casadi::SX::sym("f");  // friction force
    u_ = casadi::SX::sym("u");  // control force
    v_ = casadi::SX::sym("v");  // slack for positive relative velocity
    w_ = casadi::SX::sym("w");  // slack for negative relative velocity

    // Build state and control vectors
    state_ = casadi::SX::vertcat({x1_, x2_, x1_dot_, x2_dot_});
    control_ = casadi::SX::vertcat({f_, u_, v_, w_});

    // Build all constraints
    buildDynamics();
    buildComplementarityConstraints();
    buildEqualityConstraints();
    buildInequalityConstraints();
}

void CartTransporter::buildDynamics() {
    // System dynamics:
    // dx1/dt = x1_dot
    // dx2/dt = x2_dot
    // dx1_dot/dt = f / m1
    // dx2_dot/dt = (u - f) / m2

    casadi::SX dx1 = x1_dot_;
    casadi::SX dx2 = x2_dot_;
    casadi::SX dx1_dot = f_ / m1_;
    casadi::SX dx2_dot = (u_ - f_) / m2_;

    casadi::SX f = casadi::SX::vertcat({dx1, dx2, dx1_dot, dx2_dot});

    // Create the dynamics function
    dyna_func_ = casadi::Function("dynamics", {state_, control_}, {f}, {"state", "control"}, {"f"});
}

void CartTransporter::buildComplementarityConstraints() {
    // Complementarity constraints: 0 <= G ⊥ H >= 0
    //
    // 3 complementarity pairs:
    // 1. v ⊥ w : can't slide in both directions
    // 2. w ⊥ (mu*m1*g - f) : if sliding backward (w > 0), friction at upper limit
    // 3. v ⊥ (f + mu*m1*g) : if sliding forward (v > 0), friction at lower limit

    double mu_m1_g = mu_ * m1_ * g_;

    casadi::SX G_term = casadi::SX::vertcat({
        v_,  // [0] v >= 0
        w_,  // [1] w >= 0 (for pair with friction upper bound)
        v_   // [2] v >= 0 (for pair with friction lower bound)
    });

    casadi::SX H_term = casadi::SX::vertcat({
        w_,            // [0] w >= 0 (pair with v)
        mu_m1_g - f_,  // [1] mu*m1*g - f >= 0 (friction upper bound)
        f_ + mu_m1_g   // [2] f + mu*m1*g >= 0 (friction lower bound)
    });

    G_func_ =
        casadi::Function("G_matrix", {state_, control_}, {G_term}, {"state", "control"}, {"G"});
    H_func_ =
        casadi::Function("H_matrix", {state_, control_}, {H_term}, {"state", "control"}, {"H"});
}

void CartTransporter::buildEqualityConstraints() {
    // Equality constraint: x1_dot - x2_dot = v - w
    // Rearranged: x1_dot - x2_dot - v + w = 0

    casadi::SX eq = x1_dot_ - x2_dot_ - v_ + w_;

    E_func_ = casadi::Function("equality", {state_, control_}, {eq}, {"state", "control"}, {"eq"});
}

void CartTransporter::buildInequalityConstraints() {
    // Inequality constraints: g(x, u) <= 0
    //
    // 1. -(mu*m1*g - f) <= 0  =>  f <= mu*m1*g  (friction upper bound)
    // 2. -(f + mu*m1*g) <= 0  =>  f >= -mu*m1*g (friction lower bound)
    // 3. -(x1 - x2 + l) <= 0  =>  x1 - x2 >= -l (gap lower)
    // 4. -(l - (x1 - x2)) <= 0 => x1 - x2 <= l  (gap upper)

    double mu_m1_g = mu_ * m1_ * g_;

    casadi::SX ineq = casadi::SX::vertcat({
        -(mu_m1_g - f_),     // f <= mu*m1*g
        -(f_ + mu_m1_g),     // f >= -mu*m1*g
        -(x1_ - x2_ + l_),   // x1 - x2 >= -l
        -(l_ - (x1_ - x2_))  // x1 - x2 <= l
    });

    I_func_ =
        casadi::Function("inequality", {state_, control_}, {ineq}, {"state", "control"}, {"ineq"});
}

std::vector<double> CartTransporter::getControlLowerBounds() const {
    // Control: [f, u, v, w]
    // f: unbounded (handled by inequality constraints)
    // u: unbounded (control force)
    // v, w: >= 0 (slack variables)
    return {-casadi::inf, -casadi::inf, 0.0, 0.0};
}

std::vector<double> CartTransporter::getControlUpperBounds() const {
    // Control: [f, u, v, w]
    return {casadi::inf, casadi::inf, casadi::inf, casadi::inf};
}

casadi::DM CartTransporter::dynamics(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = dyna_func_(arg);
    return res[0];
}

casadi::DM CartTransporter::getG(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = G_func_(arg);
    return res[0];
}

casadi::DM CartTransporter::getH(const casadi::DM& x, const casadi::DM& u) const {
    std::vector<casadi::DM> arg = {x, u};
    std::vector<casadi::DM> res = H_func_(arg);
    return res[0];
}

}  // namespace cart_transporter
