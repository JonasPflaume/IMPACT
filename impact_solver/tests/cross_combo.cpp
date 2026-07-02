// Smoke test for mixing task formulations and shooting transcriptions. Both
// builders accept the StageProblem adapter layer, so the combinations
// {MPCC, LCP} x {single, multiple} should at least build and solve to finite
// results in these small cases.
//
//   - box (an MPCCProblem) is solved with single shooting
//   - allegro (an LCPProblem) is built and solved with multiple shooting

#include <iostream>
#include <memory>
#include <vector>

#include "box_pushing.h"
#include "impact/bcd_aula_solver.h"
#include "impact/lcp_stage.h"
#include "impact/mpcc_stage.h"
#include "impact/mpcc_subproblem.h"
#include "impact/multiple_shooting.h"
#include "impact/single_shooting.h"
#include "penalty_solver/allegro_lcp_problem.h"

namespace {
int g_fail = 0;
void check(bool ok, const std::string& msg) {
    std::cout << (ok ? "[ PASS ] " : "[ FAIL ] ") << msg << std::endl;
    if (!ok) ++g_fail;
}
}  // namespace

// MPCC task (box) via single shooting.
static void boxSingleShooting() {
    auto problem = std::make_shared<box_pushing::BoxPushing>();
    impact::BCDAULAConfig c;
    c.horizon = 50;
    c.x_0 = Eigen::VectorXd::Zero(3);
    c.x_goal = (Eigen::VectorXd(3) << 0.1, 0.1, 1.0).finished();
    c.stage_cost_weight = 0.001;
    c.final_cost_weight = 100.0;
    c.rho_max = 100.0;
    c.rho_scale = 1.1;
    c.dynamics_scale = 25.0;  // unused (no defect in single shooting) but harmless
    c.comp_scale = 1.0;
    c.max_outer_iters = 500;
    c.outer_tol_h = 1e-5;
    c.outer_tol_comp = 1e-5;
    c.max_inner_iters = 50;
    c.newton_max_iter = 100;
    c.newton_regularization = 2e-5;
    c.use_saddle = true;
    c.print_level = 0;

    impact::MPCCStage stage(problem, c);
    impact::SingleShootingLayout layout = impact::buildSingleShooting(stage, c);
    impact::AulaSubproblem& sub = *layout.sub;
    sub.setParamValue(layout.off_x0, c.x_0);
    sub.setParamValue(layout.off_p, c.x_goal);
    sub.syncParams();

    const int nu = problem->getControlDim();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(sub.numOpt());

    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(sub, c, z);

    // Roll the state out to read the achieved terminal state.
    Eigen::VectorXd x = c.x_0;
    const double dt = problem->getTimeStep();
    for (int k = 0; k < c.horizon; ++k) {
        Eigen::VectorXd u = r.z.segment(k * nu, nu);
        casadi::DM xd(std::vector<double>(x.data(), x.data() + x.size()));
        casadi::DM ud(std::vector<double>(u.data(), u.data() + u.size()));
        casadi::DM f = problem->dynamics(xd, ud);
        x += dt * Eigen::Map<const Eigen::VectorXd>(f.ptr(), f.numel());
    }
    const double goal_err = (x - c.x_goal).norm();
    std::cout << "  box single-shooting: converged=" << (r.converged ? "Y" : "N")
              << " obj=" << r.objective_value << " outer=" << r.outer_iterations
              << " final=[" << x.transpose() << "] goal_err=" << goal_err << "\n";
    check(r.converged && goal_err < 5e-2,
          "MPCC task (box) solves via SINGLE shooting and reaches the goal");
}

// LCP task (allegro) via multiple shooting.
static void allegroMultipleShooting() {
    auto problem = std::make_shared<allegro_lcp::AllegroLCPProblem>();
    impact::BCDAULAConfig c;
    c.horizon = 4;
    c.dynamics_scale = 25.0;
    c.comp_scale = 1.0;
    c.rho_max = 1e3;
    c.rho_scale = 5.0;
    c.max_outer_iters = 5;
    c.max_inner_iters = 3;
    c.outer_tol_h = 1e-3;
    c.outer_tol_comp = 1e-3;
    c.outer_tol_g = 1e-3;
    c.newton_max_iter = 20;
    c.use_saddle = true;
    c.use_cmd_bounds = true;
    c.cmd_lower = Eigen::VectorXd::Constant(16, -0.1);
    c.cmd_upper = Eigen::VectorXd::Constant(16, 0.1);
    c.print_level = 0;

    impact::LCPStage stage(problem, c);
    impact::MultipleShootingLayout layout = impact::buildMultipleShooting(stage, c);
    impact::AulaSubproblem& sub = *layout.sub;

    const int nx = problem->getConfigDim();
    const int nqvel = problem->getVelocityDim();
    const int n_lam = problem->getMaxContacts() * 4;

    // A valid initial configuration + synthetic-but-finite contact data.
    Eigen::VectorXd q0(nx);
    q0.setZero();
    q0(2) = 0.04;  // object height
    q0(3) = 1.0;   // unit quaternion w
    Eigen::VectorXd p(3 + 4 + n_lam + n_lam * nqvel);
    p.setZero();
    p(2) = 0.05;  // target_p z
    p(3 + 0) = 1.0;                       // target_q w
    p.segment(7, n_lam).setConstant(0.001);  // phi
    sub.setParamValue(layout.off_x0, q0);
    sub.setParamValue(layout.off_p, p);
    sub.syncParams();

    // z = [vec(X); vec(U)]: states = q0 repeated, controls = 0.
    Eigen::VectorXd z = Eigen::VectorXd::Zero(sub.numOpt());
    for (int k = 0; k <= c.horizon; ++k) z.segment(k * nx, nx) = q0;

    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(sub, c, z);
    std::cout << "  allegro multiple-shooting: outer=" << r.outer_iterations
              << " obj=" << r.objective_value << " dyn=" << r.dynamics_violation
              << " comp=" << r.complementarity_violation << "\n";
    check(r.z.allFinite() && std::isfinite(r.objective_value),
          "LCP task (allegro) builds + solves via MULTIPLE shooting (finite result)");
}

// A generic, non-trajectory MPCC solved directly with buildMPCC().
//   minimize (z1 - 1)^2 + (z2 - 1)^2   s.t.   0 <= z1 ⊥ z2 >= 0
// The complementarity forces one of z1, z2 to zero, so the optimum is (1, 0) or
// (0, 1) with objective 1.
static void genericMPCC() {
    using casadi::SX;
    SX z = SX::sym("z", 2);
    impact::MPCCDescription desc;
    desc.z = z;
    desc.p = SX::sym("p", 0);  // no runtime parameters
    desc.cost = z - 1.0;       // ||z - 1||^2
    desc.cost_is_linear = true;
    desc.addComplementarityBlock("comp", z(0), z(1), {1.0, 1.0, 1e-8});

    impact::MPCCSubproblem m = impact::buildMPCC(desc);
    impact::BCDAULAConfig c;
    c.max_outer_iters = 300;
    c.max_inner_iters = 50;
    c.rho_scale = 1.5;
    c.rho_max = 1e8;
    c.outer_tol_h = 1e-7;
    c.outer_tol_comp = 1e-7;
    c.newton_max_iter = 50;
    c.use_saddle = true;
    c.print_level = 0;

    Eigen::VectorXd z0(2);
    z0 << 0.5, 0.5;
    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(*m.sub, c, z0);
    const double comp = std::abs(r.z(0) * r.z(1));
    std::cout << "  generic MPCC  min||z-1||^2 s.t. 0<=z1 perp z2>=0:  z=[" << r.z.transpose()
              << "] obj=" << r.objective_value << " |z1*z2|=" << comp << "\n";
    check(r.converged && comp < 1e-5 && std::abs(r.objective_value - 1.0) < 1e-2,
          "generic (non-trajectory) MPCC solved directly via buildMPCC");
}

// Same generic MPCC through the classical use_saddle=false backend.
// Guards the normal-equations Gauss-Newton path end-to-end through BCDAULASolver:
// every other full-solver test uses the saddle backend, so without this the
// opt-out default path could regress unnoticed.
static void genericMPCCNoSaddle() {
    using casadi::SX;
    SX z = SX::sym("z", 2);
    impact::MPCCDescription desc;
    desc.z = z;
    desc.p = SX::sym("p", 0);
    desc.cost = z - 1.0;
    desc.cost_is_linear = true;
    desc.addComplementarityBlock("comp", z(0), z(1), {1.0, 1.0, 1e-8});

    impact::MPCCSubproblem m = impact::buildMPCC(desc);
    impact::BCDAULAConfig c;
    c.max_outer_iters = 300;
    c.max_inner_iters = 50;
    c.rho_scale = 1.5;
    c.rho_max = 1e8;
    c.outer_tol_h = 1e-7;
    c.outer_tol_comp = 1e-7;
    c.newton_max_iter = 50;
    c.use_saddle = false;  // <-- classical normal-equations backend
    c.print_level = 0;

    Eigen::VectorXd z0(2);
    z0 << 0.5, 0.5;
    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(*m.sub, c, z0);
    const double comp = std::abs(r.z(0) * r.z(1));
    std::cout << "  generic MPCC (no-saddle backend):  z=[" << r.z.transpose()
              << "] obj=" << r.objective_value << " |z1*z2|=" << comp
              << " status=" << static_cast<int>(r.status) << "\n";
    check(r.converged && r.status == impact::BCDAULAStatus::Converged && comp < 1e-5 &&
              std::abs(r.objective_value - 1.0) < 1e-2,
          "generic MPCC solved via the classical use_saddle=false backend");
}

// Direct MPCC with two independent complementarity groups.
// Guards the generic buildMPCC path that cannot be represented by the old single
// CompBlock metadata without silently dropping one group's AuLa state.
static void genericMPCCMultiComp() {
    using casadi::SX;
    SX z = SX::sym("z", 4);
    SX target = casadi::DM(std::vector<double>{0.8, 0.2, 0.25, 0.75});
    impact::MPCCDescription desc;
    desc.z = z;
    desc.p = SX::sym("p", 0);
    desc.cost = z - target;
    desc.cost_is_linear = true;
    desc.addComplementarityBlock("axis_a", z(0), z(1), {1.0, 1.0, 1e-8});
    desc.addComplementarityBlock("axis_b", z(2), z(3), {0.75, 2.0, 1e-8});

    impact::MPCCSubproblem m = impact::buildMPCC(desc);
    impact::BCDAULAConfig c;
    c.max_outer_iters = 300;
    c.max_inner_iters = 50;
    c.rho_scale = 1.5;
    c.rho_max = 1e8;
    c.outer_tol_h = 1e-7;
    c.outer_tol_comp = 1e-7;
    c.newton_max_iter = 50;
    c.use_saddle = true;
    c.print_level = 0;

    Eigen::VectorXd z0 = Eigen::VectorXd::Zero(4);
    impact::BCDAULASolver solver;
    impact::BCDAULAResult r = solver.solve(*m.sub, c, z0);
    const double comp0 = std::abs(r.z(0) * r.z(1));
    const double comp1 = std::abs(r.z(2) * r.z(3));
    std::cout << "  generic MPCC (two comp groups): z=[" << r.z.transpose()
              << "] obj=" << r.objective_value << " comp=[" << comp0 << ", " << comp1
              << "]\n";
    check(r.converged && comp0 < 1e-5 && comp1 < 1e-5 &&
              std::abs(r.objective_value - 0.1025) < 1e-2,
          "generic MPCC solved with two independent complementarity blocks");
}

int main() {
    std::cout << "=== cross-combo: shooting is orthogonal to formulation ===" << std::endl;
    boxSingleShooting();
    allegroMultipleShooting();
    genericMPCC();
    genericMPCCNoSaddle();
    genericMPCCMultiComp();
    std::cout << (g_fail == 0 ? "ALL PASS" : (std::to_string(g_fail) + " FAILED")) << std::endl;
    return g_fail == 0 ? 0 : 1;
}
