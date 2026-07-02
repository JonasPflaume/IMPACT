/**
 * @file allegro_lcp_penalty_solver.cpp
 * @brief Allegro LCP Penalty Solver Implementation
 *
 * This implements the same optimization as Python test_lcp_ipopt.py:
 * - LCP dynamics with penalty-based complementarity
 * - Quaternion integration matching Python exactly
 * - Cost functions from Python init_cost_fns() with full FK
 *
 * Forward kinematics for Allegro hand fingertips is implemented to match
 * the Python allegro_fkin.py exactly.
 */

#include "penalty_solver/allegro_lcp_penalty_solver.h"

#include <iostream>

namespace allegro_lcp {

// Helper functions for CasADi transformations.

// Translation matrix (4x4 homogeneous)
casadi::SX ttmat(double x, double y, double z) {
    casadi::SX T = casadi::SX::eye(4);
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
}

// Quaternion to rotation matrix (DCM) then to 4x4 homogeneous transform
// q = [w, x, y, z]
casadi::SX quattmat(double w, double x, double y, double z) {
    // Normalize
    double n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n;
    x /= n;
    y /= n;
    z /= n;

    casadi::SX T = casadi::SX::zeros(4, 4);
    T(0, 0) = 1 - 2 * (y * y + z * z);
    T(0, 1) = 2 * (x * y - w * z);
    T(0, 2) = 2 * (x * z + w * y);
    T(1, 0) = 2 * (x * y + w * z);
    T(1, 1) = 1 - 2 * (x * x + z * z);
    T(1, 2) = 2 * (y * z - w * x);
    T(2, 0) = 2 * (x * z - w * y);
    T(2, 1) = 2 * (y * z + w * x);
    T(2, 2) = 1 - 2 * (x * x + y * y);
    T(3, 3) = 1;
    return T;
}

// Rotation around X axis (4x4 homogeneous) - symbolic
casadi::SX rxtmat(const casadi::SX& alpha) {
    casadi::SX T = casadi::SX::eye(4);
    T(1, 1) = casadi::SX::cos(alpha);
    T(1, 2) = -casadi::SX::sin(alpha);
    T(2, 1) = casadi::SX::sin(alpha);
    T(2, 2) = casadi::SX::cos(alpha);
    return T;
}

// Rotation around Y axis (4x4 homogeneous) - symbolic
casadi::SX rytmat(const casadi::SX& beta) {
    casadi::SX T = casadi::SX::eye(4);
    T(0, 0) = casadi::SX::cos(beta);
    T(0, 2) = casadi::SX::sin(beta);
    T(2, 0) = -casadi::SX::sin(beta);
    T(2, 2) = casadi::SX::cos(beta);
    return T;
}

// Rotation around Z axis (4x4 homogeneous) - symbolic
casadi::SX rztmat(const casadi::SX& theta) {
    casadi::SX T = casadi::SX::eye(4);
    T(0, 0) = casadi::SX::cos(theta);
    T(0, 1) = -casadi::SX::sin(theta);
    T(1, 0) = casadi::SX::sin(theta);
    T(1, 1) = casadi::SX::cos(theta);
    return T;
}

// Extract position from 4x4 homogeneous transform
casadi::SX extractPos(const casadi::SX& T) {
    return casadi::SX::vertcat({T(0, 3), T(1, 3), T(2, 3)});
}

AllegroLCPPenaltySolver::AllegroLCPPenaltySolver(std::shared_ptr<AllegroLCPProblem> problem,
                                                 const AllegroLCPPenaltyConfig& config)
    : problem_(problem), config_(config) {
    buildSolver();
}

void AllegroLCPPenaltySolver::buildSolver() {
    int n_qpos = problem_->getConfigDim();      // 23
    int n_qvel = problem_->getVelocityDim();    // 22
    int n_cmd = problem_->getCommandDim();      // 16
    int max_ncon = problem_->getMaxContacts();  // 20
    int horizon = config_.horizon;              // 3
    double h = problem_->getTimeStep();         // 0.1

    // Decision variables per step: [cmd(16), lam(80), vel(22), q_next(23)]
    int n_lam = max_ncon * 4;                          // 80
    vars_per_step_ = n_cmd + n_lam + n_qvel + n_qpos;  // 16 + 80 + 22 + 23 = 141
    total_vars_ = horizon * vars_per_step_;

    // Constraints per step: [integration(23), H>=0(80), G>=0(80), dynamics(22)]
    g_per_step_ = n_qpos + n_lam + n_lam + n_qvel;  // 23 + 80 + 80 + 22 = 205

    casadi::SX w = casadi::SX::sym("w", total_vars_);

    // Parameters: [q0(23), phi_vec(80), jac_mat_flat(80*22), target_p(3), target_q(4)]
    int jac_size = n_lam * n_qvel;                    // 80 * 22 = 1760
    param_size_ = n_qpos + n_lam + jac_size + 3 + 4;  // 23 + 80 + 1760 + 3 + 4 = 1870
    casadi::SX p = casadi::SX::sym("p", param_size_);

    // Extract parameters
    int idx = 0;
    casadi::SX q0_sx = p(casadi::Slice(idx, idx + n_qpos));
    idx += n_qpos;
    casadi::SX phi_sx = p(casadi::Slice(idx, idx + n_lam));
    idx += n_lam;
    casadi::SX jac_flat = p(casadi::Slice(idx, idx + jac_size));
    idx += jac_size;
    casadi::SX jac_sx = casadi::SX::reshape(jac_flat, n_lam, n_qvel);
    casadi::SX target_p_sx = p(casadi::Slice(idx, idx + 3));
    idx += 3;
    casadi::SX target_q_sx = p(casadi::Slice(idx, idx + 4));

    // Get problem matrices
    Eigen::MatrixXd Q_eigen = problem_->getInertiaMatrix();
    Eigen::VectorXd b_gravity_eigen = problem_->getGravityBias();
    Eigen::MatrixXd robot_stiff_eigen = problem_->getRobotStiffness();

    // Convert Q to CasADi
    casadi::SX Q_sx = casadi::SX::zeros(n_qvel, n_qvel);
    for (int i = 0; i < n_qvel; ++i) {
        for (int j = 0; j < n_qvel; ++j) {
            Q_sx(i, j) = Q_eigen(i, j);
        }
    }

    casadi::SX b_gravity_sx = casadi::SX::zeros(n_qvel, 1);
    for (int i = 0; i < n_qvel; ++i) {
        b_gravity_sx(i) = b_gravity_eigen(i);
    }

    // Robot stiffness for dynamics (from Python: b_r = robot_stiff_ @ cmd)
    casadi::SX robot_stiff_sx = casadi::SX::zeros(n_cmd, n_cmd);
    for (int i = 0; i < n_cmd; ++i) {
        for (int j = 0; j < n_cmd; ++j) {
            robot_stiff_sx(i, j) = robot_stiff_eigen(i, j);
        }
    }

    // Cost weights from problem (matching Python init_cost_fns)
    double control_weight = problem_->getControlCostWeight();     // 0.1
    double contact_weight = problem_->getContactCostWeight();     // 1.0
    double pos_weight = problem_->getPositionCostWeight();        // 1.0
    double orient_weight = problem_->getOrientationCostWeight();  // 0.05
    double final_mult = problem_->getFinalCostMultiplier();       // 10.0
    double final_pos_weight = 100.0;   // From Python: 100 * position_cost
    double final_orient_weight = 5.0;  // From Python: 5.0 * quaternion_cost

    // Complementarity penalty (from Python: comp_penalty = 1e4)
    double comp_penalty = config_.complementarity_penalty;

    // Constant palm transform.
    // t_palm = quattmat_fn(np.array([0, 1, 0, 1]) / np.linalg.norm([0, 1, 0, 1]))
    // norm = sqrt(0 + 1 + 0 + 1) = sqrt(2), so q = [0, 1/sqrt(2), 0, 1/sqrt(2)]
    double sq2 = std::sqrt(2.0);
    casadi::SX t_palm = quattmat(0.0, 1.0 / sq2, 0.0, 1.0 / sq2);

    // Objective and constraints
    casadi::SX J = 0;
    std::vector<casadi::SX> g_vec;
    lbg_.clear();
    ubg_.clear();

    casadi::SX qk = q0_sx;

    for (int i = 0; i < horizon; ++i) {
        int offset = i * vars_per_step_;

        // Extract decision variables
        casadi::SX cmd = w(casadi::Slice(offset, offset + n_cmd));
        casadi::SX lam = w(casadi::Slice(offset + n_cmd, offset + n_cmd + n_lam));
        casadi::SX vel = w(casadi::Slice(offset + n_cmd + n_lam, offset + n_cmd + n_lam + n_qvel));
        casadi::SX q_next = w(casadi::Slice(offset + n_cmd + n_lam + n_qvel,
                                            offset + n_cmd + n_lam + n_qvel + n_qpos));

        // Kinematics constraint: q_next = integrate(qk, vel).
        casadi::SX obj_pos = qk(casadi::Slice(0, 3));
        casadi::SX obj_quat = qk(casadi::Slice(3, 7));
        casadi::SX robot_qpos = qk(casadi::Slice(7, n_qpos));

        casadi::SX vel_obj_lin = vel(casadi::Slice(0, 3));
        casadi::SX vel_obj_ang = vel(casadi::Slice(3, 6));
        casadi::SX vel_robot = vel(casadi::Slice(6, n_qvel));

        casadi::SX next_obj_pos = obj_pos + h * vel_obj_lin;
        casadi::SX next_robot_qpos = robot_qpos + h * vel_robot;

        // Quaternion integration from Python
        casadi::SX H_q_body = casadi::SX::vertcat(
            {casadi::SX::horzcat({-obj_quat(1), obj_quat(0), obj_quat(3), -obj_quat(2)}),
             casadi::SX::horzcat({-obj_quat(2), -obj_quat(3), obj_quat(0), obj_quat(1)}),
             casadi::SX::horzcat({-obj_quat(3), obj_quat(2), -obj_quat(1), obj_quat(0)})});
        casadi::SX next_obj_quat =
            obj_quat + 0.5 * h * casadi::SX::mtimes(H_q_body.T(), vel_obj_ang);

        casadi::SX pred_q = casadi::SX::vertcat({next_obj_pos, next_obj_quat, next_robot_qpos});

        // Integration constraint: pred_q - q_next = 0
        g_vec.push_back(pred_q - q_next);
        for (int j = 0; j < n_qpos; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(0.0);
        }

        // Complementarity constraints.
        casadi::SX Ht = lam;
        casadi::SX Gt = casadi::SX::mtimes(jac_sx, vel) + phi_sx / h;

        // H >= 0 (non-negative contact force)
        g_vec.push_back(Ht);
        for (int j = 0; j < n_lam; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(casadi::inf);
        }

        // G >= 0 (non-penetration)
        g_vec.push_back(Gt);
        for (int j = 0; j < n_lam; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(casadi::inf);
        }

        // LCP dynamics constraint.
        casadi::SX b_robot = casadi::SX::mtimes(robot_stiff_sx, cmd);
        casadi::SX b = casadi::SX::vertcat({b_gravity_sx(casadi::Slice(0, 6)), b_robot});

        casadi::SX dynamics_eq =
            casadi::SX::mtimes(Q_sx, vel) - b / h - casadi::SX::mtimes(jac_sx.T(), lam);
        g_vec.push_back(dynamics_eq);
        for (int j = 0; j < n_qvel; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(0.0);
        }

        // Stage cost.
        // Complementarity penalty
        J += comp_penalty * casadi::SX::sumsqr(Ht * Gt);

        // Extract robot joint positions from q_next for FK
        // Python: ff_qpos = x[7:11], mf_qpos = x[11:15], rf_qpos = x[15:19], tm_qpos = x[19:23]
        casadi::SX ff_qpos = q_next(casadi::Slice(7, 11));   // First finger
        casadi::SX mf_qpos = q_next(casadi::Slice(11, 15));  // Middle finger
        casadi::SX rf_qpos = q_next(casadi::Slice(15, 19));  // Ring finger
        casadi::SX th_qpos = q_next(casadi::Slice(19, 23));  // Thumb

        // Forward kinematics for each fingertip.
        // First finger (ff)
        // ff_t_base = t_palm @ ttmat([0, 0.0435, -0.001542]) @ quattmat([0.999048, -0.0436194, 0,
        // 0])
        casadi::SX ff_t_base =
            casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(0, 0.0435, -0.001542)),
                               quattmat(0.999048, -0.0436194, 0, 0));
        casadi::SX ff_t_proximal = casadi::SX::mtimes(
            casadi::SX::mtimes(ff_t_base, rztmat(ff_qpos(0))), ttmat(0, 0, 0.0164));
        casadi::SX ff_t_medial = casadi::SX::mtimes(
            casadi::SX::mtimes(ff_t_proximal, rytmat(ff_qpos(1))), ttmat(0, 0, 0.054));
        casadi::SX ff_t_distal = casadi::SX::mtimes(
            casadi::SX::mtimes(ff_t_medial, rytmat(ff_qpos(2))), ttmat(0, 0, 0.0384));
        casadi::SX ff_t_ftp = casadi::SX::mtimes(
            casadi::SX::mtimes(ff_t_distal, rytmat(ff_qpos(3))), ttmat(0, 0, 0.0384));
        casadi::SX ftp_1_position = extractPos(ff_t_ftp);

        // Middle finger (mf)
        // mf_t_base = t_palm @ ttmat([0, 0, 0.0007])
        casadi::SX mf_t_base = casadi::SX::mtimes(t_palm, ttmat(0, 0, 0.0007));
        casadi::SX mf_t_proximal = casadi::SX::mtimes(
            casadi::SX::mtimes(mf_t_base, rztmat(mf_qpos(0))), ttmat(0, 0, 0.0164));
        casadi::SX mf_t_medial = casadi::SX::mtimes(
            casadi::SX::mtimes(mf_t_proximal, rytmat(mf_qpos(1))), ttmat(0, 0, 0.054));
        casadi::SX mf_t_distal = casadi::SX::mtimes(
            casadi::SX::mtimes(mf_t_medial, rytmat(mf_qpos(2))), ttmat(0, 0, 0.0384));
        casadi::SX mf_t_ftp = casadi::SX::mtimes(
            casadi::SX::mtimes(mf_t_distal, rytmat(mf_qpos(3))), ttmat(0, 0, 0.0384));
        casadi::SX ftp_2_position = extractPos(mf_t_ftp);

        // Ring finger (rf)
        // rf_t_base = t_palm @ ttmat([0, -0.0435, -0.001542]) @ quattmat([0.999048, 0.0436194, 0,
        // 0])
        casadi::SX rf_t_base =
            casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(0, -0.0435, -0.001542)),
                               quattmat(0.999048, 0.0436194, 0, 0));
        casadi::SX rf_t_proximal = casadi::SX::mtimes(
            casadi::SX::mtimes(rf_t_base, rztmat(rf_qpos(0))), ttmat(0, 0, 0.0164));
        casadi::SX rf_t_medial = casadi::SX::mtimes(
            casadi::SX::mtimes(rf_t_proximal, rytmat(rf_qpos(1))), ttmat(0, 0, 0.054));
        casadi::SX rf_t_distal = casadi::SX::mtimes(
            casadi::SX::mtimes(rf_t_medial, rytmat(rf_qpos(2))), ttmat(0, 0, 0.0384));
        casadi::SX rf_t_ftp = casadi::SX::mtimes(
            casadi::SX::mtimes(rf_t_distal, rytmat(rf_qpos(3))), ttmat(0, 0, 0.0384));
        casadi::SX ftp_3_position = extractPos(rf_t_ftp);

        // Thumb (th)
        // th_t_base = t_palm @ ttmat([-0.0182, 0.019333, -0.045987]) @ quattmat([0.477714,
        // -0.521334, -0.521334, -0.477714])
        casadi::SX th_t_base =
            casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(-0.0182, 0.019333, -0.045987)),
                               quattmat(0.477714, -0.521334, -0.521334, -0.477714));
        // th_t_proximal = th_t_base @ rxtmat(-th_qpos[0]) @ ttmat([-0.027, 0.005, 0.0399])
        casadi::SX th_t_proximal = casadi::SX::mtimes(
            casadi::SX::mtimes(th_t_base, rxtmat(-th_qpos(0))), ttmat(-0.027, 0.005, 0.0399));
        casadi::SX th_t_medial = casadi::SX::mtimes(
            casadi::SX::mtimes(th_t_proximal, rztmat(th_qpos(1))), ttmat(0, 0, 0.0177));
        casadi::SX th_t_distal = casadi::SX::mtimes(
            casadi::SX::mtimes(th_t_medial, rytmat(th_qpos(2))), ttmat(0, 0, 0.0514));
        casadi::SX th_t_ftp = casadi::SX::mtimes(
            casadi::SX::mtimes(th_t_distal, rytmat(th_qpos(3))), ttmat(0, 0, 0.054));
        casadi::SX ftp_4_position = extractPos(th_t_ftp);

        // Object position from q_next
        casadi::SX obj_pos_next = q_next(casadi::Slice(0, 3));
        casadi::SX obj_quat_next = q_next(casadi::Slice(3, 7));

        // Position cost
        casadi::SX position_cost = casadi::SX::sumsqr(obj_pos_next - target_p_sx);

        // Quaternion cost (from Python: 1 - dot(obj_quat, target_q)^2)
        casadi::SX quat_dot = casadi::SX::dot(obj_quat_next, target_q_sx);
        casadi::SX quaternion_cost = 1.0 - quat_dot * quat_dot;

        // Contact cost: sum of distances from fingertips to object center
        // From Python: contact_cost = sumsqr(obj_pose[0:3] - ftp_i_position) for each fingertip
        casadi::SX contact_cost = casadi::SX::sumsqr(obj_pos_next - ftp_1_position) +
                                  casadi::SX::sumsqr(obj_pos_next - ftp_2_position) +
                                  casadi::SX::sumsqr(obj_pos_next - ftp_3_position) +
                                  casadi::SX::sumsqr(obj_pos_next - ftp_4_position);

        // Control cost
        casadi::SX control_cost = casadi::SX::sumsqr(cmd);

        // Base path cost (from Python: base_cost = 1*contact + 1*pos + 0.05*quat)
        // path_cost_fn = base_cost + 0.1*control
        casadi::SX path_cost = contact_weight * contact_cost + pos_weight * position_cost +
                               orient_weight * quaternion_cost + control_weight * control_cost;
        J += path_cost;

        // Update qk for next iteration
        qk = q_next;
    }

    // Terminal cost.
    // From Python: final_cost = 100 * position_cost + 5.0 * quaternion_cost
    //              final_cost_fn = 10 * final_cost
    casadi::SX obj_pos_final = qk(casadi::Slice(0, 3));
    casadi::SX obj_quat_final = qk(casadi::Slice(3, 7));

    casadi::SX final_pos_cost = casadi::SX::sumsqr(obj_pos_final - target_p_sx);
    casadi::SX final_quat_dot = casadi::SX::dot(obj_quat_final, target_q_sx);
    casadi::SX final_quat_cost = 1.0 - final_quat_dot * final_quat_dot;

    J += final_mult * (final_pos_weight * final_pos_cost + final_orient_weight * final_quat_cost);

    // Concatenate constraints
    casadi::SX g = casadi::SX::vertcat(g_vec);

    // Variable bounds (from Python build_ipopt_solver)
    double cmd_lb = problem_->getControlLowerBound();   // -0.05
    double cmd_ub = problem_->getControlUpperBound();   // 0.05
    double vel_lb = problem_->getVelocityLowerBound();  // -100.0
    double vel_ub = problem_->getVelocityUpperBound();  // 100.0
    double q_lb = problem_->getConfigLowerBound();      // -100.0
    double q_ub = problem_->getConfigUpperBound();      // 100.0

    lbw_.resize(total_vars_);
    ubw_.resize(total_vars_);
    w0_.resize(total_vars_, 0.0);

    for (int i = 0; i < horizon; ++i) {
        int offset = i * vars_per_step_;

        // cmd bounds [-0.05, 0.05]
        for (int j = 0; j < n_cmd; ++j) {
            lbw_[offset + j] = cmd_lb;
            ubw_[offset + j] = cmd_ub;
            w0_[offset + j] = 0.0;
        }

        // lam bounds (from Python: lblam = -inf, ublam = inf)
        for (int j = 0; j < n_lam; ++j) {
            lbw_[offset + n_cmd + j] = -casadi::inf;
            ubw_[offset + n_cmd + j] = casadi::inf;
            w0_[offset + n_cmd + j] = 0.0;
        }

        // vel bounds [-100, 100]
        for (int j = 0; j < n_qvel; ++j) {
            lbw_[offset + n_cmd + n_lam + j] = vel_lb;
            ubw_[offset + n_cmd + n_lam + j] = vel_ub;
            w0_[offset + n_cmd + n_lam + j] = 0.0;
        }

        // q_next bounds [-100, 100]
        for (int j = 0; j < n_qpos; ++j) {
            lbw_[offset + n_cmd + n_lam + n_qvel + j] = q_lb;
            ubw_[offset + n_cmd + n_lam + n_qvel + j] = q_ub;
        }
    }

    // Create NLP and solver
    casadi::SXDict nlp = {{"x", w}, {"f", J}, {"g", g}, {"p", p}};

    // IPOPT options from Python
    casadi::Dict opts;
    opts["ipopt.print_level"] = config_.ipopt_print_level;
    opts["ipopt.sb"] = "yes";
    opts["print_time"] = config_.print_time ? 1 : 0;
    opts["ipopt.max_iter"] = config_.ipopt_max_iter;

    // Warm start options (from Python)
    if (config_.warm_start) {
        opts["ipopt.warm_start_init_point"] = "yes";
        opts["ipopt.warm_start_bound_push"] = 1e-9;
        opts["ipopt.warm_start_bound_frac"] = 1e-9;
        opts["ipopt.warm_start_slack_bound_frac"] = 1e-9;
        opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
        opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
    }

    // Convergence tolerances (from Python)
    opts["ipopt.tol"] = config_.ipopt_tol;
    opts["ipopt.acceptable_tol"] = config_.acceptable_tol;
    opts["ipopt.acceptable_iter"] = config_.acceptable_iter;

    solver_ = casadi::nlpsol("allegro_lcp_solver", "ipopt", nlp, opts);

    if (config_.print_time) {
        std::cout << "Allegro LCP Penalty Solver built: " << total_vars_ << " variables, "
                  << lbg_.size() << " constraints" << std::endl;
    }
}

AllegroLCPSolution AllegroLCPPenaltySolver::solve(const Eigen::VectorXd& q0,
                                                  const Eigen::VectorXd& phi_vec,
                                                  const Eigen::MatrixXd& jac_mat,
                                                  const Eigen::Vector3d& target_p,
                                                  const Eigen::Vector4d& target_q,
                                                  const AllegroWarmStart* warm_start) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int n_qpos = problem_->getConfigDim();
    int n_qvel = problem_->getVelocityDim();
    int n_cmd = problem_->getCommandDim();
    int n_lam = problem_->getMaxContacts() * 4;

    // Prepare parameter vector
    std::vector<double> p_val(param_size_);
    int idx = 0;

    // q0 (23)
    for (int i = 0; i < n_qpos; ++i) p_val[idx++] = q0(i);

    // phi_vec (80)
    for (int i = 0; i < n_lam; ++i) p_val[idx++] = phi_vec(i);

    // jac_mat flattened column-major (80 x 22 = 1760)
    for (int j = 0; j < n_qvel; ++j) {
        for (int i = 0; i < n_lam; ++i) {
            p_val[idx++] = jac_mat(i, j);
        }
    }

    // target_p (3)
    for (int i = 0; i < 3; ++i) p_val[idx++] = target_p(i);

    // target_q (4)
    for (int i = 0; i < 4; ++i) p_val[idx++] = target_q(i);

    // Initial guess
    std::vector<double> x0;
    if (warm_start && warm_start->valid &&
        warm_start->primal.size() == static_cast<size_t>(total_vars_)) {
        x0 = warm_start->primal;
    } else {
        // Use default initial guess, set q_next to q0
        x0 = w0_;
        for (int i = 0; i < config_.horizon; ++i) {
            int offset = i * vars_per_step_;
            for (int j = 0; j < n_qpos; ++j) {
                x0[offset + n_cmd + n_lam + n_qvel + j] = q0(j);
            }
        }
    }

    // Solve
    std::map<std::string, casadi::DM> arg;
    arg["x0"] = x0;
    arg["lbx"] = lbw_;
    arg["ubx"] = ubw_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    arg["p"] = p_val;

    // Add dual warm start if available
    if (warm_start && warm_start->valid && config_.warm_start) {
        if (warm_start->dual_g.size() == lbg_.size()) {
            arg["lam_g0"] = warm_start->dual_g;
        }
        if (warm_start->dual_x.size() == static_cast<size_t>(total_vars_)) {
            arg["lam_x0"] = warm_start->dual_x;
        }
    }

    std::map<std::string, casadi::DM> sol = solver_(arg);

    auto end_time = std::chrono::high_resolution_clock::now();

    // Extract solution
    casadi::Dict stats = solver_.stats();
    casadi::DM x_sol = sol["x"];
    casadi::DM lam_g_sol = sol["lam_g"];
    casadi::DM lam_x_sol = sol["lam_x"];

    AllegroLCPSolution result;

    // Extract action (first command)
    result.action = Eigen::VectorXd::Zero(n_cmd);
    for (int i = 0; i < n_cmd; ++i) {
        result.action(i) = static_cast<double>(x_sol(i));
    }

    // Store full solution for warm start
    result.full_solution.resize(total_vars_);
    for (int i = 0; i < total_vars_; ++i) {
        result.full_solution[i] = static_cast<double>(x_sol(i));
    }

    // Store dual variables for warm start
    result.lam_g.resize(lbg_.size());
    for (size_t i = 0; i < lbg_.size(); ++i) {
        result.lam_g[i] = static_cast<double>(lam_g_sol(i));
    }

    result.lam_x.resize(total_vars_);
    for (int i = 0; i < total_vars_; ++i) {
        result.lam_x[i] = static_cast<double>(lam_x_sol(i));
    }

    result.cost = static_cast<double>(sol["f"]);
    result.success = (stats["success"].to_int() == 1);
    result.status = stats["return_status"].to_string();
    result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

AllegroWarmStart AllegroLCPPenaltySolver::timeShiftWarmStart(
    const AllegroLCPSolution& current_sol) {
    AllegroWarmStart shifted;
    int horizon = config_.horizon;

    // Time-shift primal variables: [step1, step2, ..., stepN] -> [step2, step3, ..., stepN, stepN]
    shifted.primal.resize(total_vars_, 0.0);
    for (int i = 0; i < horizon - 1; ++i) {
        int src_start = (i + 1) * vars_per_step_;
        int dst_start = i * vars_per_step_;
        for (int j = 0; j < vars_per_step_; ++j) {
            shifted.primal[dst_start + j] = current_sol.full_solution[src_start + j];
        }
    }
    // Last step: copy from previous last step
    int last_start = (horizon - 1) * vars_per_step_;
    int prev_last_start = (horizon - 2) * vars_per_step_;
    if (horizon > 1) {
        for (int j = 0; j < vars_per_step_; ++j) {
            shifted.primal[last_start + j] = current_sol.full_solution[prev_last_start + j];
        }
    }

    // Time-shift constraint dual variables
    shifted.dual_g.resize(lbg_.size(), 0.0);
    for (int i = 0; i < horizon - 1; ++i) {
        int src_start = (i + 1) * g_per_step_;
        int dst_start = i * g_per_step_;
        for (int j = 0; j < g_per_step_; ++j) {
            if (src_start + j < static_cast<int>(current_sol.lam_g.size()) &&
                dst_start + j < static_cast<int>(shifted.dual_g.size())) {
                shifted.dual_g[dst_start + j] = current_sol.lam_g[src_start + j];
            }
        }
    }
    // Last step
    int g_last_start = (horizon - 1) * g_per_step_;
    int g_prev_last_start = (horizon - 2) * g_per_step_;
    if (horizon > 1) {
        for (int j = 0; j < g_per_step_; ++j) {
            if (g_prev_last_start + j < static_cast<int>(current_sol.lam_g.size()) &&
                g_last_start + j < static_cast<int>(shifted.dual_g.size())) {
                shifted.dual_g[g_last_start + j] = current_sol.lam_g[g_prev_last_start + j];
            }
        }
    }

    // Time-shift bound dual variables
    shifted.dual_x.resize(total_vars_, 0.0);
    for (int i = 0; i < horizon - 1; ++i) {
        int src_start = (i + 1) * vars_per_step_;
        int dst_start = i * vars_per_step_;
        for (int j = 0; j < vars_per_step_; ++j) {
            if (src_start + j < static_cast<int>(current_sol.lam_x.size())) {
                shifted.dual_x[dst_start + j] = current_sol.lam_x[src_start + j];
            }
        }
    }
    if (horizon > 1) {
        for (int j = 0; j < vars_per_step_; ++j) {
            if (prev_last_start + j < static_cast<int>(current_sol.lam_x.size())) {
                shifted.dual_x[last_start + j] = current_sol.lam_x[prev_last_start + j];
            }
        }
    }

    shifted.valid = true;
    return shifted;
}

}  // namespace allegro_lcp
