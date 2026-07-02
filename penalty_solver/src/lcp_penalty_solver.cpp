#include "penalty_solver/lcp_penalty_solver.h"

#include <iostream>

namespace aula {

LCPPenaltySolver::LCPPenaltySolver(std::shared_ptr<LCPProblem> problem,
                                   const LCPPenaltyConfig& config)
    : problem_(problem), config_(config) {
    buildSolver();
}

void LCPPenaltySolver::buildSolver() {
    int n_qpos = problem_->getConfigDim();
    int n_qvel = problem_->getVelocityDim();
    int n_cmd = problem_->getCommandDim();
    int max_ncon = problem_->getMaxContacts();
    int horizon = config_.horizon;
    double h = problem_->getTimeStep();

    // Decision variables per step: [cmd, lam, vel, q_next]
    // lam has max_ncon * 4 components (normal + 3 friction directions per contact)
    int n_lam = max_ncon * 4;
    vars_per_step_ = n_cmd + n_lam + n_qvel + n_qpos;
    total_vars_ = horizon * vars_per_step_;

    casadi::SX w = casadi::SX::sym("w", total_vars_);

    // Parameters: [q0, phi_vec, jac_mat_flat, target_p, target_q]
    int jac_size = n_lam * n_qvel;
    param_size_ = n_qpos + n_lam + jac_size + 3 + 4;
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

    // Convert to CasADi
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

    // Cost weights
    double control_weight = problem_->getControlCostWeight();
    double contact_weight = problem_->getContactCostWeight();
    double grasp_weight = problem_->getGraspClosureWeight();
    double vel_penalty = problem_->getVelocityPenalty();
    double pos_weight = problem_->getPositionCostWeight();
    double orient_weight = problem_->getOrientationCostWeight();
    double final_mult = problem_->getFinalCostMultiplier();

    // Objective and constraints
    casadi::SX J = 0;
    std::vector<casadi::SX> g_vec;
    lbg_.clear();
    ubg_.clear();

    casadi::SX qk = q0_sx;
    casadi::SX vel_last;

    for (int i = 0; i < horizon; ++i) {
        int offset = i * vars_per_step_;

        // Extract decision variables
        casadi::SX cmd = w(casadi::Slice(offset, offset + n_cmd));
        casadi::SX lam = w(casadi::Slice(offset + n_cmd, offset + n_cmd + n_lam));
        casadi::SX vel = w(casadi::Slice(offset + n_cmd + n_lam, offset + n_cmd + n_lam + n_qvel));
        casadi::SX q_next = w(casadi::Slice(offset + n_cmd + n_lam + n_qvel,
                                            offset + n_cmd + n_lam + n_qvel + n_qpos));

        vel_last = vel;

        // Kinematics constraint: q_next = integrate(qk, vel).
        // Object position
        casadi::SX obj_pos = qk(casadi::Slice(0, 3));
        casadi::SX obj_quat = qk(casadi::Slice(3, 7));
        casadi::SX robot_qpos = qk(casadi::Slice(7, n_qpos));

        casadi::SX vel_obj_lin = vel(casadi::Slice(0, 3));
        casadi::SX vel_obj_ang = vel(casadi::Slice(3, 6));
        casadi::SX vel_robot = vel(casadi::Slice(6, n_qvel));

        casadi::SX next_obj_pos = obj_pos + h * vel_obj_lin;
        casadi::SX next_robot_qpos = robot_qpos + h * vel_robot;

        // Quaternion integration: q_new = q + 0.5 * h * H(q)^T * omega
        casadi::SX H_q_body = casadi::SX::vertcat(
            {casadi::SX::horzcat({-obj_quat(1), obj_quat(0), obj_quat(3), -obj_quat(2)}),
             casadi::SX::horzcat({-obj_quat(2), -obj_quat(3), obj_quat(0), obj_quat(1)}),
             casadi::SX::horzcat({-obj_quat(3), obj_quat(2), -obj_quat(1), obj_quat(0)})});
        casadi::SX next_obj_quat =
            obj_quat + 0.5 * h * casadi::SX::mtimes(H_q_body.T(), vel_obj_ang);

        casadi::SX pred_q = casadi::SX::vertcat({next_obj_pos, next_obj_quat, next_robot_qpos});

        g_vec.push_back(pred_q - q_next);
        for (int j = 0; j < n_qpos; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(0.0);
        }

        // LCP dynamics constraint: h * Q * vel = b + J^T * lam.
        // Build bias vector: b = b_gravity + stiffness * cmd
        casadi::SX b = b_gravity_sx;
        // Add stiffness contribution (robot joints)
        for (int j = 0; j < n_cmd; ++j) {
            b(6 + j) = b(6 + j) + Q_sx(6 + j, 6 + j) * cmd(j);
        }

        casadi::SX lcp_lhs = casadi::SX::mtimes(Q_sx, vel);
        casadi::SX lcp_rhs = b / h + casadi::SX::mtimes(jac_sx.T(), lam);
        casadi::SX lcp_eq = lcp_lhs - lcp_rhs;

        g_vec.push_back(lcp_eq);
        for (int j = 0; j < n_qvel; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(0.0);
        }

        // Complementarity constraints.
        // H = lam >= 0
        casadi::SX H = lam;
        // G = J @ vel + phi / h >= 0
        casadi::SX G = casadi::SX::mtimes(jac_sx, vel) + phi_sx / h;

        // H >= 0
        g_vec.push_back(H);
        for (int j = 0; j < n_lam; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(casadi::inf);
        }

        // G >= 0
        g_vec.push_back(G);
        for (int j = 0; j < n_lam; ++j) {
            lbg_.push_back(0.0);
            ubg_.push_back(casadi::inf);
        }

        // Stage cost.
        // Complementarity penalty
        J += config_.complementarity_penalty * casadi::SX::sumsqr(H * G);

        // Velocity penalty
        J += config_.velocity_penalty * casadi::SX::sumsqr(vel(casadi::Slice(0, 6)));

        // Control cost
        J += control_weight * casadi::SX::sumsqr(cmd);

        // Contact cost (fingertips close to object) - assumes 3 fingertips at indices 7-16
        casadi::SX obj_pos_next = q_next(casadi::Slice(0, 3));
        casadi::SX obj_quat_next = q_next(casadi::Slice(3, 7));
        casadi::SX ftp0 = q_next(casadi::Slice(7, 10));
        casadi::SX ftp1 = q_next(casadi::Slice(10, 13));
        casadi::SX ftp2 = q_next(casadi::Slice(13, 16));

        J += contact_weight *
             (casadi::SX::sumsqr(obj_pos_next - ftp0) + casadi::SX::sumsqr(obj_pos_next - ftp1) +
              casadi::SX::sumsqr(obj_pos_next - ftp2));

        // Grasp closure cost
        casadi::SX dcm = casadi::SX::vertcat(
            {casadi::SX::horzcat(
                 {1 - 2 * (obj_quat_next(2) * obj_quat_next(2) +
                           obj_quat_next(3) * obj_quat_next(3)),
                  2 * (obj_quat_next(1) * obj_quat_next(2) - obj_quat_next(0) * obj_quat_next(3)),
                  2 * (obj_quat_next(1) * obj_quat_next(3) + obj_quat_next(0) * obj_quat_next(2))}),
             casadi::SX::horzcat(
                 {2 * (obj_quat_next(1) * obj_quat_next(2) + obj_quat_next(0) * obj_quat_next(3)),
                  1 - 2 * (obj_quat_next(1) * obj_quat_next(1) +
                           obj_quat_next(3) * obj_quat_next(3)),
                  2 * (obj_quat_next(2) * obj_quat_next(3) - obj_quat_next(0) * obj_quat_next(1))}),
             casadi::SX::horzcat(
                 {2 * (obj_quat_next(1) * obj_quat_next(3) - obj_quat_next(0) * obj_quat_next(2)),
                  2 * (obj_quat_next(2) * obj_quat_next(3) + obj_quat_next(0) * obj_quat_next(1)),
                  1 - 2 * (obj_quat_next(1) * obj_quat_next(1) +
                           obj_quat_next(2) * obj_quat_next(2))})});

        casadi::SX v0 = casadi::SX::mtimes(dcm.T(), ftp0 - obj_pos_next);
        casadi::SX v1 = casadi::SX::mtimes(dcm.T(), ftp1 - obj_pos_next);
        casadi::SX v2 = casadi::SX::mtimes(dcm.T(), ftp2 - obj_pos_next);

        casadi::SX grasp_sum =
            v0 / casadi::SX::norm_2(v0) + v1 / casadi::SX::norm_2(v1) + v2 / casadi::SX::norm_2(v2);
        J += grasp_weight * casadi::SX::sumsqr(grasp_sum);

        // Update qk for next iteration
        qk = q_next;
    }

    // Terminal cost.
    casadi::SX obj_pos_final = qk(casadi::Slice(0, 3));
    casadi::SX obj_quat_final = qk(casadi::Slice(3, 7));

    casadi::SX pos_cost = casadi::SX::sumsqr(obj_pos_final - target_p_sx);
    casadi::SX quat_dot = casadi::SX::dot(obj_quat_final, target_q_sx);
    casadi::SX quat_cost = 1.0 - quat_dot * quat_dot;

    casadi::SX terminal_vel_cost = casadi::SX::sumsqr(vel_last(casadi::Slice(0, 6)));

    J += final_mult * (pos_weight * pos_cost + orient_weight * quat_cost +
                       config_.velocity_penalty * terminal_vel_cost);

    // Concatenate constraints
    casadi::SX g = casadi::SX::vertcat(g_vec);

    // Variable bounds
    double vel_lb = problem_->getVelocityLowerBound();
    double vel_ub = problem_->getVelocityUpperBound();
    double q_lb = problem_->getConfigLowerBound();
    double q_ub = problem_->getConfigUpperBound();
    double cmd_lb = problem_->getControlLowerBound();
    double cmd_ub = problem_->getControlUpperBound();

    lbw_.resize(total_vars_);
    ubw_.resize(total_vars_);
    for (int i = 0; i < horizon; ++i) {
        int offset = i * vars_per_step_;

        // cmd bounds
        for (int j = 0; j < n_cmd; ++j) {
            lbw_[offset + j] = cmd_lb;
            ubw_[offset + j] = cmd_ub;
        }
        // lam bounds (unbounded, positivity enforced via constraint)
        for (int j = 0; j < n_lam; ++j) {
            lbw_[offset + n_cmd + j] = -casadi::inf;
            ubw_[offset + n_cmd + j] = casadi::inf;
        }
        // vel bounds
        for (int j = 0; j < n_qvel; ++j) {
            lbw_[offset + n_cmd + n_lam + j] = vel_lb;
            ubw_[offset + n_cmd + n_lam + j] = vel_ub;
        }
        // q_next bounds
        for (int j = 0; j < n_qpos; ++j) {
            lbw_[offset + n_cmd + n_lam + n_qvel + j] = q_lb;
            ubw_[offset + n_cmd + n_lam + n_qvel + j] = q_ub;
        }
    }

    // Create NLP and solver
    casadi::SXDict nlp = {{"x", w}, {"f", J}, {"g", g}, {"p", p}};

    casadi::Dict opts;
    opts["ipopt.print_level"] = config_.ipopt_print_level;
    opts["ipopt.max_iter"] = config_.ipopt_max_iter;
    opts["ipopt.tol"] = config_.ipopt_tol;
    opts["ipopt.sb"] = "yes";
    opts["print_time"] = config_.print_time ? 1 : 0;

    solver_ = casadi::nlpsol("lcp_solver", "ipopt", nlp, opts);

    if (config_.print_time) {
        std::cout << "LCP Penalty Solver built: " << total_vars_ << " variables, " << lbg_.size()
                  << " constraints" << std::endl;
    }
}

LCPSolution LCPPenaltySolver::solve(const Eigen::VectorXd& q0, const Eigen::VectorXd& phi_vec,
                                    const Eigen::MatrixXd& jac_mat, const Eigen::Vector3d& target_p,
                                    const Eigen::Vector4d& target_q,
                                    const std::vector<double>* warm_start) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int n_qpos = problem_->getConfigDim();
    int n_qvel = problem_->getVelocityDim();
    int n_cmd = problem_->getCommandDim();
    int n_lam = problem_->getMaxContacts() * 4;

    // Prepare parameter vector (column-major for jac_mat)
    std::vector<double> p_val(param_size_);
    int idx = 0;
    for (int i = 0; i < n_qpos; ++i) p_val[idx++] = q0(i);
    for (int i = 0; i < n_lam; ++i) p_val[idx++] = phi_vec(i);
    // Column-major flattening of jac_mat
    for (int j = 0; j < n_qvel; ++j) {
        for (int i = 0; i < n_lam; ++i) {
            p_val[idx++] = jac_mat(i, j);
        }
    }
    for (int i = 0; i < 3; ++i) p_val[idx++] = target_p(i);
    for (int i = 0; i < 4; ++i) p_val[idx++] = target_q(i);

    // Initial guess
    std::vector<double> x0;
    if (warm_start && warm_start->size() == static_cast<size_t>(total_vars_)) {
        x0 = *warm_start;
    } else {
        x0.resize(total_vars_, 0.0);
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

    std::map<std::string, casadi::DM> sol = solver_(arg);

    auto end_time = std::chrono::high_resolution_clock::now();

    // Extract solution
    casadi::Dict stats = solver_.stats();
    casadi::DM x_sol = sol["x"];

    LCPSolution result;
    result.action = Eigen::VectorXd::Zero(n_cmd);
    for (int i = 0; i < n_cmd; ++i) {
        result.action(i) = static_cast<double>(x_sol(i));
    }

    result.full_solution.resize(total_vars_);
    for (int i = 0; i < total_vars_; ++i) {
        result.full_solution[i] = static_cast<double>(x_sol(i));
    }

    result.cost = static_cast<double>(sol["f"]);
    result.success = (stats["success"].to_int() == 1);
    result.status = stats["return_status"].to_string();
    result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

}  // namespace aula
