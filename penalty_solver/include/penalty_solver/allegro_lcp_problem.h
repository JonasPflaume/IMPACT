#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include "impact/lcp_problem.h"

namespace allegro_lcp {

/**
 * @brief Allegro hand cube manipulation problem as an LCP problem
 *
 * This defines the 4-finger Allegro hand cube manipulation problem with:
 * - Configuration: [obj_pos(3), obj_quat(4), robot_qpos(16)] = 23D
 * - Velocity: [obj_lin_vel(3), obj_ang_vel(3), robot_vel(16)] = 22D
 * - Command: joint angle increment for 16 DOF Allegro hand = 16D
 * - Contacts: up to 20 contacts, each with 4 directions (normal + 3 friction)
 *
 * Parameters matched exactly from Python ipopt_param.py:
 * - h_ = 0.1
 * - obj_inertia: 50*I for linear, 0.1*I for angular
 * - robot_stiff: 1.0 * I_16
 * - obj_mass = 0.01
 * - mu_object = 0.5
 * - mpc_horizon = 3
 * - ipopt_max_iter = 50
 * - complementarity_penalty = 1e4
 */
class AllegroLCPProblem : public impact::LCPProblem {
   public:
    struct Parameters {
        // System dimensions (from Python ipopt_param.py)
        int n_qpos = 23;    // [obj_pos(3), obj_quat(4), robot_qpos(16)]
        int n_qvel = 22;    // [obj_lin_vel(3), obj_ang_vel(3), robot_vel(16)]
        int n_cmd = 16;     // Robot command dimension (16 DOF Allegro hand)
        int max_ncon = 20;  // Maximum contacts

        // Time step (from Python: h_ = 0.1)
        double h = 0.1;

        // Physical parameters (from Python ipopt_param.py)
        double obj_mass = 0.01;
        Eigen::Vector<double, 6> gravity =
            (Eigen::Vector<double, 6>() << 0.0, 0.0, -9.8, 0.0, 0.0, 0.0).finished();

        // Inertia matrices (from Python ipopt_param.py)
        // obj_inertia_[0:3, 0:3] = 50 * eye(3)
        // obj_inertia_[3:6, 3:6] = 0.1 * eye(3)
        Eigen::Matrix<double, 6, 6> obj_inertia;

        // robot_stiff_ = np.diag(n_cmd_ * [1]) = I_16
        Eigen::MatrixXd robot_stiff;

        // Cost weights (from Python init_cost_fns)
        // Python: base_cost = 1 * contact_cost + 1 * position_cost + 0.05 * quaternion_cost
        //         final_cost = 100 * position_cost + 5.0 * quaternion_cost
        //         path_cost_fn = base_cost + 0.1 * control_cost
        //         final_cost_fn = 10 * final_cost
        double velocity_penalty = 1.0;         // From Python: velocity_penalty = 1.0
        double position_cost_weight = 1.0;     // From Python: 1 * position_cost in base
        double quaternion_cost_weight = 0.05;  // From Python: 0.05 * quaternion_cost in base
        double contact_cost_weight = 1.0;      // From Python: 1 * contact_cost in base
        double grasp_closure_weight =
            0.0;  // Python doesn't have explicit grasp closure in this version
        double control_cost_weight = 0.1;     // From Python: 0.1 * control_cost
        double final_cost_multiplier = 10.0;  // From Python: 10 * final_cost

        // Final cost weights (Python: 100 * position_cost + 5.0 * quaternion_cost)
        double final_position_weight = 100.0;
        double final_quaternion_weight = 5.0;

        // Control bounds (from Python: mpc_u_lb_ = -0.1, mpc_u_ub_ = 0.1)
        // But in build_ipopt_solver: lbu = -0.05, ubu = 0.05
        double control_lb = -0.05;
        double control_ub = 0.05;

        // Velocity bounds (from Python: lbvel = -100.0, ubvel = 100.0)
        double velocity_lb = -100.0;
        double velocity_ub = 100.0;

        Parameters() {
            // Initialize obj_inertia (from Python ipopt_param.py)
            obj_inertia.setIdentity();
            obj_inertia.block<3, 3>(0, 0) = 50.0 * Eigen::Matrix3d::Identity();
            obj_inertia.block<3, 3>(3, 3) = 0.1 * Eigen::Matrix3d::Identity();

            // Initialize robot_stiff (from Python: np.diag(n_cmd_ * [1]))
            robot_stiff = Eigen::MatrixXd::Identity(16, 16);
        }
    };

    explicit AllegroLCPProblem(const Parameters& param = Parameters());

    // LCPProblem interface.
    int getConfigDim() const override { return param_.n_qpos; }
    int getVelocityDim() const override { return param_.n_qvel; }
    int getCommandDim() const override { return param_.n_cmd; }
    int getMaxContacts() const override { return param_.max_ncon; }
    double getTimeStep() const override { return param_.h; }

    Eigen::MatrixXd getInertiaMatrix() const override { return Q_; }
    Eigen::VectorXd getGravityBias() const override { return b_gravity_; }

    double getControlCostWeight() const override { return param_.control_cost_weight; }
    double getContactCostWeight() const override { return param_.contact_cost_weight; }
    double getGraspClosureWeight() const override { return param_.grasp_closure_weight; }
    double getVelocityPenalty() const override { return param_.velocity_penalty; }
    double getPositionCostWeight() const override { return param_.position_cost_weight; }
    double getOrientationCostWeight() const override { return param_.quaternion_cost_weight; }
    double getFinalCostMultiplier() const override { return param_.final_cost_multiplier; }

    Eigen::MatrixXd getRobotStiffness() const override { return param_.robot_stiff; }

    // Control bounds (from Python: lbu = -0.05, ubu = 0.05)
    double getControlLowerBound() const override { return param_.control_lb; }
    double getControlUpperBound() const override { return param_.control_ub; }

    // Velocity bounds (from Python: lbvel = -100.0, ubvel = 100.0)
    double getVelocityLowerBound() const override { return param_.velocity_lb; }
    double getVelocityUpperBound() const override { return param_.velocity_ub; }

    // Config bounds (from Python: lbq = -1e2, ubq = 1e2)
    double getConfigLowerBound() const override { return -100.0; }
    double getConfigUpperBound() const override { return 100.0; }

    // CasADi functions (not used by LCPPenaltySolver, it builds its own)
    casadi::Function getKinematicsFunction() const override { return casadi::Function(); }
    casadi::Function getLCPDynamicsFunction() const override { return casadi::Function(); }
    casadi::Function getStageCostFunction() const override { return casadi::Function(); }
    casadi::Function getTerminalCostFunction() const override { return casadi::Function(); }

    // Fingertip kinematics.
    int getNumFingertips() const override { return 4; }  // Allegro has 4 fingers

    /**
     * @brief Compute fingertip positions from configuration via Forward Kinematics
     * @param q_sx CasADi symbolic configuration [obj_pos(3), obj_quat(4), robot_joints(16)]
     * @return CasADi symbolic vector of 4 fingertip positions (12D)
     */
    casadi::SX computeFingertipPositionsSX(const casadi::SX& q_sx) const override;

    // State integration.
    /**
     * @brief Integrate state forward (handles quaternion properly)
     * @param q Current configuration [obj_pos(3), obj_quat(4), robot_qpos(16)]
     * @param vel Velocity [obj_lin_vel(3), obj_ang_vel(3), robot_vel(16)]
     * @param dt Time step
     * @return Next configuration
     */
    Eigen::VectorXd integrateState(const Eigen::VectorXd& q, const Eigen::VectorXd& vel,
                                   double dt) const override {
        Eigen::VectorXd q_next(param_.n_qpos);

        // Object position integration: p_next = p + dt * v_lin
        q_next.segment<3>(0) = q.segment<3>(0) + dt * vel.segment<3>(0);

        // Quaternion integration (from Python IPOPTMPCModel):
        // H(q)^T maps angular velocity to quaternion derivative
        // next_obj_quat = quat + 0.5 * h * H(q)^T * omega
        Eigen::Vector4d quat = q.segment<4>(3);
        Eigen::Vector3d omega = vel.segment<3>(3);

        // H(q)^T from Python:
        // H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
        //                       cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
        //                       cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        // So H_q_body.T is 4x3
        Eigen::Matrix<double, 4, 3> H_T;
        H_T << -quat(1), -quat(2), -quat(3),  // Row for dqw
            quat(0), -quat(3), quat(2),       // Row for dqx
            quat(3), quat(0), -quat(1),       // Row for dqy
            -quat(2), quat(1), quat(0);       // Row for dqz

        Eigen::Vector4d quat_next = quat + 0.5 * dt * H_T * omega;
        // Note: Python doesn't normalize in the optimization, but we can for simulation
        // quat_next.normalize();
        q_next.segment<4>(3) = quat_next;

        // Robot position integration: p_robot_next = p_robot + dt * v_robot
        q_next.segment(7, param_.n_cmd) =
            q.segment(7, param_.n_cmd) + dt * vel.segment(6, param_.n_cmd);

        return q_next;
    }

    // Accessors.
    const Parameters& getParams() const { return param_; }

    // Final cost weight getters (for custom solver use)
    double getFinalPositionWeight() const override { return param_.final_position_weight; }
    double getFinalOrientationWeight() const override { return param_.final_quaternion_weight; }

   private:
    Parameters param_;
    Eigen::MatrixXd Q_;          // Combined inertia matrix
    Eigen::VectorXd b_gravity_;  // Gravity bias vector
};

}  // namespace allegro_lcp
