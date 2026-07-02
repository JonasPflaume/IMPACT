#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <stdexcept>

namespace impact {

/**
 * @brief Per-stage formulation interface for a single-shooting LCP contact task.
 *
 * The configuration q is rolled out from velocities (single shooting); the free
 * variables are [cmd, lam, vel] per step. The single-shooting builder
 * (buildSingleShooting) assembles the LCP dynamics penalty, the lam ⊥ (J*vel +
 * phi/h) complementarity, the manipulation cost (from the weights / fingertip FK
 * below), and the AuLa command-bound channel.
 */
class LCPProblem {
   public:
    virtual ~LCPProblem() = default;

    // Dimensions.
    virtual int getConfigDim() const = 0;
    virtual int getVelocityDim() const = 0;
    virtual int getCommandDim() const = 0;
    virtual int getMaxContacts() const = 0;
    virtual double getTimeStep() const = 0;

    // Inertia, stiffness and gravity terms.
    virtual Eigen::MatrixXd getInertiaMatrix() const = 0;  // Q (n_qvel x n_qvel)
    virtual Eigen::VectorXd getGravityBias() const = 0;    // b_gravity (n_qvel)
    virtual Eigen::MatrixXd getRobotStiffness() const {
        int n_cmd = getCommandDim();
        return 100.0 * Eigen::MatrixXd::Identity(n_cmd, n_cmd);
    }

    // Cost weights.
    virtual double getControlCostWeight() const = 0;
    virtual double getContactCostWeight() const = 0;
    virtual double getGraspClosureWeight() const = 0;
    virtual double getVelocityPenalty() const = 0;
    virtual double getPositionCostWeight() const = 0;
    virtual double getOrientationCostWeight() const = 0;
    virtual double getFinalCostMultiplier() const = 0;
    virtual double getFinalPositionWeight() const { return getPositionCostWeight(); }
    virtual double getFinalOrientationWeight() const { return getOrientationCostWeight(); }

    // Variable bounds.
    virtual double getVelocityLowerBound() const { return -1.0; }
    virtual double getVelocityUpperBound() const { return 1.0; }
    virtual double getConfigLowerBound() const { return -100.0; }
    virtual double getConfigUpperBound() const { return 100.0; }
    virtual double getControlLowerBound() const { return -casadi::inf; }
    virtual double getControlUpperBound() const { return casadi::inf; }

    // Fingertip geometry.
    virtual int getNumFingertips() const { return 3; }
    virtual casadi::SX computeFingertipPositionsSX(const casadi::SX& q_sx) const {
        int n_ftp = getNumFingertips();
        return q_sx(casadi::Slice(7, 7 + 3 * n_ftp));
    }

    // Optional CasADi functions; the single-shooting builder inlines its own.
    virtual casadi::Function getKinematicsFunction() const = 0;
    virtual casadi::Function getLCPDynamicsFunction() const = 0;
    virtual casadi::Function getStageCostFunction() const = 0;
    virtual casadi::Function getTerminalCostFunction() const = 0;

    // Eigen-side state integration used for solution rollout.
    virtual Eigen::VectorXd integrateState(const Eigen::VectorXd& q, const Eigen::VectorXd& vel,
                                           double dt) const {
        throw std::runtime_error("integrateState not implemented");
    }

    int getQposSize() const { return getConfigDim(); }
    int getQvelSize() const { return getVelocityDim(); }
    int getCommandSize() const { return getCommandDim(); }
};

}  // namespace impact
