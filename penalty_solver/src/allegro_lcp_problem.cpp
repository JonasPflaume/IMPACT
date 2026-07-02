#include "penalty_solver/allegro_lcp_problem.h"

#include <cmath>

namespace allegro_lcp {

// Helper functions for CasADi FK transformations.

namespace {

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

}  // anonymous namespace

AllegroLCPProblem::AllegroLCPProblem(const Parameters& param) : param_(param) {
    // Build combined inertia matrix Q (from Python ipopt_param.py)
    // Q = np.zeros((n_qvel_, n_qvel_))
    // Q[:6, :6] = obj_inertia_
    // Q[6:, 6:] = robot_stiff_
    Q_ = Eigen::MatrixXd::Zero(param_.n_qvel, param_.n_qvel);
    Q_.block<6, 6>(0, 0) = param_.obj_inertia;
    Q_.block(6, 6, param_.n_cmd, param_.n_cmd) = param_.robot_stiff;

    // Build gravity bias vector (from Python ipopt_param.py)
    // b_o = obj_mass_ * gravity_ (6D vector in Python)
    // b_r = robot_stiff_ @ cmd (handled at solve time)
    // Base gravity bias (without robot stiffness term which is added at solve time)
    b_gravity_ = Eigen::VectorXd::Zero(param_.n_qvel);
    b_gravity_.head<6>() = param_.obj_mass * param_.gravity;
    // Robot part is zero in gravity bias (stiffness * cmd is added during solve)
}

casadi::SX AllegroLCPProblem::computeFingertipPositionsSX(const casadi::SX& q_sx) const {
    // Extract robot joint positions
    // q = [obj_pos(3), obj_quat(4), robot_joints(16)]
    // ff_qpos = q[7:11], mf_qpos = q[11:15], rf_qpos = q[15:19], th_qpos = q[19:23]
    casadi::SX ff_qpos = q_sx(casadi::Slice(7, 11));   // First finger
    casadi::SX mf_qpos = q_sx(casadi::Slice(11, 15));  // Middle finger
    casadi::SX rf_qpos = q_sx(casadi::Slice(15, 19));  // Ring finger
    casadi::SX th_qpos = q_sx(casadi::Slice(19, 23));  // Thumb

    // Palm transform (constant)
    // t_palm = quattmat([0, 1, 0, 1] / norm) = quattmat([0, 1/sqrt(2), 0, 1/sqrt(2)])
    double sq2 = std::sqrt(2.0);
    casadi::SX t_palm = quattmat(0.0, 1.0 / sq2, 0.0, 1.0 / sq2);

    // Forward kinematics for each fingertip.

    // First finger (ff)
    casadi::SX ff_t_base =
        casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(0, 0.0435, -0.001542)),
                           quattmat(0.999048, -0.0436194, 0, 0));
    casadi::SX ff_t_proximal =
        casadi::SX::mtimes(casadi::SX::mtimes(ff_t_base, rztmat(ff_qpos(0))), ttmat(0, 0, 0.0164));
    casadi::SX ff_t_medial = casadi::SX::mtimes(
        casadi::SX::mtimes(ff_t_proximal, rytmat(ff_qpos(1))), ttmat(0, 0, 0.054));
    casadi::SX ff_t_distal = casadi::SX::mtimes(casadi::SX::mtimes(ff_t_medial, rytmat(ff_qpos(2))),
                                                ttmat(0, 0, 0.0384));
    casadi::SX ff_t_ftp = casadi::SX::mtimes(casadi::SX::mtimes(ff_t_distal, rytmat(ff_qpos(3))),
                                             ttmat(0, 0, 0.0384));
    casadi::SX ftp_1 = extractPos(ff_t_ftp);

    // Middle finger (mf)
    casadi::SX mf_t_base = casadi::SX::mtimes(t_palm, ttmat(0, 0, 0.0007));
    casadi::SX mf_t_proximal =
        casadi::SX::mtimes(casadi::SX::mtimes(mf_t_base, rztmat(mf_qpos(0))), ttmat(0, 0, 0.0164));
    casadi::SX mf_t_medial = casadi::SX::mtimes(
        casadi::SX::mtimes(mf_t_proximal, rytmat(mf_qpos(1))), ttmat(0, 0, 0.054));
    casadi::SX mf_t_distal = casadi::SX::mtimes(casadi::SX::mtimes(mf_t_medial, rytmat(mf_qpos(2))),
                                                ttmat(0, 0, 0.0384));
    casadi::SX mf_t_ftp = casadi::SX::mtimes(casadi::SX::mtimes(mf_t_distal, rytmat(mf_qpos(3))),
                                             ttmat(0, 0, 0.0384));
    casadi::SX ftp_2 = extractPos(mf_t_ftp);

    // Ring finger (rf)
    casadi::SX rf_t_base =
        casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(0, -0.0435, -0.001542)),
                           quattmat(0.999048, 0.0436194, 0, 0));
    casadi::SX rf_t_proximal =
        casadi::SX::mtimes(casadi::SX::mtimes(rf_t_base, rztmat(rf_qpos(0))), ttmat(0, 0, 0.0164));
    casadi::SX rf_t_medial = casadi::SX::mtimes(
        casadi::SX::mtimes(rf_t_proximal, rytmat(rf_qpos(1))), ttmat(0, 0, 0.054));
    casadi::SX rf_t_distal = casadi::SX::mtimes(casadi::SX::mtimes(rf_t_medial, rytmat(rf_qpos(2))),
                                                ttmat(0, 0, 0.0384));
    casadi::SX rf_t_ftp = casadi::SX::mtimes(casadi::SX::mtimes(rf_t_distal, rytmat(rf_qpos(3))),
                                             ttmat(0, 0, 0.0384));
    casadi::SX ftp_3 = extractPos(rf_t_ftp);

    // Thumb (th)
    casadi::SX th_t_base =
        casadi::SX::mtimes(casadi::SX::mtimes(t_palm, ttmat(-0.0182, 0.019333, -0.045987)),
                           quattmat(0.477714, -0.521334, -0.521334, -0.477714));
    casadi::SX th_t_proximal = casadi::SX::mtimes(
        casadi::SX::mtimes(th_t_base, rxtmat(-th_qpos(0))), ttmat(-0.027, 0.005, 0.0399));
    casadi::SX th_t_medial = casadi::SX::mtimes(
        casadi::SX::mtimes(th_t_proximal, rztmat(th_qpos(1))), ttmat(0, 0, 0.0177));
    casadi::SX th_t_distal = casadi::SX::mtimes(casadi::SX::mtimes(th_t_medial, rytmat(th_qpos(2))),
                                                ttmat(0, 0, 0.0514));
    casadi::SX th_t_ftp =
        casadi::SX::mtimes(casadi::SX::mtimes(th_t_distal, rytmat(th_qpos(3))), ttmat(0, 0, 0.054));
    casadi::SX ftp_4 = extractPos(th_t_ftp);

    // Return concatenated fingertip positions (12D: 4 fingertips * 3D)
    return casadi::SX::vertcat({ftp_1, ftp_2, ftp_3, ftp_4});
}

}  // namespace allegro_lcp
