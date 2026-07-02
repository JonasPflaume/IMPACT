#include "impact/lcp_stage.h"

#include <cmath>

namespace impact {

using casadi::Slice;
using casadi::SX;

namespace {
SX eigenToSX(const Eigen::VectorXd& v) {
    return casadi::DM(std::vector<double>(v.data(), v.data() + v.size()));
}
}  // namespace

LCPStage::LCPStage(std::shared_ptr<LCPProblem> problem, const BCDAULAConfig& config)
    : problem_(std::move(problem)) {
    n_qpos_ = problem_->getConfigDim();
    n_qvel_ = problem_->getVelocityDim();
    n_cmd_ = problem_->getCommandDim();
    n_lam_ = problem_->getMaxContacts() * 4;
    n_ftp_ = problem_->getNumFingertips();
    h_ = problem_->getTimeStep();
    control_w_ = problem_->getControlCostWeight();
    contact_w_ = problem_->getContactCostWeight();
    grasp_w_ = problem_->getGraspClosureWeight();
    vel_w_ = problem_->getVelocityPenalty();
    final_mult_ = problem_->getFinalCostMultiplier();
    final_pos_w_ = problem_->getFinalPositionWeight();
    final_ori_w_ = problem_->getFinalOrientationWeight();
    Q_ = problem_->getInertiaMatrix();
    robot_stiff_ = problem_->getRobotStiffness();
    b_o_ = problem_->getGravityBias().head<3>();
    use_cmd_ = config.use_cmd_bounds && config.cmd_lower.size() == n_cmd_ &&
               config.cmd_upper.size() == n_cmd_;
    if (use_cmd_) {
        cmd_lower_ = config.cmd_lower;
        cmd_upper_ = config.cmd_upper;
    }
}

SX LCPStage::cmdOf(const SX& u) const { return u(Slice(0, n_cmd_)); }
SX LCPStage::lamOf(const SX& u) const { return u(Slice(n_cmd_, n_cmd_ + n_lam_)); }
SX LCPStage::velOf(const SX& u) const {
    return u(Slice(n_cmd_ + n_lam_, n_cmd_ + n_lam_ + n_qvel_));
}
SX LCPStage::targetPOf(const SX& p) const { return p(Slice(0, 3)); }
SX LCPStage::targetQOf(const SX& p) const { return p(Slice(3, 7)); }
SX LCPStage::phiOf(const SX& p) const { return p(Slice(7, 7 + n_lam_)); }
SX LCPStage::jacOf(const SX& p) const {
    return SX::reshape(p(Slice(7 + n_lam_, 7 + n_lam_ + n_lam_ * n_qvel_)), n_lam_, n_qvel_);
}

SX LCPStage::step(const SX& x, const SX& u, const SX&) const {
    SX vel = velOf(u);
    SX obj_pos = x(Slice(0, 3)), obj_quat = x(Slice(3, 7)), robot_q = x(Slice(7, n_qpos_));
    SX v_lin = vel(Slice(0, 3)), v_ang = vel(Slice(3, 6)), v_robot = vel(Slice(6, n_qvel_));
    SX next_pos = obj_pos + h_ * v_lin;
    SX next_robot = robot_q + h_ * v_robot;
    SX Hqb = SX::vertcat({SX::horzcat({-obj_quat(1), obj_quat(0), obj_quat(3), -obj_quat(2)}),
                          SX::horzcat({-obj_quat(2), -obj_quat(3), obj_quat(0), obj_quat(1)}),
                          SX::horzcat({-obj_quat(3), obj_quat(2), -obj_quat(1), obj_quat(0)})});
    SX next_quat = obj_quat + 0.5 * h_ * SX::mtimes(Hqb.T(), v_ang);
    next_quat = next_quat / SX::norm_2(next_quat);
    return SX::vertcat({next_pos, next_quat, next_robot});
}

SX LCPStage::G(const SX&, const SX& u, const SX&) const { return lamOf(u); }

SX LCPStage::H(const SX&, const SX& u, const SX& p) const {
    return SX::mtimes(jacOf(p), velOf(u)) + phiOf(p) / h_;
}

SX LCPStage::dynamicsResidual(const SX&, const SX& u, const SX& p) const {
    SX Q_sx = SX::reshape(eigenToSX(Eigen::Map<const Eigen::VectorXd>(Q_.data(), Q_.size())),
                          n_qvel_, n_qvel_);
    SX stiff_sx = SX::reshape(
        eigenToSX(Eigen::Map<const Eigen::VectorXd>(robot_stiff_.data(), robot_stiff_.size())),
        n_cmd_, n_cmd_);
    SX lhs = SX::mtimes(Q_sx, velOf(u));
    SX b_r = SX::mtimes(stiff_sx, cmdOf(u));
    SX b_full = SX::vertcat({eigenToSX(b_o_), SX::zeros(3, 1), b_r});
    SX rhs = b_full / h_ + SX::mtimes(jacOf(p).T(), lamOf(u)) / h_;
    return lhs - rhs;
}

SX LCPStage::ineq(const SX&, const SX& u, const SX&) const {
    SX cmd = cmdOf(u);
    return SX::vertcat({eigenToSX(cmd_lower_) - cmd, cmd - eigenToSX(cmd_upper_)});
}

SX LCPStage::costResidual(const std::vector<SX>& X, const std::vector<SX>& U, const SX& p) const {
    const int H = static_cast<int>(U.size());
    auto quatDcm = [](const SX& q) {
        return SX::vertcat(
            {SX::horzcat({1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)),
                          2 * (q(1) * q(3) + q(0) * q(2))}),
             SX::horzcat({2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)),
                          2 * (q(2) * q(3) - q(0) * q(1))}),
             SX::horzcat({2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)),
                          1 - 2 * (q(1) * q(1) + q(2) * q(2))})});
    };

    std::vector<SX> res;
    for (int k = 0; k < H; ++k) {
        const SX& q_next = X[k + 1];
        res.push_back(std::sqrt(control_w_) * cmdOf(U[k]));
        SX obj_pos_next = q_next(Slice(0, 3));
        SX ftp = problem_->computeFingertipPositionsSX(q_next);
        for (int f = 0; f < n_ftp_; ++f)
            res.push_back(std::sqrt(contact_w_) * (obj_pos_next - ftp(Slice(3 * f, 3 * f + 3))));
        SX dcm = quatDcm(q_next(Slice(3, 7)));
        SX grasp_sum = SX::zeros(3, 1);
        for (int f = 0; f < n_ftp_; ++f) {
            SX v_f = SX::mtimes(dcm.T(), ftp(Slice(3 * f, 3 * f + 3)) - obj_pos_next);
            grasp_sum += v_f / SX::norm_2(v_f);
        }
        res.push_back(std::sqrt(grasp_w_) * grasp_sum);
        res.push_back(std::sqrt(vel_w_) * velOf(U[k])(Slice(0, 6)));
    }
    // Terminal position + quaternion-log error.
    SX q_final = X[H];
    SX target_p = targetPOf(p), target_q = targetQOf(p);
    res.push_back(std::sqrt(final_mult_ * final_pos_w_) * (q_final(Slice(0, 3)) - target_p));
    SX qf = q_final(Slice(3, 7));
    SX w1 = qf(0), x1 = qf(1), y1 = qf(2), z1 = qf(3);
    SX w2 = target_q(0), x2 = target_q(1), y2 = target_q(2), z2 = target_q(3);
    SX qrw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2;
    SX qrx = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2;
    SX qry = -w1 * y2 + y1 * w2 - z1 * x2 + x1 * z2;
    SX qrz = -w1 * z2 + z1 * w2 - x1 * y2 + y1 * x2;
    SX vnorm = SX::sqrt(qrx * qrx + qry * qry + qrz * qrz + 1e-12);
    SX theta = 2 * SX::atan2(vnorm, SX::abs(qrw));
    SX qscale = SX::if_else(vnorm > 1e-6, theta / vnorm, 2.0);
    SX quat_log = SX::vertcat({qscale * qrx, qscale * qry, qscale * qrz});
    res.push_back(std::sqrt(final_mult_ * final_ori_w_) * quat_log);
    return SX::vertcat(res);
}

}  // namespace impact
