#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace simulation {

/**
 * @brief Convert axis-angle representation to quaternion
 * @param axis The rotation axis (will be normalized)
 * @param angle The rotation angle in radians
 * @return Quaternion as [w, x, y, z]
 */
inline Eigen::Vector4d axisAngleToQuat(const Eigen::Vector3d& axis, double angle) {
    Eigen::Vector3d dir = axis.normalized();
    double half_angle = angle / 2.0;
    Eigen::Vector4d quat;
    quat(0) = std::cos(half_angle);           // w
    quat(1) = std::sin(half_angle) * dir(0);  // x
    quat(2) = std::sin(half_angle) * dir(1);  // y
    quat(3) = std::sin(half_angle) * dir(2);  // z
    return quat;
}

/**
 * @brief Compute position error (L2 norm)
 */
inline double computePositionError(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2) {
    return (pos1 - pos2).norm();
}

/**
 * @brief Compute quaternion error (geodesic distance on SO(3))
 * @param q1 First quaternion [w, x, y, z]
 * @param q2 Second quaternion [w, x, y, z]
 * @return Geodesic distance in radians
 */
inline double computeQuaternionError(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
    double dot = std::abs(q1.dot(q2));
    dot = std::min(1.0, std::max(-1.0, dot));
    return 2.0 * std::acos(dot);
}

/**
 * @brief Normalize a quaternion
 */
inline Eigen::Vector4d normalizeQuat(const Eigen::Vector4d& quat) { return quat.normalized(); }

/**
 * @brief Convert roll-pitch-yaw (ZYX Euler angles) to quaternion
 * @param yaw Rotation around Z axis (radians)
 * @param pitch Rotation around Y axis (radians)
 * @param roll Rotation around X axis (radians)
 * @return Quaternion as [w, x, y, z]
 */
inline Eigen::Vector4d rpyToQuat(double yaw, double pitch, double roll) {
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);

    Eigen::Vector4d q;
    q(0) = cr * cp * cy + sr * sp * sy;  // w
    q(1) = sr * cp * cy - cr * sp * sy;  // x
    q(2) = cr * sp * cy + sr * cp * sy;  // y
    q(3) = cr * cp * sy - sr * sp * cy;  // z
    return q;
}

}  // namespace simulation
