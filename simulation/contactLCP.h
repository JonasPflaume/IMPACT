#pragma once

#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <string>
#include <unordered_set>
#include <vector>

class MjSimulator;

class Contact {
   public:
    struct Parameters {
        std::vector<std::string> object_names;
        double mu_object = 0.5;  // friction coefficient
        int max_ncon = 20;       // maximum number of contacts
        int n_mj_v = 22;         // MuJoCo velocity DOFs
        int n_qvel = 22;         // total velocity DOFs
    };

    struct ContactResult {
        Eigen::VectorXd phi_vec;
        Eigen::MatrixXd jac_mat;
    };

    Contact(const Parameters& param);

    ContactResult detect_once(MjSimulator& simulator);

   private:
    Parameters param_;

    struct ContactInfo {
        Eigen::Vector3d pos;
        double phi;
        Eigen::Matrix3d frame;
        Eigen::VectorXd jac;
    };

    ContactResult reformat(const std::vector<ContactInfo>& contacts);
};