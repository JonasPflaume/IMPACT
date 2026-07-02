#include "contactLCP.h"

#include <algorithm>
#include <iostream>

#include "mj_simulator.h"

Contact::Contact(const Parameters& param) : param_(param) {
    std::cout << "Contact detector initialized with " << param_.object_names.size()
              << " object names" << std::endl;
}

Contact::ContactResult Contact::detect_once(MjSimulator& simulator) {
    // Forward dynamics and collision detection
    mj_forward(simulator.model_, simulator.data_);
    mj_collision(simulator.model_, simulator.data_);

    // Extract contacts
    int n_con = simulator.data_->ncon;
    mjContact* contacts = simulator.data_->contact;

    std::vector<ContactInfo> contact_infos;

    // Convert object names to set for fast lookup
    std::unordered_set<std::string> object_name_set(param_.object_names.begin(),
                                                    param_.object_names.end());

    for (int i = 0; i < n_con; ++i) {
        mjContact& contact_i = contacts[i];

        // Get geometry names
        const char* geom1_name = mj_id2name(simulator.model_, mjOBJ_GEOM, contact_i.geom1);
        const char* geom2_name = mj_id2name(simulator.model_, mjOBJ_GEOM, contact_i.geom2);

        if (!geom1_name) geom1_name = "";
        if (!geom2_name) geom2_name = "";

        // Get body IDs
        int body1_id = simulator.model_->geom_bodyid[contact_i.geom1];
        int body2_id = simulator.model_->geom_bodyid[contact_i.geom2];

        bool process_contact = false;
        bool geom1_is_object = object_name_set.count(geom1_name) > 0;
        bool geom2_is_object = object_name_set.count(geom2_name) > 0;

        if (geom1_is_object || geom2_is_object) {
            ContactInfo info;

            // Contact position and distance
            for (int j = 0; j < 3; ++j) {
                info.pos[j] = contact_i.pos[j];
            }
            info.phi = contact_i.dist * 0.5;

            // Contact frame (3x3 matrix from 9-element array)
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    info.frame(row, col) = contact_i.frame[row * 3 + col];
                }
            }

            // Create extended contact frame [n, t1, t2, -t1, -t2]
            Eigen::MatrixXd con_frame_pmd(3, 5);
            con_frame_pmd.block<3, 3>(0, 0) = info.frame;
            con_frame_pmd.block<3, 2>(0, 3) = -info.frame.block<3, 2>(0, 1);

            // Compute Jacobians
            Eigen::MatrixXd jacp1 = Eigen::MatrixXd::Zero(3, param_.n_mj_v);
            Eigen::MatrixXd jacp2 = Eigen::MatrixXd::Zero(3, param_.n_mj_v);

            mj_jac(simulator.model_, simulator.data_, jacp1.data(), nullptr, info.pos.data(),
                   body1_id);
            mj_jac(simulator.model_, simulator.data_, jacp2.data(), nullptr, info.pos.data(),
                   body2_id);

            // Transform Jacobians to contact frame
            Eigen::MatrixXd con_jacp1 = con_frame_pmd.transpose() * jacp1;
            Eigen::MatrixXd con_jacp2 = con_frame_pmd.transpose() * jacp2;

            Eigen::MatrixXd con_jacp;
            if (geom1_is_object) {
                // Jacobian direction: from contact pair to object
                con_jacp = -(con_jacp2 - con_jacp1);
            } else {
                // geom2_is_object
                con_jacp = (con_jacp2 - con_jacp1);
            }

            // Extract normal and friction components
            Eigen::VectorXd con_jacp_n = con_jacp.row(0);                         // Normal
            Eigen::MatrixXd con_jacp_f = con_jacp.block(1, 0, 4, param_.n_mj_v);  // Friction

            // Combine with friction coefficient
            info.jac = Eigen::VectorXd::Zero(4 * param_.n_mj_v);
            for (int k = 0; k < 4; ++k) {
                // row_k = con_jacp_n + μ * con_jacp_f[k]
                Eigen::VectorXd row = con_jacp_n + param_.mu_object * con_jacp_f.row(k).transpose();

                // flatten: row-major
                info.jac.segment(k * param_.n_mj_v, param_.n_mj_v) = row;
            }

            // info.jac = con_jacp_n + param_.mu_object * con_jacp_f.colwise().sum();

            contact_infos.push_back(info);
        }
    }

    return reformat(contact_infos);
}

Contact::ContactResult Contact::reformat(const std::vector<ContactInfo>& contacts) {
    ContactResult result;

    // Initialize with 0.1 for non-contact regions (smaller than 1.0 for better optimization)
    result.phi_vec = Eigen::VectorXd::Constant(param_.max_ncon * 4, 0.01);
    result.jac_mat = Eigen::MatrixXd::Zero(param_.max_ncon * 4, param_.n_qvel);

    // Fill in the contact data
    for (size_t i = 0; i < contacts.size() && i < param_.max_ncon; ++i) {
        const ContactInfo& contact = contacts[i];

        // Fill phi_vec
        for (int j = 0; j < 4; ++j) {
            result.phi_vec[4 * i + j] = contact.phi;
        }

        // Fill jac_mat (simplified version - in real implementation,
        // you'd need to handle the proper jacobian structure)
        for (int j = 0; j < 4; ++j) {
            if (contact.jac.size() >= param_.n_mj_v) {
                result.jac_mat.block(4 * i + j, 0, 1,
                                     std::min(param_.n_qvel, (int)contact.jac.size())) =
                    contact.jac.head(std::min(param_.n_qvel, (int)contact.jac.size())).transpose();
            }
        }
    }

    // Reorder jacobian: [object_dofs, robot_dofs] -> [robot_dofs, object_dofs]
    Eigen::MatrixXd jac_mat_reorder = Eigen::MatrixXd::Zero(param_.max_ncon * 4, param_.n_qvel);
    if (param_.n_qvel >= 22) {
        // Assuming last 6 DOFs are object, first 16 are robot
        jac_mat_reorder.block(0, 0, param_.max_ncon * 4, 6) =
            result.jac_mat.block(0, param_.n_qvel - 6, param_.max_ncon * 4, 6);
        jac_mat_reorder.block(0, 6, param_.max_ncon * 4, 16) =
            result.jac_mat.block(0, 0, param_.max_ncon * 4, 16);
    }

    result.jac_mat = jac_mat_reorder;

    return result;
}