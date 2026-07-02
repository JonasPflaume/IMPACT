#include "fingertips_simulator.h"

#include <iostream>
#include <stdexcept>

namespace simulation {

FingertipsSimulator::FingertipsSimulator(const Parameters& param)
    : param_(param),
      window_(nullptr),
      rendering_enabled_(false),
      button_left_(false),
      button_middle_(false),
      button_right_(false),
      lastx_(0),
      lasty_(0) {
    // Load MuJoCo model
    char error[1000] = "Could not load model";
    model_ = mj_loadXML(param_.model_path.c_str(), nullptr, error, 1000);
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + std::string(error));
    }

    data_ = mj_makeData(model_);

    // Reset to initial state
    reset();

    // Set goal visualization
    setGoal(param_.target_p, param_.target_q);
}

FingertipsSimulator::~FingertipsSimulator() {
    closeRendering();
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

void FingertipsSimulator::reset() {
    // MuJoCo model layout: [obj_qpos(7), robot_qpos(9)]
    // qpos[0:7] = object [pos(3), quat(4)]
    // qpos[7:16] = fingertips [ftp0(3), ftp1(3), ftp2(3)]

    // Set object pose (first 7 DOFs)
    for (int i = 0; i < 7; ++i) {
        data_->qpos[i] = param_.init_obj_qpos(i);
    }

    // Set fingertip positions (next 9 DOFs)
    for (int i = 0; i < 9; ++i) {
        data_->qpos[7 + i] = param_.init_robot_qpos(i);
    }

    // Zero velocities
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }

    mj_forward(model_, data_);
}

void FingertipsSimulator::step(const Eigen::VectorXd& action) {
    // action: 9D position increment for fingertips
    // MuJoCo layout: qpos[0:7] = object, qpos[7:16] = fingertips
    // MuJoCo layout: qvel[0:6] = object, qvel[6:15] = fingertips

    // Get current fingertip positions (indices 7-15 in qpos)
    Eigen::VectorXd curr_robot_qpos(9);
    for (int i = 0; i < 9; ++i) {
        curr_robot_qpos(i) = data_->qpos[7 + i];
    }

    Eigen::VectorXd desired_pos = curr_robot_qpos + action;

    // Compute mass matrix for gravity compensation ONCE before the loop
    // (fingertip mass matrix is approximately constant for small motions)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fullM_rm(model_->nv,
                                                                                    model_->nv);
    mj_fullM(model_, fullM_rm.data(), data_->qM);
    Eigen::MatrixXd fullM = fullM_rm;  // Convert to column-major
    // Fingertip mass matrix is the bottom-right 9x9 block
    Eigen::MatrixXd fingertipM = fullM.block(6, 6, 9, 9);

    // Pre-compute gravity compensation (constant throughout the loop)
    Eigen::VectorXd gravity_tiled(9);
    gravity_tiled << model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2],
        model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2],
        model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2];
    Eigen::VectorXd grav_comp = fingertipM * gravity_tiled;

    for (int iter = 0; iter < param_.frame_skip; ++iter) {
        // Get current fingertip positions
        for (int i = 0; i < 9; ++i) {
            curr_robot_qpos(i) = data_->qpos[7 + i];
        }

        Eigen::VectorXd dpos = curr_robot_qpos - desired_pos;

        // Get fingertip velocities (indices 6-14 in qvel)
        Eigen::VectorXd dvel(9);
        for (int i = 0; i < 9; ++i) {
            dvel(i) = data_->qvel[6 + i];
        }

        // PD control with gravity compensation
        Eigen::VectorXd control = -param_.robot_stiff * dpos - 2.0 * dvel - grav_comp;

        // Apply control
        for (int i = 0; i < 9 && i < model_->nu; ++i) {
            data_->ctrl[i] = control(i);
        }

        mj_step(model_, data_);

        // Render each physics step for smooth animation
        if (rendering_enabled_) {
            render();
        }
    }
}

void FingertipsSimulator::setStateDirectly(const Eigen::VectorXd& state) {
    // Directly set state without simulation (for trajectory visualization)
    // state: [obj_pos(3), obj_quat(4), robot_qpos(9)]
    // MuJoCo layout: qpos[0:7] = object, qpos[7:16] = fingertips

    // Object pose (first 7 DOFs in qpos)
    for (int i = 0; i < 3; ++i) {
        data_->qpos[i] = state(i);  // obj_pos
    }
    for (int i = 0; i < 4; ++i) {
        data_->qpos[3 + i] = state(3 + i);  // obj_quat
    }

    // Fingertip positions (DOFs 7-15 in qpos)
    for (int i = 0; i < 9; ++i) {
        data_->qpos[7 + i] = state(7 + i);  // robot_qpos
    }

    // Set velocities to zero (we're just visualizing, not simulating)
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }

    // Forward kinematics to update all positions
    mj_forward(model_, data_);
}

Eigen::VectorXd FingertipsSimulator::getState() const {
    // Return state: [obj_pos(3), obj_quat(4), robot_qpos(9)]
    // MuJoCo layout: qpos[0:7] = object, qpos[7:16] = fingertips
    Eigen::VectorXd state(16);

    // Object pose (first 7 DOFs in qpos)
    for (int i = 0; i < 3; ++i) {
        state(i) = data_->qpos[i];  // obj_pos
    }
    for (int i = 0; i < 4; ++i) {
        state(3 + i) = data_->qpos[3 + i];  // obj_quat
    }

    // Robot positions (next 9 DOFs)
    for (int i = 0; i < 9; ++i) {
        state(7 + i) = data_->qpos[7 + i];  // robot_qpos
    }

    return state;
}

FingertipsContactResult FingertipsSimulator::detectContactsAt(const Eigen::VectorXd& state) {
    // Save current MuJoCo state
    std::vector<double> saved_qpos(model_->nq);
    std::vector<double> saved_qvel(model_->nv);
    for (int i = 0; i < model_->nq; ++i) {
        saved_qpos[i] = data_->qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
        saved_qvel[i] = data_->qvel[i];
    }

    // Set to the requested state
    setStateDirectly(state);

    // Detect contacts
    FingertipsContactResult result = detectContacts();

    // Restore original state
    for (int i = 0; i < model_->nq; ++i) {
        data_->qpos[i] = saved_qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = saved_qvel[i];
    }
    mj_forward(model_, data_);

    return result;
}

FingertipsContactResult FingertipsSimulator::detectContacts() {
    mj_forward(model_, data_);
    mj_collision(model_, data_);

    int n_con = data_->ncon;
    constexpr int max_ncon = FingertipsContactResult::MAX_NCON;
    constexpr int n_qvel = FingertipsContactResult::N_QVEL;

    FingertipsContactResult result;
    result.phi_vec = Eigen::VectorXd::Ones(max_ncon * 4);
    result.jac_mat = Eigen::MatrixXd::Zero(max_ncon * 4, n_qvel);

    std::vector<double> phi_list;
    std::vector<Eigen::MatrixXd> jac_list;

    for (int i = 0; i < n_con; ++i) {
        mjContact& contact = data_->contact[i];

        const char* geom1_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom1);
        const char* geom2_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom2);

        bool geom1_is_obj = geom1_name && std::string(geom1_name) == "obj";
        bool geom2_is_obj = geom2_name && std::string(geom2_name) == "obj";

        if (geom1_is_obj || geom2_is_obj) {
            // Contact involves the object
            double con_dist = contact.dist * 0.5;
            double mu = 0.5;

            // Contact frame
            // MuJoCo stores frame as 9 elements (3 rows x 3 cols, row-major)
            // Python does: contact_i.frame.reshape((-1, 3)).T
            Eigen::Matrix3d con_frame_raw;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    con_frame_raw(r, c) = contact.frame[r * 3 + c];
                }
            }
            Eigen::Matrix3d con_frame = con_frame_raw.transpose();

            // Extended contact frame [n, t1, t2, -t1, -t2]
            Eigen::MatrixXd con_frame_pmd(3, 5);
            con_frame_pmd.block<3, 3>(0, 0) = con_frame;
            con_frame_pmd.block<3, 2>(0, 3) = -con_frame.block<3, 2>(0, 1);

            // Compute Jacobians
            int body1_id = model_->geom_bodyid[contact.geom1];
            int body2_id = model_->geom_bodyid[contact.geom2];

            // MuJoCo fills Jacobians in row-major order
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp1_rm(3, n_qvel);
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp2_rm(3, n_qvel);
            jacp1_rm.setZero();
            jacp2_rm.setZero();

            mj_jac(model_, data_, jacp1_rm.data(), nullptr, contact.pos, body1_id);
            mj_jac(model_, data_, jacp2_rm.data(), nullptr, contact.pos, body2_id);

            // Convert to column-major for Eigen operations
            Eigen::MatrixXd jacp1 = jacp1_rm;
            Eigen::MatrixXd jacp2 = jacp2_rm;

            // Transform to contact frame
            Eigen::MatrixXd con_jacp1 = con_frame_pmd.transpose() * jacp1;
            Eigen::MatrixXd con_jacp2 = con_frame_pmd.transpose() * jacp2;

            Eigen::MatrixXd con_jacp;
            if (geom1_is_obj) {
                con_jacp = -(con_jacp2 - con_jacp1);
            } else {
                con_jacp = (con_jacp2 - con_jacp1);
            }

            // Extract normal and friction components
            Eigen::VectorXd con_jacp_n = con_jacp.row(0);
            Eigen::MatrixXd con_jacp_f = con_jacp.block(1, 0, 4, n_qvel);

            // Combined Jacobian with friction: J_i = J_n + mu * J_f[i]
            Eigen::MatrixXd con_jac(4, n_qvel);
            for (int k = 0; k < 4; ++k) {
                con_jac.row(k) = con_jacp_n.transpose() + mu * con_jacp_f.row(k);
            }

            phi_list.push_back(con_dist);
            jac_list.push_back(con_jac);
        }
    }

    // Fill result
    for (size_t i = 0; i < phi_list.size() && i < static_cast<size_t>(max_ncon); ++i) {
        for (int j = 0; j < 4; ++j) {
            result.phi_vec(4 * i + j) = phi_list[i];
            result.jac_mat.row(4 * i + j) = jac_list[i].row(j);
        }
    }

    return result;
}

void FingertipsSimulator::setGoal(const Eigen::Vector3d& pos, const Eigen::Vector4d& quat) {
    param_.target_p = pos;
    param_.target_q = quat;

    int body_id = mj_name2id(model_, mjOBJ_BODY, "goal");
    if (body_id >= 0) {
        model_->body_pos[3 * body_id + 0] = pos(0);
        model_->body_pos[3 * body_id + 1] = pos(1);
        model_->body_pos[3 * body_id + 2] = pos(2);
        model_->body_quat[4 * body_id + 0] = quat(0);
        model_->body_quat[4 * body_id + 1] = quat(1);
        model_->body_quat[4 * body_id + 2] = quat(2);
        model_->body_quat[4 * body_id + 3] = quat(3);
        mj_forward(model_, data_);
    }
}

void FingertipsSimulator::initRendering() {
    if (!glfwInit()) {
        throw std::runtime_error("Could not initialize GLFW");
    }

    window_ = glfwCreateWindow(1200, 900, "Fingertips LCP MPC", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Could not create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    mjv_makeScene(model_, &scn_, 2000);
    mjr_makeContext(model_, &con_, mjFONTSCALE_150);

    // Set up camera for fingertips scene
    cam_.azimuth = 90;
    cam_.elevation = -30;
    cam_.distance = 0.5;
    cam_.lookat[0] = 0;
    cam_.lookat[1] = 0;
    cam_.lookat[2] = 0.05;

    // Install callbacks
    glfwSetWindowUserPointer(window_, this);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, mouseMoveCallback);
    glfwSetScrollCallback(window_, scrollCallback);

    rendering_enabled_ = true;
}

void FingertipsSimulator::render() {
    if (!rendering_enabled_ || !window_) return;

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    mjv_updateScene(model_, data_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

bool FingertipsSimulator::shouldClose() const { return window_ && glfwWindowShouldClose(window_); }

void FingertipsSimulator::closeRendering() {
    if (window_) {
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
    rendering_enabled_ = false;
}

void FingertipsSimulator::mouseButtonCallback(GLFWwindow* window, int button, int action,
                                              int mods) {
    FingertipsSimulator* sim = static_cast<FingertipsSimulator*>(glfwGetWindowUserPointer(window));
    if (!sim) return;

    sim->button_left_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    sim->button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    sim->button_right_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    glfwGetCursorPos(window, &sim->lastx_, &sim->lasty_);
}

void FingertipsSimulator::mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
    FingertipsSimulator* sim = static_cast<FingertipsSimulator*>(glfwGetWindowUserPointer(window));
    if (!sim) return;

    double dx = xpos - sim->lastx_;
    double dy = ypos - sim->lasty_;
    sim->lastx_ = xpos;
    sim->lasty_ = ypos;

    // Left button: rotate view
    if (sim->button_left_) {
        sim->cam_.azimuth -= dx * 0.3;
        sim->cam_.elevation -= dy * 0.3;
    }
    // Right button: also rotate view (alternative)
    else if (sim->button_right_) {
        sim->cam_.azimuth -= dx * 0.3;
        sim->cam_.elevation -= dy * 0.3;
    }
    // Middle button: pan
    else if (sim->button_middle_) {
        sim->cam_.lookat[0] -= dx * 0.001 * sim->cam_.distance;
        sim->cam_.lookat[1] += dy * 0.001 * sim->cam_.distance;
    }
}

void FingertipsSimulator::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    FingertipsSimulator* sim = static_cast<FingertipsSimulator*>(glfwGetWindowUserPointer(window));
    if (!sim) return;

    sim->cam_.distance -= yoffset * 0.05;
    if (sim->cam_.distance < 0.1) sim->cam_.distance = 0.1;
}

}  // namespace simulation
