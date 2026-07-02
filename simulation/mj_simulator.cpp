#include "mj_simulator.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

MjSimulator::MjSimulator(const Parameters& param)
    : param_(param),
      rendering_enabled_(false),
      window_(nullptr),
      button_left_(false),
      button_middle_(false),
      button_right_(false),
      lastx_(0),
      lasty_(0),
      ui_enabled_(false) {
    // Load MuJoCo model
    char error[1000] = "Could not load binary model";
    model_ = mj_loadXML(param_.model_path.c_str(), 0, error, 1000);
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + std::string(error));
    }

    // Create data
    data_ = mj_makeData(model_);

    // Initialize fingertip names
    fingertip_names_ = {"ftp_0", "ftp_1", "ftp_2", "ftp_3"};

    // Set goal and reset environment
    set_goal(&param_.target_p, &param_.target_q);
    reset_env();

    // Initialize rotation functions
    init_rotation_functions();

    // Initialize forward kinematics
    allegro_fd_fn();

    std::cout << "MjSimulator initialized successfully" << std::endl;
}

MjSimulator::~MjSimulator() {
    close_rendering();
    if (data_) {
        mj_deleteData(data_);
    }
    if (model_) {
        mj_deleteModel(model_);
    }
}

void MjSimulator::reset_env() {
    // Set initial robot and object positions
    int robot_dofs = param_.init_robot_qpos.size();
    int obj_dofs = param_.init_obj_qpos.size();

    // Copy robot positions
    for (int i = 0; i < robot_dofs && i < model_->nq; ++i) {
        data_->qpos[i] = param_.init_robot_qpos[i];
    }

    // Copy object positions
    for (int i = 0; i < obj_dofs && (robot_dofs + i) < model_->nq; ++i) {
        data_->qpos[robot_dofs + i] = param_.init_obj_qpos[i];
    }

    // Zero velocities
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }

    mj_forward(model_, data_);
}

void MjSimulator::step(const Eigen::VectorXd& jpos_cmd) {
    Eigen::VectorXd curr_jpos = get_jpos();
    Eigen::VectorXd target_jpos = curr_jpos + jpos_cmd;

    for (int i = 0; i < param_.frame_skip; ++i) {
        // Set control commands (assuming first 16 actuators are for the robot)
        for (int j = 0; j < std::min(target_jpos.size(), (long)model_->nu); ++j) {
            data_->ctrl[j] = target_jpos[j];
        }

        mj_step(model_, data_);
    }
}

void MjSimulator::reset_fingers_qpos() {
    for (int iter = 0; iter < param_.frame_skip; ++iter) {
        for (int i = 0; i < std::min(param_.init_robot_qpos.size(), (long)model_->nu); ++i) {
            data_->ctrl[i] = param_.init_robot_qpos[i];
        }
        mj_step(model_, data_);
    }
}

Eigen::VectorXd MjSimulator::get_state() {
    // Object position (last 7 DOFs) + robot position (first 16 DOFs)
    Eigen::VectorXd obj_pos(7);
    Eigen::VectorXd robot_pos(16);

    int start_obj = model_->nq - 7;
    for (int i = 0; i < 7; ++i) {
        obj_pos[i] = data_->qpos[start_obj + i];
    }

    for (int i = 0; i < 16 && i < model_->nq; ++i) {
        robot_pos[i] = data_->qpos[i];
    }

    Eigen::VectorXd state(23);
    state << obj_pos, robot_pos;
    return state;
}

Eigen::VectorXd MjSimulator::get_jpos() {
    Eigen::VectorXd jpos(16);
    for (int i = 0; i < 16 && i < model_->nq; ++i) {
        jpos[i] = data_->qpos[i];
    }
    return jpos;
}

Eigen::VectorXd MjSimulator::get_fingertips_position() {
    Eigen::VectorXd fts_pos(12);  // 4 fingertips * 3 coordinates

    for (size_t i = 0; i < fingertip_names_.size(); ++i) {
        int site_id = get_site_id(fingertip_names_[i]);
        if (site_id >= 0) {
            fts_pos[3 * i + 0] = data_->site_xpos[3 * site_id + 0];
            fts_pos[3 * i + 1] = data_->site_xpos[3 * site_id + 1];
            fts_pos[3 * i + 2] = data_->site_xpos[3 * site_id + 2];
        }
    }

    return fts_pos;
}

void MjSimulator::set_goal(const Eigen::Vector3d* goal_pos, const Eigen::Vector4d* goal_quat) {
    int body_id = mj_name2id(model_, mjOBJ_BODY, "goal");
    if (body_id >= 0) {
        if (goal_pos) {
            model_->body_pos[3 * body_id + 0] = (*goal_pos)[0];
            model_->body_pos[3 * body_id + 1] = (*goal_pos)[1];
            model_->body_pos[3 * body_id + 2] = (*goal_pos)[2];
        }
        if (goal_quat) {
            model_->body_quat[4 * body_id + 0] = (*goal_quat)[0];  // w
            model_->body_quat[4 * body_id + 1] = (*goal_quat)[1];  // x
            model_->body_quat[4 * body_id + 2] = (*goal_quat)[2];  // y
            model_->body_quat[4 * body_id + 3] = (*goal_quat)[3];  // z
        }
        mj_forward(model_, data_);
    }
}

// Initialize rotation functions using CasADi symbolic expressions
void MjSimulator::init_rotation_functions() {
    // Translation to homogeneous transformation matrix
    casadi::SX pos = casadi::SX::sym("pos", 3);
    casadi::SX ttmat = casadi::SX::vertcat(
        {casadi::SX::horzcat({1, 0, 0, pos(0)}), casadi::SX::horzcat({0, 1, 0, pos(1)}),
         casadi::SX::horzcat({0, 0, 1, pos(2)}), casadi::SX::horzcat({0, 0, 0, 1})});
    ttmat_fn_ = casadi::Function("ttmat_fn", {pos}, {ttmat});

    // X-axis rotation
    casadi::SX alpha = casadi::SX::sym("alpha", 1);
    casadi::SX rxtmat = casadi::SX::vertcat(
        {casadi::SX::horzcat({1, 0, 0, 0}),
         casadi::SX::horzcat({0, casadi::SX::cos(alpha), -casadi::SX::sin(alpha), 0}),
         casadi::SX::horzcat({0, casadi::SX::sin(alpha), casadi::SX::cos(alpha), 0}),
         casadi::SX::horzcat({0, 0, 0, 1})});
    rxtmat_fn_ = casadi::Function("rxtmat_fn", {alpha}, {rxtmat});

    // Y-axis rotation
    casadi::SX beta = casadi::SX::sym("beta", 1);
    casadi::SX rytmat = casadi::SX::vertcat(
        {casadi::SX::horzcat({casadi::SX::cos(beta), 0, casadi::SX::sin(beta), 0}),
         casadi::SX::horzcat({0, 1, 0, 0}),
         casadi::SX::horzcat({-casadi::SX::sin(beta), 0, casadi::SX::cos(beta), 0}),
         casadi::SX::horzcat({0, 0, 0, 1})});
    rytmat_fn_ = casadi::Function("rytmat_fn", {beta}, {rytmat});

    // Z-axis rotation
    casadi::SX theta = casadi::SX::sym("theta", 1);
    casadi::SX rztmat = casadi::SX::vertcat(
        {casadi::SX::horzcat({casadi::SX::cos(theta), -casadi::SX::sin(theta), 0, 0}),
         casadi::SX::horzcat({casadi::SX::sin(theta), casadi::SX::cos(theta), 0, 0}),
         casadi::SX::horzcat({0, 0, 1, 0}), casadi::SX::horzcat({0, 0, 0, 1})});
    rztmat_fn_ = casadi::Function("rztmat_fn", {theta}, {rztmat});

    // Quaternion to homogeneous transformation matrix
    casadi::SX q = casadi::SX::sym("q", 4);
    casadi::SX dcm = casadi::SX::vertcat(
        {casadi::SX::horzcat({1 - 2 * (q(2) * q(2) + q(3) * q(3)), 2 * (q(1) * q(2) - q(0) * q(3)),
                              2 * (q(1) * q(3) + q(0) * q(2))}),
         casadi::SX::horzcat({2 * (q(1) * q(2) + q(0) * q(3)), 1 - 2 * (q(1) * q(1) + q(3) * q(3)),
                              2 * (q(2) * q(3) - q(0) * q(1))}),
         casadi::SX::horzcat({2 * (q(1) * q(3) - q(0) * q(2)), 2 * (q(2) * q(3) + q(0) * q(1)),
                              1 - 2 * (q(1) * q(1) + q(2) * q(2))})});
    casadi::SX quattmat = casadi::SX::vertcat(
        {casadi::SX::horzcat({dcm, casadi::SX::zeros(3, 1)}), casadi::SX::horzcat({0, 0, 0, 1})});
    quattmat_fn_ = casadi::Function("quattmat_fn", {q}, {quattmat});

    // Quaternion to DCM only
    quat2dcm_fn_ = casadi::Function("quat2dcm_fn", {q}, {dcm});
}

// Utility functions
Eigen::Vector4d MjSimulator::normalize_quat(const Eigen::Vector4d& quat) {
    return quat / quat.norm();
}

casadi::DM MjSimulator::vector_to_dm(const std::vector<double>& vec) {
    casadi::DM dm = casadi::DM::zeros(vec.size(), 1);
    for (size_t i = 0; i < vec.size(); ++i) {
        dm(i) = vec[i];
    }
    return dm;
}

void MjSimulator::allegro_fd_fn() {
    // Palm transformation
    Eigen::Vector4d palm_quat_eigen(0, 1, 0, 1);
    palm_quat_eigen = normalize_quat(palm_quat_eigen);

    casadi::DM palm_quat_dm = vector_to_dm(
        {palm_quat_eigen[0], palm_quat_eigen[1], palm_quat_eigen[2], palm_quat_eigen[3]});
    casadi::DM t_palm = quattmat_fn_(std::vector<casadi::DM>{palm_quat_dm}).at(0);

    // First finger forward kinematics
    casadi::SX ff_qpos = casadi::SX::sym("ff_qpos", 4);

    // Base transformation matrices
    casadi::DM ff_base_trans_dm = vector_to_dm({0, 0.0435, -0.001542});
    casadi::DM ff_base_quat_dm = vector_to_dm({0.999048, -0.0436194, 0, 0});

    casadi::DM ff_t_base_trans = ttmat_fn_(std::vector<casadi::DM>{ff_base_trans_dm}).at(0);
    casadi::DM ff_t_base_quat = quattmat_fn_(std::vector<casadi::DM>{ff_base_quat_dm}).at(0);
    casadi::DM ff_t_base =
        casadi::DM::mtimes(t_palm, casadi::DM::mtimes(ff_t_base_trans, ff_t_base_quat));

    // Complete forward kinematics chain (simplified version for now)
    // In full implementation, you would chain all the transformations as in Python
    casadi::SX ff_result = casadi::SX::zeros(3, 1);
    ff_result(0) = ff_qpos(0) * 0.05 + 0.1;  // Simplified with base offset
    ff_result(1) = ff_qpos(1) * 0.05 + 0.0;
    ff_result(2) = ff_qpos(2) * 0.05 + 0.15;

    fftp_pos_fd_fn_ = casadi::Function("ff_t_ftp_fn", {ff_qpos}, {ff_result});

    // Similar simplified implementations for other fingers
    casadi::SX mf_qpos = casadi::SX::sym("mf_qpos", 4);
    casadi::SX mf_result = casadi::SX::zeros(3, 1);
    mf_result(0) = mf_qpos(0) * 0.05 + 0.12;
    mf_result(1) = mf_qpos(1) * 0.05 + 0.0;
    mf_result(2) = mf_qpos(2) * 0.05 + 0.15;
    mftp_pos_fd_fn_ = casadi::Function("mftp_pos_fd_fn", {mf_qpos}, {mf_result});

    casadi::SX rf_qpos = casadi::SX::sym("rf_qpos", 4);
    casadi::SX rf_result = casadi::SX::zeros(3, 1);
    rf_result(0) = rf_qpos(0) * 0.05 + 0.1;
    rf_result(1) = rf_qpos(1) * 0.05 - 0.05;
    rf_result(2) = rf_qpos(2) * 0.05 + 0.15;
    rftp_pos_fd_fn_ = casadi::Function("rftp_pos_fd_fn", {rf_qpos}, {rf_result});

    casadi::SX th_qpos = casadi::SX::sym("th_qpos", 4);
    casadi::SX th_result = casadi::SX::zeros(3, 1);
    th_result(0) = th_qpos(0) * 0.05 + 0.05;
    th_result(1) = th_qpos(1) * 0.05 + 0.08;
    th_result(2) = th_qpos(2) * 0.05 + 0.10;
    thtp_pos_fd_fn_ = casadi::Function("thtp_pos_fd_fn", {th_qpos}, {th_result});

    std::cout << "Forward kinematics functions initialized" << std::endl;
}

int MjSimulator::get_site_id(const std::string& name) {
    return mj_name2id(model_, mjOBJ_SITE, name.c_str());
}

// ************************************ //
void MjSimulator::run_simple_test() {
    std::cout << "Running simple test..." << std::endl;

    // Test basic functionality
    reset_env();
    std::cout << "Environment reset successfully" << std::endl;

    // Get initial joint positions
    Eigen::VectorXd init_jpos = get_jpos();
    std::cout << "Initial joint positions: ";
    for (int i = 0; i < std::min(8, (int)init_jpos.size()); ++i) {
        std::cout << init_jpos[i] << " ";
    }
    std::cout << std::endl;

    // Test a small step
    Eigen::VectorXd small_cmd = Eigen::VectorXd::Zero(16);
    small_cmd[0] = 0.1;  // Small movement in first joint
    step(small_cmd);
    std::cout << "Step executed successfully" << std::endl;

    // Get fingertip positions
    Eigen::VectorXd ft_pos = get_fingertips_position();
    std::cout << "Fingertip positions: ";
    for (int i = 0; i < std::min(12, (int)ft_pos.size()); ++i) {
        std::cout << ft_pos[i] << " ";
    }
    std::cout << std::endl;

    // Test forward kinematics
    std::vector<casadi::DM> ff_input = {casadi::DM::vertcat({0.1, 0.2, 0.1, 0.0})};
    std::vector<casadi::DM> ff_result = fftp_pos_fd_fn_(ff_input);
    std::cout << "Forward kinematics test result: " << ff_result[0] << std::endl;

    std::cout << "Simple test completed successfully!" << std::endl;
}

void MjSimulator::print_model_info() {
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Number of bodies: " << model_->nbody << std::endl;
    std::cout << "Number of geoms: " << model_->ngeom << std::endl;
    std::cout << "Number of DOFs: " << model_->nv << std::endl;

    std::cout << "\nGeometry names:" << std::endl;
    for (int i = 0; i < model_->ngeom; ++i) {
        const char* geom_name = mj_id2name(model_, mjOBJ_GEOM, i);
        if (geom_name) {
            std::cout << "  " << i << ": " << geom_name << std::endl;
        }
    }
}

// Rendering implementation
void MjSimulator::init_rendering(bool enable_rendering) {
    rendering_enabled_ = enable_rendering;
    if (!rendering_enabled_) return;

    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Could not initialize GLFW");
    }

    // Create window
    window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulator", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Could not create GLFW window");
    }

    // Make context current
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    // Initialize MuJoCo visualization
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    // Create scene and context
    mjv_makeScene(model_, &scn_, 2000);
    mjr_makeContext(model_, &con_, mjFONTSCALE_150);

    // Set up camera
    cam_.azimuth = 90;
    cam_.elevation = -20;
    cam_.distance = 1.5;
    cam_.lookat[0] = 0;
    cam_.lookat[1] = 0;
    cam_.lookat[2] = 0.1;

    // Install callbacks
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, keyboard_callback);
    glfwSetCursorPosCallback(window_, mouse_move_callback);
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetWindowCloseCallback(window_, window_close_callback);

    // Print help
    print_help();

    std::cout << "Rendering initialized successfully" << std::endl;
}

void MjSimulator::render() {
    if (!rendering_enabled_ || !window_) return;

    // Get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    // Update scene
    mjv_updateScene(model_, data_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);

    // Render
    mjr_render(viewport, &scn_, &con_);

    // Swap front and back buffers
    glfwSwapBuffers(window_);

    // Poll for and process events
    glfwPollEvents();
}

void MjSimulator::close_rendering() {
    if (!rendering_enabled_) return;

    if (window_) {
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
    rendering_enabled_ = false;
}

bool MjSimulator::should_close() const {
    return !rendering_enabled_ || !window_ || glfwWindowShouldClose(window_);
}

void MjSimulator::run_rendering_test() {
    std::cout << "Starting rendering test..." << std::endl;

    // Initialize rendering
    init_rendering(true);

    // Reset environment
    reset_env();

    // Main rendering loop
    int frame_count = 0;
    while (!should_close() && frame_count < 1000) {  // Run for ~10 seconds at 60fps
        // Simple animation - move finger joints
        Eigen::VectorXd cmd = Eigen::VectorXd::Zero(16);
        double t = frame_count * 0.01;
        cmd[0] = 0.3 * std::sin(t);
        cmd[1] = 0.5 * std::sin(t * 0.7);
        cmd[4] = 0.3 * std::sin(t * 1.2);
        cmd[5] = 0.4 * std::sin(t * 0.9);

        step(cmd);
        render();

        // Control frame rate (approximately 60 FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        frame_count++;
    }

    std::cout << "Rendering test completed!" << std::endl;
    close_rendering();
}

// Static callback functions
void MjSimulator::keyboard_callback(GLFWwindow* window, int key, int scancode, int action,
                                    int mods) {
    if (action == GLFW_PRESS) {
        MjSimulator* sim = static_cast<MjSimulator*>(glfwGetWindowUserPointer(window));
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                if (sim) {
                    sim->reset_env();
                    std::cout << "Environment reset" << std::endl;
                }
                break;
            case GLFW_KEY_H:
                if (sim) {
                    sim->print_help();
                }
                break;
            case GLFW_KEY_R:
                if (sim) {
                    // Reset camera to default position
                    sim->cam_.azimuth = 90;
                    sim->cam_.elevation = -20;
                    sim->cam_.distance = 1.5;
                    sim->cam_.lookat[0] = 0;
                    sim->cam_.lookat[1] = 0;
                    sim->cam_.lookat[2] = 0.1;
                    std::cout << "Camera reset to default view" << std::endl;
                }
                break;
        }
    }
}

void MjSimulator::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (MjSimulator* sim = static_cast<MjSimulator*>(glfwGetWindowUserPointer(window))) {
        // Update button state
        sim->button_left_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        sim->button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        sim->button_right_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

        // Update mouse position
        glfwGetCursorPos(window, &sim->lastx_, &sim->lasty_);
    }
}

void MjSimulator::mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
    if (MjSimulator* sim = static_cast<MjSimulator*>(glfwGetWindowUserPointer(window))) {
        if (!sim->button_left_ && !sim->button_middle_ && !sim->button_right_) {
            return;
        }

        // Compute mouse displacement
        double dx = xpos - sim->lastx_;
        double dy = ypos - sim->lasty_;
        sim->lastx_ = xpos;
        sim->lasty_ = ypos;

        // Get current window size
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        // Determine action based on mouse button
        int action = 0;
        if (sim->button_right_)
            action = mjMOUSE_MOVE_H;  // Horizontal translation
        else if (sim->button_middle_)
            action = mjMOUSE_MOVE_V;  // Vertical translation
        else if (sim->button_left_)
            action = mjMOUSE_ROTATE_H;  // Horizontal rotation

        // Update camera
        sim->update_camera_from_mouse(dx, dy, action, 0);
    }
}

void MjSimulator::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (MjSimulator* sim = static_cast<MjSimulator*>(glfwGetWindowUserPointer(window))) {
        // Zoom with scroll wheel
        sim->update_camera_from_mouse(0, -yoffset * 20, mjMOUSE_ZOOM, 0);
    }
}

void MjSimulator::window_close_callback(GLFWwindow* window) {
    std::cout << "Window close requested" << std::endl;
}

// Camera control implementation
void MjSimulator::update_camera_from_mouse(double dx, double dy, int action, int mods) {
    // Get current window size
    int width, height;
    glfwGetWindowSize(window_, &width, &height);

    // No change if no action
    if (action == 0) return;

    // Scaling factors
    double scale_rotate = 0.3;
    double scale_translate = 0.005;
    double scale_zoom = 0.01;

    switch (action) {
        case mjMOUSE_ROTATE_H:  // Horizontal rotation
            cam_.azimuth -= scale_rotate * dx;
            cam_.elevation -= scale_rotate * dy;
            // Clamp elevation
            if (cam_.elevation > 90) cam_.elevation = 90;
            if (cam_.elevation < -90) cam_.elevation = -90;
            break;

        case mjMOUSE_MOVE_H:  // Horizontal translation
        {
            double factor = scale_translate * cam_.distance;
            cam_.lookat[0] -= factor * dx;
            cam_.lookat[1] += factor * dy;
        } break;

        case mjMOUSE_MOVE_V:  // Vertical translation
        {
            double factor = scale_translate * cam_.distance;
            cam_.lookat[2] += factor * dy;
        } break;

        case mjMOUSE_ZOOM:  // Zoom
            cam_.distance += scale_zoom * dy * 2.0;
            if (cam_.distance < 0.1) cam_.distance = 0.1;
            if (cam_.distance > 20.0) cam_.distance = 20.0;
            break;
    }
}

// Interactive rendering with full mouse control
void MjSimulator::run_interactive_rendering() {
    std::cout << "Starting interactive rendering..." << std::endl;

    // Initialize rendering
    init_rendering(true);

    // Reset environment
    reset_env();

    // Main interactive loop
    double time = 0;
    while (!should_close()) {
        // Simple animation
        Eigen::VectorXd cmd = Eigen::VectorXd::Zero(16);
        cmd[0] = 0.2 * std::sin(time * 0.5);
        cmd[1] = 0.3 * std::sin(time * 0.7);
        cmd[4] = 0.2 * std::sin(time * 0.9);
        cmd[5] = 0.3 * std::sin(time * 1.1);

        step(cmd);
        render();

        // Control frame rate
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        time += 0.016;
    }

    close_rendering();
    std::cout << "Interactive rendering ended" << std::endl;
}

void MjSimulator::print_help() {
    std::cout << "\n=== MuJoCo Interactive Controls ===" << std::endl;
    std::cout << "Mouse Controls:" << std::endl;
    std::cout << "  Left drag:   Rotate camera" << std::endl;
    std::cout << "  Right drag:  Translate camera horizontally" << std::endl;
    std::cout << "  Middle drag: Translate camera vertically" << std::endl;
    std::cout << "  Scroll:      Zoom in/out" << std::endl;
    std::cout << "\nKeyboard Controls:" << std::endl;
    std::cout << "  ESC:         Exit" << std::endl;
    std::cout << "  SPACE:       Reset environment" << std::endl;
    std::cout << "  H:           Show/hide this help" << std::endl;
    std::cout << "  R:           Reset camera view" << std::endl;
    std::cout << "=================================" << std::endl;
}