#include "allegro_simulator.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace simulation {

AllegroSimulator::AllegroSimulator(const Parameters& param)
    : param_(param),
      window_(nullptr),
      rendering_enabled_(false),
      button_left_(false),
      button_middle_(false),
      button_right_(false),
      lastx_(0),
      lasty_(0),
      ffmpeg_pipe_(nullptr),
      video_width_(0),
      video_height_(0) {
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

AllegroSimulator::~AllegroSimulator() {
    stopVideoRecording();
    closeRendering();
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

void AllegroSimulator::reset() {
    // MuJoCo model layout for Allegro:
    // qpos[0:16] = robot joints
    // qpos[16:23] = object [pos(3), quat(4)]

    // Set robot joint positions (first 16 DOFs)
    for (int i = 0; i < 16; ++i) {
        data_->qpos[i] = param_.init_robot_qpos(i);
    }

    // Set object pose (next 7 DOFs)
    for (int i = 0; i < 7; ++i) {
        data_->qpos[16 + i] = param_.init_obj_qpos(i);
    }

    // Zero velocities
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }

    mj_forward(model_, data_);
}

void AllegroSimulator::step(const Eigen::VectorXd& action) {
    // action: 16D joint angle increment for Allegro hand
    // MuJoCo layout: qpos[0:16] = robot, qpos[16:23] = object

    // Get current joint positions
    Eigen::VectorXd curr_jpos(16);
    for (int i = 0; i < 16; ++i) {
        curr_jpos(i) = data_->qpos[i];
    }

    Eigen::VectorXd target_jpos = curr_jpos + action;

    // Step simulation with position control
    for (int iter = 0; iter < param_.frame_skip; ++iter) {
        // Set control to target joint positions
        for (int i = 0; i < 16 && i < model_->nu; ++i) {
            data_->ctrl[i] = target_jpos(i);
        }

        mj_step(model_, data_);
    }
    // Note: Rendering is done separately via render() call after step()
}

void AllegroSimulator::stepWithRender(const Eigen::VectorXd& action, int render_every) {
    // action: 16D joint angle increment for Allegro hand
    // render_every: render every N physics steps for smoother visualization

    // Get current joint positions
    Eigen::VectorXd curr_jpos(16);
    for (int i = 0; i < 16; ++i) {
        curr_jpos(i) = data_->qpos[i];
    }

    Eigen::VectorXd target_jpos = curr_jpos + action;

    // Step simulation with position control and intermediate rendering
    for (int iter = 0; iter < param_.frame_skip; ++iter) {
        // Set control to target joint positions
        for (int i = 0; i < 16 && i < model_->nu; ++i) {
            data_->ctrl[i] = target_jpos(i);
        }

        mj_step(model_, data_);

        // Render at specified intervals
        if (rendering_enabled_ && render_every > 0 && (iter + 1) % render_every == 0) {
            render();
        }
    }
}

void AllegroSimulator::setStateDirectly(const Eigen::VectorXd& state) {
    // Directly set state without simulation (for trajectory visualization)
    // state: [obj_pos(3), obj_quat(4), robot_qpos(16)]
    // MuJoCo layout: qpos[0:16] = robot, qpos[16:23] = object

    // Robot joint positions (first 16 in MuJoCo qpos)
    for (int i = 0; i < 16; ++i) {
        data_->qpos[i] = state(7 + i);  // robot_qpos starts at index 7 in state
    }

    // Object pose (next 7 in MuJoCo qpos)
    for (int i = 0; i < 3; ++i) {
        data_->qpos[16 + i] = state(i);  // obj_pos
    }
    for (int i = 0; i < 4; ++i) {
        data_->qpos[19 + i] = state(3 + i);  // obj_quat
    }

    // Set velocities to zero
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
    }

    // Forward kinematics to update all positions
    mj_forward(model_, data_);
}

Eigen::VectorXd AllegroSimulator::getState() const {
    // Return state: [obj_pos(3), obj_quat(4), robot_qpos(16)]
    // MuJoCo layout: qpos[0:16] = robot, qpos[16:23] = object
    Eigen::VectorXd state(23);

    // Object pose (indices 16-22 in MuJoCo qpos)
    for (int i = 0; i < 3; ++i) {
        state(i) = data_->qpos[16 + i];  // obj_pos
    }
    for (int i = 0; i < 4; ++i) {
        state(3 + i) = data_->qpos[19 + i];  // obj_quat
    }

    // Robot joint positions (indices 0-15 in MuJoCo qpos)
    for (int i = 0; i < 16; ++i) {
        state(7 + i) = data_->qpos[i];  // robot_qpos
    }

    return state;
}

Eigen::VectorXd AllegroSimulator::getJointPos() const {
    Eigen::VectorXd jpos(16);
    for (int i = 0; i < 16; ++i) {
        jpos(i) = data_->qpos[i];
    }
    return jpos;
}

Eigen::VectorXd AllegroSimulator::getFingertipsPosition() const {
    Eigen::VectorXd fts_pos(12);
    int idx = 0;
    for (const auto& ft_name : fingertip_names_) {
        int site_id = mj_name2id(model_, mjOBJ_SITE, ft_name.c_str());
        if (site_id >= 0) {
            fts_pos(idx++) = data_->site_xpos[3 * site_id + 0];
            fts_pos(idx++) = data_->site_xpos[3 * site_id + 1];
            fts_pos(idx++) = data_->site_xpos[3 * site_id + 2];
        } else {
            // Site not found, fill with zeros
            fts_pos(idx++) = 0;
            fts_pos(idx++) = 0;
            fts_pos(idx++) = 0;
        }
    }
    return fts_pos;
}

AllegroContactResult AllegroSimulator::detectContactsAt(const Eigen::VectorXd& state) {
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
    AllegroContactResult result = detectContacts();

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

AllegroContactResult AllegroSimulator::detectContacts() {
    mj_forward(model_, data_);
    mj_collision(model_, data_);

    int n_con = data_->ncon;
    constexpr int max_ncon = AllegroContactResult::MAX_NCON;
    constexpr int n_qvel = AllegroContactResult::N_QVEL;

    AllegroContactResult result;
    result.phi_vec = Eigen::VectorXd::Ones(max_ncon * 4);
    result.jac_mat = Eigen::MatrixXd::Zero(max_ncon * 4, n_qvel);

    std::vector<double> phi_list;
    std::vector<Eigen::MatrixXd> jac_list;

    for (int i = 0; i < n_con; ++i) {
        mjContact& contact = data_->contact[i];

        const char* geom1_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom1);
        const char* geom2_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom2);

        bool geom1_is_obj = false, geom2_is_obj = false;
        for (const auto& obj_name : param_.object_names) {
            if (geom1_name && std::string(geom1_name) == obj_name) geom1_is_obj = true;
            if (geom2_name && std::string(geom2_name) == obj_name) geom2_is_obj = true;
        }

        if (geom1_is_obj || geom2_is_obj) {
            // Contact involves the object
            double con_dist = contact.dist * 0.5;
            double mu = param_.mu_object;

            // Contact frame (MuJoCo stores 9 elements, 3x3 row-major)
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

            // MuJoCo fills Jacobians (model_->nv columns)
            // For Allegro: nv = 22 (16 robot + 6 object)
            int nv = model_->nv;
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp1_rm(3, nv);
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp2_rm(3, nv);
            jacp1_rm.setZero();
            jacp2_rm.setZero();

            mj_jac(model_, data_, jacp1_rm.data(), nullptr, contact.pos, body1_id);
            mj_jac(model_, data_, jacp2_rm.data(), nullptr, contact.pos, body2_id);

            // Convert to column-major
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
            Eigen::MatrixXd con_jacp_f = con_jacp.block(1, 0, 4, nv);

            // Combined Jacobian with friction: J_i = J_n + mu * J_f[i]
            Eigen::MatrixXd con_jac(4, nv);
            for (int k = 0; k < 4; ++k) {
                con_jac.row(k) = con_jacp_n.transpose() + mu * con_jacp_f.row(k);
            }

            phi_list.push_back(con_dist);
            jac_list.push_back(con_jac);
        }
    }

    // Reorder Jacobian columns: MuJoCo has [robot(16), obj(6)], we need [obj(6), robot(16)]
    // Fill result with reordering
    for (size_t i = 0; i < phi_list.size() && i < static_cast<size_t>(max_ncon); ++i) {
        for (int j = 0; j < 4; ++j) {
            result.phi_vec(4 * i + j) = phi_list[i];

            // Reorder: [robot(16), obj(6)] -> [obj(6), robot(16)]
            for (int k = 0; k < 6; ++k) {
                result.jac_mat(4 * i + j, k) = jac_list[i](j, 16 + k);  // obj velocities
            }
            for (int k = 0; k < 16; ++k) {
                result.jac_mat(4 * i + j, 6 + k) = jac_list[i](j, k);  // robot velocities
            }
        }
    }

    return result;
}

RealContactInfo AllegroSimulator::getRealContactInfo(double h) {
    // Make sure forward dynamics is computed
    mj_forward(model_, data_);

    RealContactInfo info;
    info.num_contacts = 0;
    info.comp_violation = 0.0;
    info.mean_comp_violation = 0.0;

    std::vector<double> phi_list;
    std::vector<double> lambda_list;
    std::vector<double> H_list;

    int n_con = data_->ncon;

    for (int i = 0; i < n_con; ++i) {
        mjContact& contact = data_->contact[i];

        // Check if contact involves the object
        const char* geom1_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom1);
        const char* geom2_name = mj_id2name(model_, mjOBJ_GEOM, contact.geom2);

        bool geom1_is_obj = false, geom2_is_obj = false;
        for (const auto& obj_name : param_.object_names) {
            if (geom1_name && std::string(geom1_name) == obj_name) geom1_is_obj = true;
            if (geom2_name && std::string(geom2_name) == obj_name) geom2_is_obj = true;
        }

        if (geom1_is_obj || geom2_is_obj) {
            // Get contact gap (signed distance)
            double phi = contact.dist * 0.5;  // Same scaling as detectContacts
            phi_list.push_back(phi);

            // Get real contact force from MuJoCo
            // mj_contactForce returns force in contact frame: [normal, tangent1, tangent2, ...]
            mjtNum force[6];  // 6D force (3 linear + 3 torque for soft contacts)
            mj_contactForce(model_, data_, i, force);

            // Normal force magnitude (first component is normal force)
            double normal_force = std::abs(force[0]);
            lambda_list.push_back(normal_force);

            // Compute gap rate H = J @ v + phi / h
            // Get contact Jacobian
            int body1_id = model_->geom_bodyid[contact.geom1];
            int body2_id = model_->geom_bodyid[contact.geom2];

            int nv = model_->nv;
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp1(3, nv);
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp2(3, nv);
            jacp1.setZero();
            jacp2.setZero();

            mj_jac(model_, data_, jacp1.data(), nullptr, contact.pos, body1_id);
            mj_jac(model_, data_, jacp2.data(), nullptr, contact.pos, body2_id);

            // Contact normal (first 3 elements of contact.frame, row-major)
            Eigen::Vector3d normal;
            normal << contact.frame[0], contact.frame[1], contact.frame[2];

            // Relative velocity in contact normal direction
            Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(data_->qvel, nv);
            Eigen::Vector3d v1 = jacp1 * qvel;
            Eigen::Vector3d v2 = jacp2 * qvel;

            // Relative velocity: positive if separating
            double v_rel;
            if (geom1_is_obj) {
                v_rel = normal.dot(v1 - v2);  // Object velocity - robot velocity
            } else {
                v_rel = normal.dot(v2 - v1);  // Object velocity - robot velocity
            }

            // Gap rate: H = v_rel + phi / h
            double H = v_rel + phi / h;
            H_list.push_back(H);

            info.num_contacts++;
        }
    }

    // Fill result vectors
    int n = static_cast<int>(phi_list.size());
    info.phi_vec.resize(n);
    info.lambda_vec.resize(n);
    info.H_vec.resize(n);

    double sum_comp = 0.0;
    double max_comp = 0.0;

    for (int i = 0; i < n; ++i) {
        info.phi_vec(i) = phi_list[i];
        info.lambda_vec(i) = lambda_list[i];
        info.H_vec(i) = H_list[i];

        // Complementarity violation: |lambda * H|
        double comp = std::abs(lambda_list[i] * H_list[i]);
        sum_comp += comp;
        if (comp > max_comp) {
            max_comp = comp;
        }
    }

    info.comp_violation = max_comp;
    info.mean_comp_violation = (n > 0) ? (sum_comp / n) : 0.0;

    return info;
}

void AllegroSimulator::setGoal(const Eigen::Vector3d& pos, const Eigen::Vector4d& quat) {
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

void AllegroSimulator::initRendering() {
    if (!glfwInit()) {
        throw std::runtime_error("Could not initialize GLFW");
    }

    window_ = glfwCreateWindow(1200, 900, "Allegro LCP MPC", nullptr, nullptr);
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

    // Set up camera for Allegro hand scene
    cam_.azimuth = 90;
    cam_.elevation = -30;
    cam_.distance = 0.5;
    cam_.lookat[0] = 0;
    cam_.lookat[1] = 0;
    cam_.lookat[2] = 0.1;

    // Install callbacks
    glfwSetWindowUserPointer(window_, this);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, mouseMoveCallback);
    glfwSetScrollCallback(window_, scrollCallback);

    rendering_enabled_ = true;
}

void AllegroSimulator::render() {
    if (!rendering_enabled_ || !window_) return;

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);

    mjv_updateScene(model_, data_, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    // Capture frame if recording
    if (ffmpeg_pipe_) {
        captureFrame();
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

bool AllegroSimulator::shouldClose() const { return window_ && glfwWindowShouldClose(window_); }

void AllegroSimulator::closeRendering() {
    if (window_) {
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
    rendering_enabled_ = false;
}

void AllegroSimulator::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    AllegroSimulator* sim = static_cast<AllegroSimulator*>(glfwGetWindowUserPointer(window));
    if (!sim) return;

    sim->button_left_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    sim->button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    sim->button_right_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    glfwGetCursorPos(window, &sim->lastx_, &sim->lasty_);
}

void AllegroSimulator::mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
    AllegroSimulator* sim = static_cast<AllegroSimulator*>(glfwGetWindowUserPointer(window));
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
    // Right button: also rotate view
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

void AllegroSimulator::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    AllegroSimulator* sim = static_cast<AllegroSimulator*>(glfwGetWindowUserPointer(window));
    if (!sim) return;

    sim->cam_.distance -= yoffset * 0.05;
    if (sim->cam_.distance < 0.1) sim->cam_.distance = 0.1;
}

void AllegroSimulator::startVideoRecording(const std::string& video_path, int fps) {
    if (!rendering_enabled_ || !window_) {
        std::cerr << "Error: Cannot start video recording without rendering enabled" << std::endl;
        return;
    }

    if (ffmpeg_pipe_) {
        std::cerr << "Warning: Already recording, stopping previous recording" << std::endl;
        stopVideoRecording();
    }

    // Get current framebuffer size
    glfwGetFramebufferSize(window_, &video_width_, &video_height_);

    // Allocate frame buffer (RGB)
    frame_buffer_.resize(3 * video_width_ * video_height_);

    // Build ffmpeg command
    // Using libx264 for good quality/compression, yuv420p for compatibility
    std::string cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " +
                      std::to_string(video_width_) + "x" + std::to_string(video_height_) +
                      " -framerate " + std::to_string(fps) +
                      " -i - -vf vflip -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \"" +
                      video_path + "\" 2>/dev/null";

    ffmpeg_pipe_ = popen(cmd.c_str(), "w");
    if (!ffmpeg_pipe_) {
        std::cerr << "Error: Failed to open ffmpeg pipe for video recording" << std::endl;
        return;
    }

    std::cout << "Started video recording: " << video_path << " (" << video_width_ << "x"
              << video_height_ << " @ " << fps << "fps)" << std::endl;
}

void AllegroSimulator::stopVideoRecording() {
    if (ffmpeg_pipe_) {
        pclose(ffmpeg_pipe_);
        ffmpeg_pipe_ = nullptr;
        frame_buffer_.clear();
        std::cout << "Video recording stopped" << std::endl;
    }
}

void AllegroSimulator::captureFrame() {
    if (!ffmpeg_pipe_ || !rendering_enabled_ || !window_) return;

    // Read pixels from OpenGL framebuffer
    mjr_readPixels(frame_buffer_.data(), nullptr, {0, 0, video_width_, video_height_}, &con_);

    // Write to ffmpeg pipe
    fwrite(frame_buffer_.data(), 1, frame_buffer_.size(), ffmpeg_pipe_);
}

}  // namespace simulation
