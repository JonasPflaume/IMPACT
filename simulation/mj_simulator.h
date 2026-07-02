#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjui.h>
#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <array>
#include <casadi/casadi.hpp>
#include <string>
#include <vector>

class MjSimulator {
   public:
    struct Parameters {
        std::string model_path;
        Eigen::Vector3d target_p;
        Eigen::Vector4d target_q;
        Eigen::VectorXd init_robot_qpos;
        Eigen::VectorXd init_obj_qpos;
        int frame_skip = 10;
    };

    MjSimulator(const Parameters& param);
    ~MjSimulator();

    // Core simulation methods
    void reset_env();
    void step(const Eigen::VectorXd& jpos_cmd);
    void reset_fingers_qpos();

    // State getters
    Eigen::VectorXd get_state();
    Eigen::VectorXd get_jpos();
    Eigen::VectorXd get_fingertips_position();

    // Goal setting
    void set_goal(const Eigen::Vector3d* goal_pos = nullptr,
                  const Eigen::Vector4d* goal_quat = nullptr);

    // Forward kinematics
    void allegro_fd_fn();

    // Rendering functions
    void init_rendering(bool enable_rendering = true);
    void render();
    void close_rendering();
    bool should_close() const;

    // Interactive rendering with full mouse control
    void run_interactive_rendering();

    // Test functions
    void run_simple_test();
    void run_rendering_test();
    void print_model_info();

    // Public access for Contact class
    mjModel* model_;
    mjData* data_;

   private:
    Parameters param_;

    std::vector<std::string> fingertip_names_;

    // Rendering variables
    GLFWwindow* window_;
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;
    bool rendering_enabled_;

    // Mouse interaction state
    bool button_left_, button_middle_, button_right_;
    double lastx_, lasty_;

    // Interaction state
    bool ui_enabled_;

    // Rendering callbacks
    static void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void window_close_callback(GLFWwindow* window);

    // Internal helpers
    void update_camera_from_mouse(double dx, double dy, int action, int mods);
    void print_help();

    // CasADi forward kinematics functions
    casadi::Function fftp_pos_fd_fn_;  // First finger
    casadi::Function mftp_pos_fd_fn_;  // Middle finger
    casadi::Function rftp_pos_fd_fn_;  // Ring finger
    casadi::Function thtp_pos_fd_fn_;  // Thumb

    // CasADi rotation functions
    casadi::Function ttmat_fn_;
    casadi::Function rxtmat_fn_;
    casadi::Function rytmat_fn_;
    casadi::Function rztmat_fn_;
    casadi::Function quattmat_fn_;
    casadi::Function quat2dcm_fn_;

    // Initialize rotation functions
    void init_rotation_functions();

    // Helper utility functions
    Eigen::Vector4d normalize_quat(const Eigen::Vector4d& quat);
    casadi::DM vector_to_dm(const std::vector<double>& vec);

    // Utility functions
    int get_site_id(const std::string& name);
};