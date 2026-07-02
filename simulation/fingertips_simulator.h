#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjui.h>
#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <memory>
#include <string>

namespace simulation {

// Forward declaration
struct FingertipsContactResult;

/**
 * @brief Simulator for the 3-fingertip cube manipulation environment
 *
 * MuJoCo model layout:
 * - qpos[0:7] = object [pos(3), quat(4)]
 * - qpos[7:16] = fingertips [ftp0(3), ftp1(3), ftp2(3)]
 * - qvel[0:6] = object [linear(3), angular(3)]
 * - qvel[6:15] = fingertips [ftp0(3), ftp1(3), ftp2(3)]
 */
class FingertipsSimulator {
   public:
    struct Parameters {
        std::string model_path;
        Eigen::Vector3d target_p;
        Eigen::Vector4d target_q;
        Eigen::VectorXd init_robot_qpos;  // 9D for 3 fingertips
        Eigen::VectorXd init_obj_qpos;    // 7D [pos(3), quat(4)]
        int frame_skip = 50;
        double robot_stiff = 100.0;
    };

    FingertipsSimulator(const Parameters& param);
    ~FingertipsSimulator();

    // Core simulation methods
    void reset();
    void step(const Eigen::VectorXd& action);

    // Directly set state (for trajectory visualization without simulation)
    void setStateDirectly(const Eigen::VectorXd& state);

    // State getters (returns [obj_pos(3), obj_quat(4), robot_qpos(9)])
    Eigen::VectorXd getState() const;

    // Contact detection
    FingertipsContactResult detectContacts();

    // Contact detection at a given state (for trajectory optimization)
    // Temporarily sets MuJoCo state, computes contacts, then restores
    FingertipsContactResult detectContactsAt(const Eigen::VectorXd& state);

    // Goal setting
    void setGoal(const Eigen::Vector3d& pos, const Eigen::Vector4d& quat);

    // Rendering
    void initRendering();
    void render();
    bool shouldClose() const;
    void closeRendering();

    // Access to MuJoCo model (for contact detection if needed externally)
    mjModel* model() const { return model_; }
    mjData* data() const { return data_; }

   private:
    Parameters param_;
    mjModel* model_;
    mjData* data_;

    // Rendering
    GLFWwindow* window_;
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;
    bool rendering_enabled_;
    bool button_left_, button_middle_, button_right_;
    double lastx_, lasty_;

    // Mouse callbacks (static)
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};

/**
 * @brief Contact detection result for fingertips environment
 */
struct FingertipsContactResult {
    Eigen::VectorXd phi_vec;  // Contact distances (max_ncon * 4)
    Eigen::MatrixXd jac_mat;  // Contact Jacobians (max_ncon * 4, n_qvel)

    // Default: 8 contacts max, 15 velocity DOFs
    static constexpr int MAX_NCON = 8;
    static constexpr int N_QVEL = 15;
};

}  // namespace simulation
