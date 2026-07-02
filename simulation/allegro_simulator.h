#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjui.h>
#include <mujoco/mujoco.h>

#include <Eigen/Dense>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace simulation {

// Forward declaration
struct AllegroContactResult;
struct RealContactInfo;

/**
 * @brief Real contact forces and complementarity info from MuJoCo
 */
struct RealContactInfo {
    Eigen::VectorXd phi_vec;     // Contact gaps (signed distances)
    Eigen::VectorXd lambda_vec;  // Real contact forces (normal force magnitude)
    Eigen::VectorXd H_vec;       // Gap rate: J @ v + phi / h
    double comp_violation;       // max |lambda * H|
    double mean_comp_violation;  // mean |lambda * H|
    int num_contacts;            // Number of active contacts
};

/**
 * @brief Simulator for the Allegro hand cube manipulation environment
 *
 * MuJoCo model layout (Allegro hand):
 * - qpos[0:16] = robot joints (16 DOF)
 * - qpos[16:23] = object [pos(3), quat(4)]
 * - qvel[0:16] = robot joint velocities
 * - qvel[16:22] = object [linear(3), angular(3)]
 *
 * State convention for LCP (reordered for consistency):
 * - state[0:7] = object [pos(3), quat(4)]
 * - state[7:23] = robot joints (16 DOF)
 */
class AllegroSimulator {
   public:
    struct Parameters {
        std::string model_path;
        Eigen::Vector3d target_p;
        Eigen::Vector4d target_q;
        Eigen::VectorXd init_robot_qpos;  // 16D for Allegro hand joints
        Eigen::VectorXd init_obj_qpos;    // 7D [pos(3), quat(4)]
        int frame_skip = 50;              // Default from Python
        double robot_stiff = 1.0;         // Joint stiffness (Python uses 1.0)
        std::vector<std::string> object_names = {"obj"};
        double mu_object = 0.5;  // Friction coefficient
    };

    AllegroSimulator(const Parameters& param);
    ~AllegroSimulator();

    // Core simulation methods
    void reset();
    void step(const Eigen::VectorXd& action);

    // Step with intermediate rendering for smoother visualization
    // render_every: render every N physics steps (0 = no intermediate rendering)
    void stepWithRender(const Eigen::VectorXd& action, int render_every = 5);

    // Directly set state (for trajectory visualization without simulation)
    void setStateDirectly(const Eigen::VectorXd& state);

    // State getters (returns [obj_pos(3), obj_quat(4), robot_qpos(16)])
    Eigen::VectorXd getState() const;

    // Get robot joint positions only
    Eigen::VectorXd getJointPos() const;

    // Get fingertip positions from MuJoCo sites (12D: 4 fingertips × 3D)
    Eigen::VectorXd getFingertipsPosition() const;

    // Contact detection
    AllegroContactResult detectContacts();

    // Contact detection at a given state
    AllegroContactResult detectContactsAt(const Eigen::VectorXd& state);

    // Get real contact forces and complementarity violation from MuJoCo
    // h: timestep for computing gap rate H = Jv + phi/h
    RealContactInfo getRealContactInfo(double h = 0.1);

    // Goal setting
    void setGoal(const Eigen::Vector3d& pos, const Eigen::Vector4d& quat);

    // Rendering
    void initRendering();
    void render();
    bool shouldClose() const;
    void closeRendering();

    // Video recording
    void startVideoRecording(const std::string& video_path, int fps = 30);
    void stopVideoRecording();
    bool isRecording() const { return ffmpeg_pipe_ != nullptr; }
    void captureFrame();  // Call this each time you want to add a frame

    // Access to MuJoCo model
    mjModel* model() const { return model_; }
    mjData* data() const { return data_; }

   private:
    Parameters param_;
    mjModel* model_;
    mjData* data_;

    // Fingertip site names
    std::vector<std::string> fingertip_names_ = {"ftp_0", "ftp_1", "ftp_2", "ftp_3"};

    // Rendering
    GLFWwindow* window_;
    mjvCamera cam_;
    mjvOption opt_;
    mjvScene scn_;
    mjrContext con_;
    bool rendering_enabled_;
    bool button_left_, button_middle_, button_right_;
    double lastx_, lasty_;

    // Video recording
    FILE* ffmpeg_pipe_;
    std::vector<unsigned char> frame_buffer_;
    int video_width_, video_height_;

    // Mouse callbacks (static)
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};

/**
 * @brief Contact detection result for Allegro environment
 */
struct AllegroContactResult {
    Eigen::VectorXd phi_vec;  // Contact distances (max_ncon * 4)
    Eigen::MatrixXd jac_mat;  // Contact Jacobians (max_ncon * 4, n_qvel)

    // Default: 20 contacts max, 22 velocity DOFs (6 object + 16 robot)
    static constexpr int MAX_NCON = 20;
    static constexpr int N_QVEL = 22;
};

}  // namespace simulation
