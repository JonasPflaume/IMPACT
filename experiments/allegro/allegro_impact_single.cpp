/**
 * @file allegro_impact_single.cpp
 * @brief Allegro hand receding-horizon MPC driver.
 *
 * Main MPC loop using:
 * - simulation::AllegroSimulator for physics simulation and rendering
 * - impact::SingleShootingSolver for single-shooting BCD-AuLa optimization
 *
 * Based on fingertips_bcd_aula.cpp and adapted for the 16-DOF Allegro hand.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "allegro_simulator.h"
#include "impact/single_shooting.h"
#include "penalty_solver/allegro_lcp_problem.h"
#include "rotation_utils.h"

// Project source directory (set via CMake)
#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

/**
 * @brief Solver parameters used by the Allegro MPC example.
 * Default values are universal parameters from grid search weighted voting:
 *   velocity_penalty=0.1, final_cost_multiplier=10, control_cost_weight=0.1,
 * final_quaternion_weight=7.5
 */
struct ObjectSolverParams {
    // Task weights.
    double position_cost_weight = 0.0;     // Universal: 0.0 (contact-based control)
    double quaternion_cost_weight = 0.0;   // Universal: 0.0 (contact-based control)
    double contact_cost_weight = 1.0;      // Universal: 1.0
    double grasp_closure_weight = 0.0;     // Universal: 0.0
    double control_cost_weight = 0.2;      // Universal: 0.2 (tuned 0.1->0.2; +stability)
    double velocity_penalty = 0.4;         // Universal: 0.4 (tuned 0.1->0.4; +stability)
    double final_cost_multiplier = 10.0;   // Universal: 10.0 (from grid search)
    double final_position_weight = 100.0;  // Universal: 100.0
    double final_quaternion_weight = 9.0;  // Universal: 9.0 (from grid search)
    // ^ control_cost_weight & velocity_penalty retuned via 17-object x 10-seed sweep:
    //   success rate 85.9% -> 91.2% (146 -> 155 / 170). fcm/fpw/fqw unchanged (no gain).
    double cmd_bound = 0.1;                // Universal: 0.1 (matches grid search)

    // Solver settings.
    int horizon = 4;
    double rho_dynamics_init = 1.0;
    double rho_comp_init = 1.0;
    double rho_cmd_init = 1.0;
    double rho_max = 1e3;
    double rho_scale = 5.0;
    double dynamics_scale = 25.0;
    double comp_scale = 1.0;
    int max_outer_iters = 10;
    double outer_tol_h = 1e-3;
    double outer_tol_comp = 1e-3;
    int max_inner_iters = 5;
    int newton_max_iter = 30;
    double newton_step_tol = 1e-5;
    double newton_obj_tol = 1e-6;
};

/**
 * @brief Convert RPY angles to quaternion (matching Python rpy_to_quaternion)
 */
Eigen::Vector4d rpyToQuaternion(double yaw, double pitch, double roll) {
    double cy = std::cos(yaw / 2);
    double sy = std::sin(yaw / 2);
    double cp = std::cos(pitch / 2);
    double sp = std::sin(pitch / 2);
    double cr = std::cos(roll / 2);
    double sr = std::sin(roll / 2);

    Eigen::Vector4d q;
    q(0) = cr * cp * cy + sr * sp * sy;  // w
    q(1) = sr * cp * cy - cr * sp * sy;  // x
    q(2) = cr * sp * cy + sr * cp * sy;  // y
    q(3) = cr * cp * sy - sr * sp * cy;  // z
    return q;
}

/**
 * @brief Compute quaternion error (matching Python metrics.comp_quat_error)
 */
double computeQuatError(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
    double dot = std::abs(q1.dot(q2));
    return 1.0 - dot * dot;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --render              Enable rendering" << std::endl;
    std::cout << "  --save-video <path>   Save video to file (requires --render)" << std::endl;
    std::cout << "  --video-fps <int>     Video framerate (default: 30)" << std::endl;
    std::cout << "  --object <name>       Object to manipulate (default: cube)" << std::endl;
    std::cout << "  --seed <int>          Random seed for trial (default: 0)" << std::endl;
    std::cout << "  --json                Output results in JSON format" << std::endl;
    std::cout << "  --output <file>       Output file path (default: stdout)" << std::endl;
    std::cout << "  --codegen             Enable CasADi CodeGen" << std::endl;
    std::cout << "  --no-saddle           Use original normal-equations GN (default: IMPACT saddle)"
              << std::endl;
    std::cout << "  --no-forcing          Disable forcing-sequence inexact X-update (default: on)"
              << std::endl;
    std::cout << "Available objects: airplane, binoculars, bowl, bunny, camera, can, cube, cup,"
              << std::endl;
    std::cout << "                   elephant, flashlight, foambrick, light_bulb, mug, piggy_bank,"
              << std::endl;
    std::cout << "                   rubber_duck, stick, teapot, torus, water_bottle" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments.
    std::string object_name = "cube";  // default object
    bool with_rendering = false;
    int trial_seed = 0;  // default seed
    bool json_output = false;
    std::string output_file = "";
    bool use_codegen = false;
    bool use_saddle = true;  // IMPACT primal saddle XCD (default); --no-saddle => original GN
    bool use_forcing = true;  // forcing-sequence inexact X-update (default); --no-forcing disables
    std::string video_path = "";  // Video output path (empty = no recording)
    int video_fps = 30;                 // Video framerate

    // Parameter tuning variables
    double velocity_penalty = -1.0;  // -1 means use default from object params
    double final_cost_multiplier = -1.0;
    double control_cost_weight = -1.0;
    double final_quaternion_weight = -1.0;
    double final_position_weight = -1.0;  // -1 means use default
    int horizon = -1;  // -1 means use default

    // Solver config tuning variables
    double rho_scale = -1.0;
    double outer_tol_comp = -1.0;
    int max_inner_iters = -1;
    int max_steps_override = -1;  // -1 means use default (1000); for fast sweeps

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--render") {
            with_rendering = true;
        } else if (arg == "--save-video" && i + 1 < argc) {
            video_path = argv[++i];
            with_rendering = true;  // Video requires rendering
        } else if (arg == "--video-fps" && i + 1 < argc) {
            video_fps = std::stoi(argv[++i]);
        } else if (arg == "--object" && i + 1 < argc) {
            object_name = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            trial_seed = std::stoi(argv[++i]);
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--codegen") {
            use_codegen = true;
        } else if (arg == "--no-saddle") {
            use_saddle = false;  // use the original normal-equations Gauss-Newton XCD
        } else if (arg == "--no-forcing") {
            use_forcing = false;  // disable the forcing-sequence inexact X-update
        } else if (arg == "--velocity-penalty" && i + 1 < argc) {
            velocity_penalty = std::stod(argv[++i]);
        } else if (arg == "--final-cost-multiplier" && i + 1 < argc) {
            final_cost_multiplier = std::stod(argv[++i]);
        } else if (arg == "--control-cost-weight" && i + 1 < argc) {
            control_cost_weight = std::stod(argv[++i]);
        } else if (arg == "--final-quaternion-weight" && i + 1 < argc) {
            final_quaternion_weight = std::stod(argv[++i]);
        } else if (arg == "--final-position-weight" && i + 1 < argc) {
            final_position_weight = std::stod(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps_override = std::stoi(argv[++i]);
        } else if (arg == "--horizon" && i + 1 < argc) {
            horizon = std::stoi(argv[++i]);
        } else if (arg == "--rho-scale" && i + 1 < argc) {
            rho_scale = std::stod(argv[++i]);
        } else if (arg == "--outer-tol-comp" && i + 1 < argc) {
            outer_tol_comp = std::stod(argv[++i]);
        } else if (arg == "--max-inner-iters" && i + 1 < argc) {
            max_inner_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Validate object name
    std::vector<std::string> valid_objects = {
        "airplane",    "binoculars", "bowl",       "bunny",     "camera",      "can", "cube",
        "cup",         "elephant",   "flashlight", "foambrick", "light_bulb",  "mug", "piggy_bank",
        "rubber_duck", "stick",      "teapot",     "torus",     "water_bottle"};
    bool valid = false;
    for (const auto& obj : valid_objects) {
        if (obj == object_name) {
            valid = true;
            break;
        }
    }
    if (!valid) {
        std::cerr << "Error: Unknown object '" << object_name << "'" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    int trial_count = trial_seed;  // Use provided seed, matching np.random.seed(100 + seed)
    std::mt19937 rng(100 + trial_count);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);
    std::normal_distribution<double> randn(0.0, 1.0);

    // XML path (allegro MuJoCo models live in the in-tree resources/ folder)
    std::string xml_path = std::string(PROJECT_SOURCE_DIR) + "/resources/xmls/env_allegro_" +
                           object_name + ".xml";

    if (!json_output) {
        std::cout << "=== Allegro Hand LCP BCD-AULA Solver ===" << std::endl;
        std::cout << "Loading model from: " << xml_path << std::endl;
    }

    // Initial robot joint positions (from Python)
    Eigen::VectorXd init_robot_qpos(16);
    init_robot_qpos << 0.125, 1.13, 1.45, 1.24,  // First finger
        -0.02, 0.445, 1.17, 1.5,                 // Middle finger
        -0.459, 1.54, 1.11, 1.23,                // Ring finger
        0.638, 1.85, 1.5, 1.26;                  // Thumb

    // Object-specific initial and target positions (matching cfree params.py)
    Eigen::Vector2d init_obj_xy_base;
    double init_height;
    Eigen::Vector3d target_p;

    // Set object-specific parameters
    if (object_name == "cube") {
        init_obj_xy_base << -0.03, -0.01;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.05;
    } else if (object_name == "bunny") {
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.04;
        target_p << 0.0, 0.0, 0.04;
    } else if (object_name == "stick") {
        init_obj_xy_base << -0.01, 0.0;
        init_height = 0.03;
        target_p << -0.01, 0.0, 0.03;
    } else if (object_name == "camera") {
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.04;
    } else if (object_name == "rubber_duck") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.05;
        target_p << 0.0, 0.0, 0.05;
    } else if (object_name == "teapot") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.05;
        target_p << 0.0, 0.0, 0.05;
    } else if (object_name == "bowl") {
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.04;
        target_p << -0.03, -0.01, 0.04;
    } else if (object_name == "airplane") {
        init_obj_xy_base << -0.03, -0.01;
        init_height = 0.03;
        target_p << -0.02, -0.01, 0.03;
    } else if (object_name == "binoculars") {
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.04;
    } else if (object_name == "can") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.06;
    } else if (object_name == "cup") {
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.03;
        target_p << 0.0, 0.0, 0.03;
    } else if (object_name == "elephant") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.04;
    } else if (object_name == "foambrick") {
        init_obj_xy_base << -0.01, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.04;
    } else if (object_name == "mug") {
        init_obj_xy_base << 0.0, 0.0;
        init_height = 0.04;
        target_p << 0.0, 0.0, 0.04;
    } else if (object_name == "piggy_bank") {
        init_obj_xy_base << 0.0, 0.0;
        init_height = 0.04;
        target_p << 0.0, 0.0, 0.04;
    } else if (object_name == "torus") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.04;
        target_p << 0.0, 0.0, 0.04;
    } else if (object_name == "water_bottle") {
        init_obj_xy_base << -0.02, 0.0;
        init_height = 0.06;
        target_p << 0.0, 0.0, 0.06;
    } else {
        // Fallback default
        init_obj_xy_base << -0.03, 0.0;
        init_height = 0.04;
        target_p << -0.01, 0.0, 0.04;
    }

    // Add random perturbation to initial position (matching Python: 0.005 * np.random.randn(2))
    Eigen::Vector2d init_obj_xy;
    init_obj_xy(0) = init_obj_xy_base(0) + 0.005 * randn(rng);
    init_obj_xy(1) = init_obj_xy_base(1) + 0.005 * randn(rng);

    // Random yaw angle for initial orientation (matching Python: np.pi * np.random.rand(1) - np.pi
    // / 2)
    double init_yaw = M_PI * rand01(rng) - M_PI / 2.0;
    Eigen::Vector4d init_obj_quat = rpyToQuaternion(init_yaw, 0.0, 0.0);

    Eigen::VectorXd init_obj_qpos(7);
    init_obj_qpos << init_obj_xy(0), init_obj_xy(1), init_height, init_obj_quat(0),
        init_obj_quat(1), init_obj_quat(2), init_obj_quat(3);

    if (!json_output) {
        std::cout << "Initial robot qpos: " << init_robot_qpos.transpose() << std::endl;
        std::cout << "Initial object pose: " << init_obj_qpos.head<3>().transpose() << std::endl;
    }

    // Target state from Python params.py, target_type='rotation'.
    // target_p is already set above based on object

    // Random target yaw: init_yaw +/- pi/2 (matching Python: np.random.choice([np.pi/2, -np.pi/2]))
    // np.random.choice uses randint internally, which consumes one random number
    double choice_rand = rand01(rng);
    double target_yaw = init_yaw + (choice_rand < 0.5 ? M_PI / 2.0 : -M_PI / 2.0);
    Eigen::Vector4d target_q = rpyToQuaternion(target_yaw, 0.0, 0.0);

    if (!json_output) {
        std::cout << "Target position: " << target_p.transpose() << std::endl;
        std::cout << "Target quaternion: " << target_q.transpose() << std::endl;
    }

    // Create simulator.
    simulation::AllegroSimulator::Parameters sim_param;
    sim_param.model_path = xml_path;
    sim_param.target_p = target_p;
    sim_param.target_q = target_q;
    sim_param.init_robot_qpos = init_robot_qpos;
    sim_param.init_obj_qpos = init_obj_qpos;
    sim_param.frame_skip = 50;  // From Python: frame_skip_ = 50
    sim_param.mu_object = 0.5;  // From Python: mu_object_ = 0.5

    simulation::AllegroSimulator sim(sim_param);

    // Initialize rendering if requested.
    if (with_rendering) {
        sim.initRendering();
        std::cout << "Rendering enabled" << std::endl;

        // Start video recording when a path was provided.
        if (!video_path.empty()) {
            sim.startVideoRecording(video_path, video_fps);
        }
    }

    // Create problem and solver.
    // Universal solver parameters for every object. Per-object tuning was removed:
    // a single grid-search parameter set yields a higher overall success rate
    // across the object set than the per-object schedule.
    ObjectSolverParams obj_params;

    // Command-line overrides.
    if (velocity_penalty >= 0.0) obj_params.velocity_penalty = velocity_penalty;
    if (final_cost_multiplier >= 0.0) obj_params.final_cost_multiplier = final_cost_multiplier;
    if (control_cost_weight >= 0.0) obj_params.control_cost_weight = control_cost_weight;
    if (final_quaternion_weight >= 0.0)
        obj_params.final_quaternion_weight = final_quaternion_weight;
    if (final_position_weight >= 0.0) obj_params.final_position_weight = final_position_weight;
    if (horizon >= 0) obj_params.horizon = horizon;

    // Solver-parameter overrides.
    if (rho_scale >= 0.0) obj_params.rho_scale = rho_scale;
    if (outer_tol_comp >= 0.0) obj_params.outer_tol_comp = outer_tol_comp;
    if (max_inner_iters >= 0) obj_params.max_inner_iters = max_inner_iters;

    // Adjust weights for the diff-log orientation cost used by this solver. The
    // default Allegro weights were tuned for the penalty solver's 1-dot^2 form.
    allegro_lcp::AllegroLCPProblem::Parameters problem_params;
    // Copy tuned weights into the problem parameters.
    problem_params.velocity_penalty = obj_params.velocity_penalty;
    problem_params.position_cost_weight = obj_params.position_cost_weight;
    problem_params.quaternion_cost_weight = obj_params.quaternion_cost_weight;
    problem_params.contact_cost_weight = obj_params.contact_cost_weight;
    problem_params.grasp_closure_weight = obj_params.grasp_closure_weight;
    problem_params.control_cost_weight = obj_params.control_cost_weight;
    problem_params.final_cost_multiplier = obj_params.final_cost_multiplier;
    problem_params.final_position_weight = obj_params.final_position_weight;
    problem_params.final_quaternion_weight = obj_params.final_quaternion_weight;

    auto problem = std::make_shared<allegro_lcp::AllegroLCPProblem>(problem_params);

    impact::SingleShootingSolver solver(problem);

    // Configure the solver from the selected parameter set.
    impact::BCDAULAConfig config;
    config.horizon = obj_params.horizon;
    config.rho_dynamics_init = obj_params.rho_dynamics_init;
    config.rho_comp_init = obj_params.rho_comp_init;
    config.rho_ineq_init = obj_params.rho_cmd_init;  // cmd bounds = the inequality channel
    config.rho_max = obj_params.rho_max;
    config.rho_scale = obj_params.rho_scale;
    config.dynamics_scale = obj_params.dynamics_scale;
    config.comp_scale = obj_params.comp_scale;
    config.max_outer_iters = obj_params.max_outer_iters;
    config.outer_tol_h = obj_params.outer_tol_h;
    config.outer_tol_comp = obj_params.outer_tol_comp;
    config.outer_tol_g = obj_params.outer_tol_comp;
    config.max_inner_iters = obj_params.max_inner_iters;
    config.newton_max_iter = obj_params.newton_max_iter;
    config.newton_step_tol = obj_params.newton_step_tol;
    config.newton_tol = 1e-5;             // legacy LCPBCDAULAConfig default
    config.newton_regularization = 1e-6;  // legacy LCPBCDAULAConfig default
    config.print_level = 0;  // 0=quiet, 1=outer iter, 2=detailed

    // Control bounds via the Augmented Lagrangian inequality channel.
    config.use_cmd_bounds = true;
    config.cmd_lower = Eigen::VectorXd::Constant(16, -obj_params.cmd_bound);
    config.cmd_upper = Eigen::VectorXd::Constant(16, obj_params.cmd_bound);

    config.use_saddle = use_saddle;             // IMPACT primal saddle XCD (default on)
    config.use_forcing_sequence = use_forcing;  // forcing-sequence inexact X-update (default on)
    config.jit = use_codegen;  // --codegen: JIT-compile the CasADi residual/Jacobian (built once)

    if (!json_output) {
        std::cout << "\nSolver configuration for " << object_name << ":" << std::endl;
        std::cout << "  Parameter mode: UNIVERSAL (per-object tuning removed)" << std::endl;
        std::cout << "  Horizon: " << config.horizon << std::endl;
        std::cout << "  Velocity penalty: " << obj_params.velocity_penalty << std::endl;
        std::cout << "  Control cost weight: " << obj_params.control_cost_weight << std::endl;
        std::cout << "  Final cost multiplier: " << obj_params.final_cost_multiplier << std::endl;
        std::cout << "  Final quaternion weight: " << obj_params.final_quaternion_weight
                  << std::endl;
        std::cout << "  Control bound: " << obj_params.cmd_bound << std::endl;
        std::cout << "  CodeGen: " << (use_codegen ? "enabled" : "disabled") << std::endl;
    }

    // MPC rollout parameters from Python run_mpc_rollout.
    int max_steps = (max_steps_override > 0) ? max_steps_override : 1000;
    int consecutive_success = 0;
    int success_threshold = 20;
    double pos_tol = 0.02;
    double quat_tol = 0.04;  // From Python: success_quat_threshold = 0.04

    if (!json_output) {
        std::cout << "\nStarting MPC rollout..." << std::endl;
    }

    // Statistics
    double total_solve_time = 0.0;
    int total_steps = 0;
    std::vector<double> mpc_times;
    std::vector<Eigen::VectorXd> actions;
    bool success = false;
    int steps_to_success = max_steps;

    // New metrics for comparison with cfree
    double max_comp_violation = 0.0;  // max |G * H| = complementarity_violation from solver
    double max_obj_velocity = 0.0;    // max ||obj_linear_vel||
    std::vector<double> comp_violations;
    std::vector<double> obj_velocities;
    Eigen::VectorXd prev_state;

    // Warm start storage
    impact::SingleShootingSolution last_sol;
    bool has_warm_start = false;

    for (int step = 0; step < max_steps; ++step) {
        if (with_rendering && sim.shouldClose()) {
            std::cout << "Window closed by user" << std::endl;
            break;
        }

        // Get current state: [obj_pos(3), obj_quat(4), robot_qpos(16)]
        Eigen::VectorXd state = sim.getState();

        // Contact detection
        simulation::AllegroContactResult contacts = sim.detectContacts();

        // Solve MPC with warm start
        impact::SingleShootingSolution sol;
        if (has_warm_start) {
            // Shift trajectories for warm start: drop first timestep, repeat last
            int horizon = config.horizon;
            int n_cmd = last_sol.command_trajectory.rows();
            int n_lam = last_sol.lambda_trajectory.rows();
            int n_vel = last_sol.velocity_trajectory.rows();

            // Shift command trajectory
            Eigen::MatrixXd cmd_init(n_cmd, horizon);
            for (int t = 0; t < horizon - 1; ++t) {
                cmd_init.col(t) = last_sol.command_trajectory.col(t + 1);
            }
            cmd_init.col(horizon - 1) = last_sol.command_trajectory.col(horizon - 1);

            // Shift lambda trajectory
            Eigen::MatrixXd lam_init(n_lam, horizon);
            for (int t = 0; t < horizon - 1; ++t) {
                lam_init.col(t) = last_sol.lambda_trajectory.col(t + 1);
            }
            lam_init.col(horizon - 1) = last_sol.lambda_trajectory.col(horizon - 1);

            // Shift velocity trajectory
            Eigen::MatrixXd vel_init(n_vel, horizon);
            for (int t = 0; t < horizon - 1; ++t) {
                vel_init.col(t) = last_sol.velocity_trajectory.col(t + 1);
            }
            vel_init.col(horizon - 1) = last_sol.velocity_trajectory.col(horizon - 1);

            // Multipliers are reset inside solveWithInitialGuess (new q0 each step).
            sol = solver.solveWithInitialGuess(config, state, contacts.phi_vec, contacts.jac_mat,
                                               target_p, target_q, cmd_init, lam_init, vel_init);
        } else {
            sol =
                solver.solve(config, state, contacts.phi_vec, contacts.jac_mat, target_p, target_q);
        }

        // Save solution for warm start
        last_sol = sol;
        has_warm_start = true;

        total_solve_time += sol.solve_time;
        total_steps++;
        mpc_times.push_back(sol.solve_time);
        actions.push_back(sol.first_command);

        // Execute action (first command from solution)
        sim.step(sol.first_command);

        // Get REAL complementarity violation from MuJoCo after stepping
        // This uses actual contact forces from the physics engine
        simulation::RealContactInfo real_contact = sim.getRealContactInfo(0.1);  // h = 0.1
        comp_violations.push_back(real_contact.comp_violation);
        if (real_contact.comp_violation > max_comp_violation) {
            max_comp_violation = real_contact.comp_violation;
        }

        // Sync viewer if rendering
        if (with_rendering) {
            sim.render();
        }

        // Check success
        Eigen::VectorXd new_state = sim.getState();
        double pos_err = (new_state.head<3>() - target_p).norm();
        double quat_err = computeQuatError(new_state.segment<4>(3), target_q);

        // Check if object fell to ground (failure condition)
        double obj_z = new_state(2);  // Object z position
        if (obj_z < 0.005) {          // Object height threshold for ground contact
            if (!json_output) {
                std::cout << "\n*** FAILURE: Object fell to ground at step " << step << " ***"
                          << std::endl;
                std::cout << "Object z position: " << obj_z << std::endl;
            }
            success = false;
            steps_to_success = max_steps;
            break;  // Early termination: no need to continue if object is lost
        }

        // Record object velocity (estimate from state change)
        if (prev_state.size() > 0) {
            Eigen::Vector3d obj_vel =
                (new_state.head<3>() - prev_state.head<3>()) / 0.1;  // h = 0.1
            double obj_vel_norm = obj_vel.norm();
            obj_velocities.push_back(obj_vel_norm);
            if (obj_vel_norm > max_obj_velocity) {
                max_obj_velocity = obj_vel_norm;
            }
        }
        prev_state = new_state;

        if (quat_err < quat_tol) {
            consecutive_success++;
        } else {
            consecutive_success = 0;
        }

        if (!json_output && step % 10 == 0) {
            std::cout << "Step " << step << ": solve_time=" << std::fixed << std::setprecision(4)
                      << sol.solve_time << "s"
                      << ", converged=" << (sol.converged ? "Y" : "N")
                      << ", obj=" << std::setprecision(2) << sol.objective_value
                      << ", dyn_viol=" << std::setprecision(4) << sol.dynamics_violation
                      << ", pos_err=" << pos_err << ", quat_err=" << quat_err << std::endl;
            std::cout << "  first_cmd: " << sol.first_command.transpose() << std::endl;

            // Debug: compare FK output with MuJoCo fingertip positions
            if (step == 0) {
                Eigen::VectorXd mj_ftp = sim.getFingertipsPosition();
                std::cout << "  MuJoCo fingertips: " << mj_ftp.transpose() << std::endl;
                std::cout << "  Object position: " << new_state.head<3>().transpose() << std::endl;
            }
            std::cout << "  cmd norm: " << sol.first_command.norm() << std::endl;
        }

        if (consecutive_success >= success_threshold) {
            success = true;
            steps_to_success = step;
            if (!json_output) {
                std::cout << "\n*** SUCCESS at step " << step << "! ***" << std::endl;
                std::cout << "Position error: " << pos_err << std::endl;
                std::cout << "Quaternion error: " << quat_err << std::endl;
            }
            break;
        }
    }

    // Compute control variance (mean variance across all control dimensions)
    double control_variance = 0.0;
    if (actions.size() > 1) {
        int n_cmd = actions[0].size();
        Eigen::VectorXd mean_action = Eigen::VectorXd::Zero(n_cmd);
        for (const auto& a : actions) {
            mean_action += a;
        }
        mean_action /= actions.size();

        Eigen::VectorXd var_action = Eigen::VectorXd::Zero(n_cmd);
        for (const auto& a : actions) {
            var_action += (a - mean_action).array().square().matrix();
        }
        var_action /= (actions.size() - 1);
        control_variance = var_action.mean();
    }

    // Compute control smoothness: mean(||u_{t+1} - u_t||^2)
    double control_smoothness = 0.0;
    if (actions.size() > 1) {
        double sum_diff_sq = 0.0;
        for (size_t i = 1; i < actions.size(); ++i) {
            sum_diff_sq += (actions[i] - actions[i - 1]).squaredNorm();
        }
        control_smoothness = sum_diff_sq / (actions.size() - 1);
    }

    // Compute control energy: sum(||u_t||^2)
    double control_energy = 0.0;
    for (const auto& a : actions) {
        control_energy += a.squaredNorm();
    }

    // Compute MPC frequency
    double mpc_freq = 0.0;
    if (!mpc_times.empty()) {
        double avg_time = total_solve_time / mpc_times.size();
        if (avg_time > 0) {
            mpc_freq = 1.0 / avg_time;
        }
    }

    // Compute mean complementarity violation
    double mean_comp_violation = 0.0;
    if (!comp_violations.empty()) {
        for (double v : comp_violations) mean_comp_violation += v;
        mean_comp_violation /= comp_violations.size();
    }

    // Output results
    if (json_output) {
        // Redirect output to file if specified
        std::ostream* out = &std::cout;
        std::ofstream file_out;
        if (!output_file.empty()) {
            file_out.open(output_file);
            out = &file_out;
        }

        *out << "{" << std::endl;
        *out << "  \"success\": " << (success ? "true" : "false") << "," << std::endl;
        *out << "  \"steps_to_success\": " << steps_to_success << "," << std::endl;
        *out << "  \"mpc_freq\": " << std::fixed << std::setprecision(4) << mpc_freq << ","
             << std::endl;
        *out << "  \"control_variance\": " << std::setprecision(8) << control_variance << ","
             << std::endl;
        *out << "  \"control_smoothness\": " << std::setprecision(8) << control_smoothness << ","
             << std::endl;
        *out << "  \"control_energy\": " << std::setprecision(8) << control_energy << ","
             << std::endl;
        *out << "  \"max_comp_violation\": " << std::setprecision(8) << max_comp_violation << ","
             << std::endl;
        *out << "  \"mean_comp_violation\": " << std::setprecision(8) << mean_comp_violation << ","
             << std::endl;
        *out << "  \"max_obj_velocity\": " << std::setprecision(8) << max_obj_velocity << ","
             << std::endl;
        // Compute mean object velocity
        double mean_obj_velocity = 0.0;
        if (!obj_velocities.empty()) {
            for (double v : obj_velocities) mean_obj_velocity += v;
            mean_obj_velocity /= obj_velocities.size();
        }
        *out << "  \"mean_obj_velocity\": " << std::setprecision(8) << mean_obj_velocity << ","
             << std::endl;
        *out << "  \"total_solve_time\": " << total_solve_time << "," << std::endl;
        *out << "  \"num_mpc_calls\": " << mpc_times.size() << "," << std::endl;
        *out << "  \"mpc_times\": [";
        for (size_t i = 0; i < mpc_times.size(); ++i) {
            *out << mpc_times[i];
            if (i < mpc_times.size() - 1) *out << ", ";
        }
        *out << "]" << std::endl;
        *out << "}" << std::endl;

        if (file_out.is_open()) {
            file_out.close();
        }
    } else {
        // Print statistics
        std::cout << "\n=== Statistics ===" << std::endl;
        std::cout << "Success: " << (success ? "Yes" : "No") << std::endl;
        std::cout << "Steps to success: " << steps_to_success << std::endl;
        std::cout << "Total steps: " << total_steps << std::endl;
        std::cout << "Total solve time: " << total_solve_time << "s" << std::endl;
        if (total_steps > 0) {
            std::cout << "Average solve time: " << (total_solve_time / total_steps) << "s"
                      << std::endl;
            std::cout << "MPC frequency: " << mpc_freq << " Hz" << std::endl;
        }
        std::cout << "Control variance: " << control_variance << std::endl;
        std::cout << "Max complementarity violation (G*H): " << max_comp_violation << std::endl;
        std::cout << "Max object velocity: " << max_obj_velocity << " m/s" << std::endl;

        // Compute averages
        if (!comp_violations.empty()) {
            double avg_comp = 0.0;
            for (double v : comp_violations) avg_comp += v;
            avg_comp /= comp_violations.size();
            std::cout << "Avg complementarity violation: " << avg_comp << std::endl;
        }
        if (!obj_velocities.empty()) {
            double avg_vel = 0.0;
            for (double v : obj_velocities) avg_vel += v;
            avg_vel /= obj_velocities.size();
            std::cout << "Avg object velocity: " << avg_vel << " m/s" << std::endl;
        }
    }

    if (with_rendering) {
        // Stop video recording before closing
        if (sim.isRecording()) {
            sim.stopVideoRecording();
        }

        std::cout << "\nKeeping window open for 3 seconds..." << std::endl;
        auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < end_time && !sim.shouldClose()) {
            sim.render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        sim.closeRendering();
    }

    if (!json_output) {
        std::cout << "\n=== Done ===" << std::endl;
    }

    return 0;
}
