#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "mj_simulator.h"

int main(int argc, char** argv) {
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = std::string() + "/home/jiayun/MPCC/model/xmls/env_allegro_bunny.xml";
    }

    MjSimulator::Parameters param;
    param.model_path = model_path;
    param.target_p = Eigen::Vector3d::Zero();
    param.target_q = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    param.frame_skip = 5;
    // Leave init_robot_qpos and init_obj_qpos empty (defaults to model qpos)

    try {
        MjSimulator sim(param);
        sim.init_rendering(true);

        while (!sim.should_close()) {
            sim.render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        sim.close_rendering();
    } catch (const std::exception& e) {
        std::cerr << "Failed to run Allegro viewer: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
