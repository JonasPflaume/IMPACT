add_library(mujoco_simulation_lib
    contactLCP.cpp
    contactLCP.h
    fingertips_simulator.cpp
    fingertips_simulator.h
    allegro_simulator.cpp
    allegro_simulator.h
    mj_simulator.cpp
    mj_simulator.h
    rotation_utils.h
)

target_include_directories(mujoco_simulation_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CASADI_INCLUDE_DIRS}
)

target_link_libraries(mujoco_simulation_lib
    PUBLIC
        mujoco::mujoco
        Eigen3::Eigen
        glfw
        casadi
)

add_executable(mujoco_viewer
    test_viewer.cpp
)

target_link_libraries(mujoco_viewer
    PRIVATE
        mujoco_simulation_lib
        glfw
        casadi
)

target_include_directories(mujoco_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(mujoco_viewer PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)