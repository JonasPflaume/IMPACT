# Allegro Hand MPC controller — single-shooting LCP BCD-AuLa solver.
add_executable(allegro_impact_single
    allegro_impact_single.cpp
)

target_include_directories(allegro_impact_single
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/simulation
)

target_link_libraries(allegro_impact_single
    PRIVATE
        impact
        penalty_solver  # For AllegroLCPProblem definition
        mujoco_simulation_lib
        casadi
        glfw
        Eigen3::Eigen
)

target_compile_definitions(allegro_impact_single PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
