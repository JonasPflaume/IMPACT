# Minimal non-trajectory MPCC, solved directly via the buildMPCC solver layer.
add_executable(toy_mpcc toy_mpcc.cpp)
target_link_libraries(toy_mpcc PRIVATE impact casadi Eigen3::Eigen)
