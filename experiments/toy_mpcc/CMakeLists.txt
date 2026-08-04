# Minimal non-trajectory MPCC, solved directly via the buildMPCC solver layer.
add_executable(toy_mpcc toy_mpcc.cpp)
target_include_directories(toy_mpcc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(toy_mpcc PRIVATE impact ${IMPACT_CASADI_TARGET} Eigen3::Eigen)
