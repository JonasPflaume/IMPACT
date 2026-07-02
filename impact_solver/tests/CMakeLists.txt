# Component and parity tests for the solver library.
# Enabled with -DIMPACT_BUILD_TESTS=ON.

set(EXP_DIR ${CMAKE_SOURCE_DIR}/experiments)

# Component tests; no task problem definitions needed.
add_executable(impact_test_components test_components.cpp)
target_link_libraries(impact_test_components PRIVATE impact casadi Eigen3::Eigen)

# Multiple-shooting parity checks using the task definitions.
add_executable(impact_parity_ms
    parity_ms.cpp
    ${EXP_DIR}/box/box_pushing.cpp
    ${EXP_DIR}/push_t/push_t.cpp
    ${EXP_DIR}/cart_transporter/cart_transporter.cpp
)
target_include_directories(impact_parity_ms PRIVATE
    ${EXP_DIR}/box
    ${EXP_DIR}/push_t
    ${EXP_DIR}/cart_transporter
)
target_link_libraries(impact_parity_ms PRIVATE impact casadi Eigen3::Eigen)

# Cross-combo demo: {MPCC, LCP} x {single, multiple}.
add_executable(impact_cross_combo
    cross_combo.cpp
    ${EXP_DIR}/box/box_pushing.cpp
)
target_include_directories(impact_cross_combo PRIVATE ${EXP_DIR}/box)
target_link_libraries(impact_cross_combo PRIVATE impact penalty_solver casadi Eigen3::Eigen)

# Register with ctest.
add_test(NAME components       COMMAND impact_test_components)
add_test(NAME cross_combo      COMMAND impact_cross_combo)
# parity_ms takes <task> <x0...> <goal...> and exits nonzero if it fails to converge.
add_test(NAME parity_box    COMMAND impact_parity_ms box    0 0 0    0.1 0.1 1.0)
add_test(NAME parity_push_t COMMAND impact_parity_ms push_t 0 0 0    0.05 0.05 1.5708)
add_test(NAME parity_cart   COMMAND impact_parity_ms cart   0 0 0 0  1 0 0 0)
