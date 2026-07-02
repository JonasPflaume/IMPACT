# Penalty (IPOPT) solver executable
add_executable(cart_transporter_penalty
    cart_transporter_penalty.cpp
    cart_transporter.cpp
)

target_include_directories(cart_transporter_penalty PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cart_transporter_penalty PRIVATE penalty_solver impact casadi)

# Scholtes Relaxation solver executable
add_executable(cart_transporter_relaxation
    cart_transporter_relaxation.cpp
    cart_transporter.cpp
)

target_include_directories(cart_transporter_relaxation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cart_transporter_relaxation PRIVATE relaxation_solver impact casadi)

# IMPACT solver — multiple shooting (state free, dynamics as defect)
add_executable(cart_transporter_impact_multiple
    cart_transporter_impact_multiple.cpp
    cart_transporter.cpp
)

target_include_directories(cart_transporter_impact_multiple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cart_transporter_impact_multiple PRIVATE impact casadi)

# IMPACT solver — single shooting (state rolled out from x_0)
add_executable(cart_transporter_impact_single
    cart_transporter_impact_single.cpp
    cart_transporter.cpp
)

target_include_directories(cart_transporter_impact_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cart_transporter_impact_single PRIVATE impact casadi)
