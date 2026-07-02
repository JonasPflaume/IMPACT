# Penalty (IPOPT) solver executable
add_executable(push_t_penalty
    push_t_penalty.cpp
    push_t.cpp
)

target_include_directories(push_t_penalty PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_t_penalty PRIVATE penalty_solver impact casadi)

# Scholtes Relaxation solver executable
add_executable(push_t_relaxation
    push_t_relaxation.cpp
    push_t.cpp
)

target_include_directories(push_t_relaxation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_t_relaxation PRIVATE relaxation_solver impact casadi)

# IMPACT solver — multiple shooting (state free, dynamics as defect)
add_executable(push_t_impact_multiple
    push_t_impact_multiple.cpp
    push_t.cpp
)

target_include_directories(push_t_impact_multiple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_t_impact_multiple PRIVATE impact casadi)

# IMPACT solver — single shooting (state rolled out from x_0)
add_executable(push_t_impact_single
    push_t_impact_single.cpp
    push_t.cpp
)

target_include_directories(push_t_impact_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_t_impact_single PRIVATE impact casadi)
