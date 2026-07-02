# Penalty (IPOPT) solver executable
add_executable(box_penalty_ipopt
    penalty_ipopt.cpp
    box_pushing.cpp
)

target_include_directories(box_penalty_ipopt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(box_penalty_ipopt PRIVATE penalty_solver impact casadi)

# Scholtes Relaxation solver executable
add_executable(box_relaxation
    box_relaxation.cpp
    box_pushing.cpp
)

target_include_directories(box_relaxation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(box_relaxation PRIVATE relaxation_solver impact casadi)

# IMPACT solver — multiple shooting (state free, dynamics as defect)
add_executable(box_impact_multiple
    box_impact_multiple.cpp
    box_pushing.cpp
)

target_include_directories(box_impact_multiple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(box_impact_multiple PRIVATE impact casadi)

# IMPACT solver — single shooting (state rolled out from x_0)
add_executable(box_impact_single
    box_impact_single.cpp
    box_pushing.cpp
)

target_include_directories(box_impact_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(box_impact_single PRIVATE impact casadi)
