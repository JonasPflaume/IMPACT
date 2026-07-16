# IMPACT solver — multiple shooting (state free, dynamics as defect)
add_executable(push_circle_impact_multiple
    push_circle_impact_multiple.cpp
    push_circle.cpp
)

target_include_directories(push_circle_impact_multiple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_circle_impact_multiple PRIVATE impact casadi)
