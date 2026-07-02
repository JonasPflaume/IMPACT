
file(GLOB_RECURSE PENALTY_SOLVER_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(GLOB_RECURSE PENALTY_SOLVER_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

add_library(penalty_solver ${PENALTY_SOLVER_SOURCES} ${PENALTY_SOLVER_HEADERS})

target_include_directories(penalty_solver 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Baseline solvers reuse the problem interfaces from impact_solver.
target_link_libraries(penalty_solver
    PUBLIC
        impact
        Eigen3::Eigen
        casadi
)

target_compile_features(penalty_solver PUBLIC cxx_std_17)

set_target_properties(penalty_solver PROPERTIES
    VERSION 0.0.0
    SOVERSION 1
    PUBLIC_HEADER "${PENALTY_SOLVER_HEADERS}"
)
