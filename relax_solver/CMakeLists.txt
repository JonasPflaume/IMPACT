file(GLOB_RECURSE RELAXATION_SOLVER_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(GLOB_RECURSE RELAXATION_SOLVER_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

add_library(relaxation_solver ${RELAXATION_SOLVER_SOURCES} ${RELAXATION_SOLVER_HEADERS})

target_include_directories(relaxation_solver 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Baseline solvers reuse the problem interfaces from impact_solver.
target_link_libraries(relaxation_solver
    PUBLIC
        impact
        Eigen3::Eigen
        casadi
)

target_compile_features(relaxation_solver PUBLIC cxx_std_17)

set_target_properties(relaxation_solver PROPERTIES
    VERSION 0.0.0
    SOVERSION 1
    PUBLIC_HEADER "${RELAXATION_SOLVER_HEADERS}"
)
