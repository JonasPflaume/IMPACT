# IMPACT solver library.
#
# Tasks provide symbolic problem data; the single- and multiple-shooting builders
# assemble the subproblems consumed by the same BCD-AuLa solver.

file(GLOB_RECURSE IMPACT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(GLOB_RECURSE IMPACT_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

add_library(impact ${IMPACT_SOURCES} ${IMPACT_HEADERS})

target_include_directories(impact
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(impact
    PUBLIC
        Eigen3::Eigen
        BLAS::BLAS
        LAPACK::LAPACK
        casadi
)

target_compile_features(impact PUBLIC cxx_std_17)

set_target_properties(impact PROPERTIES
    VERSION 0.0.0
    SOVERSION 1
    PUBLIC_HEADER "${IMPACT_HEADERS}"
)

# Optional component / parity tests (built only when explicitly requested).
if(IMPACT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
