cmake_minimum_required(VERSION 3.15)
project(IMPACT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(IMPACT_NATIVE_ARCH "Build with host-specific -march/-mtune release flags" ON)
option(IMPACT_BUILD_ALLEGRO "Build MuJoCo/Allegro simulation examples" ON)

# Extra release flags used in the benchmark runs.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND IMPACT_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -ffast-math")
endif()

# Core dependencies.
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)

# Linear algebra backends.
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(IMPACT_BUILD_ALLEGRO)
    # MuJoCo and rendering are needed by the Allegro examples.
    find_package(mujoco REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
endif()

# Symbolic model construction.
find_package(casadi REQUIRED)

# Test infrastructure (tests themselves are built only with -DIMPACT_BUILD_TESTS=ON,
# but enabling here lets `ctest` discover them from the build root).
enable_testing()

# Project targets.
if(IMPACT_BUILD_ALLEGRO)
    add_subdirectory(simulation)
endif()
add_subdirectory(impact_solver)
add_subdirectory(penalty_solver)
add_subdirectory(relax_solver)
# add_subdirectory(direct_solver)
add_subdirectory(experiments)
