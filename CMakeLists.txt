cmake_minimum_required(VERSION 3.15)
project(IMPACT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(IMPACT_NATIVE_ARCH "Build with host-specific -march/-mtune release flags" ON)
option(IMPACT_BUILD_ALLEGRO "Build MuJoCo/Allegro simulation examples" ON)
# The Python extension and the C++ drivers are independent front-ends onto the
# same solver library, so each gets an option rather than sharing one tree. A
# wheel build wants only the extension: `pip install .` configures with
# -DIMPACT_BUILD_PYTHON=ON -DIMPACT_BUILD_EXPERIMENTS=OFF -DIMPACT_BUILD_ALLEGRO=OFF
# and therefore needs neither MuJoCo nor GLFW nor a system CasADi.
option(IMPACT_BUILD_PYTHON "Build the pybind11 extension module" OFF)
option(IMPACT_BUILD_EXPERIMENTS "Build the C++ experiment drivers" ON)

# Extra release flags used in the benchmark runs.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND IMPACT_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -ffast-math")
endif()

# Core dependencies.
find_package(Eigen3 REQUIRED)

if(IMPACT_BUILD_PYTHON)
    # The extension is a shared object linking the solver library.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Prefer the CasADi that ships inside the `casadi` wheel over any system
    # install. Both sides of the boundary then share one libcasadi build, so the
    # serialized functions the Python modelling layer produces are read back by
    # exactly the version that wrote them -- the cross-version compatibility
    # question disappears instead of having to be managed. The wheel ships headers
    # and a CMake package config, so this is an ordinary find_package against a
    # different prefix.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import casadi, os, sys; sys.stdout.write(os.path.dirname(casadi.__file__))"
        OUTPUT_VARIABLE IMPACT_CASADI_PY_DIR
        RESULT_VARIABLE IMPACT_CASADI_PY_STATUS
        ERROR_QUIET)
    if(IMPACT_CASADI_PY_STATUS EQUAL 0 AND
       EXISTS "${IMPACT_CASADI_PY_DIR}/cmake/casadi-config.cmake")
        message(STATUS "IMPACT: using CasADi from the Python wheel at ${IMPACT_CASADI_PY_DIR}")
        list(INSERT CMAKE_PREFIX_PATH 0 "${IMPACT_CASADI_PY_DIR}/cmake")
        set(IMPACT_CASADI_FROM_WHEEL ON)
    else()
        message(STATUS "IMPACT: no CasADi Python wheel found; falling back to a system CasADi")
        set(IMPACT_CASADI_FROM_WHEEL OFF)
    endif()

    # Ask the interpreter for its pybind11 before searching the system. A distro
    # pybind11 can easily be older than the NumPy in this environment, and the two
    # are coupled: NumPy 2 changed the array object layout, so a pre-2.12 pybind11
    # miscasts every real array it is handed. That does not fail to build and does
    # not fail on import -- it corrupts memory on the first non-empty array that
    # crosses the boundary. Hence both the search order and the version floor.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE IMPACT_PYBIND11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(IMPACT_PYBIND11_DIR)
        list(INSERT CMAKE_PREFIX_PATH 0 "${IMPACT_PYBIND11_DIR}")
    endif()
    find_package(pybind11 2.12 CONFIG REQUIRED)
    message(STATUS "IMPACT: using pybind11 ${pybind11_VERSION} from ${pybind11_DIR}")
endif()

# Linear algebra backends. The solver's factorisations are Eigen's, which are
# header-only, so an external BLAS/LAPACK is only pulled in for callers that want
# one; requiring it in a wheel build would add a system dependency for nothing.
if(IMPACT_BUILD_EXPERIMENTS)
    find_package(PkgConfig REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
else()
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
endif()

if(IMPACT_BUILD_ALLEGRO)
    # MuJoCo and rendering are needed by the Allegro examples.
    find_package(mujoco REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
endif()

# Symbolic model construction.
find_package(casadi REQUIRED)

# The CasADi package config exports the imported target `casadi::casadi` carrying
# both its include directory and, crucially, the libstdc++ string ABI that binary
# was built with (_GLIBCXX_USE_CXX11_ABI). The two ABIs are incompatible and the
# CasADi wheels use the pre-GCC5 one while a locally built CasADi normally uses
# the default, so this is not a detail a caller can assume -- and getting it wrong
# does not fail the link of a shared module, only its import, on whichever
# std::string-taking overload is reached first. Link the imported target and the
# question answers itself. The bare `casadi` name below is kept as a fallback for
# installs predating the namespaced target; it degrades to `-lcasadi`.
if(TARGET casadi::casadi)
    set(IMPACT_CASADI_TARGET casadi::casadi)
else()
    set(IMPACT_CASADI_TARGET casadi)
endif()

# Test infrastructure (tests themselves are built only with -DIMPACT_BUILD_TESTS=ON,
# but enabling here lets `ctest` discover them from the build root).
enable_testing()

# Project targets.
if(IMPACT_BUILD_ALLEGRO)
    add_subdirectory(simulation)
endif()
add_subdirectory(impact_solver)
if(IMPACT_BUILD_EXPERIMENTS)
    add_subdirectory(penalty_solver)
    add_subdirectory(relax_solver)
    # add_subdirectory(direct_solver)
    add_subdirectory(experiments)
endif()
if(IMPACT_BUILD_PYTHON)
    add_subdirectory(python)
    if(NOT IMPACT_BUILD_EXPERIMENTS)
        # The parity harness compares the Python build against the C++ builders,
        # and the comparison travels as serialized CasADi functions. CasADi's
        # serialization is forward- but not backward-compatible, so the two sides
        # have to agree on the library: built here, parity_dump links the same
        # CasADi wheel the extension does. (The drivers' own build tree keeps the
        # system CasADi, so their numerics are untouched by any of this.)
        add_subdirectory(experiments/parity_dump)
    endif()
endif()
