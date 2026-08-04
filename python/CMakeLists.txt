# Python extension module for the IMPACT solver.
#
# Only the solver crosses this boundary. `python/impact/` is pure Python on top
# of the CasADi Python API and hands the solver serialized CasADi functions, so
# nothing here needs to know what a task looks like -- and the tasks themselves
# live outside the package entirely, in `python/examples/`.

pybind11_add_module(_impact_core MODULE src/bindings.cpp)

target_link_libraries(_impact_core PRIVATE impact)

# Compile with -ffast-math (the benchmark build's flags, kept so the solver's
# arithmetic matches the C++ drivers), but do not *link* with it.
#
# GCC's driver reacts to -ffast-math on the link line by pulling in
# crtfastmath.o, whose only job is to set FTZ/DAZ in MXCSR at load time. In an
# executable that is contained. In a Python extension it is not: the mode is
# per-thread CPU state, so merely importing this module would flush denormals to
# zero for NumPy and every other library in the interpreter -- silently, and long
# after the import. Measured before this line: `np.finfo(np.float64)
# .smallest_subnormal` became 0 and `np.float64(1e-320) * 1.0` returned 0.0 once
# `import impact` had run.
#
# Trailing -fno-fast-math leaves every compile-time optimisation in place (the
# objects are already built) and only stops crtfastmath.o being linked, so the
# process keeps standard IEEE denormal handling. Denormals are ~1e-308, far below
# anything this solver's tolerances can see, so declining to flush them does not
# meaningfully separate it from the drivers.
if(CMAKE_CXX_FLAGS_RELEASE MATCHES "ffast-math")
    target_link_options(_impact_core PRIVATE -fno-fast-math)
endif()

if(DEFINED SKBUILD_PROJECT_VERSION)
    target_compile_definitions(_impact_core PRIVATE IMPACT_VERSION="${SKBUILD_PROJECT_VERSION}")
endif()

# Runtime search path for libcasadi. `$ORIGIN/../casadi` is where the `casadi`
# wheel unpacks next to us in site-packages, which is the same libcasadi this
# module was compiled against -- see the wheel-preferring find_package in the top
# level CMakeLists. Keeping both sides on one build is what makes the serialized
# functions readable by construction rather than by version negotiation.
set_target_properties(_impact_core PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../casadi"
    BUILD_WITH_INSTALL_RPATH FALSE
)

if(IMPACT_CASADI_FROM_WHEEL)
    # In-tree builds (cmake --build build) run from the build directory, where
    # $ORIGIN/../casadi does not exist yet.
    set_target_properties(_impact_core PROPERTIES
        BUILD_RPATH "${IMPACT_CASADI_PY_DIR}")
endif()

install(TARGETS _impact_core LIBRARY DESTINATION impact)
