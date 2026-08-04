# Writes out a C++-assembled subproblem for python/tests/test_parity.py to check
# the Python modelling layer against. Links each task's model directly, so the
# comparison is against the same code the drivers run.
add_executable(parity_dump
    parity_dump.cpp
    ../push_circle/push_circle.cpp
    ../box/box_pushing.cpp
    ../push_t/push_t.cpp
    ../cart_transporter/cart_transporter.cpp
)

target_include_directories(parity_dump PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../push_circle
    ${CMAKE_CURRENT_SOURCE_DIR}/../box
    ${CMAKE_CURRENT_SOURCE_DIR}/../push_t
    ${CMAKE_CURRENT_SOURCE_DIR}/../cart_transporter
)
target_link_libraries(parity_dump PRIVATE impact ${IMPACT_CASADI_TARGET})
