cmake_minimum_required(VERSION 3.16)

SET(CMAKE_CXX_STANDARD 17)

# custom macros
# (PROJECT_SOURCE_DIR is still empty here because project() comes later)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# For bindings directly with dealii
# Called before project definition
find_package(deal.II 9.7 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

deal_ii_initialize_cached_variables()


project(dealii_with_pybind LANGUAGES CXX)

find_package(EXADG REQUIRED
  HINTS ${EXADG_DIR})

add_subdirectory(extern/pybind11)
add_subdirectory(python)
add_subdirectory(exec)

# Testsuite: C++ tests via deal_ii_pickup_tests(), python tests via
# pybind_pickup_tests() (cmake/PybindPickupTests.cmake)
include(PybindPickupTests)

enable_testing()
add_subdirectory(tests)
