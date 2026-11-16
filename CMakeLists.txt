# 3.23: FILE_SET in apps/, ENVIRONMENT_MODIFICATION in the python tests
cmake_minimum_required(VERSION 3.23)

SET(CMAKE_CXX_STANDARD 17)

# custom macros, functions and scripts
# (PROJECT_SOURCE_DIR is still empty here because project() comes later)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# deal.II has to be found before project(): deal_ii_initialize_cached_variables()
# selects deal.II's compiler and clears CMake's default per-build-type flags,
# which only takes effect before project() sets up the compiler.
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

# Install into <source>/install unless a prefix is given explicitly
# (-DCMAKE_INSTALL_PREFIX=... or cmake --install build --prefix ...).
# Only takes effect when a build directory is configured for the first time.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH
    "Install path prefix" FORCE)
endif()

# deal.II only defines its imported targets (dealii::dealii, ...) by itself
# if find_package() runs after project(), so load them explicitly.
include(${DEAL_II_TARGET_CONFIG})
set(DEAL_II_TARGET_CONFIG_INCLUDED TRUE)

find_package(EXADG REQUIRED
  HINTS ${EXADG_DIR})

# EXADG::exadg is linked against one fixed deal.II variant (the build type
# ExaDG was configured with). Mixing it with the other variant would load
# libdeal_II.so and libdeal_II.g.so into the same process.
get_target_property(_exadg_links EXADG::exadg INTERFACE_LINK_LIBRARIES)
if("${_exadg_links}" MATCHES "dealii::dealii_(debug|release)")
  string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type)
  if(NOT "${CMAKE_MATCH_1}" STREQUAL "${_build_type}")
    message(FATAL_ERROR
      "\nExaDG (${EXADG_DIR}) was built against the ${CMAKE_MATCH_1} version "
      "of deal.II, but this project is configured with "
      "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}.\n"
      "Use the same build type, or point EXADG_DIR to a matching ExaDG build.\n")
  endif()
endif()

# Use CMake's FindPython module and add pybind11
set(PYBIND11_FINDPYTHON NEW)
add_subdirectory(extern/pybind11)

# Function that adds a python extension module via pybind11_add_module(),
# and links it against targets.
include(DealiiPybindModule)

# Add projects libraray of applications, for the install this library will be
# copied to /lib. (Note that dealii and exaDG are of course not copied)
add_subdirectory(apps)

add_subdirectory(python)

# Testsuite: C++ tests via deal_ii_pickup_tests(), python tests via
# pybind_pickup_tests() (cmake/PybindPickupTests.cmake)
include(PybindPickupTests)

enable_testing()
add_subdirectory(tests)
