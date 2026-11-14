cmake_minimum_required(VERSION 3.16)

SET(CMAKE_CXX_STANDARD 17)

# custom macros
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


# This project uses deal.II
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

find_package(EXADG REQUIRED
  HINTS ${EXADG_DIR})


deal_ii_initialize_cached_variables()

project(dealii_with_pybind LANGUAGES CXX)

add_subdirectory(extern/pybind11)
add_subdirectory(python)
add_subdirectory(exec)
