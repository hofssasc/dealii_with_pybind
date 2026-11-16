# Python extension modules, see cmake/DealiiPybindModule.cmake.
# Each module links only what its bindings need.

dealii_pybind_add_module(example
  SOURCES bindings/example.cc
  LINK    applications
  )

dealii_pybind_add_module(dealii_core_module
  SOURCES bindings/dealii_core_module.cc
  LINK    dealii::dealii
  )

# install python scripts
install(
    DIRECTORY scripts/
    DESTINATION python/scripts
    FILES_MATCHING PATTERN "*.py"
)
