# Python extension modules, see cmake/DealiiPybindModule.cmake.
# Each module links only what its bindings need.

dealii_pybind_add_module(example
  SOURCES bindings/example.cc
  LINK    applications
  )

dealii_pybind_add_module(gaussian_source
  SOURCES bindings/gaussian_source.cc
  LINK    applications
  )

# applications provides the interfaces (apps/include/interfaces)
dealii_pybind_add_module(dealii_core_module
  SOURCES bindings/dealii_core_module.cc
  LINK    dealii::dealii applications
  )

# pyMOR adapter (pure python), installed next to the modules it wraps
install(
    DIRECTORY dealii_pymor
    DESTINATION python
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)

# install python scripts
install(
    DIRECTORY scripts
    DESTINATION python
    FILES_MATCHING PATTERN "*.py"
)
