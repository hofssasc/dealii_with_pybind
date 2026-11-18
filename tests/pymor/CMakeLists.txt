# pyMOR tests: like tests/python, plus the dealii_pymor package from the
# source tree on PYTHONPATH. Only added if pyMOR is installed (tests/CMakeLists.txt).
get_property(TEST_PYTHON_MODULES GLOBAL PROPERTY DEALII_PYBIND_MODULES)
set(TEST_PYTHON_PATH "${PROJECT_SOURCE_DIR}/python")

pybind_pickup_tests()
