# Python tests: <name>.py + <name>.output (or .run_only), picked up by
# pybind_pickup_tests() from cmake/PybindPickupTests.cmake.
# All modules added with dealii_pybind_add_module() are put on PYTHONPATH.
get_property(TEST_PYTHON_MODULES GLOBAL PROPERTY DEALII_PYBIND_MODULES)

pybind_pickup_tests()
