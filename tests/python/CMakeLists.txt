# Python tests: <name>.py + <name>.output (or .run_only), picked up by
# pybind_pickup_tests() from cmake/PybindPickupTests.cmake.
# The directories of module targets are added to PYTHONPATH.
set(TEST_PYTHON_MODULES dealii_core_module)

pybind_pickup_tests()
