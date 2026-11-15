# C++ tests: <name>.cc + <name>.output (or .run_only), picked up by deal.II.
# Every test executable is linked against the libraries listed here.
set(TEST_LIBRARIES code)

deal_ii_pickup_tests()
