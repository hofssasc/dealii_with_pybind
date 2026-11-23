# gaussian_source_rb.py for the matrix-free model with ExaDG
# (apps/include/implementations/gaussian_source_matrix_free.h).
# ExaDG prints its setup to stdout, which is part of the output.

import common  # first, see common.py

import gaussian_source
from gaussian_source_rb import run_tests

run_tests(gaussian_source.make_matrix_free_model(n_refinements=5, width=0.2))
