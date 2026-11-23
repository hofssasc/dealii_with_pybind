"""Helpers shared by the pyMOR tests. Not a test itself: it has no .output file.

A test file covers one topic, with its expensive setup (models, solutions, snapshots) computed
once at module level, and one function per check, `test_<name>`. `run(test_a, test_b, ...)` calls
them in order, each in a `section` that prefixes its output lines with `<name>::`, like deal.II's
`LogStream::Prefix`, so that a diff names the check that changed.

Import this module first in a test, as it imports dealii_pymor before pyMOR.
"""

import contextlib

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
from pymor.core.logger import set_log_levels
from pymor.parameters.functionals import GenericParameterFunctional

import dealii_core_module as dealii

# The only warnings are the expected ones about projecting operators generically.
set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

_prefixes = []


def log(*args):
    """Print on rank 0, prefixed with the current sections."""
    if dealii.this_mpi_process() == 0:
        print(''.join(f'{prefix}::' for prefix in _prefixes) + ' '.join(str(a) for a in args))


@contextlib.contextmanager
def section(name):
    """Prefix the lines logged inside with `name::`; sections nest."""
    _prefixes.append(name)
    try:
        yield
    finally:
        _prefixes.pop()


def run(*tests):
    """Call the test functions in order, each in a section named after it without `test_`."""
    for test in tests:
        with section(test.__name__.removeprefix('test_')):
            test()


# ------------------ periodic hill ---------------------------

# Steepest slope 15 h / (8 w) up to 1.4 (54 degrees), the narrowest gap at the crest 1.5.
PERIODIC_HILL_RANGES = {'hill_height': (0.5, 1.5), 'hill_width': (2., 4.5)}


def periodic_hill_flow_rate(model, fom, rhs):
    """The flow rate Q(mu) = (|Omega(mu)| - f(mu)^T phi_tilde) / 9 as a compliant output, with the
    right-hand side `rhs` of `fom` (or its interpolation) as functional."""
    import periodic_hill
    L = periodic_hill.length
    volume = GenericParameterFunctional(lambda mu: model.volume(mu) / L, fom.parameters)
    return dealii_pymor.compliant_output(rhs, factor=-1 / L, offset=volume, name='flow_rate')


def periodic_hill_fom(n_refinements=2, degree=2):
    """The C++ periodic hill model and its pyMOR model with the flow rate as output."""
    import periodic_hill
    model = periodic_hill.make_potential_flow_model(n_refinements=n_refinements, degree=degree)
    fom = dealii_pymor.stationary_model(model, name='periodic_hill_potential_flow')
    return model, fom.with_(output_functional=periodic_hill_flow_rate(model, fom, fom.rhs))
