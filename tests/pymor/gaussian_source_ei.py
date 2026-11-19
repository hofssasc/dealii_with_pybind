# The reduced model of gaussian_source_rb.py with an empirically interpolated right-hand side: f(mu)
# is interpolated from M of its entries, which the C++ model computes on the cells around them only
# (GaussianSourceRestrictedRhs in apps/include/implementations/gaussian_source.h). The error of the
# interpolated right-hand side decreases with M, until the error of the reduced solution is the one
# of the reduced basis alone. Runs serially and on 2 ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor

import dealii_core_module as dealii
import gaussian_source


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

fom = dealii_pymor.stationary_model(gaussian_source.make_model(n_refinements=5, width=0.2),
                                    name='gaussian_source')
h1 = fom.h1_0_semi_product
parameter_space = fom.parameters.space(0.2, 0.8)

# The reduced basis of gaussian_source_rb.py.
snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(6):
    snapshots.append(fom.solve(mu))
basis, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)

# Interpolation dofs and collateral basis from right-hand sides on a finer grid of centers: they
# only cost an assembly each, and the narrow source needs more of them than the solutions.
rhs_snapshots = fom.rhs.range.empty()
for mu in parameter_space.sample_uniformly(12):
    rhs_snapshots.append(fom.rhs.as_range_array(mu))
dofs, collateral_basis, _ = ei_greedy(rhs_snapshots, max_interpolation_dofs=64)
log(f'right-hand side snapshots: {len(rhs_snapshots)}, interpolation dofs: {len(dofs)}')

test_set = parameter_space.sample_randomly(10)
rhs_test = [fom.rhs.as_range_array(mu) for mu in test_set]
solutions = [fom.solve(mu) for mu in test_set]


def solution_error(reductor, rom):
    return max(((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]
               for mu, u in zip(test_set, solutions))


reductor = StationaryRBReductor(fom, RB=basis, product=h1)
log(f'reduced basis alone: largest relative error of the solution {solution_error(reductor, reductor.reduce()):.1e}')

for M in (16, 32, 64):
    # The first M dofs and basis vectors of ei_greedy are the interpolation with M dofs.
    rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, dofs[:M], collateral_basis[:M], triangular=True)
    reductor = dealii_pymor.DealIIStationaryRBReductor(fom.with_(rhs=rhs_ei), RB=basis, product=h1)
    rom = reductor.reduce()
    rhs_error = max(((f - rhs_ei.as_range_array(mu)).norm() / f.norm())[0] for mu, f in zip(test_set, rhs_test))
    log(f'interpolation dofs {M:2d}: largest relative error of the right-hand side {rhs_error:.1e}, '
        f'of the solution {solution_error(reductor, rom):.1e}')

# Evaluated at the interpolation dofs only, without a vector of the full space.
log(f'reduced right-hand side: {type(rom.rhs).__name__}')
