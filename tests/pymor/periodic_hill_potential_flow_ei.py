# The reduced model of periodic_hill_potential_flow_rb.py with an empirically interpolated system
# operator and right-hand side: A(mu) and f(mu) are only assembled at a few dofs, on the cells
# around them (PeriodicHillRestrictedOperator and PeriodicHillRestrictedRhs in
# apps/include/implementations/periodic_hill_potential_flow.h), so no reduced solve touches the
# full mesh. The flow rate as output, with the interpolated right-hand side as its functional, which
# the reduction projects like the right-hand side. Runs serially and on 2 ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
import numpy as np
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.reductors.basic import StationaryRBReductor

import dealii_core_module as dealii
import periodic_hill


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

model = periodic_hill.make_potential_flow_model(n_refinements=2, degree=2)
fom = dealii_pymor.stationary_model(model, name='periodic_hill_potential_flow')
L = periodic_hill.length
volume = GenericParameterFunctional(lambda mu: model.volume(mu) / L, fom.parameters)


def flow_rate(rhs):
    return dealii_pymor.compliant_output(rhs, factor=-1 / L, offset=volume, name='flow_rate')


fom = fom.with_(output_functional=flow_rate(fom.rhs))
h1 = fom.h1_semi_product
parameter_space = fom.parameters.space({'hill_height': (0.5, 1.5), 'hill_width': (2., 4.5)})

# Restricted to all unconstrained dofs (the nonzero entries of an admissible vector of ones), A(mu)
# and f(mu) are the full ones. This includes the dofs on the periodic boundary, whose rows also
# collect the cells at the other end of the channel.
mu = fom.parameters.parse({'hill_height': 1.2, 'hill_width': 2.3})
dofs = np.flatnonzero(fom.solution_space.ones().to_numpy()[:, 0])
U = fom.solve(mu)
restricted_operator, source_dofs = fom.operator.restricted(dofs)
AU = fom.operator.apply(U, mu=mu).dofs(dofs)
AU_restricted = restricted_operator.apply(restricted_operator.source.make_array(U.dofs(source_dofs)),
                                          mu=mu).to_numpy()
restricted_rhs, _ = fom.rhs.restricted(dofs)
F = fom.rhs.as_range_array(mu).dofs(dofs)
log(f'restricted to all {len(dofs)} unconstrained dofs, same A(mu) u: '
    f'{np.abs(AU_restricted - AU).max() < 1e-12 * np.abs(AU).max()}, same f(mu): '
    f'{np.abs(restricted_rhs.as_range_array(mu).to_numpy() - F).max() < 1e-12 * np.abs(F).max()}')

# A reduced basis of 16 POD modes.
training_set = parameter_space.sample_uniformly(8)
snapshots = fom.solution_space.empty()
for mu in training_set:
    snapshots.append(fom.solve(mu))
basis, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
basis = basis[:16]

# The reduced operator applies A(mu) to the basis vectors, so these are what is interpolated.
operator_snapshots = fom.solution_space.empty()
for mu in training_set:
    operator_snapshots.append(fom.operator.apply(basis, mu=mu))
operator_dofs, operator_basis, _ = ei_greedy(operator_snapshots, max_interpolation_dofs=200)
# f(mu) is linear in the hill height, but needs many more widths than the training set has: with
# its 8 widths, the snapshots only span 8 dimensions.
rhs_snapshots = fom.solution_space.empty()
for width in np.linspace(2., 4.5, 100):
    rhs_snapshots.append(fom.rhs.as_range_array(fom.parameters.parse({'hill_height': 1.,
                                                                      'hill_width': width})))
rhs_dofs, rhs_basis, _ = ei_greedy(rhs_snapshots, max_interpolation_dofs=64, rtol=1e-6)
log(f'interpolation dofs: operator {len(operator_dofs)} (from {len(operator_snapshots)} snapshots), '
    f'right-hand side {len(rhs_dofs)} (from {len(rhs_snapshots)})')

test_set = parameter_space.sample_randomly(10)
solutions = [fom.solve(mu) for mu in test_set]
outputs = [fom.output(mu)[0, 0] for mu in test_set]


def solution_error(reductor, rom):
    return max(((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]
               for mu, u in zip(test_set, solutions))


def output_error(rom):
    return max(abs(rom.output(mu)[0, 0] - Q) / Q for mu, Q in zip(test_set, outputs))


reductor = StationaryRBReductor(fom, RB=basis, product=h1)
log(f'reduced basis alone: largest relative error {solution_error(reductor, reductor.reduce()):.1e}')

rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, rhs_dofs, rhs_basis, triangular=True)
for M in (50, 100, 200):
    operator_ei = EmpiricalInterpolatedOperator(fom.operator, operator_dofs[:M], operator_basis[:M],
                                                triangular=True)
    reductor = dealii_pymor.DealIIStationaryRBReductor(
        fom.with_(operator=operator_ei, rhs=rhs_ei, output_functional=flow_rate(rhs_ei)),
        RB=basis, product=h1)
    rom = reductor.reduce()
    # The number of source dofs depends on the dof numbering, i.e. on the number of ranks.
    log(f'operator interpolation dofs {M:3d}: largest relative error {solution_error(reductor, rom):.1e}, '
        f'of the flow rate {output_error(rom):.1e}')

log(f'reduced operator: {type(rom.operator).__name__}, right-hand side: {type(rom.rhs).__name__}, '
    f'in the flow rate: {type(rom.output_functional.operators[0].rhs).__name__}')
