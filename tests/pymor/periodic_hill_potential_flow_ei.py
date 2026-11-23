# The reduced model of periodic_hill_potential_flow_rb.py with an empirically interpolated system
# operator and right-hand side: A(mu) and f(mu) are only assembled at a few dofs, on the cells
# around them (PeriodicHillRestrictedOperator and PeriodicHillRestrictedRhs in
# apps/include/implementations/periodic_hill_potential_flow.h), so no reduced solve touches the
# full mesh.
#  - restricted: restricted to all unconstrained dofs, A(mu) and f(mu) are the full ones.
#  - interpolation: the interpolation dofs of A(mu) and f(mu).
#  - reduction: the errors of the reduced solutions and of the flow rate, whose functional is the
#    interpolated right-hand side, projected like the right-hand side.
# Runs serially and on 2 ranks (SPMD).

import common  # first, see common.py
from common import log, run
import dealii_pymor
import numpy as np
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor

model, fom = common.periodic_hill_fom()
h1 = fom.h1_semi_product
parameter_space = fom.parameters.space(common.PERIODIC_HILL_RANGES)

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

test_set = parameter_space.sample_randomly(10)
solutions = [fom.solve(mu) for mu in test_set]
outputs = [fom.output(mu)[0, 0] for mu in test_set]


def solution_error(reductor, rom):
    return max(((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]
               for mu, u in zip(test_set, solutions))


def output_error(rom):
    return max(abs(rom.output(mu)[0, 0] - Q) / Q for mu, Q in zip(test_set, outputs))


def test_restricted():
    # The unconstrained dofs are the nonzero entries of an admissible vector of ones. They include
    # the dofs on the periodic boundary, whose rows also collect the cells at the other end of the
    # channel.
    mu = fom.parameters.parse({'hill_height': 1.2, 'hill_width': 2.3})
    dofs = np.flatnonzero(fom.solution_space.ones().to_numpy()[:, 0])
    U = fom.solve(mu)
    restricted_operator, source_dofs = fom.operator.restricted(dofs)
    AU = fom.operator.apply(U, mu=mu).dofs(dofs)
    AU_restricted = restricted_operator.apply(
        restricted_operator.source.make_array(U.dofs(source_dofs)), mu=mu).to_numpy()
    restricted_rhs, _ = fom.rhs.restricted(dofs)
    F = fom.rhs.as_range_array(mu).dofs(dofs)
    log(f'all {len(dofs)} unconstrained dofs, same A(mu) u: '
        f'{np.abs(AU_restricted - AU).max() < 1e-12 * np.abs(AU).max()}, same f(mu): '
        f'{np.abs(restricted_rhs.as_range_array(mu).to_numpy() - F).max() < 1e-12 * np.abs(F).max()}')


def test_interpolation():
    log(f'operator {len(operator_dofs)} dofs (from {len(operator_snapshots)} snapshots), '
        f'right-hand side {len(rhs_dofs)} (from {len(rhs_snapshots)})')


def test_reduction():
    reductor = StationaryRBReductor(fom, RB=basis, product=h1)
    log(f'reduced basis alone: largest relative error {solution_error(reductor, reductor.reduce()):.1e}')

    rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, rhs_dofs, rhs_basis, triangular=True)
    for M in (50, 100, 200):
        operator_ei = EmpiricalInterpolatedOperator(fom.operator, operator_dofs[:M],
                                                    operator_basis[:M], triangular=True)
        reductor = dealii_pymor.DealIIStationaryRBReductor(
            fom.with_(operator=operator_ei, rhs=rhs_ei,
                      output_functional=common.periodic_hill_flow_rate(model, fom, rhs_ei)),
            RB=basis, product=h1)
        rom = reductor.reduce()
        log(f'operator interpolation dofs {M:3d}: largest relative error '
            f'{solution_error(reductor, rom):.1e}, of the flow rate {output_error(rom):.1e}')

    log(f'reduced operator: {type(rom.operator).__name__}, right-hand side: {type(rom.rhs).__name__}, '
        f'in the flow rate: {type(rom.output_functional.operators[0].rhs).__name__}')


run(test_restricted, test_interpolation, test_reduction)
