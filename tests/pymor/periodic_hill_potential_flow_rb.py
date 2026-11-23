# A POD reduced basis for the periodic hill model (periodic_hill_potential_flow_fom.py), without
# empirical interpolation: the reduced operator still assembles A(mu) on the full mesh for every mu.
#  - pod: the singular values of the snapshots.
#  - reduced_basis: the errors of the reduced solutions and of the flow rate Q(mu), a compliant
#    output, whose reduced error is ||e||^2 / 9 in the energy norm of A(mu) by Galerkin
#    orthogonality.
# Runs serially and on 2 ranks (SPMD).

import common  # first, see common.py
from common import log, run
from pymor.algorithms.pod import pod
from pymor.reductors.basic import StationaryRBReductor

import periodic_hill

model, fom = common.periodic_hill_fom()
h1 = fom.h1_semi_product
parameter_space = fom.parameters.space(common.PERIODIC_HILL_RANGES)

snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(6):
    snapshots.append(fom.solve(mu))
basis, singular_values = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)

test_set = parameter_space.sample_randomly(10)
solutions = [fom.solve(mu) for mu in test_set]
outputs = [fom.output(mu)[0, 0] for mu in test_set]


def test_pod():
    log(f'snapshots: {len(snapshots)}, POD modes: {len(basis)}')
    log('singular values relative to the first: '
        + ' '.join(f'{s:.1e}' for s in singular_values[1:16:3] / singular_values[0]))


def test_reduced_basis():
    reductor = StationaryRBReductor(fom, RB=basis, product=h1)
    for n in (4, 8, 16):
        rom = reductor.reduce(n)
        errors, output_errors, compliant = [], [], True
        for mu, u, Q in zip(test_set, solutions, outputs):
            e = u - reductor.reconstruct(rom.solve(mu))
            errors.append((e.norm(h1) / u.norm(h1))[0])
            output_errors.append(abs(rom.output(mu)[0, 0] - Q) / Q)
            energy = fom.operator.apply(e, mu=mu).inner(e)[0, 0] / periodic_hill.length
            compliant = compliant and abs(rom.output(mu)[0, 0] - Q - energy) < 1e-6 * energy
        log(f'reduced dimension {n:2d}: largest relative error {max(errors):.1e}, '
            f'of the flow rate {max(output_errors):.1e}, the latter is ||e||^2 / 9: {compliant}')
    log(f'reduced operator: {type(rom.operator).__name__}')


run(test_pod, test_reduced_basis)
