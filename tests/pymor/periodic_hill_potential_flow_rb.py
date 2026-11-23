# Potential flow through the periodic hill (apps/include/implementations/
# periodic_hill_potential_flow.h) in pyMOR: the system operator depends on the hill height and
# width, the C++ model assembles it for every mu. A full-order solve with the flow written on the
# channel, and a POD reduced basis. Without empirical interpolation, the reduced operator still
# assembles A(mu) on the full mesh for every mu. The flow rate Q(mu) as output: the compliant output
# (|Omega(mu)| - f(mu)^T phi_tilde) / 9, whose reduced error is ||e||^2 / 9 in the energy norm of
# A(mu) by Galerkin orthogonality. Runs serially and on 2 ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.reductors.basic import StationaryRBReductor

import dealii_core_module as dealii
import periodic_hill


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


# The only warnings are the expected ones about projecting the operator and the right-hand side
# generically.
set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

model = periodic_hill.make_potential_flow_model(n_refinements=2, degree=2)
fom = dealii_pymor.stationary_model(model, name='periodic_hill_potential_flow')
L = periodic_hill.length
volume = GenericParameterFunctional(lambda mu: model.volume(mu) / L, fom.parameters)
fom = fom.with_(output_functional=dealii_pymor.compliant_output(fom.rhs, factor=-1 / L, offset=volume,
                                                                name='flow_rate'))
h1 = fom.h1_semi_product
log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}, '
    f'operator: {type(fom.operator).__name__}')

# Close to the ERCOFTAC hill.
mu = fom.parameters.parse({'hill_height': 1., 'hill_width': 1.93})
U = fom.solve(mu)
F = fom.rhs.as_range_array(mu)
log(f'relative residual below 1e-9: {((fom.operator.apply(U, mu=mu) - F).norm() / F.norm())[0] < 1e-9}')
log(f'h1_semi norm of phi_tilde: {fom.h1_semi_norm(U)[0]:.6e}')
Q = fom.output(mu)[0, 0]
log(f'flow rate: {Q:.6f}, same as flow_rate() of the C++ model: '
    f'{abs(Q - model.flow_rate(mu, U.vectors[0].impl)) < 1e-12 * Q}')
log(f'flow written: {model.write_flow_vtu("", "flow", mu, [U.vectors[0].impl], ["fom"])}')

# Steepest slope 15 h / (8 w) up to 1.4 (54 degrees).
parameter_space = fom.parameters.space({'hill_height': (0.5, 1.5), 'hill_width': (2., 4.5)})
snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(6):
    snapshots.append(fom.solve(mu))
basis, singular_values = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
log(f'snapshots: {len(snapshots)}, POD modes: {len(basis)}')
log('singular values relative to the first: '
    + ' '.join(f'{s:.1e}' for s in singular_values[1:16:3] / singular_values[0]))

reductor = StationaryRBReductor(fom, RB=basis, product=h1)
test_set = parameter_space.sample_randomly(10)
solutions = [fom.solve(mu) for mu in test_set]
outputs = [fom.output(mu)[0, 0] for mu in test_set]
for n in (4, 8, 16):
    rom = reductor.reduce(n)
    errors, output_errors, compliant = [], [], True
    for mu, u, Q in zip(test_set, solutions, outputs):
        e = u - reductor.reconstruct(rom.solve(mu))
        errors.append((e.norm(h1) / u.norm(h1))[0])
        output_errors.append(abs(rom.output(mu)[0, 0] - Q) / Q)
        energy = fom.operator.apply(e, mu=mu).inner(e)[0, 0] / L
        compliant = compliant and abs(rom.output(mu)[0, 0] - Q - energy) < 1e-6 * energy
    log(f'reduced dimension {n:2d}: largest relative error {max(errors):.1e}, '
        f'of the flow rate {max(output_errors):.1e}, the latter is ||e||^2 / 9: {compliant}')
log(f'reduced operator: {type(rom.operator).__name__}')
