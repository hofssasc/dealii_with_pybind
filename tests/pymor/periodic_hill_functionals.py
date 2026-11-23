# The quantities of interest of the periodic hill (apps/include/implementations/
# periodic_hill_functionals.h) as pyMOR operators (DealIIFunctional), for goal-oriented reduction:
# the flux through the lower third of the cross-section x = 9/4 (affine in phi_tilde) and the mean
# pressure on the hill for 0 <= x <= 2 (quadratic).
#  - functional: the values are the ones of the C++ functionals, the jacobian matches central
#    differences.
#  - adjoint: the adjoint solution z = A^-T (-dJ/du) gives the derivative of J in the direction of a
#    solution A^-1 g as -(z, g).
#  - goal_oriented: the output errors of a reduced basis of 2n POD modes of the solutions against one
#    of n modes of the solutions and n of the adjoint solutions.
# Runs serially and on 2 ranks (SPMD).

import common  # first, see common.py
from common import log, run
import dealii_pymor
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.reductors.basic import StationaryRBReductor

import periodic_hill

model, fom = common.periodic_hill_fom()
h1 = fom.h1_semi_product
parameter_space = fom.parameters.space(common.PERIODIC_HILL_RANGES)
functionals = {
    'flux': dealii_pymor.DealIIFunctional(
        periodic_hill.CrossSectionFlux(model, x=periodic_hill.length / 4, y_hat_lower=0., y_hat_upper=1 / 3),
        model, name='flux'),
    'pressure': dealii_pymor.DealIIFunctional(periodic_hill.WallPressure(model, x_lower=0., x_upper=2.),
                                              model, name='pressure'),
}

mu = fom.parameters.parse({'hill_height': 1., 'hill_width': 3.})
U = fom.solve(mu)
# A direction: the solution for another mu.
V = fom.solve(fom.parameters.parse({'hill_height': 0.7, 'hill_width': 2.5}))


def adjoint_solution(J, u, mu):
    return fom.operator.apply_inverse_adjoint(-J.jacobian(u, mu).as_source_array(mu), mu=mu)


def test_functional():
    for name, J in functionals.items():
        value = J.apply(U, mu=mu).to_numpy()[0, 0]
        same = abs(value - J.impl.value(mu, U.vectors[0].impl)) < 1e-14 * abs(value)
        step = 1e-2
        difference = (J.apply(U + step * V, mu=mu) - J.apply(U - step * V, mu=mu)).to_numpy()[0, 0] / (2 * step)
        derivative = J.jacobian(U, mu=mu).apply(V).to_numpy()[0, 0]
        log(f'{name}: value {value:.6f}, the one of the C++ functional: {same}, '
            f'jacobian matches central differences: {abs(derivative - difference) < 1e-10 * abs(derivative)}')


def test_adjoint():
    g = fom.rhs.as_range_array(fom.parameters.parse({'hill_height': 0.7, 'hill_width': 2.5}))
    direction = fom.operator.apply_inverse(g, mu=mu)
    for name, J in functionals.items():
        z = adjoint_solution(J, U, mu)
        derivative = J.jacobian(U, mu=mu).apply(direction).to_numpy()[0, 0]
        log(f'{name}: dJ/du A^-1 g = -(z, g): {abs(derivative + z.inner(g)[0, 0]) < 1e-8 * abs(derivative)}')


def test_goal_oriented():
    training_set = parameter_space.sample_uniformly(6)
    snapshots = fom.solution_space.empty()
    for mu_train in training_set:
        snapshots.append(fom.solve(mu_train))
    primal_modes, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    test_set = parameter_space.sample_randomly(10)
    for name, J in functionals.items():
        adjoint_snapshots = fom.solution_space.empty()
        for mu_train, u in zip(training_set, snapshots):
            adjoint_snapshots.append(adjoint_solution(J, u, mu_train))
        adjoint_modes, _ = pod(adjoint_snapshots, product=h1, rtol=0, atol=0, l2_err=0)
        fom_J = fom.with_(output_functional=J)
        outputs = [fom_J.output(mu_test)[0, 0] for mu_test in test_set]

        def output_error(basis):
            rom = StationaryRBReductor(fom_J, RB=basis, product=h1).reduce()
            return max(abs(rom.output(mu_test)[0, 0] - J_test) / abs(J_test)
                       for mu_test, J_test in zip(test_set, outputs))

        for n in (4, 8):
            goal_oriented = primal_modes[:n].copy()
            goal_oriented.append(adjoint_modes[:n])
            goal_oriented = gram_schmidt(goal_oriented, product=h1)
            log(f'{name}, n = {n}: largest relative output error with {2 * n} primal modes '
                f'{output_error(primal_modes[:2 * n]):.1e}, with {n} + {n} primal and adjoint modes '
                f'{output_error(goal_oriented):.1e}')


run(test_functional, test_adjoint, test_goal_oriented)
