# The full-order periodic hill model (apps/include/implementations/periodic_hill_potential_flow.h)
# in pyMOR: the system operator depends on the hill height and width, the C++ model assembles it for
# every mu (DealIIParametricOperator).
#  - solve: a full-order solve and the flow written on the channel.
#  - flow_rate: the output Q(mu) = (|Omega(mu)| - f(mu)^T phi_tilde) / 9 is flow_rate() of the C++
#    model.
#  - adjoint: the adjoint of A(mu) and the adjoint solve, which the C++ operator implements by
#    itself, being symmetric.
# Runs serially and on 2 ranks (SPMD).

import common  # first, see common.py
from common import log, run

model, fom = common.periodic_hill_fom()
# Close to the ERCOFTAC hill.
mu = fom.parameters.parse({'hill_height': 1., 'hill_width': 1.93})
U = fom.solve(mu)


def test_solve():
    log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}, '
        f'operator: {type(fom.operator).__name__}')
    F = fom.rhs.as_range_array(mu)
    log(f'relative residual below 1e-9: {((fom.operator.apply(U, mu=mu) - F).norm() / F.norm())[0] < 1e-9}')
    log(f'h1_semi norm of phi_tilde: {fom.h1_semi_norm(U)[0]:.6e}')
    log(f'flow written: {model.write_flow_vtu("", "flow", mu, [U.vectors[0].impl], ["fom"])}')


def test_flow_rate():
    Q = fom.output(mu)[0, 0]
    log(f'flow rate: {Q:.6f}, same as flow_rate() of the C++ model: '
        f'{abs(Q - model.flow_rate(mu, U.vectors[0].impl)) < 1e-12 * Q}')


def test_adjoint():
    # The right-hand side of another mu as test vector: (A^T G, U) = (G, A U), and the adjoint solve
    # A^T Z = G.
    G = fom.rhs.as_range_array(fom.parameters.parse({'hill_height': 0.7, 'hill_width': 3.}))
    AG_U = fom.operator.apply_adjoint(G, mu=mu).inner(U)[0, 0]
    G_AU = G.inner(fom.operator.apply(U, mu=mu))[0, 0]
    log(f'(A^T G, U) = (G, A U): {abs(AG_U - G_AU) < 1e-12 * abs(G_AU)}')
    Z = fom.operator.apply_inverse_adjoint(G, mu=mu)
    log(f'adjoint solve, relative residual below 1e-9: '
        f'{((fom.operator.apply_adjoint(Z, mu=mu) - G).norm() / G.norm())[0] < 1e-9}')


run(test_solve, test_flow_rate, test_adjoint)
