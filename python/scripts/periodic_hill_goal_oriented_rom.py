# Goal-oriented reduced bases for the potential flow through the periodic hill
# (apps/include/implementations/periodic_hill_potential_flow.h), for two quantities of interest
# (apps/include/implementations/periodic_hill_functionals.h):
#  - the flux through the lower third of the cross-section x = 9/4, which lies on the flank of the
#    hill for wide hills and in the valley for narrow ones (affine in phi_tilde), and
#  - the mean pressure on the hill for 0 <= x <= 2, i.e. on its flank for every width (quadratic),
# both for a prescribed flow rate Q0 instead of the jump 9 of phi that the model solves for.
#
# A reduced basis that only approximates the solutions approximates a quantity of interest J only as
# well as the solutions. With the adjoint solutions z(mu), A(mu)^T z = -dJ/du, in the basis as well,
# Galerkin orthogonality gives J(u) - J(u_N) ~ (A (u - u_N), z - z_N), roughly the product of the
# errors of both, as for the dual-weighted residual correction, but without a reduced adjoint
# problem. For every quantity, a basis of 2n POD modes of the solutions is compared to one of n POD
# modes of the solutions and n of the adjoint solutions, so both reduced models have the same size.
#
# No empirical interpolation: the reduced operator assembles A(mu) on the full mesh for every mu,
# and the outputs are evaluated on the reconstructed solutions. Both only cost time and do not change
# the errors, which are what is compared here. Runs serially and on several ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.operators.constructions import LincombOperator
from pymor.operators.interface import Operator
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.reductors.basic import StationaryRBReductor

import dealii_core_module as dealii
import periodic_hill

import os
import time


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


# The only warnings are the expected ones about projecting the operator generically.
set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})


class AtFlowRate(Operator):
    """A quantity of interest J of the computed flow, for the flow with the flow rate Q0 instead.

    The model solves for the potential with the jump 9 over one period, whose flow has the flow rate
    Q(mu; phi_tilde), see the flow rate output below. The problem is linear, so the flow with the
    flow rate Q0 has the velocity u Q0 / Q. A quantity that is homogeneous of the degree k in the
    velocity (k = 1 for a flux, 2 for the pressure -|u|^2 / 2) therefore is

        J_Q0(mu; phi_tilde) = (Q0 / Q)^k J(mu; phi_tilde).

    This is not linear in phi_tilde, even for a linear J. Its derivative, the right-hand side of the
    adjoint problem, follows by the product and the chain rule:

        dJ_Q0 / dphi_tilde = (Q0 / Q)^k (dJ / dphi_tilde - k J / Q dQ / dphi_tilde).

    pyMOR needs `apply` for the output and `jacobian` for the adjoint problem; J and Q provide
    theirs, and the parameters of this operator are the ones of J and Q.
    """

    linear = False

    def __init__(self, J, flow_rate, Q0, degree, name=None):
        self.__auto_init(locals())
        self.source = J.source
        self.range = J.range

    def apply(self, U, mu=None):
        J = self.J.apply(U, mu=mu).to_numpy()
        Q = self.flow_rate.apply(U, mu=mu).to_numpy()
        return self.range.make_array((self.Q0 / Q) ** self.degree * J)

    def jacobian(self, U, mu=None):
        assert len(U) == 1
        J = self.J.apply(U, mu=mu).to_numpy()[0, 0]
        Q = self.flow_rate.apply(U, mu=mu).to_numpy()[0, 0]
        scaling = (self.Q0 / Q) ** self.degree
        return LincombOperator([self.J.jacobian(U, mu=mu), self.flow_rate.jacobian(U, mu=mu)],
                               [scaling, -scaling * self.degree * J / Q])


def run(model, filename, flux_x=periodic_hill.length / 4, flux_y_hat=(0., 1 / 3),
        pressure_x=(0., 2.), Q0=2.036, n_train=8, n_test=20, ns=(1, 2, 4, 6, 8, 12)):
    """Compare the reduced bases for both quantities of interest.

    Parameters
    ----------
    model
        The C++ model from periodic_hill.make_potential_flow_model().
    filename
        Basename of the files written to output/.
    flux_x, flux_y_hat
        The cross-section x = flux_x and its part y_hat_lower <= y_hat <= y_hat_upper of the flux,
        in coordinates of the reference box: y_hat in [0, 1] from the hill to the top wall.
    pressure_x
        The interval of x of the mean pressure on the hill.
    Q0
        The prescribed flow rate; 2.036 is the one of the ERCOFTAC case, with the bulk velocity 1
        in the gap of the height 3.036 - 1 above the crest.
    n_train
        Training parameters per parameter, on a uniform grid.
    n_test
        Random test parameters.
    ns
        The numbers n of primal and of adjoint modes, the reduced dimension is 2 n.
    """
    os.makedirs('output', exist_ok=True)

    # ------------------Full-Order Model and Quantities of Interest ---------------------------
    fom = dealii_pymor.stationary_model(model, name='periodic_hill_potential_flow')
    h1 = fom.h1_semi_product
    # The flow rate Q(mu) = (|Omega(mu)| - f(mu)^T phi_tilde) / 9, see
    # periodic_hill_potential_flow_ei.py.
    L = periodic_hill.length
    volume = GenericParameterFunctional(lambda mu: model.volume(mu) / L, fom.parameters)
    flow_rate = dealii_pymor.compliant_output(fom.rhs, factor=-1 / L, offset=volume, name='flow_rate')
    # The C++ functionals as pyMOR operators, for the jump 9 of phi, and at the flow rate Q0.
    flux = dealii_pymor.DealIIFunctional(
        periodic_hill.CrossSectionFlux(model, x=flux_x, y_hat_lower=flux_y_hat[0], y_hat_upper=flux_y_hat[1]),
        model, name='flux')
    pressure = dealii_pymor.DealIIFunctional(
        periodic_hill.WallPressure(model, x_lower=pressure_x[0], x_upper=pressure_x[1]),
        model, name='pressure')
    qois = {'flux': AtFlowRate(flux, flow_rate, Q0, 1, name='flux'),
            'pressure': AtFlowRate(pressure, flow_rate, Q0, 2, name='pressure')}
    log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}')
    log(f'flux through {flux_y_hat[0]:.3g} <= y_hat <= {flux_y_hat[1]:.3g} at x = {flux_x:.3g}, '
        f'mean pressure on the hill for {pressure_x[0]:.3g} <= x <= {pressure_x[1]:.3g}, '
        f'at the flow rate {Q0}')

    # Steepest slope 15 h / (8 w) up to 1.4 (54 degrees), the narrowest gap at the crest 1.5.
    parameter_space = fom.parameters.space({'hill_height': (0.5, 1.5), 'hill_width': (2., 4.5)})

    # Close to the ERCOFTAC hill: the quantities for the jump 9 and at the flow rate Q0.
    mu = fom.parameters.parse({'hill_height': 1., 'hill_width': 1.93})
    u = fom.solve(mu)
    Q = flow_rate.apply(u, mu=mu).to_numpy()[0, 0]
    log(f'hill height 1, width 1.93: flow rate {Q:.4f}; flux {flux.apply(u, mu=mu).to_numpy()[0, 0]:.4f}, '
        f'at Q0 {qois["flux"].apply(u, mu=mu).to_numpy()[0, 0]:.4f}; pressure '
        f'{pressure.apply(u, mu=mu).to_numpy()[0, 0]:.4f}, at Q0 {qois["pressure"].apply(u, mu=mu).to_numpy()[0, 0]:.4f}')


    # ------------------Primal Snapshots ---------------------------
    training_set = parameter_space.sample_uniformly(n_train)
    start = time.time()
    snapshots = fom.solution_space.empty()
    for mu_train in training_set:
        snapshots.append(fom.solve(mu_train))
    primal_modes, primal_values = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    log(f'primal snapshots: {len(snapshots)}, time with POD {time.time() - start:.1f} s')
    log(f'visualized: {fom.visualize(primal_modes, filename=f"output/{filename}_modes", legend=[f"mode_{i}" for i in range(1, len(primal_modes) + 1)])}')



    # ------------------Adjoint Snapshots ---------------------------
    # For every quantity J and training parameter, the adjoint problem A(mu)^T z = -dJ/du(mu; u(mu))
    # at the primal snapshot u(mu): its right-hand side depends on u(mu), unless J is affine, so the
    # primal snapshots are reused. A(mu) is symmetric, the C++ operator solves the adjoint problem
    # like the primal one.
    adjoint_modes, adjoints = {}, {}
    for name, J in qois.items():
        start = time.time()
        adjoint_snapshots = fom.solution_space.empty()
        for mu_train, u_train in zip(training_set, snapshots):
            gradient = J.jacobian(u_train, mu=mu_train).as_source_array(mu_train)
            adjoint_snapshots.append(fom.operator.apply_inverse_adjoint(-gradient, mu=mu_train))
        adjoint_modes[name], adjoint_values = pod(adjoint_snapshots, product=h1, rtol=0, atol=0, l2_err=0)
        log(f'{name}: adjoint snapshots {len(adjoint_snapshots)}, time with POD {time.time() - start:.1f} s, '
            'singular values relative to the first, primal and adjoint:')
        log('  ' + ' '.join(f'{s:.1e}' for s in primal_values[:12] / primal_values[0]))
        log('  ' + ' '.join(f'{s:.1e}' for s in adjoint_values[:12] / adjoint_values[0]))
        gradient = J.jacobian(u, mu=mu).as_source_array(mu)
        adjoints[name] = fom.operator.apply_inverse_adjoint(-gradient, mu=mu)
    # The adjoint solutions for the ERCOFTAC-like hill: where the quantities "look".
    log(f'visualized: {model.write_flow_vtu("output/", f"{filename}_adjoints", mu, [u.vectors[0].impl], ["fom"], [z.vectors[0].impl for z in adjoints.values()], [f"adjoint_{name}" for name in adjoints])}')
    log(f'visualized: {fom.visualize(adjoint_modes["flux"], filename=f"output/{filename}_modes_adjoint_flux", legend=[f"mode_{i}" for i in range(1, len(adjoint_modes["flux"]) + 1)])}')
    log(f'visualized: {fom.visualize(adjoint_modes["pressure"], filename=f"output/{filename}_modes_adjoint_pressure", legend=[f"mode_{i}" for i in range(1, len(adjoint_modes["pressure"]) + 1)])}')

    # ------------------Comparison on the Test Set ---------------------------
    # The full-order solutions and quantities once; for every quantity and n, the reduced models of
    # 2n primal modes and of n primal and n adjoint modes, orthonormalized together.
    test_set = parameter_space.sample_randomly(n_test)
    solutions = [fom.solve(mu_test) for mu_test in test_set]
    results = {}
    for name, J in qois.items():
        fom_J = fom.with_(output_functional=J)
        values = [J.apply(u_test, mu=mu_test).to_numpy()[0, 0] for mu_test, u_test in zip(test_set, solutions)]

        def errors(basis):
            reductor = StationaryRBReductor(fom_J, RB=basis, product=h1)
            rom = reductor.reduce()
            output_errors, solution_errors = [], []
            for mu_test, u_test, value in zip(test_set, solutions, values):
                # The reduced output functional evaluates J on the reconstructed solution.
                u_N = rom.solve(mu_test)
                J_N = rom.output_functional.apply(u_N, mu=mu_test).to_numpy()[0, 0]
                output_errors.append(abs(J_N - value) / abs(value))
                solution_errors.append(((u_test - reductor.reconstruct(u_N)).norm(h1) / u_test.norm(h1))[0])
            return max(output_errors), np.mean(output_errors), max(solution_errors)

        start = time.time()
        results[name] = []
        log(f'{name}: largest and mean relative error of the quantity, largest relative error of the solution')
        log('     n | 2n primal modes             | n primal + n adjoint modes')
        for n in ns:
            goal_oriented = primal_modes[:n].copy()
            goal_oriented.append(adjoint_modes[name][:n])
            goal_oriented = gram_schmidt(goal_oriented, product=h1)
            plain, goal = errors(primal_modes[:2 * n]), errors(goal_oriented)
            results[name].append((n, plain, goal))
            log(f'  {n:4d} | {plain[0]:.1e} {plain[1]:.1e} {plain[2]:.1e} | {goal[0]:.1e} {goal[1]:.1e} {goal[2]:.1e}')
        log(f'  time {time.time() - start:.1f} s')

    log(f'plotted: {plot(results, f"output/{filename}_errors.png")}')


def plot(results, filename):
    # The largest relative errors over the reduced dimension 2n, if matplotlib is installed.
    if dealii.this_mpi_process() != 0:
        return filename
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'not, matplotlib is not installed'
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), results.items()):
        dims = [2 * n for n, _, _ in rows]
        ax.semilogy(dims, [plain[0] for _, plain, _ in rows], 'o-', color='C0', label='quantity, 2n primal')
        ax.semilogy(dims, [goal[0] for _, _, goal in rows], 's-', color='C1', label='quantity, n primal + n adjoint')
        ax.semilogy(dims, [plain[2] for _, plain, _ in rows], 'o--', color='C0', alpha=0.5, label='solution, 2n primal')
        ax.semilogy(dims, [goal[2] for _, _, goal in rows], 's--', color='C1', alpha=0.5,
                    label='solution, n primal + n adjoint')
        ax.set_title(name)
        ax.set_xlabel('reduced dimension 2n')
        ax.grid(True, which='both', alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel('largest relative error')
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    return filename


if __name__ == '__main__':
    run(periodic_hill.make_potential_flow_model(n_refinements=4, degree=2),
        filename='periodic_hill_goal_oriented_rom')
