# Potential flow through the periodic hill (apps/include/implementations/
# periodic_hill_potential_flow.h) in pyMOR, reduced with empirical interpolation of the system
# operator and the right-hand side, like tests/pymor/periodic_hill_potential_flow_ei.py. For every
# test parameter, the full-order and the reduced flow and their difference are written to output/
# on the channel of the parameter; <filename>_test.pvd collects them, one time step per test
# parameter. Runs serially and on several ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
import numpy as np
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parameters.functionals import GenericParameterFunctional

import dealii_core_module as dealii
import periodic_hill

import os
import time


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


def write_pvd(filename, files):
    # A ParaView collection of the given .pvtu files, the k-th as time step k.
    if dealii.this_mpi_process() == 0:
        with open(filename, 'w') as pvd:
            pvd.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
            for k, file in enumerate(files):
                pvd.write(f'  <DataSet timestep="{k}" file="{os.path.basename(file)}"/>\n')
            pvd.write('</Collection>\n</VTKFile>\n')
    return filename


set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})


def run(model, filename, n_basis=16, n_operator_dofs=200):
    # Wrap the C++ model from periodic_hill.make_potential_flow_model() in a pyMOR
    # StationaryModel. Its system operator depends on the parameters: the C++ model assembles
    # A(mu) on the reference box for every mu (DealIIParametricOperator).
    fom = dealii_pymor.stationary_model(model, name='periodic_hill_potential_flow')
    # The flow rate Q(mu) = (|Omega(mu)| - f(mu)^T phi_tilde) / 9 as output, a compliant output: its
    # reduced error is quadratic in the error of phi_tilde. To prescribe a flow rate instead of the
    # jump 9 of phi, scale the velocity by it over Q(mu).
    L = periodic_hill.length
    volume = GenericParameterFunctional(lambda mu: model.volume(mu) / L, fom.parameters)

    def flow_rate(rhs):
        return dealii_pymor.compliant_output(rhs, factor=-1 / L, offset=volume, name='flow_rate')

    fom = fom.with_(output_functional=flow_rate(fom.rhs))
    h1 = fom.h1_semi_product
    log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}')

    # Steepest slope 15 h / (8 w) up to 1.4 (54 degrees), the narrowest gap at the crest 1.5.
    parameter_space = fom.parameters.space({'hill_height': (0.5, 1.5), 'hill_width': (2., 4.5)})


    # ------------------Reduced Basis ---------------------------
    # All snapshots live on the same mesh of the reference box, so POD needs no transformation.
    # h1 is the H1 seminorm on the channel of a fixed parameter (product_mu of the model).
    training_set = parameter_space.sample_uniformly(8)
    snapshots = fom.solution_space.empty()
    start = time.time()
    for mu in training_set:
        snapshots.append(fom.solve(mu))
    end = time.time()
    log(f'time for snapshots: {end - start:.2f} s ({len(snapshots)} solves)')

    start = time.time()
    basis, singular_values = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    basis = basis[:n_basis]
    end = time.time()
    log(f'time for POD: {end - start:.2f} s')
    log('singular values relative to the first: '
        + ' '.join(f'{s:.1e}' for s in singular_values[:n_basis] / singular_values[0]))
    # The modes as phi_tilde on the reference box, the space pyMOR works in.
    log(f'visualized: {fom.visualize(basis, filename=f"output/{filename}_modes", legend=[f"mode_{i}" for i in range(1, len(basis) + 1)])}')


    # ------------------Empirical Interpolation of the Operator ---------------------------
    # The reduced operator applies A(mu) to the basis vectors, so the collateral basis has to span
    # A(mu) v for the basis vectors v. (For a linear problem, A(mu) u(mu) is just f(mu) again.)
    operator_snapshots = fom.solution_space.empty()
    start = time.time()
    for mu in training_set:
        operator_snapshots.append(fom.operator.apply(basis, mu=mu))
    end = time.time()
    log(f'time for operator snapshots: {end - start:.2f} s')

    start = time.time()
    operator_dofs, operator_basis, _ = ei_greedy(operator_snapshots, max_interpolation_dofs=n_operator_dofs)
    end = time.time()
    log(f'time for empirical interpolation of the operator: {end - start:.2f} s')
    log(f'operator snapshots: {len(operator_snapshots)}, interpolation dofs: {len(operator_dofs)}')


    # ------------------Empirical Interpolation of the Right-Hand Side ---------------------------
    # f(mu) is linear in the hill height, but needs many more widths than the training set has:
    # with its 8 widths, the snapshots only span 8 dimensions, and further interpolation dofs would
    # be chosen from round-off. Right-hand sides are cheap, an assembly each.
    rhs_snapshots = fom.rhs.range.empty()
    start = time.time()
    for width in np.linspace(2., 4.5, 100):
        rhs_snapshots.append(fom.rhs.as_range_array(fom.parameters.parse({'hill_height': 1.,
                                                                          'hill_width': width})))
    end = time.time()
    log(f'time for right-hand side snapshots: {end - start:.2f} s')

    start = time.time()
    rhs_dofs, rhs_basis, _ = ei_greedy(rhs_snapshots, max_interpolation_dofs=64, rtol=1e-6)
    end = time.time()
    log(f'time for empirical interpolation of the right-hand side: {end - start:.2f} s')
    log(f'right-hand side snapshots: {len(rhs_snapshots)}, interpolation dofs: {len(rhs_dofs)}')


    # ------------------Reduced Model ---------------------------
    # DealIIStationaryRBReductor instead of StationaryRBReductor, so that the interpolated
    # right-hand side is projected efficiently as well (see dealii_pymor/reductors.py).
    rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, rhs_dofs, rhs_basis, triangular=True)
    operator_ei = EmpiricalInterpolatedOperator(fom.operator, operator_dofs, operator_basis, triangular=True)
    start = time.time()
    # The flow rate with the interpolated right-hand side, projected like it.
    reductor = dealii_pymor.DealIIStationaryRBReductor(
        fom.with_(operator=operator_ei, rhs=rhs_ei, output_functional=flow_rate(rhs_ei)),
        RB=basis, product=h1)
    rom = reductor.reduce()
    end = time.time()
    log(f'time for reduction: {end - start:.2f} s, '
        f'source dofs of the interpolated operator: {len(operator_ei.source_dofs)}')


    # ------------------Comparison on the Test Set ---------------------------
    # For every test parameter: the full-order and the reduced solution, their relative error in the
    # h1 product, and on the channel of the parameter the flows fom_* and rom_* (potential,
    # velocity, pressure) and their difference error_* (potential, velocity).
    test_set = parameter_space.sample_randomly(20)
    files, errors, output_errors, fom_times, rom_times = [], [], [], [], []
    for k, mu in enumerate(test_set):
        start = time.time()
        u = fom.solve(mu)
        fom_times.append(time.time() - start)

        start = time.time()
        u_rb = rom.solve(mu)
        rom_times.append(time.time() - start)

        u_reconstructed = reductor.reconstruct(u_rb)
        error = u - u_reconstructed
        errors.append((error.norm(h1) / u.norm(h1))[0])
        Q = fom.output_functional.apply(u, mu=mu).to_numpy()[0, 0]
        output_errors.append(abs(rom.output(mu)[0, 0] - Q) / Q)
        files.append(model.write_flow_vtu('output/', f'{filename}_test_{k}', mu,
                                          [u.vectors[0].impl, u_reconstructed.vectors[0].impl],
                                          ['fom', 'rom'], [error.vectors[0].impl], ['error']))
        log(f'test {k}: hill height {mu["hill_height"][0]:.3f}, width {mu["hill_width"][0]:.3f}, '
            f'relative error {errors[-1]:.1e}, flow rate {Q:.4f} (error {output_errors[-1]:.1e}), time fom {fom_times[-1]:.3f} s, rom {rom_times[-1]:.4f} s')

    log(f'visualized: {write_pvd(f"output/{filename}_test.pvd", files)}')
    log(f'largest relative error {max(errors):.1e}, of the flow rate {max(output_errors):.1e}, '
        f'mean speedup {np.mean(fom_times) / np.mean(rom_times):.0f}')


if __name__ == '__main__':
    run(periodic_hill.make_potential_flow_model(n_refinements=4, degree=2),
        filename='periodic_hill_potential_flow_ei', n_basis=16, n_operator_dofs=100)
