# The Gaussian source model (apps/include/implementations/gaussian_source.h) in pyMOR: a
# full-order solve and a POD reduced basis from snapshots. Runs serially and on 2 ranks (SPMD).
# gaussian_source_matrix_free_rb.py runs the same for the matrix-free model.

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.reductors.basic import StationaryRBReductor

import dealii_core_module as dealii
import gaussian_source


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


# Every rank logs, in no particular order. The only warning is the expected one about projecting
# the non-affine right-hand side generically.
set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

def run(model):
    # Wrap the C++ model from gaussian_source.make_model() (or make_matrix_free_model()) in a pyMOR
    # StationaryModel.
    fom = dealii_pymor.stationary_model(model, name='gaussian_source')
    h1 = fom.h1_0_semi_product
    log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}')

    mu = fom.parameters.parse([0.3, 0.6])
    U = fom.solve(mu)
    F = fom.rhs.as_range_array(mu)
    log(f'relative residual below 1e-9: {((fom.operator.apply(U) - F).norm() / F.norm())[0] < 1e-9}')
    log(f'h1_0_semi norm of the solution: {fom.h1_0_semi_norm(U)[0]:.6e}')
    log(f'visualized: {fom.visualize(U, filename="solution", legend="u")}')

    parameter_space = fom.parameters.space(0.2, 0.8)
    training_set = parameter_space.sample_uniformly(6)
    snapshots = fom.solution_space.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))
    basis, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    log(f'snapshots: {len(snapshots)}, POD modes: {len(basis)}')

    reductor = StationaryRBReductor(fom, RB=basis, product=h1)

    def relative_error(rom, mu, u):
        return ((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]

    # Galerkin projection reproduces every snapshot once all modes are used.
    rom = reductor.reduce()
    log(f'snapshot reproduced (relative error below 1e-8): {relative_error(rom, training_set[7], snapshots[7]) < 1e-8}')

    test_set = parameter_space.sample_randomly(10)
    solutions = [fom.solve(mu) for mu in test_set]
    for n in (4, 8, 16, 32):
        rom = reductor.reduce(n)
        log(f'reduced dimension {n:2d}: largest relative error {max(relative_error(rom, mu, u) for mu, u in zip(test_set, solutions)):.1e}')


if __name__ == '__main__':
    run(gaussian_source.make_model(n_refinements=5, width=0.2))
