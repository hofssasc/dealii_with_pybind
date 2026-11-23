# The Gaussian source model (apps/include/implementations/gaussian_source.h) in pyMOR.
#  - solve: a full-order solve.
#  - snapshot: a POD reduced basis of all snapshots reproduces each of them.
#  - reduced_basis: the errors of the reduced solutions for growing reduced dimensions.
# Runs serially and on 2 ranks (SPMD). gaussian_source_matrix_free_rb.py runs the same for the
# matrix-free model.

import common  # first, see common.py
from common import log, run
import dealii_pymor
from pymor.algorithms.pod import pod
from pymor.reductors.basic import StationaryRBReductor

import gaussian_source


def run_tests(model):
    # The C++ model from gaussian_source.make_model() (or make_matrix_free_model()) as a pyMOR
    # StationaryModel.
    fom = dealii_pymor.stationary_model(model, name='gaussian_source')
    h1 = fom.h1_0_semi_product
    parameter_space = fom.parameters.space(0.2, 0.8)

    training_set = parameter_space.sample_uniformly(6)
    snapshots = fom.solution_space.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))
    basis, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    reductor = StationaryRBReductor(fom, RB=basis, product=h1)

    def relative_error(rom, mu, u):
        return ((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]

    def test_solve():
        log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}')
        mu = fom.parameters.parse([0.3, 0.6])
        U = fom.solve(mu)
        F = fom.rhs.as_range_array(mu)
        log(f'relative residual below 1e-9: {((fom.operator.apply(U) - F).norm() / F.norm())[0] < 1e-9}')
        log(f'h1_0_semi norm of the solution: {fom.h1_0_semi_norm(U)[0]:.6e}')
        log(f'visualized: {fom.visualize(U, filename="solution", legend="u")}')

    def test_snapshot():
        # Galerkin projection reproduces every snapshot once all modes are used.
        log(f'snapshots: {len(snapshots)}, POD modes: {len(basis)}')
        log(f'snapshot reproduced (relative error below 1e-8): '
            f'{relative_error(reductor.reduce(), training_set[7], snapshots[7]) < 1e-8}')

    def test_reduced_basis():
        test_set = parameter_space.sample_randomly(10)
        solutions = [fom.solve(mu) for mu in test_set]
        for n in (4, 8, 16, 32):
            rom = reductor.reduce(n)
            error = max(relative_error(rom, mu, u) for mu, u in zip(test_set, solutions))
            log(f'reduced dimension {n:2d}: largest relative error {error:.1e}')

    run(test_solve, test_snapshot, test_reduced_basis)


if __name__ == '__main__':
    run_tests(gaussian_source.make_model(n_refinements=5, width=0.2))
