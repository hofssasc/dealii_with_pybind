# The Gaussian source model (apps/include/implementations/gaussian_source.h) in pyMOR: a
# full-order solve and a POD reduced basis from snapshots. Runs serially and on 2 ranks (SPMD).
# gaussian_source_matrix_free_rb.py runs the same for the matrix-free model.

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.algorithms.ei import ei_greedy
from pymor.operators.ei import EmpiricalInterpolatedOperator

import dealii_core_module as dealii
import gaussian_source


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


# Every rank logs, in no particular order. The only warning is the expected one about projecting
# the non-affine right-hand side generically.
set_log_levels({'pymor': 'ERROR', 'dealii_pymor': 'ERROR'})

def run(model, filename):
    # Wrap the C++ model from gaussian_source.make_model() (or make_matrix_free_model()) in a pyMOR
    # StationaryModel.
    # This allows to use cpp functions (e.g system_operator(), parameters(), solution_space(),
    # products(), assemble_rhs() , ...) through the python interfaced defined by pyMORs StationaryModel.
    fom = dealii_pymor.stationary_model(model, name='gaussian_source')
    h1 = fom.h1_0_semi_product
    log(f'parameters: {fom.parameters}, dimension: {fom.solution_space.dim}')

    mu = fom.parameters.parse([0.3, 0.6])
    U = fom.solve(mu)
    F = fom.rhs.as_range_array(mu)
    log(f'relative residual below 1e-9: {((fom.operator.apply(U) - F).norm() / F.norm())[0] < 1e-9}')
    log(f'h1_0_semi norm of the solution: {fom.h1_0_semi_norm(U)[0]:.6e}')
    log(f'visualized: {fom.visualize(U, filename=f"output/{filename}", legend="u")}')


    # ------------------Reduction of the Poisson Operator ---------------------------
    # Define parameter space such that source doesnt leave the domain and uniformly sample a grid of 6x6.
    parameter_space = fom.parameters.space(0.2, 0.8)
    training_set = parameter_space.sample_uniformly(6)
    snapshots = fom.solution_space.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    # Minimize reconstruction error in the H1 norm (alternative could be L2)
    basis, _ = pod(snapshots, product=h1, rtol=0, atol=0, l2_err=0)
    log(f'snapshots: {len(snapshots)}, POD modes: {len(basis)}')

    log(f'visualized: {fom.visualize(basis, filename=f"output/{filename}_modes", legend=[f"mode_{i}" for i in range(1, len(basis) + 1)])}')



    # ------------------Emiprical Interpolation of the Right-Hand Side ---------------------------
    # The right-hand side is non-affine, so we use empirical interpolation to approximate it
    # with a reduced basis and interpolation dofs. The reduced basis is obtained from snapshots of
    # the right-hand side on a finer grid of centers, since assembling the rhs is cheaper.
    rhs_training_set = parameter_space.sample_uniformly(12)
    rhs_snapshots = fom.rhs.range.empty()
    for mu in rhs_training_set:
        rhs_snapshots.append(fom.rhs.as_range_array(mu))
    dofs, collateral_basis, _ = ei_greedy(rhs_snapshots, max_interpolation_dofs=64)
    log(f'right-hand side snapshots: {len(rhs_snapshots)}, interpolation dofs: {len(dofs)}')

    def relative_error(rom, mu, u):
        return ((u - reductor.reconstruct(rom.solve(mu))).norm(h1) / u.norm(h1))[0]



    # ------------------Reduced Basis and Empirical Interpolation Error Estimation ---------------------------
    test_set = parameter_space.sample_randomly(10)
    solutions = [fom.solve(mu) for mu in test_set]
    rhs_test = [fom.rhs.as_range_array(mu) for mu in test_set]

    # No need to recompute dofs and collateral basis, they are hierarchically ordered and the first M are the interpolation with M dofs.
    for M in (16, 32, 64):
        rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, dofs[:M], collateral_basis[:M], triangular=True)
        for n in (4, 8, 16, 32):
            # Note that we use a custom reductor instead of the standard StationaryRBReductor,
            # since we need to project the non-affine right-hand side.
            # Generally for a StationaryRBReductor, the fom is needed to project via the apply function, simply said
            # something like basis_i.inner(fom.apply(basis_j)) has to be done for each reduced matrix entry ij.
            # The h1 norm is also projected down for use in reconstruct and error estimation.
            reductor = dealii_pymor.DealIIStationaryRBReductor(fom.with_(rhs=rhs_ei), RB=basis, product=h1)
            rom = reductor.reduce(n)

            rhs_error = max(((f - rhs_ei.as_range_array(mu)).norm() / f.norm())[0] for mu, f in zip(test_set, rhs_test))
            log(f'interpolation dofs {M:2d}: largest relative error of the right-hand side {rhs_error:.1e}, '
                f'reduced dimension {n:2d}: largest relative error {max(relative_error(rom, mu, u) for mu, u in zip(test_set, solutions)):.1e}')


if __name__ == '__main__':
    run(gaussian_source.make_model(
        n_refinements=6, degree=2, width=0.2),
        filename='gaussian_source_rb')
    run(gaussian_source.make_matrix_free_model(
        n_refinements=6, degree=2, width=0.2),
        filename='gaussian_source_matrix_free_rb')
