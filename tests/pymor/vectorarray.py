# pyMOR's VectorArray interface on deal.II vectors, checked against pyMOR's
# own NumPy implementation. Runs serially and on 2 ranks (SPMD).

import dealii_pymor  # before pymor, see dealii_pymor/__init__.py
import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import set_log_levels
from pymor.tools.random import new_rng
from pymor.vectorarrays.numpy import NumpyVectorSpace

import dealii_core_module as dealii
import example


def check(name, value, expected):
    ok = np.allclose(value, expected, rtol=1e-12, atol=1e-12)
    if dealii.this_mpi_process() == 0:
        print(f"{name}: {'ok' if ok else f'FAILED, {value} != {expected}'}")


# every rank logs, in no particular order
set_log_levels({'pymor': 'WARN'})

space = dealii_pymor.DealIIVectorSpace(example.make_uniform_space(10))
reference = NumpyVectorSpace(10)

check('equal to a space wrapping the same C++ space', space == dealii_pymor.DealIIVectorSpace(space.impl), True)
check('equal to a space of the same dimension', space == dealii_pymor.DealIIVectorSpace(example.make_uniform_space(10)),
      False)

data = np.random.default_rng(0).normal(size=(10, 3))
U, V = space.from_numpy(data), reference.from_numpy(data)
check('to_numpy', U.to_numpy(), data)
check('inner', U.inner(U), V.inner(V))
check('pairwise_inner', U.pairwise_inner(U[::-1]), V.pairwise_inner(V[::-1]))
check('norm', U.norm(), V.norm())
check('norm2', U.norm2(), V.norm2())
check('sup_norm', U.sup_norm(), V.sup_norm())
check('dofs', U.dofs([0, 5, 9]), V.dofs([0, 5, 9]))
check('amax', np.array(U.amax()), np.array(V.amax()))

coefficients = np.array([[1.0, -2.0], [0.5, 0.0], [3.0, 1.0]])
check('lincomb', U.lincomb(coefficients).to_numpy(), V.lincomb(coefficients).to_numpy())

W = U.copy()
W.axpy(2.0, U[0])
W.scal(-0.5)
check('scal, axpy', W.to_numpy(), -0.5 * (data + 2.0 * data[:, [0]]))
check('copy is not changed by its original', U.to_numpy(), data)

C = U * (1 + 2j)
check('complex inner', C.inner(U), (V * (1 + 2j)).inner(V))
check('complex amax', np.array(C.amax()), np.array((V * (1 + 2j)).amax()))

Q = gram_schmidt(U)
check('gram_schmidt', Q.to_numpy(), gram_schmidt(V).to_numpy())
check('gram_schmidt gramian', Q.gramian(), np.eye(3))

with new_rng(0):
    R = space.random(2)
with new_rng(0):
    check('random', R.to_numpy(), reference.random(2).to_numpy())
check('zeros', space.zeros(2).norm(), [0.0, 0.0])
check('full', space.full(3.0).to_numpy(), np.full((10, 1), 3.0))
