# The VectorSpace bindings with a space created by an application module
# (example.make_uniform_space). Runs serially and on 2 ranks; the printed
# results must not depend on the number of ranks.

import numpy as np

import dealii_core_module as dealii
import example


def log(*args):
    if dealii.this_mpi_process() == 0:
        print(*args)


space = example.make_uniform_space(10)
log(f"space: {space!r}")
log(f"equal to itself: {space == space}")
log(f"equal to another space of the same dimension: {space == example.make_uniform_space(10)}")

# -9 and 9 tie for the largest magnitude; on 2 ranks they are owned by different ranks.
data = np.array([1.0, -2.0, -9.0, 4.0, 0.5, 3.0, 2.0, 7.0, 9.0, -1.0])
v = space.from_numpy(data)
log(f"to_numpy(from_numpy(data)) == data: {np.array_equal(space.to_numpy(v), data)}")
log(f"l2_norm matches numpy: {np.isclose(v.l2_norm(), np.linalg.norm(data))}")
log(f"dofs [0, 4, 9]: {space.dofs(v, [0, 4, 9])}")
log(f"amax: {space.amax(v)}")

zero = space.create_vector()
log(f"create_vector: {zero!r}, all_zero: {zero.all_zero()}")

try:
    space.from_numpy(np.zeros(3))
except ValueError as e:
    log(f"from_numpy with the wrong length: ValueError: {e}")

try:
    space.dofs(v, [10])
except IndexError as e:
    log(f"dofs out of range: IndexError: {e}")

try:
    space.to_numpy(example.make_uniform_space(11).create_vector())
except ValueError as e:
    log(f"vector of another space: ValueError: {e}")

# A serial space lives on MPI_COMM_SELF, so every rank has a complete copy.
serial = example.make_serial_space(4)
w = serial.from_numpy(np.array([3.0, -4.0, 0.0, 1.0]))
log(f"serial space: {serial!r}, vector: {w!r}")
log(f"serial to_numpy: {serial.to_numpy(w)}, amax: {serial.amax(w)}, dofs [1, 3]: {serial.dofs(w, [1, 3])}")
