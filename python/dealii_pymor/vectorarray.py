"""pyMOR vector arrays of deal.II vectors.

Both classes work for every vector type bound in dealii_core_module: they only
use the methods `bind_vector` and `bind_vector_space` define for all of them.
Only real vectors are supported.
"""

from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


class DealIIVector(CopyOnWriteVector):
    """A deal.II vector, together with the C++ space it belongs to.

    The space is needed for everything that collects entries over the ranks:
    `to_numpy`, `dofs` and `amax`.
    """

    def __init__(self, impl, space_impl):
        self.impl = impl
        self.space_impl = space_impl

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl, instance.space_impl)

    def _copy_data(self):
        self.impl = self.impl.copy()

    def _scal(self, alpha):
        self.impl.scal(alpha)

    def _axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def inner(self, other):
        return self.impl.inner(other.impl)

    def norm(self):
        return self.impl.l2_norm()

    def norm2(self):
        return self.impl.norm_sqr()

    def sup_norm(self):
        return self.impl.linfty_norm()

    def dofs(self, dof_indices):
        return self.space_impl.dofs(self.impl, dof_indices)

    def amax(self):
        return self.space_impl.amax(self.impl)

    def to_numpy(self, ensure_copy=False):
        # gathered from all ranks, so always a new array
        return self.space_impl.to_numpy(self.impl)


class DealIIVectorSpace(ListVectorSpace):
    """pyMOR vector space of the vectors of a C++ `VectorSpace`.

    Parameters
    ----------
    impl
        The C++ space, e.g. a `dealii_core_module.DistributedVectorSpace`
        returned by an application module.
    id
        See :attr:`~pymor.vectorarrays.interface.VectorSpace.id`.
    """

    vector_type = DealIIVector

    def __init__(self, impl, id=None):
        self.__auto_init(locals())

    @property
    def dim(self):
        return self.impl.dim

    def __eq__(self, other):
        return type(other) is type(self) and self.impl == other.impl and self.id == other.id

    def __hash__(self):
        return hash((self.impl, self.id))

    def zero_vector(self):
        return DealIIVector(self.impl.create_vector(), self.impl)

    def make_vector(self, obj):
        return DealIIVector(obj, self.impl)

    def vector_from_numpy(self, data, ensure_copy=False):
        # always copies; random and full vectors are built through here as well
        impl = self.impl.from_numpy(data)
        self.impl.make_admissible(impl)
        return DealIIVector(impl, self.impl)
