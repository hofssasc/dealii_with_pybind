"""pyMOR operators of C++ `Operators` and `StationaryModels`."""

import numpy as np
from pymor.operators.interface import Operator
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.solvers.list import ListVectorArrayBasedSolver
from pymor.vectorarrays.numpy import NumpyVectorSpace

from dealii_pymor.vectorarray import DealIIVectorSpace


class DealIISolver(ListVectorArrayBasedSolver):
    """Solves with the `apply_inverse` of the C++ operator."""

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        u = operator.source.zero_vector() if initial_guess is None else initial_guess.copy(deep=True)
        operator.impl.apply_inverse(u.impl, v.impl)
        return u


class DealIIOperator(ListVectorArrayOperatorBase):
    """A C++ `Operator`; inverted by its own `apply_inverse` if it has one."""

    linear = True

    def __init__(self, impl, solver=None, name=None):
        if solver is None and impl.has_inverse():
            solver = DealIISolver()
        self.__auto_init(locals())
        self.source = DealIIVectorSpace(impl.source())
        self.range = DealIIVectorSpace(impl.range())

    def _apply_one_vector(self, u, mu=None):
        v = self.range.zero_vector()
        self.impl.apply(v.impl, u.impl)
        return v


class DealIIParametricOperator(Operator):
    """The system operator A(μ) of a C++ `StationaryModel` that depends on μ.

    For every μ, the C++ model assembles a new operator, which `assemble` returns as a
    `DealIIOperator`; pyMOR's default solver inverts it with its own `apply_inverse`.
    """

    linear = True

    def __init__(self, model, name=None):
        self.__auto_init(locals())
        self.source = self.range = DealIIVectorSpace(model.solution_space())
        self.parameters_own = model.parameters()

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        return DealIIOperator(self.model.system_operator(mu), name=f'{self.name}_assembled')

    def apply(self, U, mu=None):
        assert U in self.source
        return self.assemble(mu).apply(U)

    def restricted(self, dofs):
        """For pyMOR's `EmpiricalInterpolatedOperator`: A(μ) only at `dofs`, see
        `restricted_system_operator` of the C++ `StationaryModel`."""
        impl = self.model.restricted_system_operator(dofs)
        if impl is None:
            raise NotImplementedError
        source_dofs = np.array(impl.source_dofs(), dtype=np.int64)
        return DealIIRestrictedOperator(impl, len(dofs), len(source_dofs), self.parameters), source_dofs


class DealIIRestrictedOperator(Operator):
    """`DealIIParametricOperator` restricted to a few dofs, see `DealIIParametricOperator.restricted`:
    the small matrix of A(μ) at these dofs and the source dofs, assembled by C++ for every μ."""

    linear = True

    def __init__(self, impl, n_dofs, n_source_dofs, parameters_own, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(n_source_dofs)
        self.range = NumpyVectorSpace(n_dofs)

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        matrix = np.array(self.impl.assemble(mu)).reshape(self.n_dofs, self.n_source_dofs)
        return NumpyMatrixOperator(matrix, name=f'{self.name}_assembled')

    def apply(self, U, mu=None):
        assert U in self.source
        return self.assemble(mu).apply(U)


class DealIIRhsOperator(Operator):
    """The right-hand side f(μ) of a C++ `StationaryModel`, as a vector-like operator."""

    linear = True

    def __init__(self, model, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(1)
        self.range = DealIIVectorSpace(model.solution_space())
        self.parameters_own = model.parameters()

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        F = self.range.make_array([self.model.assemble_rhs(mu)])
        return F.lincomb(U.to_numpy())

    def restricted(self, dofs):
        """For pyMOR's `EmpiricalInterpolatedOperator`: f(μ) only at `dofs`, see `restricted_rhs`
        of the C++ `StationaryModel`. The source R^1 is kept, as its only dof 0."""
        impl = self.model.restricted_rhs(dofs)
        if impl is None:
            raise NotImplementedError
        return DealIIRestrictedRhsOperator(impl, len(dofs), self.parameters), np.array([0])


class DealIIRestrictedRhsOperator(Operator):
    """`DealIIRhsOperator` restricted to a few dofs, see `DealIIRhsOperator.restricted`."""

    linear = True

    def __init__(self, impl, n_dofs, parameters_own, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(1)
        self.range = NumpyVectorSpace(n_dofs)

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        F = self.range.make_array(np.array(self.impl.evaluate(mu))[:, np.newaxis])
        return F.lincomb(U.to_numpy())
