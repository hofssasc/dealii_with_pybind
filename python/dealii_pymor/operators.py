"""pyMOR operators of C++ `Operators` and `StationaryModels`."""

import numpy as np
from pymor.operators.constructions import ConstantOperator, LincombOperator, VectorFunctional
from pymor.operators.interface import Operator
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.solvers.list import ListVectorArrayBasedSolver
from pymor.vectorarrays.numpy import NumpyVectorSpace

from dealii_pymor.vectorarray import DealIIVectorSpace


class DealIISolver(ListVectorArrayBasedSolver):
    """Solves with the `apply_inverse` of the C++ operator, and with its `apply_inverse_adjoint` if
    it has an adjoint."""

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        u = operator.source.zero_vector() if initial_guess is None else initial_guess.copy(deep=True)
        operator.impl.apply_inverse(u.impl, v.impl)
        return u

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        if not operator.impl.has_adjoint():
            raise NotImplementedError
        v = operator.range.zero_vector() if initial_guess is None else initial_guess.copy(deep=True)
        operator.impl.apply_inverse_adjoint(v.impl, u.impl)
        return v


class DealIIOperator(ListVectorArrayOperatorBase):
    """A C++ `Operator`; inverted by its own `apply_inverse` if it has one, and with its own
    `apply_adjoint` and `apply_inverse_adjoint` if it has an adjoint."""

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

    def _apply_adjoint_one_vector(self, v, mu=None):
        if not self.impl.has_adjoint():
            raise NotImplementedError
        u = self.source.zero_vector()
        self.impl.apply_adjoint(u.impl, v.impl)
        return u


class DealIIParametricOperator(Operator):
    """The system operator A(μ) of a C++ `StationaryModel` that depends on μ.

    For every μ, the C++ model assembles a new operator, which `assemble` returns as a
    `DealIIOperator`; pyMOR's default solver inverts it with its own `apply_inverse`, and
    `apply_inverse_adjoint` likewise.
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

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.assemble(mu).apply_adjoint(V)

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


class DealIIFunctional(Operator):
    """A C++ `Functional` J(μ; u), a quantity of interest of the solutions of a C++
    `StationaryModel`, as an operator from its solution space to R^1, e.g. as output functional.

    Not linear in general. Its `jacobian` at u is the linear functional v ↦ dJ/du(μ; u) v, whose
    `as_source_array` is the vector dJ/du(μ; u), so that the adjoint problem of the model for J is
    `fom.operator.apply_inverse_adjoint(-J.jacobian(u, mu).as_source_array(mu), mu=mu)`.

    Parameters
    ----------
    impl
        The C++ `Functional`.
    model
        The C++ `StationaryModel` it belongs to, for its solution space and parameters.
    """

    linear = False

    def __init__(self, impl, model, name=None):
        self.__auto_init(locals())
        self.source = DealIIVectorSpace(model.solution_space())
        self.range = NumpyVectorSpace(1)
        self.parameters_own = model.parameters()

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        return self.range.make_array(np.array([[self.impl.value(mu, u.impl) for u in U.vectors]]))

    def jacobian(self, U, mu=None):
        assert U in self.source and len(U) == 1
        assert self.parameters.assert_compatible(mu)
        gradient = self.source.make_array([self.impl.gradient(mu, U.vectors[0].impl)])
        return VectorFunctional(gradient, name=f'{self.name}_jacobian')


class RhsFunctional(Operator):
    """u ↦ f(μ)ᵀu for a right-hand side f(μ), an operator from R^1 like `DealIIRhsOperator`, on
    the full or on a reduced space.

    `DealIIStationaryRBReductor` projects it by projecting `rhs` like the right-hand side of the
    model, so that an interpolated right-hand side stays efficient here as well; pyMOR's generic
    projection would evaluate f(μ) in the full space for every μ.
    """

    linear = True

    def __init__(self, rhs, name=None):
        assert rhs.linear and rhs.source.dim == 1
        self.__auto_init(locals())
        self.source = rhs.range
        self.range = NumpyVectorSpace(1)

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(self.rhs.as_range_array(mu).inner(U))

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.rhs.as_range_array(mu).lincomb(V.to_numpy())


def compliant_output(rhs, factor=1., offset=None, name='output'):
    """The output functional offset(μ) + factor(μ) f(μ)ᵀu of the right-hand side f(μ) `rhs`.

    For a symmetric coercive system operator, the error of this output for a Galerkin reduced
    solution is quadratic in the error of the solution. `factor` and `offset` are numbers or
    `ParameterFunctionals`, without `offset` the output is linear.
    """
    operators, coefficients = [RhsFunctional(rhs)], [factor]
    if offset is not None:
        operators.append(ConstantOperator(NumpyVectorSpace(1).ones(), rhs.range))
        coefficients.append(offset)
    return LincombOperator(operators, coefficients, name=name)
