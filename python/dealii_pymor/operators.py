"""pyMOR operators of C++ `Operators` and `StationaryModels`."""

from pymor.operators.interface import Operator
from pymor.operators.list import ListVectorArrayOperatorBase
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
