"""pyMOR models of C++ models."""

import os

from pymor.core.base import ImmutableObject
from pymor.models.basic import StationaryModel

from dealii_pymor.operators import DealIIOperator, DealIIParametricOperator, DealIIRhsOperator


def stationary_model(model, name=None):
    """The pyMOR `StationaryModel` of a C++ `StationaryModel`."""
    if model.system_operator_is_parametric():
        operator = DealIIParametricOperator(model, name='operator')
    else:
        operator = DealIIOperator(model.system_operator({}), name='operator')
    return StationaryModel(
        operator,
        DealIIRhsOperator(model, name='rhs'),
        products={k: DealIIOperator(op, name=f'{k}_product') for k, op in model.products().items()},
        visualizer=DealIIVisualizer(model),
        name=name,
    )


class DealIIVisualizer(ImmutableObject):
    """Writes vector arrays with the `write_vtu` of a C++ model."""

    def __init__(self, model):
        self.__auto_init(locals())

    def visualize(self, U, filename, legend=None):
        """Write each vector of `U` as one field of `<filename>_0.pvtu` and return its path.

        Parameters
        ----------
        U
            The vector array.
        filename
            Path without extension, e.g. 'results/solution'.
        legend
            One field name per vector, `u_0`, `u_1`, ... by default.
        """
        if legend is None:
            legend = [f'u_{i}' for i in range(len(U))]
        elif isinstance(legend, str):
            legend = [legend]
        directory, basename = os.path.split(filename)
        return self.model.write_vtu(os.path.join(directory, ''), basename,
                                    [v.impl for v in U.vectors], list(legend))
