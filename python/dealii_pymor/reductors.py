"""pyMOR reductors for the models of dealii_pymor."""

from pymor.algorithms.projection import project
from pymor.reductors.basic import StationaryRBReductor


class DealIIStationaryRBReductor(StationaryRBReductor):
    """pyMOR's `StationaryRBReductor`, but efficient for an empirically interpolated right-hand side.

    pyMOR's right-hand side is an operator from R^1, which `StationaryRBReductor` projects with
    `source_basis=None`: "do not reduce the source". pyMOR's projection of an
    `EmpiricalInterpolatedOperator` needs a source basis, though. Without one it falls back to a
    generic projection, which evaluates the interpolant in the full space in every reduced solve.
    Projecting with the identity of R^1, the single vector [1], is the same and gives pyMOR's
    efficient `ProjectedEmpiricalInterpolatedOperator`.
    """

    def project_operators(self):
        fom = self.fom
        RB = self.bases['RB']
        return {
            'operator':          project(fom.operator, RB, RB),
            'rhs':               project(fom.rhs, RB, fom.rhs.source.ones()),
            'products':          {k: project(v, RB, RB) for k, v in fom.products.items()},
            'output_functional': project(fom.output_functional, None, RB),
        }
