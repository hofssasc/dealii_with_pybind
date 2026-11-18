#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <implementations/trilinos_matrix_operator.h>

#include <utility>

TrilinosMatrixOperator::TrilinosMatrixOperator(
  std::shared_ptr<const dealii::TrilinosWrappers::SparseMatrix> matrix,
  std::shared_ptr<const VectorSpace<VectorType>>                 space,
  const double                                                   relative_tolerance)
  : matrix(std::move(matrix)), space(std::move(space)), relative_tolerance(relative_tolerance)
{
  preconditioner.initialize(*this->matrix);
}

std::shared_ptr<const VectorSpace<TrilinosMatrixOperator::VectorType>>
TrilinosMatrixOperator::source() const
{
  return space;
}

std::shared_ptr<const VectorSpace<TrilinosMatrixOperator::VectorType>>
TrilinosMatrixOperator::range() const
{
  return space;
}

void
TrilinosMatrixOperator::apply(VectorType & dst, const VectorType & src) const
{
  matrix->vmult(dst, src);
}

bool
TrilinosMatrixOperator::has_inverse() const
{
  return true;
}

void
TrilinosMatrixOperator::apply_inverse(VectorType & dst, const VectorType & rhs) const
{
  // Logs the number of iterations to deallog, which prints nothing unless a stream is attached,
  // as in the tests.
  dealii::SolverControl        control(matrix->m(),
                                       relative_tolerance * rhs.l2_norm(),
                                       /*log_history*/ false,
                                       /*log_result*/ true);
  dealii::SolverCG<VectorType> cg(control);
  cg.solve(*matrix, dst, rhs, preconditioner);
}
