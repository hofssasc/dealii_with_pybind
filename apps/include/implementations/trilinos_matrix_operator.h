#ifndef APPLICATIONS_TRILINOS_MATRIX_OPERATOR_H
#define APPLICATIONS_TRILINOS_MATRIX_OPERATOR_H

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <interfaces/operator.h>
#include <interfaces/vector_space.h>

#include <memory>

/**
 * A symmetric positive definite Trilinos matrix as an Operator, inverted by CG with an algebraic
 * multigrid preconditioner. Being symmetric, it is its own adjoint.
 */
class TrilinosMatrixOperator : public Operator<dealii::LinearAlgebra::distributed::Vector<double>>
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  // The preconditioner is set up here, once.
  TrilinosMatrixOperator(std::shared_ptr<const dealii::TrilinosWrappers::SparseMatrix> matrix,
                         std::shared_ptr<const VectorSpace<VectorType>>                 space,
                         const double relative_tolerance = 1e-10);

  std::shared_ptr<const VectorSpace<VectorType>>
  source() const override;

  std::shared_ptr<const VectorSpace<VectorType>>
  range() const override;

  void
  apply(VectorType & dst, const VectorType & src) const override;

  bool
  has_inverse() const override;

  void
  apply_inverse(VectorType & dst, const VectorType & rhs) const override;

  bool
  has_adjoint() const override;

  // apply(), as the matrix is symmetric.
  void
  apply_adjoint(VectorType & dst, const VectorType & src) const override;

  // apply_inverse(), as the matrix is symmetric.
  void
  apply_inverse_adjoint(VectorType & dst, const VectorType & rhs) const override;

private:
  std::shared_ptr<const dealii::TrilinosWrappers::SparseMatrix> matrix;
  std::shared_ptr<const VectorSpace<VectorType>>                 space;
  dealii::TrilinosWrappers::PreconditionAMG                      preconditioner;
  double                                                         relative_tolerance;
};

#endif /* APPLICATIONS_TRILINOS_MATRIX_OPERATOR_H */
