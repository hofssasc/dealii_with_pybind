#ifndef APPLICATIONS_EXADG_POISSON_OPERATOR_H
#define APPLICATIONS_EXADG_POISSON_OPERATOR_H

#include <deal.II/lac/la_parallel_vector.h>

#include <exadg/poisson/spatial_discretization/operator.h>

#include <interfaces/operator.h>
#include <interfaces/vector_space.h>

#include <memory>

/**
 * The (scalar) Laplace operator of an ExaDG Poisson solver as an Operator. It is applied
 * matrix-free and inverted by the Krylov solver and preconditioner set in the solver's parameters.
 */
template<int dim>
class ExaDGPoissonOperator : public Operator<dealii::LinearAlgebra::distributed::Vector<double>>
{
public:
  using VectorType      = dealii::LinearAlgebra::distributed::Vector<double>;
  using PoissonOperator = ExaDG::Poisson::Operator<dim, 1, double>;

  // The space has to create its vectors with the partitioner of the solver's MatrixFree object.
  ExaDGPoissonOperator(std::shared_ptr<const PoissonOperator>         pde_operator,
                       std::shared_ptr<const VectorSpace<VectorType>> space);

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

private:
  std::shared_ptr<const PoissonOperator>         pde_operator;
  std::shared_ptr<const VectorSpace<VectorType>> space;
};

#endif /* APPLICATIONS_EXADG_POISSON_OPERATOR_H */
