#include <deal.II/base/logstream.h>

#include <implementations/exadg_poisson_operator.h>

#include <utility>

template<int dim>
ExaDGPoissonOperator<dim>::ExaDGPoissonOperator(
  std::shared_ptr<const PoissonOperator>         pde_operator,
  std::shared_ptr<const VectorSpace<VectorType>> space)
  : pde_operator(std::move(pde_operator)), space(std::move(space))
{
}

template<int dim>
std::shared_ptr<const VectorSpace<typename ExaDGPoissonOperator<dim>::VectorType>>
ExaDGPoissonOperator<dim>::source() const
{
  return space;
}

template<int dim>
std::shared_ptr<const VectorSpace<typename ExaDGPoissonOperator<dim>::VectorType>>
ExaDGPoissonOperator<dim>::range() const
{
  return space;
}

template<int dim>
void
ExaDGPoissonOperator<dim>::apply(VectorType & dst, const VectorType & src) const
{
  pde_operator->vmult(dst, src);
}

template<int dim>
bool
ExaDGPoissonOperator<dim>::has_inverse() const
{
  return true;
}

template<int dim>
void
ExaDGPoissonOperator<dim>::apply_inverse(VectorType & dst, const VectorType & rhs) const
{
  const unsigned int n_iterations = pde_operator->solve(dst, rhs, 0.0 /*time*/);

  // Logs the number of iterations to deallog like TrilinosMatrixOperator;
  // prints nothing unless a stream is attached (see tests).
  dealii::deallog << "Convergence step " << n_iterations << std::endl;
}

template class ExaDGPoissonOperator<2>;
