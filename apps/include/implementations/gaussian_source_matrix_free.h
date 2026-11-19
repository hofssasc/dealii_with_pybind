#ifndef APPLICATIONS_GAUSSIAN_SOURCE_MATRIX_FREE_H
#define APPLICATIONS_GAUSSIAN_SOURCE_MATRIX_FREE_H

#include <deal.II/base/point.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/parameters.h>

#include <implementations/exadg_poisson_operator.h>
#include <implementations/gaussian_source.h>
#include <interfaces/stationary_model.h>
#include <interfaces/vector_space.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * The problem of GaussianSource,
 *
 *   -Laplace u = exp(-|x - center|^2 / width^2)   in (0, 1)^dim,
 *            u = 0                                on the boundary,
 *
 * with ExaDG's matrix-free Poisson solver instead of an assembled matrix: CG preconditioned by
 * polynomial and geometric multigrid. The discretization is the same (continuous Q_k elements on
 * the same mesh), so both models have the same discrete solution up to the solver tolerance.
 *
 * The only product is "h1_0_semi", the Laplacian itself. The restricted right-hand side is the one
 * of GaussianSource (GaussianSourceRestrictedRhs), as ExaDG has no restricted assembly.
 */
template<int dim>
class GaussianSourceMatrixFree
  : public StationaryModel<dealii::LinearAlgebra::distributed::Vector<double>>
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  GaussianSourceMatrixFree(const unsigned int n_refinements,
                           const unsigned int degree,
                           const double       width,
                           const MPI_Comm     communicator);

  std::shared_ptr<const VectorSpace<VectorType>>
  solution_space() const override;

  std::map<std::string, unsigned int>
  parameters() const override;

  std::shared_ptr<const Operator<VectorType>>
  system_operator() const override;

  std::shared_ptr<VectorType>
  assemble_rhs(const ParameterValues & mu) const override;

  std::shared_ptr<const RestrictedRhs>
  restricted_rhs(const std::vector<dealii::types::global_dof_index> & dofs) const override;

  std::map<std::string, std::shared_ptr<const Operator<VectorType>>>
  products() const override;

  std::string
  write_vtu(const std::string &                                    directory,
            const std::string &                                    basename,
            const std::vector<std::shared_ptr<const VectorType>> & fields,
            const std::vector<std::string> &                       names) const override;

private:
  MPI_Comm mpi_communicator;
  double   width;

  // ExaDG's operator keeps a reference to its parameters.
  ExaDG::Poisson::Parameters param;

  // Shared with the right-hand side function of the solver, set by assemble_rhs().
  std::shared_ptr<dealii::Point<dim>> center;

  std::shared_ptr<ExaDG::Poisson::Operator<dim, 1, double>> pde_operator;
  dealii::AffineConstraints<double>                         constraints;

  std::shared_ptr<const ConstrainedVectorSpace<VectorType>> space;
  std::shared_ptr<const ExaDGPoissonOperator<dim>>          laplace_operator;
};

#endif /* APPLICATIONS_GAUSSIAN_SOURCE_MATRIX_FREE_H */
