#ifndef APPLICATIONS_GAUSSIAN_SOURCE_H
#define APPLICATIONS_GAUSSIAN_SOURCE_H

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <implementations/restricted_assembly.h>
#include <implementations/trilinos_matrix_operator.h>
#include <interfaces/stationary_model.h>
#include <interfaces/vector_space.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * Poisson's equation with a Gaussian source on the unit square,
 *
 *   -Laplace u = exp(-|x - center|^2 / width^2)   in (0, 1)^dim,
 *            u = 0                                on the boundary,
 *
 * with the parameter "center" in (0, 1)^dim. Continuous Q_k elements, matrix-based: the Laplacian
 * is assembled into a Trilinos matrix and inverted by CG with algebraic multigrid. The mesh is
 * distributed over the ranks of the given communicator.
 *
 * The only product is "h1_0_semi", the Laplacian itself.
 */
template<int dim>
class GaussianSource : public StationaryModel<dealii::LinearAlgebra::distributed::Vector<double>>
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  GaussianSource(const unsigned int n_refinements,
                 const unsigned int degree,
                 const double       width,
                 const MPI_Comm     communicator);

  std::shared_ptr<const VectorSpace<VectorType>>
  solution_space() const override;

  std::map<std::string, unsigned int>
  parameters() const override;

  std::shared_ptr<const Operator<VectorType>>
  system_operator(const ParameterValues & mu) const override;

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
  std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix>
  assemble_laplace_matrix() const;

  MPI_Comm                                          mpi_communicator;
  double                                            width;
  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::FE_Q<dim>                                 fe;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::IndexSet                                  locally_relevant_dofs;
  dealii::AffineConstraints<double>                 constraints;

  std::shared_ptr<const ConstrainedVectorSpace<VectorType>> space;
  std::shared_ptr<const TrilinosMatrixOperator>             laplace_operator;
};

/**
 * The right-hand side of GaussianSource at a few dofs: the source is only integrated over the
 * cells around them, see RestrictedAssembly. GaussianSourceMatrixFree has the same right-hand side
 * and uses it as well. The boundary dofs are the constrained ones, f(mu) is zero there, so
 * empirical interpolation never chooses them.
 */
template<int dim>
class GaussianSourceRestrictedRhs : public RestrictedRhs
{
public:
  // Refers to the cells of dof_handler, which has to outlive this object.
  GaussianSourceRestrictedRhs(const dealii::DoFHandler<dim> &                      dof_handler,
                              const dealii::AffineConstraints<double> &            constraints,
                              const double                                         width,
                              const std::vector<dealii::types::global_dof_index> & dofs,
                              const MPI_Comm                                       communicator);

  std::vector<double>
  evaluate(const ParameterValues & mu) const override;

private:
  const dealii::FiniteElement<dim> & fe;
  double                             width;
  RestrictedAssembly<dim>            assembly;
};

#endif /* APPLICATIONS_GAUSSIAN_SOURCE_H */
