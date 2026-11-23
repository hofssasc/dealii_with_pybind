#ifndef APPLICATIONS_PERIODIC_HILL_POTENTIAL_FLOW_H
#define APPLICATIONS_PERIODIC_HILL_POTENTIAL_FLOW_H

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <implementations/periodic_hill_geometry.h>
#include <implementations/restricted_assembly.h>
#include <implementations/trilinos_matrix_operator.h>
#include <interfaces/stationary_model.h>
#include <interfaces/vector_space.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * Potential flow through the periodic hill channel of PeriodicHillGeometry, with the parameters
 * "hill_height" and "hill_width" and a fixed stretching.
 *
 * The velocity is u = grad phi with
 *
 *   Laplace phi = 0,   u.n = 0 on the walls,   phi(x + 9) = phi(x) + 9,
 *
 * so that the integral of u along every path over one period is 9, e.g. u_x has the mean 1 on
 * horizontal lines above the crests. With phi = x + phi_tilde, the unknown phi_tilde is periodic
 * and solves
 *
 *   (grad v, grad phi_tilde) = -(grad v, e_x)   for all periodic v.
 *
 * Constants are removed by phi_tilde = 0 in the middle of the valley floor.
 *
 * The flow rate Q, the volume flux through every cross-section x = const (div u = 0, u.n = 0 on
 * the walls), depends on the geometry, and the flow is linear in it: for a given flow rate, scale
 * the velocity by it over Q. Averaged over one period, Q is the integral of u_x over the channel
 * over 9, and since (grad v, e_x) = -f(mu)_v,
 *
 *   Q(mu) = (|Omega(mu)| - f(mu)^T phi_tilde) / 9 = (|Omega(mu)| - phi_tilde^T A(mu) phi_tilde) / 9,
 *
 * see flow_rate(). It is linear in phi_tilde with the right-hand side as functional, a compliant
 * output, and it does not depend on the constant in phi_tilde, as the entries of f(mu) sum to zero.
 *
 * Everything is solved on one fixed mesh of the reference box, with continuous Q_k elements,
 * periodic in x (and z). The dofs are those of phi_tilde o T, the geometry only enters the
 * coefficients: with J = dT/dx_hat at the quadrature points,
 *
 *   A(mu)_ij = (grad_hat phi_i, K grad_hat phi_j),   f(mu)_i = -(grad_hat phi_i, K e_x),
 *
 * K = det(J) J^-1 J^-T, both evaluated exactly. A(mu) is assembled into a Trilinos matrix for every
 * mu and inverted by CG with algebraic multigrid. For empirical interpolation, both can also be
 * assembled at a few dofs only (PeriodicHillRestrictedRhs, PeriodicHillRestrictedOperator).
 *
 * The only product is "h1_semi", the H^1 seminorm on the channel of product_mu, i.e. A(product_mu):
 * one fixed inner product for all snapshots, close to the physical one for parameters near
 * product_mu.
 */
template<int dim>
class PeriodicHillPotentialFlow
  : public StationaryModel<dealii::LinearAlgebra::distributed::Vector<double>>
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  PeriodicHillPotentialFlow(const unsigned int      n_refinements,
                            const unsigned int      degree,
                            const double            stretching,
                            const ParameterValues & product_mu,
                            const MPI_Comm          communicator);

  std::shared_ptr<const VectorSpace<VectorType>>
  solution_space() const override;

  std::map<std::string, unsigned int>
  parameters() const override;

  std::shared_ptr<const Operator<VectorType>>
  system_operator(const ParameterValues & mu) const override;

  bool
  system_operator_is_parametric() const override;

  std::shared_ptr<VectorType>
  assemble_rhs(const ParameterValues & mu) const override;

  std::shared_ptr<const RestrictedRhs>
  restricted_rhs(const std::vector<dealii::types::global_dof_index> & dofs) const override;

  std::shared_ptr<const RestrictedOperator>
  restricted_system_operator(
    const std::vector<dealii::types::global_dof_index> & dofs) const override;

  std::map<std::string, std::shared_ptr<const Operator<VectorType>>>
  products() const override;

  // phi_tilde on the reference box.
  std::string
  write_vtu(const std::string &                                    directory,
            const std::string &                                    basename,
            const std::vector<std::shared_ptr<const VectorType>> & fields,
            const std::vector<std::string> &                       names) const override;

  // Solutions phi_tilde for mu on the channel of mu, written like write_vtu(). For each of flows,
  // the flow: <name>_phi (phi = x + phi_tilde), <name>_velocity (u = e_x + grad phi_tilde) and
  // <name>_pressure (p = -|u|^2 / 2, Bernoulli, zero at rest). For each of differences, e.g. the
  // error of a reduced solution: <name>_phi and <name>_velocity, the differences of the potentials
  // and velocities. The difference of the pressures is not linear in it, compute it from two flows.
  std::string
  write_flow_vtu(const std::string &                                    directory,
                 const std::string &                                    basename,
                 const ParameterValues &                                mu,
                 const std::vector<std::shared_ptr<const VectorType>> & flows,
                 const std::vector<std::string> &                       flow_names,
                 const std::vector<std::shared_ptr<const VectorType>> & differences,
                 const std::vector<std::string> &                       difference_names) const;

  // The flow rate Q(mu) of the solution phi_tilde for mu (see above). Collective.
  double
  flow_rate(const ParameterValues & mu, const VectorType & phi_tilde) const;

  PeriodicHillGeometry<dim>
  geometry(const ParameterValues & mu) const;

  const dealii::DoFHandler<dim> &
  get_dof_handler() const;

  const dealii::AffineConstraints<double> &
  get_constraints() const;

private:
  // A(mu) into matrix and/or f(mu) into rhs, either may be null.
  void
  assemble(const ParameterValues &                  mu,
           dealii::TrilinosWrappers::SparseMatrix * matrix,
           VectorType *                             rhs) const;

  MPI_Comm                                          mpi_communicator;
  double                                            stretching;
  dealii::parallel::distributed::Triangulation<dim> triangulation;
  dealii::FE_Q<dim>                                 fe;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::IndexSet                                  locally_relevant_dofs;
  dealii::AffineConstraints<double>                 constraints;
  dealii::DynamicSparsityPattern                    sparsity_pattern;

  std::shared_ptr<const ConstrainedVectorSpace<VectorType>> space;
  std::shared_ptr<const Operator<VectorType>>               h1_semi_product;
};

/**
 * The right-hand side of PeriodicHillPotentialFlow at a few dofs, assembled only on the cells
 * around them; RestrictedAssembly takes care of the periodicity.
 */
template<int dim>
class PeriodicHillRestrictedRhs : public RestrictedRhs
{
public:
  // Refers to model, which has to outlive this object.
  PeriodicHillRestrictedRhs(const PeriodicHillPotentialFlow<dim> &               model,
                            const std::vector<dealii::types::global_dof_index> & dofs);

  std::vector<double>
  evaluate(const ParameterValues & mu) const override;

private:
  const PeriodicHillPotentialFlow<dim> & model;
  RestrictedAssembly<dim>                assembly;
};

/**
 * The system operator of PeriodicHillPotentialFlow at a few dofs, like PeriodicHillRestrictedRhs.
 */
template<int dim>
class PeriodicHillRestrictedOperator : public RestrictedOperator
{
public:
  // Refers to model, which has to outlive this object.
  PeriodicHillRestrictedOperator(const PeriodicHillPotentialFlow<dim> &               model,
                                 const std::vector<dealii::types::global_dof_index> & dofs);

  std::vector<dealii::types::global_dof_index>
  source_dofs() const override;

  std::vector<double>
  assemble(const ParameterValues & mu) const override;

private:
  const PeriodicHillPotentialFlow<dim> & model;
  RestrictedAssembly<dim>                assembly;
};

#endif /* APPLICATIONS_PERIODIC_HILL_POTENTIAL_FLOW_H */
