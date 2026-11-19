// PeriodicHillPotentialFlow, which solves on the reference box with the coefficients of the exact
// map T, against the textbook way: the plain Laplacian with the rhs -(grad v, e_x) on the same mesh
// mapped onto the channel by a MappingQCache that interpolates T with polynomials of degree p. Both
// are the same problem on the same dofs, only the geometry differs, so the difference of the
// solutions decreases with p like the interpolation error of T:
//  - hill width 2: the feet of the hill, where the profile is only C^2, are vertices of the mesh
//    (cells of width 1/8). Inside the cells, T is a polynomial of degree 5, so for p = 5 both
//    solutions agree up to round-off.
//  - hill width 1.93: the feet lie inside cells, the interpolation error of T stays O(h^3) for all
//    p >= 3, and so does the difference.
// A flat channel has the exact solution phi_tilde = 0 (the flow u = e_x). The mpirun=2 variant has
// to give the same.

#include "../tests.h"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <implementations/periodic_hill_potential_flow.h>
#include <implementations/trilinos_matrix_operator.h>

#include <memory>
#include <vector>

using Model      = PeriodicHillPotentialFlow<2>;
using VectorType = Model::VectorType;

std::shared_ptr<VectorType>
solve(const Model & model, const ParameterValues & mu)
{
  const auto rhs = model.assemble_rhs(mu);
  const auto u   = model.solution_space()->create_vector();
  model.system_operator(mu)->apply_inverse(*u, *rhs);
  return u;
}

// The Laplacian on the mesh of model mapped by a MappingQCache of the given degree.
std::shared_ptr<VectorType>
solve_on_mapped_mesh(const Model & model, const ParameterValues & mu, const unsigned int degree)
{
  const DoFHandler<2> &             dof_handler = model.get_dof_handler();
  const AffineConstraints<double> & constraints = model.get_constraints();
  const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  MappingQCache<2> mapping(degree);
  model.geometry(mu).initialize_mapping(mapping, dof_handler.get_triangulation());

  DynamicSparsityPattern dsp(relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             dof_handler.get_mpi_communicator(),
                                             relevant_dofs);
  auto matrix = std::make_shared<TrilinosWrappers::SparseMatrix>();
  matrix->reinit(dof_handler.locally_owned_dofs(),
                 dof_handler.locally_owned_dofs(),
                 dsp,
                 dof_handler.get_mpi_communicator());
  const auto rhs = model.solution_space()->create_vector();

  const FiniteElement<2> & fe = dof_handler.get_fe();
  FEValues<2>              fe_values(mapping,
                        fe,
                        QGauss<2>(fe.degree + 1),
                        update_gradients | update_JxW_values);
  FullMatrix<double>       cell_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  Vector<double>           cell_rhs(fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
  for(const auto & cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;
    for(const unsigned int q : fe_values.quadrature_point_indices())
      for(const unsigned int i : fe_values.dof_indices())
      {
        cell_rhs(i) -= fe_values.shape_grad(i, q)[0] * fe_values.JxW(q);
        for(const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
      }
    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, *matrix, *rhs);
  }
  matrix->compress(VectorOperation::add);
  rhs->compress(VectorOperation::add);

  const auto u = model.solution_space()->create_vector();
  TrilinosMatrixOperator(matrix, model.solution_space()).apply_inverse(*u, *rhs);
  return u;
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  // Without the CG iterations, which depend on the partitioning.
  deallog.depth_file(1);
  deallog.precision(3);

  const ParameterValues product_mu = {{"hill_height", {1.}}, {"hill_width", {3.}}};
  const Model           model(3, 2, 0., product_mu, MPI_COMM_WORLD);
  deallog << "dofs: " << model.solution_space()->n_dofs() << std::endl;

  const ParameterValues flat     = {{"hill_height", {0.}}, {"hill_width", {3.}}};
  const double          max_flat = solve(model, flat)->linfty_norm();
  deallog << "flat channel, phi_tilde = 0: " << (max_flat < 1e-12 ? "true" : "false") << std::endl;

  for(const double width : {1.93, 2.})
  {
    const ParameterValues mu = {{"hill_height", {1.}}, {"hill_width", {width}}};
    const auto            u  = solve(model, mu);
    for(const unsigned int degree : {1u, 2u, 3u, 5u})
    {
      VectorType difference = *solve_on_mapped_mesh(model, mu, degree);
      difference -= *u;
      deallog << "hill width " << width << ", mapping degree " << degree
              << ": relative difference " << difference.l2_norm() / u->l2_norm() << std::endl;
    }
  }
}
