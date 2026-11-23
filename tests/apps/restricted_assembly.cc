// RestrictedAssembly against the full condensed assembly, with all kinds of homogeneous
// constraints: Q2 on the unit square, periodic in x, zero at y = 0 and 1, and hanging nodes around
// two refined cells, one of them at the periodic boundary. The cell matrix is the one of
// -Laplace + b.grad with b = (1, 0.5), which is not symmetric, so that rows and columns cannot be
// mixed up; the cell vector the one of f = 1 + x. At all unconstrained dofs (in reverse order) and
// at every 7th of them, C^T f and (C^T A C) u for an admissible u have to be the same up to
// round-off. In the mpirun=2 variant, the cells around a dof and its periodic partner belong to
// different ranks.

#include "../tests.h"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <implementations/restricted_assembly.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using VectorType = LinearAlgebra::distributed::Vector<double>;

// All entries on every rank.
std::vector<double>
gather(const VectorType & vector)
{
  std::vector<double> values(vector.size(), 0.);
  for(const auto i : vector.locally_owned_elements())
    values[i] = vector(i);
  Utilities::MPI::sum(values, MPI_COMM_WORLD, values);
  return values;
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();

  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1., true /*colorize*/);
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<2>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  triangulation.add_periodicity(periodic_faces);
  triangulation.refine_global(2);
  for(const auto & cell : triangulation.active_cell_iterators())
    if(cell->is_locally_owned() && (cell->center().distance(Point<2>(0.375, 0.375)) < 0.1 ||
                                    cell->center().distance(Point<2>(0.125, 0.625)) < 0.1))
      cell->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();

  const FE_Q<2>  fe(2);
  DoFHandler<2>  dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const IndexSet owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<double> constraints(owned_dofs, relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, 2, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, 3, constraints);
  constraints.close();

  FEValues<2> fe_values(fe,
                        QGauss<2>(fe.degree + 1),
                        update_values | update_gradients | update_quadrature_points |
                          update_JxW_values);
  const auto integrate = [&](const auto &         cell,
                             FullMatrix<double> & cell_matrix,
                             Vector<double> &     cell_rhs) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs    = 0;
    for(const unsigned int q : fe_values.quadrature_point_indices())
      for(const unsigned int i : fe_values.dof_indices())
      {
        cell_rhs(i) += (1. + fe_values.quadrature_point(q)[0]) * fe_values.shape_value(i, q) *
                       fe_values.JxW(q);
        for(const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) += (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
                                fe_values.shape_value(i, q) *
                                  (fe_values.shape_grad(j, q) * Tensor<1, 2>({1., 0.5}))) *
                               fe_values.JxW(q);
      }
  };

  // The full condensed system.
  DynamicSparsityPattern dsp(relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, MPI_COMM_WORLD, relevant_dofs);
  TrilinosWrappers::SparseMatrix matrix;
  matrix.reinit(owned_dofs, owned_dofs, dsp, MPI_COMM_WORLD);
  VectorType rhs(owned_dofs, relevant_dofs, MPI_COMM_WORLD);

  FullMatrix<double>                   cell_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
  Vector<double>                       cell_rhs(fe.n_dofs_per_cell());
  std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
  for(const auto & cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
  {
    integrate(cell, cell_matrix, cell_rhs);
    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, matrix, rhs);
  }
  matrix.compress(VectorOperation::add);
  rhs.compress(VectorOperation::add);

  VectorType u(owned_dofs, MPI_COMM_WORLD), Au(owned_dofs, MPI_COMM_WORLD);
  for(const auto i : owned_dofs)
    u(i) = std::sin(i + 1.);
  constraints.set_zero(u);
  matrix.vmult(Au, u);

  const std::vector<double> all_f = gather(rhs), all_u = gather(u), all_Au = gather(Au);

  // The constrained dofs, known on every rank.
  std::vector<double> constrained(dof_handler.n_dofs(), 0.);
  for(const auto i : owned_dofs)
    constrained[i] = constraints.is_constrained(i) ? 1. : 0.;
  Utilities::MPI::sum(constrained, MPI_COMM_WORLD, constrained);

  std::vector<types::global_dof_index> unconstrained_dofs;
  for(types::global_dof_index i = dof_handler.n_dofs(); i-- > 0;)
    if(constrained[i] == 0.)
      unconstrained_dofs.push_back(i);
  std::vector<types::global_dof_index> every_7th;
  for(std::size_t k = 0; k < unconstrained_dofs.size(); k += 7)
    every_7th.push_back(unconstrained_dofs[k]);
  deallog << "dofs: " << dof_handler.n_dofs() << ", unconstrained: " << unconstrained_dofs.size()
          << std::endl;

  for(const auto & [name, dofs] : {std::pair{"unconstrained_dofs", unconstrained_dofs},
                                    std::pair{"every_7th", every_7th}})
  {
    LogStream::Prefix           prefix(name);
    const RestrictedAssembly<2> assembly(dof_handler, constraints, dofs, MPI_COMM_WORLD);
    const auto &                source_dofs = assembly.get_source_dofs();

    FullMatrix<double>        unused_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
    Vector<double>            unused_vector(fe.n_dofs_per_cell());
    const std::vector<double> f = assembly.assemble_vector(
      [&](const auto & cell, Vector<double> & cell_f) { integrate(cell, unused_matrix, cell_f); });
    const std::vector<double> A =
      assembly.assemble_matrix([&](const auto & cell, FullMatrix<double> & cell_A) {
        integrate(cell, cell_A, unused_vector);
      });

    double f_difference = 0., Au_difference = 0.;
    for(std::size_t k = 0; k < dofs.size(); ++k)
    {
      double Au_k = 0.;
      for(std::size_t s = 0; s < source_dofs.size(); ++s)
        Au_k += A[k * source_dofs.size() + s] * all_u[source_dofs[s]];
      f_difference  = std::max(f_difference, std::abs(f[k] - all_f[dofs[k]]));
      Au_difference = std::max(Au_difference, std::abs(Au_k - all_Au[dofs[k]]));
    }
    const bool source_unconstrained =
      std::all_of(source_dofs.begin(), source_dofs.end(), [&](const auto j) {
        return constrained[j] == 0.;
      });

    // The number of source dofs of every 7th dof depends on the numbering, i.e. on the ranks.
    deallog << "rows: " << dofs.size()
            << ", source dofs unconstrained: " << (source_unconstrained ? "true" : "false")
            << ", same f: " << (f_difference < 1e-14 * rhs.linfty_norm() ? "true" : "false")
            << ", same A u: " << (Au_difference < 1e-12 * Au.linfty_norm() ? "true" : "false")
            << std::endl;
  }
}
