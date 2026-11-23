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
// A flat channel has the exact solution phi_tilde = 0 (the flow u = e_x) and the flow rate 3.036.
// With a hill, flow_rate() (the mean of u_x over the channel times its height) against the flux of
// the discrete velocity through the cross-sections at the crest and in the valley, which is only
// conserved up to the discretization error, and against ||grad phi||^2 / 9 (Galerkin orthogonality).
// The mpirun=2 variant has to give the same.

#include "../tests.h"

#include <deal.II/base/mpi.h>
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

#include <cmath>
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

// The integral of u_x = 1 + d phi_tilde / dx over the cross-section at x_hat = x, a vertical line of
// the reference mesh, on the channel of mu.
double
cross_section_flux(const Model & model, const ParameterValues & mu, const VectorType & u, const double x)
{
  const DoFHandler<2> &   dof_handler = model.get_dof_handler();
  const PeriodicHillGeometry<2> geometry = model.geometry(mu);

  VectorType ghosted(u);
  model.get_constraints().distribute(ghosted);
  ghosted.update_ghost_values();

  const FiniteElement<2> & fe = dof_handler.get_fe();
  FEFaceValues<2>          fe_face_values(fe,
                                 QGauss<1>(fe.degree + 2),
                                 update_gradients | update_quadrature_points | update_JxW_values);
  std::vector<Tensor<1, 2>> gradients(fe_face_values.n_quadrature_points);
  double                    flux = 0.;
  // Face 1 of a cell is the one at its right end in x.
  for(const auto & cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    if(std::abs(cell->face(1)->center()[0] - x) < 1e-12)
    {
      fe_face_values.reinit(cell, 1);
      fe_face_values.get_function_gradients(ghosted, gradients);
      for(const unsigned int q : fe_face_values.quadrature_point_indices())
      {
        // The gradient on the channel is J^-T times the one on the reference box, and dy = J_yy dy_hat.
        const Tensor<2, 2> J = geometry.jacobian(fe_face_values.quadrature_point(q));
        const double u_x = 1. + (transpose(invert(J)) * gradients[q])[0];
        flux += u_x * J[1][1] * fe_face_values.JxW(q);
      }
    }
  return Utilities::MPI::sum(flux, dof_handler.get_mpi_communicator());
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
  const auto            u_flat   = solve(model, flat);
  deallog << "flat channel, phi_tilde = 0: " << (u_flat->linfty_norm() < 1e-12 ? "true" : "false")
          << ", flow rate 3.036: "
          << (std::abs(model.flow_rate(flat, *u_flat) - PeriodicHillGeometry<2>::height) < 1e-12 ?
                "true" :
                "false")
          << std::endl;

  for(const double width : {1.93, 2.})
  {
    const ParameterValues mu = {{"hill_height", {1.}}, {"hill_width", {width}}};
    const auto            u  = solve(model, mu);

    const double flow_rate = model.flow_rate(mu, *u);
    VectorType   Au(*u);
    model.system_operator(mu)->apply(Au, *u);
    const double energy = (model.geometry(mu).volume() - Au * *u) / PeriodicHillGeometry<2>::length;
    deallog << "hill width " << width << ": flow rate " << flow_rate
            << ", relative difference to the flux through the crest "
            << std::abs(cross_section_flux(model, mu, *u, PeriodicHillGeometry<2>::length) - flow_rate) /
                 flow_rate
            << ", the valley below 1e-6: "
            << (std::abs(cross_section_flux(model, mu, *u, PeriodicHillGeometry<2>::length / 2.) -
                         flow_rate) < 1e-6 * flow_rate ?
                  "true" :
                  "false")
            << ", equals ||grad phi||^2 / 9: "
            << (std::abs(energy - flow_rate) < 1e-8 * flow_rate ? "true" : "false") << std::endl;
    for(const unsigned int degree : {1u, 2u, 3u, 5u})
    {
      VectorType difference = *solve_on_mapped_mesh(model, mu, degree);
      difference -= *u;
      deallog << "hill width " << width << ", mapping degree " << degree
              << ": relative difference " << difference.l2_norm() / u->l2_norm() << std::endl;
    }
  }
}
