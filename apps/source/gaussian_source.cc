#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <implementations/gaussian_source.h>

#include <cmath>

template<int dim>
GaussianSource<dim>::GaussianSource(const unsigned int n_refinements,
                                    const unsigned int degree,
                                    const double       width,
                                    const MPI_Comm     communicator)
  : mpi_communicator(communicator),
    width(width),
    triangulation(communicator),
    fe(degree),
    dof_handler(triangulation)
{
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(n_refinements);

  dof_handler.distribute_dofs(fe);
  locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  dealii::DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // Ghosted with the locally relevant dofs, so that assembly can add into entries other ranks own.
  space = std::make_shared<const ConstrainedVectorSpace<VectorType>>(
    dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_communicator, constraints);
  laplace_operator =
    std::make_shared<const TrilinosMatrixOperator>(assemble_laplace_matrix(), space);
}

template<int dim>
std::shared_ptr<const VectorSpace<typename GaussianSource<dim>::VectorType>>
GaussianSource<dim>::solution_space() const
{
  return space;
}

template<int dim>
std::map<std::string, unsigned int>
GaussianSource<dim>::parameters() const
{
  return {{"center", dim}};
}

template<int dim>
std::shared_ptr<const Operator<typename GaussianSource<dim>::VectorType>>
GaussianSource<dim>::system_operator() const
{
  return laplace_operator;
}

template<int dim>
std::shared_ptr<typename GaussianSource<dim>::VectorType>
GaussianSource<dim>::assemble_rhs(const ParameterValues & mu) const
{
  const std::vector<double> & center_values = mu.at("center");
  AssertThrow(center_values.size() == dim, dealii::ExcDimensionMismatch(center_values.size(), dim));

  dealii::Point<dim> center;
  for(unsigned int d = 0; d < dim; ++d)
    center[d] = center_values[d];

  const dealii::QGauss<dim> quadrature(fe.degree + 1);
  dealii::FEValues<dim>     fe_values(fe,
                                      quadrature,
                                      dealii::update_values | dealii::update_quadrature_points |
                                        dealii::update_JxW_values);

  dealii::Vector<double>                       cell_rhs(fe.n_dofs_per_cell());
  std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

  auto rhs = space->create_vector();
  for(const auto & cell :
      dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
  {
    fe_values.reinit(cell);
    cell_rhs = 0;

    for(const unsigned int q : fe_values.quadrature_point_indices())
    {
      const double source =
        std::exp(-(fe_values.quadrature_point(q) - center).norm_square() / (width * width));
      for(const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += source * fe_values.shape_value(i, q) * fe_values.JxW(q);
    }

    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(cell_rhs, dof_indices, *rhs);
  }
  rhs->compress(dealii::VectorOperation::add);

  return rhs;
}

template<int dim>
std::map<std::string, std::shared_ptr<const Operator<typename GaussianSource<dim>::VectorType>>>
GaussianSource<dim>::products() const
{
  return {{"h1_0_semi", laplace_operator}};
}

template<int dim>
std::string
GaussianSource<dim>::write_vtu(const std::string &                                    directory,
                               const std::string &                                    basename,
                               const std::vector<std::shared_ptr<const VectorType>> & fields,
                               const std::vector<std::string> &                       names) const
{
  AssertThrow(fields.size() == names.size(),
              dealii::ExcDimensionMismatch(fields.size(), names.size()));

  // The full fields including constrained entries, with ghost values for DataOut. DataOut only
  // keeps references, so all of them have to exist before the first one is added.
  std::vector<VectorType> ghosted_fields;
  ghosted_fields.reserve(fields.size());
  for(const auto & field : fields)
  {
    ghosted_fields.emplace_back(*field);
    constraints.distribute(ghosted_fields.back());
    ghosted_fields.back().update_ghost_values();
  }

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  for(unsigned int k = 0; k < fields.size(); ++k)
    data_out.add_data_vector(ghosted_fields[k], names[k]);
  data_out.build_patches(fe.degree);

  return directory + data_out.write_vtu_with_pvtu_record(directory, basename, 0, mpi_communicator);
}

template<int dim>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix>
GaussianSource<dim>::assemble_laplace_matrix() const
{
  dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                     dof_handler.locally_owned_dofs(),
                                                     mpi_communicator,
                                                     locally_relevant_dofs);

  auto matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  matrix->reinit(dof_handler.locally_owned_dofs(),
                 dof_handler.locally_owned_dofs(),
                 dsp,
                 mpi_communicator);

  const dealii::QGauss<dim> quadrature(fe.degree + 1);
  dealii::FEValues<dim>     fe_values(fe,
                                      quadrature,
                                      dealii::update_gradients | dealii::update_JxW_values);

  const unsigned int                           n_dofs_per_cell = fe.n_dofs_per_cell();
  dealii::FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_per_cell);

  for(const auto & cell :
      dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;

    for(const unsigned int q : fe_values.quadrature_point_indices())
      for(const unsigned int i : fe_values.dof_indices())
        for(const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);

    cell->get_dof_indices(dof_indices);
    constraints.distribute_local_to_global(cell_matrix, dof_indices, *matrix);
  }
  matrix->compress(dealii::VectorOperation::add);

  return matrix;
}

template class GaussianSource<2>;
