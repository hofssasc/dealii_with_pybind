#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <implementations/periodic_hill_potential_flow.h>

#include <string>
#include <vector>

namespace
{
double
get_scalar(const ParameterValues & mu, const std::string & name)
{
  const std::vector<double> & values = mu.at(name);
  AssertThrow(values.size() == 1, dealii::ExcDimensionMismatch(values.size(), 1));

  return values[0];
}

// The quadrature of A(mu) and f(mu), degree + 1 Gauss points.
template<int dim>
dealii::FEValues<dim>
create_fe_values(const dealii::FiniteElement<dim> & fe)
{
  return dealii::FEValues<dim>(fe,
                               dealii::QGauss<dim>(fe.degree + 1),
                               dealii::update_gradients | dealii::update_quadrature_points |
                                 dealii::update_JxW_values);
}

// The cell matrix of A(mu) and/or the cell vector of f(mu), either may be null, on the cell
// fe_values is initialized on.
template<int dim>
void
integrate_cell(const PeriodicHillGeometry<dim> & geometry,
               const dealii::FEValues<dim> &     fe_values,
               dealii::FullMatrix<double> *      cell_matrix,
               dealii::Vector<double> *          cell_rhs)
{
  if(cell_matrix != nullptr)
    *cell_matrix = 0;
  if(cell_rhs != nullptr)
    *cell_rhs = 0;

  for(const unsigned int q : fe_values.quadrature_point_indices())
  {
    // The reference cells are rectangles, so the gradients and JxW of fe_values are the ones on
    // the reference box.
    const dealii::Tensor<2, dim> J         = geometry.jacobian(fe_values.quadrature_point(q));
    const dealii::Tensor<2, dim> J_inverse = dealii::invert(J);
    const dealii::Tensor<2, dim> K =
      dealii::determinant(J) * J_inverse * dealii::transpose(J_inverse);

    for(const unsigned int i : fe_values.dof_indices())
    {
      // K is symmetric, (grad phi_i, K e_x) is the x component of K grad phi_i.
      const dealii::Tensor<1, dim> K_grad_i = K * fe_values.shape_grad(i, q);
      if(cell_rhs != nullptr)
        (*cell_rhs)(i) -= K_grad_i[0] * fe_values.JxW(q);
      if(cell_matrix != nullptr)
        for(const unsigned int j : fe_values.dof_indices())
          (*cell_matrix)(i, j) += K_grad_i * fe_values.shape_grad(j, q) * fe_values.JxW(q);
    }
  }
}

// From phi_tilde at the points of the physical mesh: the full potential phi = x + phi_tilde, the
// velocity u = grad phi and the pressure -|u|^2 / 2. For a difference of two phi_tilde only the
// differences of the potentials and velocities, grad phi_tilde.
template<int dim>
class FlowPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
  FlowPostprocessor(const std::string & name, const bool difference)
    : name(name), difference(difference)
  {
  }

  void
  evaluate_scalar_field(const dealii::DataPostprocessorInputs::Scalar<dim> & inputs,
                        std::vector<dealii::Vector<double>> & computed_quantities) const override
  {
    for(unsigned int q = 0; q < inputs.solution_values.size(); ++q)
    {
      dealii::Tensor<1, dim> velocity = inputs.solution_gradients[q];
      computed_quantities[q](0)       = inputs.solution_values[q];
      if(!difference)
      {
        velocity[0] += 1.;
        computed_quantities[q](0) += inputs.evaluation_points[q][0];
        computed_quantities[q](1 + dim) = -velocity.norm_square() / 2.;
      }
      for(unsigned int d = 0; d < dim; ++d)
        computed_quantities[q](1 + d) = velocity[d];
    }
  }

  std::vector<std::string>
  get_names() const override
  {
    std::vector<std::string> names = {name + "_phi"};
    names.insert(names.end(), dim, name + "_velocity");
    if(!difference)
      names.emplace_back(name + "_pressure");
    return names;
  }

  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override
  {
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> interpretation(
      dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
    interpretation.insert(interpretation.begin(),
                          dealii::DataComponentInterpretation::component_is_scalar);
    if(!difference)
      interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    return interpretation;
  }

  dealii::UpdateFlags
  get_needed_update_flags() const override
  {
    return dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points;
  }

private:
  std::string name;
  bool        difference;
};
} // namespace

template<int dim>
PeriodicHillPotentialFlow<dim>::PeriodicHillPotentialFlow(const unsigned int      n_refinements,
                                                          const unsigned int      degree,
                                                          const double            stretching,
                                                          const ParameterValues & product_mu,
                                                          const MPI_Comm          communicator)
  : mpi_communicator(communicator),
    stretching(stretching),
    triangulation(communicator),
    fe(degree),
    dof_handler(triangulation)
{
  AssertThrow(n_refinements >= 1,
              dealii::ExcMessage("At least one refinement, for a vertex in the middle of the "
                                 "valley floor."));

  // Periodic in x (boundary ids 0 and 1) and z (4 and 5). A distributed mesh has to know this
  // before refinement, for its ghost layer.
  PeriodicHillGeometry<dim>::create_coarse_reference_grid(triangulation);
  std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
    periodic_faces;
  dealii::GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  if(dim == 3)
    dealii::GridTools::collect_periodic_faces(triangulation, 4, 5, 2, periodic_faces);
  triangulation.add_periodicity(periodic_faces);
  triangulation.refine_global(n_refinements);

  dof_handler.distribute_dofs(fe);
  locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
  dealii::DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
  if(dim == 3)
    dealii::DoFTools::make_periodicity_constraints(dof_handler, 4, 5, 2, constraints);

  // phi_tilde = 0 in the middle of the valley floor, on every rank that has this dof.
  dealii::Point<dim> valley_floor;
  valley_floor[0] = PeriodicHillGeometry<dim>::length / 2.;
  if(dim == 3)
    valley_floor[2] = PeriodicHillGeometry<dim>::depth / 2.;

  // This can be improved but works for now.
  for(const auto & cell : dof_handler.active_cell_iterators())
    if(!cell->is_artificial())
      for(const unsigned int v : cell->vertex_indices())
        if(const auto i = cell->vertex_dof_index(v, 0);
           cell->vertex(v).distance(valley_floor) < 1e-12 && !constraints.is_constrained(i))
          constraints.add_line(i);
  constraints.close();

  sparsity_pattern.reinit(dof_handler.n_dofs(), dof_handler.n_dofs(), locally_relevant_dofs);
  dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                                     dof_handler.locally_owned_dofs(),
                                                     mpi_communicator,
                                                     locally_relevant_dofs);

  // Ghosted with the locally relevant dofs, so that assembly can add into entries other ranks own.
  space = std::make_shared<const ConstrainedVectorSpace<VectorType>>(
    dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_communicator, constraints);
  h1_semi_product = system_operator(product_mu);
}

template<int dim>
std::shared_ptr<const VectorSpace<typename PeriodicHillPotentialFlow<dim>::VectorType>>
PeriodicHillPotentialFlow<dim>::solution_space() const
{
  return space;
}

template<int dim>
std::map<std::string, unsigned int>
PeriodicHillPotentialFlow<dim>::parameters() const
{
  return {{"hill_height", 1}, {"hill_width", 1}};
}

template<int dim>
std::shared_ptr<const Operator<typename PeriodicHillPotentialFlow<dim>::VectorType>>
PeriodicHillPotentialFlow<dim>::system_operator(const ParameterValues & mu) const
{
  auto matrix = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  matrix->reinit(dof_handler.locally_owned_dofs(),
                 dof_handler.locally_owned_dofs(),
                 sparsity_pattern,
                 mpi_communicator);
  assemble(mu, matrix.get(), nullptr);

  return std::make_shared<const TrilinosMatrixOperator>(matrix, space);
}

template<int dim>
bool
PeriodicHillPotentialFlow<dim>::system_operator_is_parametric() const
{
  return true;
}

template<int dim>
std::shared_ptr<typename PeriodicHillPotentialFlow<dim>::VectorType>
PeriodicHillPotentialFlow<dim>::assemble_rhs(const ParameterValues & mu) const
{
  auto rhs = space->create_vector();
  assemble(mu, nullptr, rhs.get());

  return rhs;
}

template<int dim>
std::map<std::string,
         std::shared_ptr<const Operator<typename PeriodicHillPotentialFlow<dim>::VectorType>>>
PeriodicHillPotentialFlow<dim>::products() const
{
  return {{"h1_semi", h1_semi_product}};
}

template<int dim>
std::string
PeriodicHillPotentialFlow<dim>::write_vtu(
  const std::string &                                    directory,
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
std::string
PeriodicHillPotentialFlow<dim>::write_flow_vtu(
  const std::string &                                    directory,
  const std::string &                                    basename,
  const ParameterValues &                                mu,
  const std::vector<std::shared_ptr<const VectorType>> & flows,
  const std::vector<std::string> &                       flow_names,
  const std::vector<std::shared_ptr<const VectorType>> & differences,
  const std::vector<std::string> &                       difference_names) const
{
  AssertThrow(flows.size() == flow_names.size(),
              dealii::ExcDimensionMismatch(flows.size(), flow_names.size()));
  AssertThrow(differences.size() == difference_names.size(),
              dealii::ExcDimensionMismatch(differences.size(), difference_names.size()));

  // The full fields with ghost values, and a postprocessor for each. DataOut only keeps references,
  // so all of them have to exist before the first one is added.
  std::vector<VectorType>             ghosted_fields;
  std::vector<FlowPostprocessor<dim>> postprocessors;
  const std::size_t                   n_fields = flows.size() + differences.size();
  ghosted_fields.reserve(n_fields);
  postprocessors.reserve(n_fields);
  for(std::size_t k = 0; k < n_fields; ++k)
  {
    const bool difference = k >= flows.size();
    ghosted_fields.emplace_back(difference ? *differences[k - flows.size()] : *flows[k]);
    constraints.distribute(ghosted_fields.back());
    ghosted_fields.back().update_ghost_values();
    postprocessors.emplace_back(difference ? difference_names[k - flows.size()] : flow_names[k],
                                difference);
  }

  // The mesh on the channel of mu; the velocity comes from the gradients on it.
  dealii::MappingQCache<dim> mapping(fe.degree + 1);
  geometry(mu).initialize_mapping(mapping, triangulation);

  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  for(std::size_t k = 0; k < n_fields; ++k)
    data_out.add_data_vector(ghosted_fields[k], postprocessors[k]);
  data_out.build_patches(mapping, fe.degree, dealii::DataOut<dim>::curved_inner_cells);

  return directory + data_out.write_vtu_with_pvtu_record(directory, basename, 0, mpi_communicator);
}

template<int dim>
PeriodicHillGeometry<dim>
PeriodicHillPotentialFlow<dim>::geometry(const ParameterValues & mu) const
{
  return PeriodicHillGeometry<dim>(get_scalar(mu, "hill_height"),
                                   get_scalar(mu, "hill_width"),
                                   stretching);
}

template<int dim>
const dealii::DoFHandler<dim> &
PeriodicHillPotentialFlow<dim>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim>
const dealii::AffineConstraints<double> &
PeriodicHillPotentialFlow<dim>::get_constraints() const
{
  return constraints;
}

template<int dim>
std::shared_ptr<const RestrictedRhs>
PeriodicHillPotentialFlow<dim>::restricted_rhs(
  const std::vector<dealii::types::global_dof_index> & dofs) const
{
  return std::make_shared<const PeriodicHillRestrictedRhs<dim>>(*this, dofs);
}

template<int dim>
std::shared_ptr<const RestrictedOperator>
PeriodicHillPotentialFlow<dim>::restricted_system_operator(
  const std::vector<dealii::types::global_dof_index> & dofs) const
{
  return std::make_shared<const PeriodicHillRestrictedOperator<dim>>(*this, dofs);
}

template<int dim>
void
PeriodicHillPotentialFlow<dim>::assemble(const ParameterValues &                  mu,
                                         dealii::TrilinosWrappers::SparseMatrix * matrix,
                                         VectorType *                             rhs) const
{
  const PeriodicHillGeometry<dim> geometry  = this->geometry(mu);
  dealii::FEValues<dim>           fe_values = create_fe_values(fe);

  const unsigned int                           n_dofs_per_cell = fe.n_dofs_per_cell();
  dealii::FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
  dealii::Vector<double>                       cell_rhs(n_dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_per_cell);

  for(const auto & cell :
      dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
  {
    fe_values.reinit(cell);
    integrate_cell(geometry,
                   fe_values,
                   matrix != nullptr ? &cell_matrix : nullptr,
                   rhs != nullptr ? &cell_rhs : nullptr);

    cell->get_dof_indices(dof_indices);
    if(matrix != nullptr)
      constraints.distribute_local_to_global(cell_matrix, dof_indices, *matrix);
    if(rhs != nullptr)
      constraints.distribute_local_to_global(cell_rhs, dof_indices, *rhs);
  }

  if(matrix != nullptr)
    matrix->compress(dealii::VectorOperation::add);
  if(rhs != nullptr)
    rhs->compress(dealii::VectorOperation::add);
}

template<int dim>
PeriodicHillRestrictedRhs<dim>::PeriodicHillRestrictedRhs(
  const PeriodicHillPotentialFlow<dim> &               model,
  const std::vector<dealii::types::global_dof_index> & dofs)
  : model(model),
    assembly(model.get_dof_handler(),
             model.get_constraints(),
             dofs,
             model.solution_space()->communicator())
{
}

template<int dim>
std::vector<double>
PeriodicHillRestrictedRhs<dim>::evaluate(const ParameterValues & mu) const
{
  const PeriodicHillGeometry<dim> geometry  = model.geometry(mu);
  dealii::FEValues<dim>           fe_values = create_fe_values(model.get_dof_handler().get_fe());

  return assembly.assemble_vector([&](const auto & cell, dealii::Vector<double> & cell_rhs) {
    fe_values.reinit(cell);
    integrate_cell(geometry, fe_values, nullptr, &cell_rhs);
  });
}

template<int dim>
PeriodicHillRestrictedOperator<dim>::PeriodicHillRestrictedOperator(
  const PeriodicHillPotentialFlow<dim> &               model,
  const std::vector<dealii::types::global_dof_index> & dofs)
  : model(model),
    assembly(model.get_dof_handler(),
             model.get_constraints(),
             dofs,
             model.solution_space()->communicator())
{
}

template<int dim>
std::vector<dealii::types::global_dof_index>
PeriodicHillRestrictedOperator<dim>::source_dofs() const
{
  return assembly.get_source_dofs();
}

template<int dim>
std::vector<double>
PeriodicHillRestrictedOperator<dim>::assemble(const ParameterValues & mu) const
{
  const PeriodicHillGeometry<dim> geometry  = model.geometry(mu);
  dealii::FEValues<dim>           fe_values = create_fe_values(model.get_dof_handler().get_fe());

  return assembly.assemble_matrix([&](const auto & cell, dealii::FullMatrix<double> & cell_matrix) {
    fe_values.reinit(cell);
    integrate_cell(geometry, fe_values, &cell_matrix, nullptr);
  });
}

template class PeriodicHillPotentialFlow<2>;
template class PeriodicHillRestrictedRhs<2>;
template class PeriodicHillRestrictedOperator<2>;
