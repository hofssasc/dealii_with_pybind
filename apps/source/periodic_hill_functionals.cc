#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/vector.h>

#include <implementations/periodic_hill_functionals.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

template<int dim>
PeriodicHillLineFunctional<dim>::PeriodicHillLineFunctional(
  const PeriodicHillPotentialFlow<dim> & model,
  const unsigned int                     axis,
  const double                           position,
  const double                           lower,
  const double                           upper,
  const bool                             mean)
  : model(model), tangential_axis(1 - axis), mean(mean)
{
  AssertThrow(dim == 2, dealii::ExcMessage("Only implemented in 2D."));
  AssertThrow(axis < dim, dealii::ExcIndexRange(axis, 0, dim));
  AssertThrow(lower < upper, dealii::ExcMessage("The segment has to have a positive length."));
  // On the periodic boundary, the cells on the other end of the channel would be missing.
  AssertThrow(axis != 0 || (0. < position && position < PeriodicHillGeometry<dim>::length),
              dealii::ExcMessage("A cross-section has to lie strictly inside the period."));

  const dealii::FiniteElement<dim> & fe = model.get_dof_handler().get_fe();
  const dealii::QGauss<1>            gauss(fe.degree + 2);
  const unsigned int                 t = tangential_axis;

  // The reference mesh consists of rectangles, vertex 0 and the last one are opposite corners.
  for(const auto & cell : model.get_dof_handler().active_cell_iterators() |
                            dealii::IteratorFilters::LocallyOwnedCell())
  {
    const dealii::Point<dim> & corner_0 = cell->vertex(0);
    const dealii::Point<dim> & corner_1 = cell->vertex(cell->n_vertices() - 1);
    const double               h_axis   = corner_1[axis] - corner_0[axis];
    const double               h_t      = corner_1[t] - corner_0[t];
    const double               xi       = (position - corner_0[axis]) / h_axis;
    const double               begin    = std::max(lower, corner_0[t]);
    const double               end      = std::min(upper, corner_1[t]);
    if(xi < -1e-12 || xi > 1. + 1e-12 || end - begin < 1e-12 * h_t)
      continue;

    // On a face of the cell, the gradient of phi_tilde jumps: take the mean of both sides, unless
    // the face is at the boundary.
    const bool on_face = xi < 1e-12 || xi > 1. - 1e-12;
    const double share =
      on_face && !cell->face(2 * axis + (xi > 0.5 ? 1 : 0))->at_boundary() ? 0.5 : 1.;

    std::vector<dealii::Point<dim>> points(gauss.size());
    std::vector<double>             weights(gauss.size());
    for(unsigned int q = 0; q < gauss.size(); ++q)
    {
      points[q][axis] = on_face ? std::round(xi) : xi;
      points[q][t]    = (begin + (end - begin) * gauss.point(q)[0] - corner_0[t]) / h_t;
      weights[q]      = gauss.weight(q) * (end - begin) * share;
    }
    parts.push_back(
      {cell,
       std::make_unique<dealii::FEValues<dim>>(fe,
                                               dealii::Quadrature<dim>(points, weights),
                                               dealii::update_gradients |
                                                 dealii::update_quadrature_points),
       std::move(weights)});
  }
}

template<int dim>
double
PeriodicHillLineFunctional<dim>::value(const ParameterValues & mu,
                                       const VectorType &      phi_tilde) const
{
  VectorType ghosted(phi_tilde);
  model.get_constraints().distribute(ghosted);
  ghosted.update_ghost_values();
  return integrate(mu, ghosted, nullptr);
}

template<int dim>
std::shared_ptr<typename PeriodicHillLineFunctional<dim>::VectorType>
PeriodicHillLineFunctional<dim>::gradient(const ParameterValues & mu,
                                          const VectorType &      phi_tilde) const
{
  VectorType ghosted(phi_tilde);
  model.get_constraints().distribute(ghosted);
  ghosted.update_ghost_values();
  auto gradient = model.solution_space()->create_vector();
  integrate(mu, ghosted, gradient.get());
  return gradient;
}

template<int dim>
double
PeriodicHillLineFunctional<dim>::integrate(const ParameterValues & mu,
                                           const VectorType &      ghosted_phi_tilde,
                                           VectorType *            gradient) const
{
  const PeriodicHillGeometry<dim> geometry = model.geometry(mu);
  const unsigned int n_dofs_per_cell       = model.get_dof_handler().get_fe().n_dofs_per_cell();
  dealii::Vector<double>                       cell_gradient(n_dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_per_cell);
  std::vector<dealii::Tensor<1, dim>>          gradients;

  double integral = 0., length = 0.;
  for(const Part & part : parts)
  {
    const dealii::FEValues<dim> & fe_values = *part.fe_values;
    part.fe_values->reinit(part.cell);
    gradients.resize(fe_values.n_quadrature_points);
    fe_values.get_function_gradients(ghosted_phi_tilde, gradients);
    cell_gradient = 0;

    for(const unsigned int q : fe_values.quadrature_point_indices())
    {
      // The reference cells are rectangles, so the quadrature points of fe_values are on the
      // reference box, and so are its gradients.
      const dealii::Tensor<2, dim> J = geometry.jacobian(fe_values.quadrature_point(q));
      const dealii::Tensor<2, dim> J_inverse_transpose = dealii::transpose(dealii::invert(J));
      dealii::Tensor<1, dim>       tau;
      for(unsigned int d = 0; d < dim; ++d)
        tau[d] = J[d][tangential_axis];
      dealii::Tensor<1, dim> velocity = J_inverse_transpose * gradients[q];
      velocity[0] += 1.;

      dealii::Tensor<1, dim> derivative;
      integral += integrand(tau, velocity, derivative) * part.weights[q];
      length += tau.norm() * part.weights[q];
      if(gradient != nullptr)
        for(const unsigned int i : fe_values.dof_indices())
          cell_gradient(i) +=
            derivative * (J_inverse_transpose * fe_values.shape_grad(i, q)) * part.weights[q];
    }

    if(gradient != nullptr)
    {
      part.cell->get_dof_indices(dof_indices);
      model.get_constraints().distribute_local_to_global(cell_gradient, dof_indices, *gradient);
    }
  }

  const MPI_Comm communicator = model.solution_space()->communicator();
  integral                    = dealii::Utilities::MPI::sum(integral, communicator);
  length                      = dealii::Utilities::MPI::sum(length, communicator);
  if(gradient != nullptr)
  {
    gradient->compress(dealii::VectorOperation::add);
    if(mean)
      *gradient /= length;
  }
  return mean ? integral / length : integral;
}

template<int dim>
PeriodicHillCrossSectionFlux<dim>::PeriodicHillCrossSectionFlux(
  const PeriodicHillPotentialFlow<dim> & model,
  const double                           x,
  const double                           y_hat_lower,
  const double                           y_hat_upper)
  : PeriodicHillLineFunctional<dim>(model, 0, x, y_hat_lower, y_hat_upper, false)
{
}

template<int dim>
double
PeriodicHillCrossSectionFlux<dim>::integrand(const dealii::Tensor<1, dim> & tau,
                                             const dealii::Tensor<1, dim> & velocity,
                                             dealii::Tensor<1, dim> &       derivative) const
{
  // u . n |tau| with the normal n = (tau_y, -tau_x) / |tau|, to the right of tau.
  derivative[0] = tau[1];
  derivative[1] = -tau[0];
  return velocity * derivative;
}

template<int dim>
PeriodicHillWallPressure<dim>::PeriodicHillWallPressure(const PeriodicHillPotentialFlow<dim> & model,
                                                        const double x_lower,
                                                        const double x_upper)
  : PeriodicHillLineFunctional<dim>(model, 1, 0., x_lower, x_upper, true)
{
}

template<int dim>
double
PeriodicHillWallPressure<dim>::integrand(const dealii::Tensor<1, dim> & tau,
                                         const dealii::Tensor<1, dim> & velocity,
                                         dealii::Tensor<1, dim> &       derivative) const
{
  // p |tau| with p = -|u|^2 / 2.
  derivative = -tau.norm() * velocity;
  return -0.5 * velocity.norm_square() * tau.norm();
}

template class PeriodicHillLineFunctional<2>;
template class PeriodicHillCrossSectionFlux<2>;
template class PeriodicHillWallPressure<2>;
