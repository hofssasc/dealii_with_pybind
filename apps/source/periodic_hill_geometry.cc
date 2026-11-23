#include <deal.II/base/exceptions.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <implementations/periodic_hill_geometry.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
// The stretching eta of the reference coordinate y_hat in [0, 1] and its derivative.
double
stretch(const double y_hat, const double beta)
{
  if(beta == 0.)
    return y_hat;
  return (1. + std::tanh(beta * (2. * y_hat - 1.)) / std::tanh(beta)) / 2.;
}

double
stretch_derivative(const double y_hat, const double beta)
{
  if(beta == 0.)
    return 1.;
  const double tanh_value = std::tanh(beta * (2. * y_hat - 1.));
  return beta * (1. - tanh_value * tanh_value) / std::tanh(beta);
}
} // namespace

template<int dim>
PeriodicHillGeometry<dim>::PeriodicHillGeometry(const double hill_height,
                                                const double hill_width,
                                                const double stretching)
  : hill_height(hill_height), hill_width(hill_width), stretching(stretching)
{
  AssertThrow(hill_height >= 0. && hill_height < height,
              dealii::ExcMessage("The hill height has to be in [0, 3.036)."));
  AssertThrow(hill_width > 0. && hill_width <= length / 2.,
              dealii::ExcMessage("The hill width has to be in (0, 4.5]."));
  AssertThrow(stretching >= 0., dealii::ExcMessage("The stretching has to be nonnegative."));
}

template<int dim>
void
PeriodicHillGeometry<dim>::create_coarse_reference_grid(dealii::Triangulation<dim> & triangulation)
{
  std::vector<unsigned int> repetitions = {9, 3};
  dealii::Point<dim>        upper_corner;
  upper_corner[0] = length;
  upper_corner[1] = 1.;
  if(dim == 3)
  {
    repetitions.push_back(4);
    upper_corner[2] = depth;
  }

  dealii::GridGenerator::subdivided_hyper_rectangle(
    triangulation, repetitions, dealii::Point<dim>(), upper_corner, true /*colorize*/);
}

template<int dim>
double
PeriodicHillGeometry<dim>::wall(const double x) const
{
  const double t = std::min(x, length - x) / hill_width;
  if(t >= 1.)
    return 0.;
  if(t <= 0.)
    return hill_height;
  return hill_height * (1. - t * t * t * (10. - 15. * t + 6. * t * t));
}

template<int dim>
double
PeriodicHillGeometry<dim>::wall_derivative(const double x) const
{
  const double t = std::min(x, length - x) / hill_width;
  if(t >= 1. || t <= 0.)
    return 0.;
  // d/dx of min(x, 9 - x) is 1 left and -1 right of the valley.
  const double sign = x < length / 2. ? 1. : -1.;
  return sign * hill_height / hill_width * (-30. * t * t * (1. - t) * (1. - t));
}

template<int dim>
double
PeriodicHillGeometry<dim>::volume() const
{
  const double area = length * height - hill_height * hill_width;
  return dim == 3 ? area * depth : area;
}

template<int dim>
dealii::Point<dim>
PeriodicHillGeometry<dim>::push_forward(const dealii::Point<dim> & x_hat) const
{
  const double y_w = wall(x_hat[0]);

  dealii::Point<dim> x = x_hat;
  x[1]                 = y_w + (height - y_w) * stretch(x_hat[1], stretching);
  return x;
}

template<int dim>
dealii::Tensor<2, dim>
PeriodicHillGeometry<dim>::jacobian(const dealii::Point<dim> & x_hat) const
{
  dealii::Tensor<2, dim> J;
  for(unsigned int d = 0; d < dim; ++d)
    J[d][d] = 1.;

  J[1][0] = wall_derivative(x_hat[0]) * (1. - stretch(x_hat[1], stretching));
  J[1][1] = (height - wall(x_hat[0])) * stretch_derivative(x_hat[1], stretching);
  return J;
}

template<int dim>
void
PeriodicHillGeometry<dim>::initialize_mapping(
  dealii::MappingQCache<dim> &       mapping,
  const dealii::Triangulation<dim> & triangulation) const
{
  mapping.initialize(
    dealii::MappingQ<dim>(1),
    triangulation,
    [this](const typename dealii::Triangulation<dim>::cell_iterator &,
           const dealii::Point<dim> & x_hat) { return push_forward(x_hat); },
    false /*function_describes_relative_displacement*/);
}

template class PeriodicHillGeometry<2>;
template class PeriodicHillGeometry<3>;
