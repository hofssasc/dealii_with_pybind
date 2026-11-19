#ifndef APPLICATIONS_PERIODIC_HILL_GEOMETRY_H
#define APPLICATIONS_PERIODIC_HILL_GEOMETRY_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/tria.h>

/**
 * The map T from the reference box onto a periodic hill channel, analytic and C^2 smooth.
 *
 * Lengths are in units of the ERCOFTAC hill height: the channel is periodic over the length 9 in
 * x, the top wall is at y = 3.036, as in ExaDG's periodic hill. The bottom wall is the symmetric
 * hill profile
 *
 *   y_w(x) = h s(d(x) / w),   d(x) = min(x, 9 - x),   s(t) = 1 - 10 t^3 + 15 t^4 - 6 t^5,
 *
 * with s = 0 for t > 1: crests of the height h at x = 0 and 9, flanks of the width w, and a flat
 * valley between the flanks if w < 4.5. The steepest slope is 15 h / (8 w); h = 1, w = 1.93 is
 * close to the ERCOFTAC hill.
 *
 * The reference box is (0, 9) x (0, 1) (x (0, 4.5) in 3D). T only shifts and stretches vertically,
 *
 *   x = x_hat,   y = y_w(x_hat) + (3.036 - y_w(x_hat)) eta(y_hat),   z = z_hat,
 *
 * with ExaDG's tanh stretching eta(y_hat) = (1 + tanh(beta (2 y_hat - 1)) / tanh(beta)) / 2, which
 * clusters cells at both walls for beta > 0 (ExaDG uses 1.6) and is the identity for beta = 0.
 * det J = (3.036 - y_w) eta' is positive for h < 3.036.
 */
template<int dim>
class PeriodicHillGeometry
{
public:
  static constexpr double length = 9.;
  static constexpr double height = 3.036;
  // Only in 3D, the extent in z.
  static constexpr double depth = 4.5;

  PeriodicHillGeometry(const double hill_height, const double hill_width, const double stretching);

  // The reference box, one coarse cell per unit length (roughly square cells in the valley for
  // beta = 0). Boundary ids: 0 and 1 at x_hat = 0 and 9, 2 at the hill, 3 at the top wall (and 4, 5
  // at z_hat = 0, 4.5).
  static void
  create_coarse_reference_grid(dealii::Triangulation<dim> & triangulation);

  // The hill profile y_w and its derivative, for x in [0, 9].
  double
  wall(const double x) const;

  double
  wall_derivative(const double x) const;

  dealii::Point<dim>
  push_forward(const dealii::Point<dim> & x_hat) const;

  // J = dT / dx_hat at x_hat.
  dealii::Tensor<2, dim>
  jacobian(const dealii::Point<dim> & x_hat) const;

  // Initializes mapping with T on all cells of triangulation, a mesh of the reference box: then
  // mapping maps it onto the channel, up to the interpolation of T by polynomials of the degree of
  // mapping. E.g. for output on the channel.
  void
  initialize_mapping(dealii::MappingQCache<dim> &       mapping,
                     const dealii::Triangulation<dim> & triangulation) const;

private:
  double hill_height;
  double hill_width;
  double stretching;
};

#endif /* APPLICATIONS_PERIODIC_HILL_GEOMETRY_H */
