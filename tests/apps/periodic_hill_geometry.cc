// PeriodicHillGeometry for a few hill shapes: the wall is periodic and symmetric with the steepest
// slope 15 h / (8 w), the bottom and top of the reference box are mapped onto the hill and the top
// wall, the Jacobian agrees with central differences of push_forward, and det J is positive at the
// quadrature points of a reference mesh, and volume() is the integral of det J. Also writes the mapped meshes with det J per cell,
// periodic_hill_<dim>d_<k>.vtu, to look at them.

#include "../tests.h"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <implementations/periodic_hill_geometry.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

struct Shape
{
  double hill_height;
  double hill_width;
  double stretching;
};

template<int dim>
void
test(const std::vector<Shape> & shapes, const unsigned int n_refinements)
{
  using Geometry = PeriodicHillGeometry<dim>;

  for(unsigned int k = 0; k < shapes.size(); ++k)
  {
    const Shape &  shape = shapes[k];
    const Geometry geometry(shape.hill_height, shape.hill_width, shape.stretching);

    // The wall at the crests, in the valley, mirrored, and its steepest slope.
    bool   wall_ok   = std::abs(geometry.wall(0.) - shape.hill_height) < 1e-14 &&
                   std::abs(geometry.wall(Geometry::length) - shape.hill_height) < 1e-14 &&
                   geometry.wall(Geometry::length / 2.) == 0.;
    double max_slope = 0.;
    for(unsigned int i = 0; i <= 9000; ++i)
    {
      const double x = i * Geometry::length / 9000;
      wall_ok =
        wall_ok && std::abs(geometry.wall(x) - geometry.wall(Geometry::length - x)) < 1e-14;
      max_slope = std::max(max_slope, std::abs(geometry.wall_derivative(x)));
    }

    Triangulation<dim> triangulation;
    Geometry::create_coarse_reference_grid(triangulation);
    triangulation.refine_global(n_refinements);

    // The hill (boundary id 2) and the top wall (3).
    bool walls_ok = true;
    for(const auto & face : triangulation.active_face_iterators())
      if(face->at_boundary() && (face->boundary_id() == 2 || face->boundary_id() == 3))
        for(const unsigned int v : face->vertex_indices())
        {
          const Point<dim> x_hat = face->vertex(v);
          const double     y =
            face->boundary_id() == 2 ? geometry.wall(x_hat[0]) : Geometry::height;
          walls_ok = walls_ok && std::abs(geometry.push_forward(x_hat)[1] - y) < 1e-14;
        }

    // The Jacobian at the quadrature points, and at the cell centers for the output.
    FE_Q<dim>      fe(1);
    FEValues<dim>  fe_values(fe, QGauss<dim>(3), update_quadrature_points);
    Vector<double> cell_determinants(triangulation.n_active_cells());
    double         min_determinant = std::numeric_limits<double>::max();
    double         jacobian_error  = 0.;
    const double   step            = 1e-6;
    for(const auto & cell : triangulation.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for(const Point<dim> & x_hat : fe_values.get_quadrature_points())
      {
        const Tensor<2, dim> J = geometry.jacobian(x_hat);
        min_determinant        = std::min(min_determinant, determinant(J));

        for(unsigned int j = 0; j < dim; ++j)
        {
          Tensor<1, dim> offset;
          offset[j] = step;
          const Tensor<1, dim> difference =
            (geometry.push_forward(x_hat + offset) - geometry.push_forward(x_hat - offset)) /
            (2. * step);
          for(unsigned int i = 0; i < dim; ++i)
            jacobian_error = std::max(jacobian_error, std::abs(difference[i] - J[i][j]));
        }
      }
      cell_determinants[cell->active_cell_index()] = determinant(geometry.jacobian(cell->center()));
    }

    // The integral of det J, with a fine quadrature in x, as the feet of the hill, where the profile
    // is only C^2, may lie inside cells; det J does not depend on z.
    const QIterated<1> fine_x(QGauss<1>(4), 20);
    const QGauss<1>    gauss_y(6);
    FEValues<dim>      fine_values(fe,
                              dim == 2 ? QAnisotropic<dim>(fine_x, gauss_y) :
                                         QAnisotropic<dim>(fine_x, gauss_y, QGauss<1>(1)),
                              update_quadrature_points | update_JxW_values);
    double        volume = 0.;
    for(const auto & cell : triangulation.active_cell_iterators())
    {
      fine_values.reinit(cell);
      for(const unsigned int q : fine_values.quadrature_point_indices())
        volume += determinant(geometry.jacobian(fine_values.quadrature_point(q))) * fine_values.JxW(q);
    }

    deallog << "dim " << dim << ", hill height " << shape.hill_height << ", width "
            << shape.hill_width << ", stretching " << shape.stretching << std::endl;
    deallog << "  steepest slope " << max_slope << " (15 h / (8 w) = "
            << 15. * shape.hill_height / (8. * shape.hill_width) << ", "
            << std::atan(max_slope) * 180. / M_PI << " degrees), min det J " << min_determinant
            << std::endl;
    deallog << "  wall periodic and symmetric: " << (wall_ok ? "true" : "false")
            << ", bottom and top on the walls: " << (walls_ok ? "true" : "false")
            << ", Jacobian matches differences: " << (jacobian_error < 1e-7 ? "true" : "false")
            << std::endl;
    deallog << "  volume " << geometry.volume() << ", equals the integral of det J: "
            << (std::abs(volume - geometry.volume()) < 1e-10 * geometry.volume() ? "true" : "false")
            << std::endl;

    GridTools::transform(
      [&geometry](const Point<dim> & x_hat) { return geometry.push_forward(x_hat); },
      triangulation);
    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);
    data_out.add_data_vector(cell_determinants, "det_J");
    data_out.build_patches();
    std::ofstream file("periodic_hill_" + std::to_string(dim) + "d_" + std::to_string(k) + ".vtu");
    data_out.write_vtu(file);
  }
}

int
main()
{
  initlog();
  deallog.precision(4);

  // Close to the ERCOFTAC hill, with and without ExaDG's stretching; low and narrow; high with
  // flanks over the whole period; high and steep.
  test<2>({{1., 1.93, 0.}, {1., 1.93, 1.6}, {0.5, 1., 0.}, {2., 4.5, 0.}, {2., 2., 0.}}, 3);
  test<3>({{1., 1.93, 1.6}}, 1);
}
