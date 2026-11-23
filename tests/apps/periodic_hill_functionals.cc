// The quantities of interest of PeriodicHillPotentialFlow (periodic_hill_functionals.h).
//  - flat_channel: u = e_x, so the flux through the lower third of a cross-section is a third of
//    the height 3.036, and the mean wall pressure -1/2.
//  - flux: the flux through a whole cross-section is the flow rate Q, up to the discretization
//    error, on a mesh line (x = 2.25, the mean of both sides) and inside cells (x = 2.3); in the
//    valley (hill width 1.93) and on the flank (3).
//  - gradient: gradient() against central differences of value(), exact up to round-off for the
//    affine flux and the quadratic pressure.
// The mpirun=2 variant has to give the same.

#include "../tests.h"

#include <deal.II/base/mpi.h>

#include <implementations/periodic_hill_functionals.h>
#include <implementations/periodic_hill_potential_flow.h>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using Model      = PeriodicHillPotentialFlow<2>;
using VectorType = Model::VectorType;

// The CG iterations are logged one level below the caller, so that depth_file() hides them.
std::shared_ptr<VectorType>
solve(const Model & model, const ParameterValues & mu)
{
  LogStream::Prefix prefix("solve");
  const auto        rhs = model.assemble_rhs(mu);
  const auto u   = model.solution_space()->create_vector();
  model.system_operator(mu)->apply_inverse(*u, *rhs);
  return u;
}

const char *
boolean(const bool value)
{
  return value ? "true" : "false";
}

void
test_flat_channel(const Model & model)
{
  LogStream::Prefix     prefix("flat_channel");
  const ParameterValues flat = {{"hill_height", {0.}}, {"hill_width", {3.}}};
  const auto            u    = solve(model, flat);
  const double          flux = PeriodicHillCrossSectionFlux<2>(model, 2.25, 0., 1. / 3.).value(flat, *u);
  const double pressure      = PeriodicHillWallPressure<2>(model, 0., 2.).value(flat, *u);
  deallog << "flux through the lower third 1.012: "
          << boolean(std::abs(flux - PeriodicHillGeometry<2>::height / 3.) < 1e-12)
          << ", mean wall pressure -0.5: " << boolean(std::abs(pressure + 0.5) < 1e-12) << std::endl;
}

void
test_flux(const Model & model, const ParameterValues & mu, const VectorType & u)
{
  LogStream::Prefix prefix("flux");
  const double      flow_rate = model.flow_rate(mu, u);
  for(const double x : {2.25, 2.3})
  {
    const double flux = PeriodicHillCrossSectionFlux<2>(model, x, 0., 1.).value(mu, u);
    deallog << "x = " << x << ": whole cross-section, relative difference to the flow rate "
            << std::abs(flux - flow_rate) / flow_rate << ", lower third "
            << PeriodicHillCrossSectionFlux<2>(model, x, 0., 1. / 3.).value(mu, u) << std::endl;
  }
}

void
test_gradient(const Model & model, const ParameterValues & mu, const VectorType & u)
{
  LogStream::Prefix prefix("gradient");
  // The direction: a solution for another mu.
  const auto v = solve(model, {{"hill_height", {0.7}}, {"hill_width", {3.}}});

  const PeriodicHillCrossSectionFlux<2> flux(model, 2.25, 0., 1. / 3.);
  const PeriodicHillWallPressure<2>     pressure(model, 0., 2.);
  for(const auto & [name, functional] :
      std::vector<std::pair<std::string, const Functional<VectorType> *>>{{"flux", &flux},
                                                                          {"pressure", &pressure}})
  {
    const double step = 1e-2;
    VectorType   u_plus(u), u_minus(u);
    u_plus.add(step, *v);
    u_minus.add(-step, *v);
    const double difference =
      (functional->value(mu, u_plus) - functional->value(mu, u_minus)) / (2. * step);
    const double derivative = *functional->gradient(mu, u) * *v;
    deallog << name << ": value " << functional->value(mu, u)
            << ", gradient matches central differences: "
            << boolean(std::abs(derivative - difference) < 1e-10 * std::abs(derivative)) << std::endl;
  }
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();
  // Without the CG iterations (one level below the nested sections), which depend on the
  // partitioning.
  deallog.depth_file(3);
  deallog.precision(4);

  const ParameterValues product_mu = {{"hill_height", {1.}}, {"hill_width", {3.}}};
  const Model           model(3, 2, 0., product_mu, MPI_COMM_WORLD);

  test_flat_channel(model);
  // With the hill foot left of the cross-section at x = 2.25, and right of it.
  for(const double width : {1.93, 3.})
  {
    LogStream::Prefix     prefix("hill_width_" + std::to_string(width).substr(0, 4));
    const ParameterValues mu = {{"hill_height", {1.}}, {"hill_width", {width}}};
    const auto            u  = solve(model, mu);
    test_flux(model, mu, *u);
    test_gradient(model, mu, *u);
  }
}
