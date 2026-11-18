// GaussianSource, used through its StationaryModel interface as the Python layer uses it, for Q1
// to Q3 elements on a sequence of uniformly refined meshes. The output shows
//  - the CG iterations of the system operator's inverse (logged by deal.II's SolverControl): with
//    the algebraic multigrid preconditioner they must stay about constant under refinement. Up to
//    about 2000 dofs the coarse level of the multigrid is the whole problem and is solved directly,
//    so CG takes one step.
//  - the energy E = f(u) = |u|^2_{H^1_0}, whose error decays like h^(2k) for Q_k elements: the
//    experimental order log2(dE_coarse / dE_fine) of the increments between meshes has to
//    approach 2k.
// The mpirun=2 variant has to give the same dofs and energies; iterations may differ slightly,
// because the multigrid hierarchy depends on the partitioning.

#include "../tests.h"

#include <implementations/gaussian_source.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();

  const ParameterValues mu = {{"center", {0.3, 0.6}}};

  for(const unsigned int degree : {1u, 2u, 3u})
  {
    deallog.push("Q" + std::to_string(degree));

    // Finest meshes with about 5e4 dofs: beyond that, the energy increments of Q3 are too close
    // to round-off to give an order.
    double energy = 0., increment = 0.;
    for(unsigned int n_refinements = 3; n_refinements <= 9 - degree; ++n_refinements)
    {
      const GaussianSource<2> model(n_refinements, degree, 0.2, MPI_COMM_WORLD);
      deallog << "refinements: " << n_refinements
              << ", dofs: " << model.solution_space()->n_dofs() << std::endl;

      const auto rhs = model.assemble_rhs(mu);
      const auto u   = model.solution_space()->create_vector();
      model.system_operator()->apply_inverse(*u, *rhs);

      const double new_energy    = *rhs * *u;
      const double new_increment = new_energy - energy;

      std::ostringstream line;
      line << "energy: " << std::scientific << std::setprecision(8) << new_energy;
      if(n_refinements >= 5)
        line << ", order: " << std::fixed << std::setprecision(2)
             << std::log2(increment / new_increment);
      deallog << line.str() << std::endl;

      energy    = new_energy;
      increment = new_increment;
    }

    deallog.pop();
  }
}
