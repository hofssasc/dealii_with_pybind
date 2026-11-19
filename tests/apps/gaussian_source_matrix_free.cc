// The test of gaussian_source.cc for GaussianSourceMatrixFree, ExaDG's matrix-free solver with
// multigrid. Dofs and energies have to agree with gaussian_source.output, iterations are those of
// the multigrid preconditioned CG (logged by ExaDGPoissonOperator).

#include "../tests.h"

#include <implementations/gaussian_source_matrix_free.h>

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
      const GaussianSourceMatrixFree<2> model(n_refinements, degree, 0.2, MPI_COMM_WORLD);
      deallog << "refinements: " << n_refinements
              << ", dofs: " << model.solution_space()->n_dofs() << std::endl;

      const auto rhs = model.assemble_rhs(mu);
      const auto u   = model.solution_space()->create_vector();
      model.system_operator(mu)->apply_inverse(*u, *rhs);

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
