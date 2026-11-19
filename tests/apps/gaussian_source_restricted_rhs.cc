// The restricted right-hand side of GaussianSource and GaussianSourceMatrixFree (both
// GaussianSourceRestrictedRhs) against their full right-hand sides, for Q1 to Q3: restricted to all
// dofs off the boundary, in reverse order, it has to give the same entries up to round-off. In the
// mpirun=2 variant, the cells around a dof can belong to different ranks.

#include "../tests.h"

#include <implementations/gaussian_source.h>
#include <implementations/gaussian_source_matrix_free.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

template<typename Model>
void
test(const std::string & name)
{
  const ParameterValues mu = {{"center", {0.3, 0.6}}};

  for(const unsigned int degree : {1u, 2u, 3u})
  {
    const Model model(3, degree, 0.2, MPI_COMM_WORLD);
    const auto  rhs = model.assemble_rhs(mu);

    // All entries on every rank; the boundary dofs are the ones that are zero.
    std::vector<double> full(rhs->size(), 0.);
    for(const auto i : rhs->locally_owned_elements())
      full[i] = (*rhs)(i);
    Utilities::MPI::sum(full, MPI_COMM_WORLD, full);

    std::vector<types::global_dof_index> dofs;
    for(types::global_dof_index i = 0; i < full.size(); ++i)
      if(full[i] != 0.)
        dofs.push_back(i);
    std::reverse(dofs.begin(), dofs.end());

    const std::vector<double> values = model.restricted_rhs(dofs)->evaluate(mu);

    double difference = 0.;
    for(std::size_t k = 0; k < dofs.size(); ++k)
      difference = std::max(difference, std::abs(values[k] - full[dofs[k]]));

    deallog << name << " Q" << degree << ": dofs " << full.size() << ", restricted to " << dofs.size()
            << ", same entries: " << (difference < 1e-14 * rhs->linfty_norm() ? "true" : "false")
            << std::endl;
  }
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();

  test<GaussianSource<2>>("matrix-based");
  test<GaussianSourceMatrixFree<2>>("matrix-free");
}
