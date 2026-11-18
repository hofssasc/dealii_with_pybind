#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include <exadg/grid/grid.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>

#include <implementations/gaussian_source_matrix_free.h>

#include <cmath>

template<int dim>
GaussianSourceMatrixFree<dim>::GaussianSourceMatrixFree(const unsigned int n_refinements,
                                                        const unsigned int degree,
                                                        const double       width,
                                                        const MPI_Comm     communicator)
  : mpi_communicator(communicator), center(std::make_shared<dealii::Point<dim>>())
{
  // Continuous Q_k elements on a uniformly refined, distributed mesh, as in GaussianSource.
  param.right_hand_side             = true;
  param.grid.triangulation_type     = ExaDG::TriangulationType::Distributed;
  param.grid.n_refine_global        = n_refinements;
  param.mapping_degree              = 1;
  param.mapping_degree_coarse_grids = 1;
  param.spatial_discretization      = ExaDG::Poisson::SpatialDiscretization::CG;
  param.degree                      = degree;

  // CG with the same tolerance as GaussianSource, preconditioned by multigrid as in ExaDG's
  // Poisson applications: polynomial coarsening, then geometric coarsening, Chebyshev smoothing,
  // and CG with algebraic multigrid on the coarsest level.
  param.solver_data    = ExaDG::SolverData(1000, 1e-20, 1e-10, ExaDG::LinearSolver::CG);
  param.preconditioner = ExaDG::Poisson::Preconditioner::Multigrid;
  param.multigrid_data.type                     = ExaDG::MultigridType::cphMG;
  param.multigrid_data.p_sequence               = ExaDG::PSequenceType::Bisect;
  param.multigrid_data.smoother_data.smoother   = ExaDG::MultigridSmoother::Chebyshev;
  param.multigrid_data.smoother_data.iterations = 5;
  param.multigrid_data.coarse_problem.solver    = ExaDG::MultigridCoarseGridSolver::CG;
  param.multigrid_data.coarse_problem.preconditioner =
    ExaDG::MultigridCoarseGridPreconditioner::AMG;
  param.multigrid_data.coarse_problem.solver_data.rel_tol = 1e-6;
  param.check();

  auto grid = std::make_shared<ExaDG::Grid<dim>>();
  ExaDG::GridUtilities::create_triangulation_with_multigrid<dim>(
    *grid,
    mpi_communicator,
    param.grid,
    param.involves_h_multigrid(),
    [](dealii::Triangulation<dim> & triangulation,
       ExaDG::GridUtilities::PeriodicFacePairs<dim> & /*periodic_face_pairs*/,
       const unsigned int global_refinements,
       const std::vector<unsigned int> & /*local_refinements*/) {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(global_refinements);
    },
    {} /*local_refinements*/);

  std::shared_ptr<dealii::Mapping<dim>>                  mapping;
  std::shared_ptr<ExaDG::MultigridMappings<dim, double>> multigrid_mappings;
  ExaDG::GridUtilities::create_mapping_with_multigrid(mapping,
                                                      multigrid_mappings,
                                                      param.grid.element_type,
                                                      param.mapping_degree,
                                                      param.mapping_degree_coarse_grids,
                                                      param.involves_h_multigrid());

  auto boundary_descriptor = std::make_shared<ExaDG::Poisson::BoundaryDescriptor<0, dim>>();
  boundary_descriptor->dirichlet_bc.insert(
    {0, std::make_shared<dealii::Functions::ZeroFunction<dim>>()});

  auto field_functions             = std::make_shared<ExaDG::Poisson::FieldFunctions<dim>>();
  field_functions->right_hand_side = std::make_shared<dealii::ScalarFunctionFromFunctionObject<dim>>(
    [center = center, width](const dealii::Point<dim> & x) {
      return std::exp(-(x - *center).norm_square() / (width * width));
    });

  pde_operator = std::make_shared<ExaDG::Poisson::Operator<dim, 1, double>>(grid,
                                                                            mapping,
                                                                            multigrid_mappings,
                                                                            boundary_descriptor,
                                                                            field_functions,
                                                                            param,
                                                                            "Poisson",
                                                                            mpi_communicator);
  pde_operator->setup();

  // The same constraints as ExaDG's (which are not accessible), for make_admissible() and output.
  const auto & dof_handler = pde_operator->get_dof_handler();
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));
  dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  dealii::DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  space = std::make_shared<const ConstrainedVectorSpace<VectorType>>(
    pde_operator->get_matrix_free()->get_vector_partitioner(pde_operator->get_dof_index()),
    constraints);
  laplace_operator = std::make_shared<const ExaDGPoissonOperator<dim>>(pde_operator, space);
}

template<int dim>
std::shared_ptr<const VectorSpace<typename GaussianSourceMatrixFree<dim>::VectorType>>
GaussianSourceMatrixFree<dim>::solution_space() const
{
  return space;
}

template<int dim>
std::map<std::string, unsigned int>
GaussianSourceMatrixFree<dim>::parameters() const
{
  return {{"center", dim}};
}

template<int dim>
std::shared_ptr<const Operator<typename GaussianSourceMatrixFree<dim>::VectorType>>
GaussianSourceMatrixFree<dim>::system_operator() const
{
  return laplace_operator;
}

template<int dim>
std::shared_ptr<typename GaussianSourceMatrixFree<dim>::VectorType>
GaussianSourceMatrixFree<dim>::assemble_rhs(const ParameterValues & mu) const
{
  const std::vector<double> & center_values = mu.at("center");
  AssertThrow(center_values.size() == dim, dealii::ExcDimensionMismatch(center_values.size(), dim));

  for(unsigned int d = 0; d < dim; ++d)
    (*center)[d] = center_values[d];

  auto rhs = space->create_vector();
  pde_operator->rhs(*rhs);

  return rhs;
}

template<int dim>
std::map<std::string,
         std::shared_ptr<const Operator<typename GaussianSourceMatrixFree<dim>::VectorType>>>
GaussianSourceMatrixFree<dim>::products() const
{
  return {{"h1_0_semi", laplace_operator}};
}

template<int dim>
std::string
GaussianSourceMatrixFree<dim>::write_vtu(
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
  data_out.attach_dof_handler(pde_operator->get_dof_handler());
  for(unsigned int k = 0; k < fields.size(); ++k)
    data_out.add_data_vector(ghosted_fields[k], names[k]);
  data_out.build_patches(param.degree);

  return directory + data_out.write_vtu_with_pvtu_record(directory, basename, 0, mpi_communicator);
}

template class GaussianSourceMatrixFree<2>;
