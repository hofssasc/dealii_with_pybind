#ifndef APPLICATIONS_RESTRICTED_ASSEMBLY_H
#define APPLICATIONS_RESTRICTED_ASSEMBLY_H

#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

/**
 * A few rows of a condensed system, for empirical interpolation: with homogeneous constraints C
 * (periodicity, hanging nodes, Dirichlet values zero), the entries of C^T f and the rows of
 * C^T A C at a few unconstrained dofs, assembled only on the cells around them.
 *
 * The row of a dof i collects the cells around i and, weighted by C_ki, the cells around every dof
 * k that is constrained to i: for a periodic boundary the cells on both sides, for a hanging node
 * the cells of the finer side. A column k of a cell matrix is distributed over the dofs k is
 * constrained to, with their weights; a dof that is constrained to zero has no column. The columns
 * that occur are the source dofs, the entries of u that the rows of A u depend on.
 *
 * Every rank assembles its locally owned cells, the results are summed over the ranks.
 */
template<int dim>
class RestrictedAssembly
{
public:
  using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

  // Refers to the cells of dof_handler, which has to outlive this object. Collective, with the same
  // dofs on every rank; none of them may be constrained.
  RestrictedAssembly(const dealii::DoFHandler<dim> &                      dof_handler,
                     const dealii::AffineConstraints<double> &            constraints,
                     const std::vector<dealii::types::global_dof_index> & dofs,
                     const MPI_Comm                                       communicator);

  // The unconstrained dofs the rows depend on, sorted, the same on every rank.
  const std::vector<dealii::types::global_dof_index> &
  get_source_dofs() const;

  // C^T f at the dofs, in their order. cell_vector(cell, cell_f) fills the cell vector of f.
  // Collective, the result is the same on every rank.
  std::vector<double>
  assemble_vector(
    const std::function<void(const CellIterator &, dealii::Vector<double> &)> & cell_vector) const;

  // C^T A C at the dofs (rows, in their order) and the source dofs (columns), row-major.
  // cell_matrix(cell, cell_A) fills the cell matrix of A. Collective, the result is the same on
  // every rank.
  std::vector<double>
  assemble_matrix(const std::function<void(const CellIterator &, dealii::FullMatrix<double> &)> &
                    cell_matrix) const;

private:
  struct Cell
  {
    CellIterator cell;
    // (local dof, row, weight) for the local dofs that contribute to a row.
    std::vector<std::tuple<unsigned int, std::size_t, double>> rows;
    // For each local dof, its (column, weight).
    std::vector<std::vector<std::pair<std::size_t, double>>> columns;
  };

  std::size_t                                  n_dofs;
  unsigned int                                 n_dofs_per_cell;
  MPI_Comm                                     mpi_communicator;
  std::vector<dealii::types::global_dof_index> source_dofs;
  std::vector<Cell>                            cells;
};

#endif /* APPLICATIONS_RESTRICTED_ASSEMBLY_H */
