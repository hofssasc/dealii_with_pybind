#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>

#include <implementations/restricted_assembly.h>

#include <algorithm>
#include <map>
#include <string>

template<int dim>
RestrictedAssembly<dim>::RestrictedAssembly(
  const dealii::DoFHandler<dim> &                      dof_handler,
  const dealii::AffineConstraints<double> &            constraints,
  const std::vector<dealii::types::global_dof_index> & dofs,
  const MPI_Comm                                       communicator)
  : n_dofs(dofs.size()),
    n_dofs_per_cell(dof_handler.get_fe().n_dofs_per_cell()),
    mpi_communicator(communicator)
{
  // The rows each global dof contributes to, with its weight: the dofs themselves with 1, the dofs
  // constrained to them with their weights.
  std::map<dealii::types::global_dof_index, std::size_t> positions;
  std::map<dealii::types::global_dof_index, std::vector<std::pair<std::size_t, double>>>
    row_weights;
  for(std::size_t k = 0; k < dofs.size(); ++k)
  {
    // Checked on the ranks that store the line of the dof.
    AssertThrow(!constraints.can_store_line(dofs[k]) || !constraints.is_constrained(dofs[k]),
                dealii::ExcMessage("Dof " + std::to_string(dofs[k]) + " is constrained."));
    positions[dofs[k]] = k;
    row_weights[dofs[k]].emplace_back(k, 1.);
  }
  // The locally stored lines include all dofs of the locally owned cells.
  for(const auto & line : constraints.get_lines())
  {
    AssertThrow(line.inhomogeneity == 0., dealii::ExcNotImplemented());
    for(const auto & [j, weight] : line.entries)
      if(const auto position = positions.find(j); position != positions.end())
        row_weights[line.index].emplace_back(position->second, weight);
  }

  // The cells with a dof that contributes to a row, their columns still as global dofs.
  std::vector<std::vector<std::vector<std::pair<dealii::types::global_dof_index, double>>>>
                                               global_columns;
  std::vector<dealii::types::global_dof_index> local_source_dofs;
  std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_per_cell);
  for(const auto & cell :
      dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
  {
    cell->get_dof_indices(dof_indices);

    Cell cell_data{cell, {}, {}};
    for(unsigned int l = 0; l < n_dofs_per_cell; ++l)
      if(const auto weights = row_weights.find(dof_indices[l]); weights != row_weights.end())
        for(const auto & [k, weight] : weights->second)
          cell_data.rows.emplace_back(l, k, weight);
    if(cell_data.rows.empty())
      continue;

    auto & columns = global_columns.emplace_back(n_dofs_per_cell);
    for(unsigned int l = 0; l < n_dofs_per_cell; ++l)
    {
      // No entries for a dof that is constrained to zero.
      if(const auto * entries = constraints.get_constraint_entries(dof_indices[l]))
        columns[l] = *entries;
      else
        columns[l] = {{dof_indices[l], 1.}};
      for(const auto & [j, weight] : columns[l])
        local_source_dofs.push_back(j);
    }
    cells.push_back(std::move(cell_data));
  }

  for(const auto & rank_source_dofs :
      dealii::Utilities::MPI::all_gather(mpi_communicator, local_source_dofs))
    source_dofs.insert(source_dofs.end(), rank_source_dofs.begin(), rank_source_dofs.end());
  std::sort(source_dofs.begin(), source_dofs.end());
  source_dofs.erase(std::unique(source_dofs.begin(), source_dofs.end()), source_dofs.end());

  for(std::size_t c = 0; c < cells.size(); ++c)
  {
    cells[c].columns.resize(n_dofs_per_cell);
    for(unsigned int l = 0; l < n_dofs_per_cell; ++l)
      for(const auto & [j, weight] : global_columns[c][l])
      {
        const auto column = std::lower_bound(source_dofs.begin(), source_dofs.end(), j);
        cells[c].columns[l].emplace_back(column - source_dofs.begin(), weight);
      }
  }
}

template<int dim>
const std::vector<dealii::types::global_dof_index> &
RestrictedAssembly<dim>::get_source_dofs() const
{
  return source_dofs;
}

template<int dim>
std::vector<double>
RestrictedAssembly<dim>::assemble_vector(
  const std::function<void(const CellIterator &, dealii::Vector<double> &)> & cell_vector) const
{
  dealii::Vector<double> cell_f(n_dofs_per_cell);
  std::vector<double>    values(n_dofs, 0.);
  for(const auto & [cell, rows, columns] : cells)
  {
    cell_vector(cell, cell_f);
    for(const auto & [l, k, weight] : rows)
      values[k] += weight * cell_f(l);
  }
  dealii::Utilities::MPI::sum(values, mpi_communicator, values);

  return values;
}

template<int dim>
std::vector<double>
RestrictedAssembly<dim>::assemble_matrix(
  const std::function<void(const CellIterator &, dealii::FullMatrix<double> &)> & cell_matrix) const
{
  const std::size_t          n_columns = source_dofs.size();
  dealii::FullMatrix<double> cell_A(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<double>        values(n_dofs * n_columns, 0.);
  for(const auto & [cell, rows, columns] : cells)
  {
    cell_matrix(cell, cell_A);
    for(const auto & [l, k, row_weight] : rows)
      for(unsigned int m = 0; m < n_dofs_per_cell; ++m)
        for(const auto & [s, column_weight] : columns[m])
          values[k * n_columns + s] += row_weight * cell_A(l, m) * column_weight;
  }
  dealii::Utilities::MPI::sum(values, mpi_communicator, values);

  return values;
}

template class RestrictedAssembly<2>;
