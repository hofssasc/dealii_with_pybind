#ifndef APPLICATIONS_VECTOR_SPACE_H
#define APPLICATIONS_VECTOR_SPACE_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <memory>
#include <utility>

/**
 * The vectors of one discretisation: how many entries they have, how the entries are distributed
 * over the MPI ranks, and which constraints they have to satisfy.
 *
 * This is everything the Python layer needs from an application to wrap its vectors for pyMOR.
 * The vector arithmetic itself comes from VectorType, everything that needs communication beyond
 * that (gathering into NumPy arrays, picking single entries) is built on locally_owned_dofs() and
 * communicator() in the bindings.
 *
 * Most applications derive from PartitionedVectorSpace below instead of implementing this directly.
 */
template<typename VectorType>
class VectorSpace
{
public:
  virtual ~VectorSpace() = default;

  // Global number of degrees of freedom, summed over ranks.
  dealii::types::global_dof_index
  n_dofs() const
  {
    return locally_owned_dofs().size();
  }

  virtual MPI_Comm
  communicator() const = 0;

  // Global indices of the entries stored on this rank. Disjoint between ranks, all of them together
  // make up [0, n_dofs()).
  virtual dealii::IndexSet const &
  locally_owned_dofs() const = 0;

  // A zero vector with this space's partitioning. Ghost entries, if the vector stores any, are zero
  // as well; operators update them themselves where they need them.
  virtual std::shared_ptr<VectorType>
  create_vector() const = 0;

  /**
   * Project onto the subspace the operators are valid on.
   *
   * Called on every vector the Python layer builds that did not come out of a solve: random
   * probes, interpolation candidates, data read from NumPy. Eliminate Dirichlet rows here if the
   * discretisation has them.
   *
   * pyMOR forms arbitrary linear combinations of vectors, so this has to be a linear projection:
   * homogeneous Dirichlet and hanging node constraints belong here, inhomogeneous boundary values
   * belong into the right-hand side.
   *
   * The default is a no-op, which is correct for a discretisation that constrains nothing -- a
   * discontinuous Galerkin velocity space, for instance, imposes its boundary conditions weakly.
   */
  virtual void
  make_admissible(VectorType & /*vector*/) const
  {
  }
};


/**
 * A VectorSpace given by index sets, typically those of a DoFHandler:
 *
 *   PartitionedVectorSpace<VectorType>(dof_handler.locally_owned_dofs(),
 *                                      DoFTools::extract_locally_relevant_dofs(dof_handler),
 *                                      dof_handler.get_mpi_communicator());
 *
 * Derive from it to add constraints via make_admissible(), or to create vectors through
 * MatrixFree::initialize_dof_vector() so they match the partitioner the operators use.
 */
template<typename VectorType>
class PartitionedVectorSpace : public VectorSpace<VectorType>
{
public:
  PartitionedVectorSpace(dealii::IndexSet locally_owned_dofs,
                         dealii::IndexSet ghost_dofs,
                         MPI_Comm const   communicator)
    : owned_dofs(std::move(locally_owned_dofs)),
      ghost_dofs(std::move(ghost_dofs)),
      mpi_communicator(communicator)
  {
  }

  // A space without ghost entries.
  PartitionedVectorSpace(dealii::IndexSet locally_owned_dofs, MPI_Comm const communicator)
    : PartitionedVectorSpace(locally_owned_dofs,
                             dealii::IndexSet(locally_owned_dofs.size()),
                             communicator)
  {
  }

  MPI_Comm
  communicator() const override
  {
    return mpi_communicator;
  }

  dealii::IndexSet const &
  locally_owned_dofs() const override
  {
    return owned_dofs;
  }

  std::shared_ptr<VectorType>
  create_vector() const override
  {
    auto vector = std::make_shared<VectorType>();
    reinit(*vector, owned_dofs, ghost_dofs, mpi_communicator);
    return vector;
  }

private:
  template<typename Number>
  static void
  reinit(dealii::Vector<Number> & vector,
         dealii::IndexSet const & locally_owned_dofs,
         dealii::IndexSet const & /*ghost_dofs*/,
         MPI_Comm const communicator)
  {
    AssertThrow(dealii::Utilities::MPI::n_mpi_processes(communicator) == 1,
                dealii::ExcMessage("A dealii::Vector cannot be distributed over several ranks, "
                                   "use LinearAlgebra::distributed::Vector instead."));
    AssertThrow(locally_owned_dofs.n_elements() == locally_owned_dofs.size(),
                dealii::ExcMessage("A dealii::Vector has to own all of its entries."));

    vector.reinit(locally_owned_dofs.size());
  }

  template<typename Number, typename MemorySpace>
  static void
  reinit(dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> & vector,
         dealii::IndexSet const &                                          locally_owned_dofs,
         dealii::IndexSet const &                                          ghost_dofs,
         MPI_Comm const                                                    communicator)
  {
    vector.reinit(locally_owned_dofs, ghost_dofs, communicator);
  }

  dealii::IndexSet owned_dofs;
  dealii::IndexSet ghost_dofs;
  MPI_Comm         mpi_communicator;
};

#endif /* APPLICATIONS_VECTOR_SPACE_H */
