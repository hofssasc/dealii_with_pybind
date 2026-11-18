#ifndef APPLICATIONS_VECTOR_SPACE_H
#define APPLICATIONS_VECTOR_SPACE_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/affine_constraints.h>
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
 * Most applications use PartitionedVectorSpace or ConstrainedVectorSpace below instead of
 * implementing this directly.
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
  virtual const dealii::IndexSet &
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
 * or by the partitioner of a MatrixFree object, so that the vectors match the ones its operators
 * work on:
 *
 *   PartitionedVectorSpace<VectorType>(matrix_free.get_vector_partitioner());
 *
 * All vectors share one partitioner, so creating a vector needs no communication.
 *
 * Derive from it to add constraints via make_admissible(), see ConstrainedVectorSpace.
 */
template<typename VectorType>
class PartitionedVectorSpace : public VectorSpace<VectorType>
{
public:
  PartitionedVectorSpace(const dealii::IndexSet & locally_owned_dofs,
                         const dealii::IndexSet & ghost_dofs,
                         const MPI_Comm           communicator)
    : partitioner(std::make_shared<const dealii::Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                                               ghost_dofs,
                                                                               communicator))
  {
  }

  // A space without ghost entries.
  PartitionedVectorSpace(const dealii::IndexSet & locally_owned_dofs, const MPI_Comm communicator)
    : PartitionedVectorSpace(locally_owned_dofs,
                             dealii::IndexSet(locally_owned_dofs.size()),
                             communicator)
  {
  }

  explicit PartitionedVectorSpace(
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner)
    : partitioner(std::move(partitioner))
  {
  }

  MPI_Comm
  communicator() const override
  {
    return partitioner->get_mpi_communicator();
  }

  const dealii::IndexSet &
  locally_owned_dofs() const override
  {
    return partitioner->locally_owned_range();
  }

  std::shared_ptr<VectorType>
  create_vector() const override
  {
    auto vector = std::make_shared<VectorType>();
    reinit(*vector, partitioner);
    return vector;
  }

private:
  template<typename Number>
  static void
  reinit(dealii::Vector<Number> &                                           vector,
         const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner)
  {
    AssertThrow(partitioner->locally_owned_size() == partitioner->size(),
                dealii::ExcMessage("A dealii::Vector cannot be distributed over several ranks, "
                                   "use LinearAlgebra::distributed::Vector instead."));

    vector.reinit(partitioner->size());
  }

  template<typename Number, typename MemorySpace>
  static void
  reinit(dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace> &  vector,
         const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner)
  {
    vector.reinit(partitioner);
  }

  std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;
};


/**
 * A PartitionedVectorSpace whose vectors are zero in all constrained entries (Dirichlet boundary,
 * hanging nodes). This is the representation the condensed system works in; apply
 * AffineConstraints::distribute() only to get the full field, e.g. for output.
 *
 * Holds its own copy of the constraints, so it stays valid after the application is gone.
 */
template<typename VectorType>
class ConstrainedVectorSpace : public PartitionedVectorSpace<VectorType>
{
public:
  using Constraints = dealii::AffineConstraints<typename VectorType::value_type>;

  ConstrainedVectorSpace(const dealii::IndexSet & locally_owned_dofs,
                         const dealii::IndexSet & ghost_dofs,
                         const MPI_Comm           communicator,
                         const Constraints &      constraints)
    : PartitionedVectorSpace<VectorType>(locally_owned_dofs, ghost_dofs, communicator),
      constraints(constraints)
  {
  }

  ConstrainedVectorSpace(std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner,
                         const Constraints &                                        constraints)
    : PartitionedVectorSpace<VectorType>(std::move(partitioner)), constraints(constraints)
  {
  }

  void
  make_admissible(VectorType & vector) const override
  {
    constraints.set_zero(vector);
  }

private:
  Constraints constraints;
};

#endif /* APPLICATIONS_VECTOR_SPACE_H */
