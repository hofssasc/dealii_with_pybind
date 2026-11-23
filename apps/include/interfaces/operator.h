#ifndef APPLICATIONS_OPERATOR_H
#define APPLICATIONS_OPERATOR_H

#include <deal.II/base/exceptions.h>

#include <interfaces/vector_space.h>

#include <memory>

/**
 * Abstract base class for linear operators,
 * a linear map between two VectorSpaces.
 * Examples:
 * a system matrix or parts of it that correspond to a certain parameter,
 * or the matrix of an inner product.
 *
 * Important: Operators do not depend on parameters.
 * Everything that depends on parameters is assembled 
 * by the model (see StationaryModel::assemble_rhs()).
 */
template<typename VectorType>
class Operator
{
public:
  virtual ~Operator() = default;

  // For range and source spaces, return pointer to space object on every call.
  // Do not use make_shared, this creates a new space object on every call.
  virtual std::shared_ptr<const VectorSpace<VectorType>>
  source() const = 0;

  virtual std::shared_ptr<const VectorSpace<VectorType>>
  range() const = 0;

  // A src = dst
  virtual void
  apply(VectorType & dst, const VectorType & src) const = 0;

  // Per default, operators do not have an inverse. If they do, override this method.
  virtual bool
  has_inverse() const
  {
    return false;
  }

  // Solve A dst = rhs. On entry, dst holds the initial guess.
  virtual void
  apply_inverse(VectorType & /*dst*/, const VectorType & /*rhs*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This operator has no inverse."));
  }

  // Per default, operators do not have an adjoint (transpose), e.g. for goal-oriented reduction.
  // If they do, override this method and apply_adjoint(), and apply_inverse_adjoint() if they have
  // an inverse as well. How is up to the operator: a symmetric one can apply itself.
  virtual bool
  has_adjoint() const
  {
    return false;
  }

  // A^T src = dst, from range() to source().
  virtual void
  apply_adjoint(VectorType & /*dst*/, const VectorType & /*src*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This operator has no adjoint."));
  }

  // Solve A^T dst = rhs. On entry, dst holds the initial guess.
  virtual void
  apply_inverse_adjoint(VectorType & /*dst*/, const VectorType & /*rhs*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This operator has no inverse adjoint."));
  }
};

#endif /* APPLICATIONS_OPERATOR_H */
