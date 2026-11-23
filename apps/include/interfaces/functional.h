#ifndef APPLICATIONS_FUNCTIONAL_H
#define APPLICATIONS_FUNCTIONAL_H

#include <interfaces/stationary_model.h>

#include <memory>

/**
 * A quantity of interest J(mu; u) of a solution u of a StationaryModel, linear or not, e.g. the
 * flux through a line or a mean pressure. For goal-oriented reduction, its derivative with respect
 * to u is the right-hand side of the adjoint problem
 *
 *   A(mu)^T z = -dJ/du(mu; u).
 *
 * Created by the application, which knows the quantities that make sense for it; may refer to the
 * model, which then has to outlive it.
 */
template<typename VectorType>
class Functional
{
public:
  virtual ~Functional() = default;

  // J(mu; u) for u in the solution space of the model. Collective, the result is the same on
  // every rank.
  virtual double
  value(const ParameterValues & mu, const VectorType & u) const = 0;

  // dJ/du(mu; u), a new vector in the solution space: the entries of the linear functional
  // v -> dJ/du(mu; u) v. For a functional that is affine in u, it does not depend on u.
  virtual std::shared_ptr<VectorType>
  gradient(const ParameterValues & mu, const VectorType & u) const = 0;
};

#endif /* APPLICATIONS_FUNCTIONAL_H */
