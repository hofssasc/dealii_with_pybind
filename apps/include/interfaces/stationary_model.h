#ifndef APPLICATIONS_STATIONARY_MODEL_H
#define APPLICATIONS_STATIONARY_MODEL_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <interfaces/operator.h>
#include <interfaces/vector_space.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

// Parameter values by name, e.g. {"center": {0.3, 0.6}}; pyMOR's Mu.
using ParameterValues = std::map<std::string, std::vector<double>>;

/**
 * The right-hand side f(mu) at a fixed set of dofs, for its empirical interpolation: the reduced
 * model only needs these few entries of f(mu) for every new mu. Created once for the interpolation
 * dofs by StationaryModel::restricted_rhs(); an evaluation must not get more expensive as the mesh
 * is refined.
 */
class RestrictedRhs
{
public:
  virtual ~RestrictedRhs() = default;

  // The entries of f(mu) at the dofs, in their order. Collective, the result is the same on every
  // rank.
  virtual std::vector<double>
  evaluate(const ParameterValues & mu) const = 0;
};

/**
 * The components of a stationary problem
 *
 *   A u(mu) = f(mu)
 *
 * with a fixed system operator A and a right-hand side that may depend on the parameters in any
 * way. This is a description for the Python layer, which turns it into a pyMOR StationaryModel;
 * each application implements it once.
 */
template<typename VectorType>
class StationaryModel
{
public:
  virtual ~StationaryModel() = default;

  virtual std::shared_ptr<const VectorSpace<VectorType>>
  solution_space() const = 0;

  // The parameters and their number of components, e.g. {"center": 2}.
  virtual std::map<std::string, unsigned int>
  parameters() const = 0;

  // A; has to have an inverse.
  virtual std::shared_ptr<const Operator<VectorType>>
  system_operator() const = 0;

  // f(mu), a new vector in solution_space().
  virtual std::shared_ptr<VectorType>
  assemble_rhs(const ParameterValues & mu) const = 0;

  // f(mu) at the given global dofs, for empirical interpolation. Collective, with the same dofs on
  // every rank. The result may refer to the model, which has to outlive it. Optional: without it
  // (nullptr), pyMOR assembles all of f(mu) for every mu, with the same result but no speedup.
  virtual std::shared_ptr<const RestrictedRhs>
  restricted_rhs(const std::vector<dealii::types::global_dof_index> & /*dofs*/) const
  {
    return nullptr;
  }

  // Inner products on solution_space() by name; pyMOR adds a <name>_product and a <name>_norm to
  // the model for each, e.g. h1_0_semi_product and h1_0_semi_norm.
  virtual std::map<std::string, std::shared_ptr<const Operator<VectorType>>>
  products() const
  {
    return {};
  }

  // Write fields to directory/basename_0.pvtu (plus one .vtu per rank) and return the path of the
  // .pvtu record. directory has to end with '/'.
  virtual std::string
  write_vtu(const std::string & /*directory*/,
            const std::string & /*basename*/,
            const std::vector<std::shared_ptr<const VectorType>> & /*fields*/,
            const std::vector<std::string> & /*names*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This model does not implement write_vtu()."));

    return {};
  }
};

#endif /* APPLICATIONS_STATIONARY_MODEL_H */
