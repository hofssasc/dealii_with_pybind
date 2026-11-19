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
 * The rows of a system operator A(mu) at a fixed set of dofs, for its empirical interpolation. The
 * rows only depend on the entries of u at a few source dofs, so the reduced model only needs the
 * small matrix of A(mu) restricted to the dofs and the source dofs for every new mu. Created once
 * for the interpolation dofs by StationaryModel::restricted_system_operator(); an evaluation must
 * not get more expensive as the mesh is refined.
 */
class RestrictedOperator
{
public:
  virtual ~RestrictedOperator() = default;

  // The global dofs of the columns, the same on every rank.
  virtual std::vector<dealii::types::global_dof_index>
  source_dofs() const = 0;

  // A(mu) at the dofs (rows, in their order) and source_dofs() (columns), row-major. Collective,
  // the result is the same on every rank.
  virtual std::vector<double>
  assemble(const ParameterValues & mu) const = 0;
};

/**
 * The components of a stationary problem
 *
 *   A(mu) u(mu) = f(mu)
 *
 * with a system operator and a right-hand side that may depend on the parameters in any way. This
 * is a description for the Python layer, which turns it into a pyMOR StationaryModel; each
 * application implements it once.
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

  // A(mu), has to have an inverse. If the operator does not depend on the parameters (see
  // below), mu is empty and the same operator is returned on every call.
  virtual std::shared_ptr<const Operator<VectorType>>
  system_operator(const ParameterValues & mu) const = 0;

  // Whether A depends on the parameters. If so, pyMOR calls system_operator(mu) for every mu,
  // which assembles a new operator; if not, it calls it once.
  virtual bool
  system_operator_is_parametric() const
  {
    return false;
  }

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

  // A(mu) at the given global dofs, for empirical interpolation of a parametric system operator.
  // Like restricted_rhs(): collective, may refer to the model, optional.
  virtual std::shared_ptr<const RestrictedOperator>
  restricted_system_operator(
    const std::vector<dealii::types::global_dof_index> & /*dofs*/) const
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
