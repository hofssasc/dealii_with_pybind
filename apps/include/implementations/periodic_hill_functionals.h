#ifndef APPLICATIONS_PERIODIC_HILL_FUNCTIONALS_H
#define APPLICATIONS_PERIODIC_HILL_FUNCTIONALS_H

#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <implementations/periodic_hill_potential_flow.h>
#include <interfaces/functional.h>

#include <memory>
#include <vector>

/**
 * A quantity of interest of PeriodicHillPotentialFlow (2D): the integral, or the mean, of a
 * function of the velocity u = e_x + grad phi_tilde along a segment of a line of the reference box,
 *
 *   J(mu; phi_tilde) = int f(tau, u) ds_hat   or   int f(tau, u) ds_hat / int |tau| ds_hat,
 *
 * with tau = J t_hat the tangent on the channel of the reference tangent t_hat, so that
 * |tau| ds_hat is the arc length on the channel. The segment is
 *
 *   {x_hat_axis = position, lower <= x_hat_other <= upper},
 *
 * e.g. axis 0 for a cross-section x = position (x = x_hat), axis 1 and position 0 for the hill.
 * It is fixed on the reference box, so the cells it touches do not depend on mu and J is smooth in
 * mu. The segment does not have to lie on mesh lines: inside a cell, the gradient there is used;
 * on a face between two cells, the mean of both sides.
 *
 * Subclasses define f and its derivative with respect to u; the derivative with respect to
 * phi_tilde follows by the chain rule, df/du . (J^-T grad_hat phi_i).
 */
template<int dim>
class PeriodicHillLineFunctional
  : public Functional<typename PeriodicHillPotentialFlow<dim>::VectorType>
{
public:
  using VectorType = typename PeriodicHillPotentialFlow<dim>::VectorType;

  // Refers to model, which has to outlive this object. Collective.
  PeriodicHillLineFunctional(const PeriodicHillPotentialFlow<dim> & model,
                             const unsigned int                     axis,
                             const double                           position,
                             const double                           lower,
                             const double                           upper,
                             const bool                             mean);

  double
  value(const ParameterValues & mu, const VectorType & phi_tilde) const override;

  std::shared_ptr<VectorType>
  gradient(const ParameterValues & mu, const VectorType & phi_tilde) const override;

protected:
  // f(tau, u), and df/du into derivative.
  virtual double
  integrand(const dealii::Tensor<1, dim> & tau,
            const dealii::Tensor<1, dim> & velocity,
            dealii::Tensor<1, dim> &       derivative) const = 0;

private:
  // The parts of the segment in the locally owned cells: the gradients of the shape functions at
  // the quadrature points on the part, and the weights in reference arc length, halved on faces
  // between two cells.
  struct Part
  {
    typename dealii::DoFHandler<dim>::active_cell_iterator cell;
    std::unique_ptr<dealii::FEValues<dim>>                 fe_values;
    std::vector<double>                                    weights;
  };

  // value() or gradient() (into gradient, if not null): the gradients of phi_tilde at the
  // quadrature points of each part are taken from ghosted_phi_tilde.
  double
  integrate(const ParameterValues & mu,
            const VectorType &      ghosted_phi_tilde,
            VectorType *            gradient) const;

  const PeriodicHillPotentialFlow<dim> & model;
  unsigned int                           tangential_axis;
  bool                                   mean;
  std::vector<Part>                      parts;
};

/**
 * The flux of the velocity through the part lower <= y_hat <= upper of the cross-section
 * x = position (0 < position < 9), in the direction of e_x: f = u_x tau_y with tau = J e_y_hat. For
 * the whole cross-section, the flux is the flow rate Q, up to the discretization error. Affine in
 * phi_tilde.
 */
template<int dim>
class PeriodicHillCrossSectionFlux : public PeriodicHillLineFunctional<dim>
{
public:
  PeriodicHillCrossSectionFlux(const PeriodicHillPotentialFlow<dim> & model,
                               const double                           x,
                               const double                           y_hat_lower,
                               const double                           y_hat_upper);

protected:
  double
  integrand(const dealii::Tensor<1, dim> & tau,
            const dealii::Tensor<1, dim> & velocity,
            dealii::Tensor<1, dim> &       derivative) const override;
};

/**
 * The mean of the pressure p = -|u|^2 / 2 (Bernoulli, zero at rest) over the hill, i.e. the
 * bottom wall, for x_lower <= x <= x_upper, with respect to the arc length. Quadratic in
 * phi_tilde.
 */
template<int dim>
class PeriodicHillWallPressure : public PeriodicHillLineFunctional<dim>
{
public:
  PeriodicHillWallPressure(const PeriodicHillPotentialFlow<dim> & model,
                           const double                           x_lower,
                           const double                           x_upper);

protected:
  double
  integrand(const dealii::Tensor<1, dim> & tau,
            const dealii::Tensor<1, dim> & velocity,
            dealii::Tensor<1, dim> &       derivative) const override;
};

#endif /* APPLICATIONS_PERIODIC_HILL_FUNCTIONALS_H */
