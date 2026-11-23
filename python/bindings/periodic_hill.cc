#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <implementations/periodic_hill_functionals.h>
#include <implementations/periodic_hill_potential_flow.h>

#include <deal.II/base/mpi.h>

#include <memory>

namespace py = pybind11;

using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
using Model      = PeriodicHillPotentialFlow<2>;

PYBIND11_MODULE(periodic_hill, m, py::mod_gil_not_used()) {
    m.doc() = "Potential flow through a periodic hill channel with parametrized hill height and "
              "width (periodic_hill_potential_flow.h)";

    // StationaryModel and the classes it hands out are bound in dealii_core_module.
    py::module_::import("dealii_core_module");

    py::class_<Model, StationaryModel<VectorType>, std::shared_ptr<Model>>(m, "PotentialFlow")
            .def("write_flow_vtu", &Model::write_flow_vtu,
                 py::arg("directory"), py::arg("basename"), py::arg("mu"), py::arg("flows"),
                 py::arg("flow_names"),
                 py::arg("differences") = std::vector<std::shared_ptr<const VectorType>>(),
                 py::arg("difference_names") = std::vector<std::string>())
            .def("flow_rate", &Model::flow_rate,
                 "The volume flux Q(mu) through every cross-section for the solution phi_tilde of mu.",
                 py::arg("mu"), py::arg("phi_tilde"))
            .def("volume",
                 [](const Model& model, const ParameterValues& mu)
                 { return model.geometry(mu).volume(); },
                 "The area of the channel of mu.", py::arg("mu"));

    // Quantities of interest, Functionals bound in dealii_core_module. They refer to the model,
    // which is kept alive as long as they are.
    py::class_<PeriodicHillCrossSectionFlux<2>, Functional<VectorType>,
               std::shared_ptr<PeriodicHillCrossSectionFlux<2>>>(
            m, "CrossSectionFlux",
            "The flux through the part y_hat_lower <= y_hat <= y_hat_upper of the cross-section x, "
            "y_hat in [0, 1] from the hill to the top wall (periodic_hill_functionals.h).")
            .def(py::init<const Model&, double, double, double>(), py::arg("model"), py::arg("x"),
                 py::arg("y_hat_lower") = 0., py::arg("y_hat_upper") = 1., py::keep_alive<1, 2>());
    py::class_<PeriodicHillWallPressure<2>, Functional<VectorType>,
               std::shared_ptr<PeriodicHillWallPressure<2>>>(
            m, "WallPressure",
            "The mean pressure -|u|^2 / 2 on the hill for x_lower <= x <= x_upper "
            "(periodic_hill_functionals.h).")
            .def(py::init<const Model&, double, double>(), py::arg("model"), py::arg("x_lower"),
                 py::arg("x_upper"), py::keep_alive<1, 2>());

    m.attr("length") = PeriodicHillGeometry<2>::length;
    m.attr("height") = PeriodicHillGeometry<2>::height;

    m.def("make_potential_flow_model",
          [](unsigned int n_refinements, unsigned int degree, double stretching,
             const ParameterValues& product_mu) -> std::shared_ptr<Model>
          {
              return std::make_shared<Model>(n_refinements, degree, stretching, product_mu,
                                             MPI_COMM_WORLD);
          },
          "The 2d model on the ranks of MPI_COMM_WORLD.",
          py::arg("n_refinements") = 3, py::arg("degree") = 2, py::arg("stretching") = 0.,
          py::arg("product_mu") = ParameterValues{{"hill_height", {1.}}, {"hill_width", {3.}}});
}
