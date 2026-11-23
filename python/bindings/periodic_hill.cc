#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

    m.attr("length") = PeriodicHillGeometry<2>::length;

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
