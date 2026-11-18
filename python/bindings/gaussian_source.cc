#include <pybind11/pybind11.h>
#include <implementations/gaussian_source.h>
#include <implementations/gaussian_source_matrix_free.h>

#include <deal.II/base/mpi.h>

#include <memory>

namespace py = pybind11;

using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

PYBIND11_MODULE(gaussian_source, m, py::mod_gil_not_used()) {
    m.doc() = "Poisson's equation with a Gaussian source of variable center, matrix-based "
              "(gaussian_source.h) and matrix-free with ExaDG (gaussian_source_matrix_free.h)";

    // StationaryModel and the classes it hands out are bound in dealii_core_module.
    py::module_::import("dealii_core_module");

    m.def("make_model",
          [](unsigned int n_refinements, unsigned int degree, double width)
              -> std::shared_ptr<StationaryModel<VectorType>>
          {
              return std::make_shared<GaussianSource<2>>(n_refinements, degree, width, MPI_COMM_WORLD);
          },
          "The 2d model on the ranks of MPI_COMM_WORLD.",
          py::arg("n_refinements") = 5, py::arg("degree") = 1, py::arg("width") = 0.2);

    m.def("make_matrix_free_model",
          [](unsigned int n_refinements, unsigned int degree, double width)
              -> std::shared_ptr<StationaryModel<VectorType>>
          {
              return std::make_shared<GaussianSourceMatrixFree<2>>(n_refinements, degree, width,
                                                                   MPI_COMM_WORLD);
          },
          "The same 2d model with ExaDG's matrix-free solver.",
          py::arg("n_refinements") = 5, py::arg("degree") = 1, py::arg("width") = 0.2);
}
