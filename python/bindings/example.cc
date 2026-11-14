#include <pybind11/pybind11.h>
#include <code.h>

namespace py = pybind11;

PYBIND11_MODULE(example, m, py::mod_gil_not_used()) {
    m.doc() = "pybind11 example plugin";

    // function using the costum library
    m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b"));

    // function using dealii
}
