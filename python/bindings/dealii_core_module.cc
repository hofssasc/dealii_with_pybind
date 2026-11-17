#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


// Function to register a dealii vector class with pybind11
#include "dealii_vectors.h"
// Currently registered vector types
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace py = pybind11;


PYBIND11_MODULE(dealii_core_module, m, py::mod_gil_not_used()) {
    m.doc() = "core bindings to use the dealii library with pymor";
    
    bindings::bind_vector<dealii::Vector<double>>(m, "Vector");
    bindings::bind_vector<dealii::LinearAlgebra::distributed::Vector<double>>(m, "DistributedVector");
}

