#ifndef BINDINGS_DEALII_STATIONARY_MODEL_H
#define BINDINGS_DEALII_STATIONARY_MODEL_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interfaces/stationary_model.h>

#include <memory>
#include <string>


namespace bindings
{
    namespace py = pybind11;

    template <typename VectorType>
    void bind_stationary_model(py::module_& module, const std::string& name)
    {
        using Model = StationaryModel<VectorType>;

        py::class_<Model, std::shared_ptr<Model>>(module, name.c_str())
                .def("solution_space", &Model::solution_space)
                .def("parameters", &Model::parameters)
                .def("system_operator", &Model::system_operator)
                .def("assemble_rhs", &Model::assemble_rhs, py::arg("mu"))
                .def("products", &Model::products)
                .def("write_vtu", &Model::write_vtu,
                     py::arg("directory"), py::arg("basename"), py::arg("fields"), py::arg("names"));
    }
}

#endif /* BINDINGS_DEALII_STATIONARY_MODEL_H */
