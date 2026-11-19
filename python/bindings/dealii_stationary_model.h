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

    // Independent of the vector type, bound once.
    inline void bind_restricted(py::module_& module)
    {
        py::class_<RestrictedRhs, std::shared_ptr<RestrictedRhs>>(module, "RestrictedRhs")
                .def("evaluate", &RestrictedRhs::evaluate, py::arg("mu"));
        py::class_<RestrictedOperator, std::shared_ptr<RestrictedOperator>>(module, "RestrictedOperator")
                .def("source_dofs", &RestrictedOperator::source_dofs)
                .def("assemble", &RestrictedOperator::assemble, py::arg("mu"));
    }

    template <typename VectorType>
    void bind_stationary_model(py::module_& module, const std::string& name)
    {
        using Model = StationaryModel<VectorType>;

        py::class_<Model, std::shared_ptr<Model>>(module, name.c_str())
                .def("solution_space", &Model::solution_space)
                .def("parameters", &Model::parameters)
                .def("system_operator", &Model::system_operator, py::arg("mu"))
                .def("system_operator_is_parametric", &Model::system_operator_is_parametric)
                .def("assemble_rhs", &Model::assemble_rhs, py::arg("mu"))
                // The restricted right-hand side and operator may refer to the model (e.g. its
                // mesh), so the model is kept alive as long as they are. None if the model does not
                // implement them.
                .def("restricted_rhs", &Model::restricted_rhs, py::arg("dofs"), py::keep_alive<0, 1>())
                .def("restricted_system_operator", &Model::restricted_system_operator, py::arg("dofs"),
                     py::keep_alive<0, 1>())
                .def("products", &Model::products)
                .def("write_vtu", &Model::write_vtu,
                     py::arg("directory"), py::arg("basename"), py::arg("fields"), py::arg("names"));
    }
}

#endif /* BINDINGS_DEALII_STATIONARY_MODEL_H */
