#ifndef BINDINGS_DEALII_FUNCTIONAL_H
#define BINDINGS_DEALII_FUNCTIONAL_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interfaces/functional.h>

#include <memory>
#include <string>


namespace bindings
{
    namespace py = pybind11;

    template <typename VectorType>
    void bind_functional(py::module_& module, const std::string& name)
    {
        using F = Functional<VectorType>;

        py::class_<F, std::shared_ptr<F>>(module, name.c_str())
                .def("value", &F::value, py::arg("mu"), py::arg("u"))
                .def("gradient", &F::gradient, py::arg("mu"), py::arg("u"));
    }
}

#endif /* BINDINGS_DEALII_FUNCTIONAL_H */
