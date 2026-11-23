#ifndef BINDINGS_DEALII_OPERATOR_H
#define BINDINGS_DEALII_OPERATOR_H


#include <pybind11/pybind11.h>

#include <interfaces/operator.h>

#include <memory>
#include <string>


namespace bindings
{
    namespace py = pybind11;

    template <typename VectorType>
    void bind_operator(py::module_& module, const std::string& name)
    {
        using Op = Operator<VectorType>;

        py::class_<Op, std::shared_ptr<Op>>(module, name.c_str())
                .def("source", &Op::source)
                .def("range", &Op::range)
                .def("apply", &Op::apply, py::arg("dst"), py::arg("src"))
                .def("has_inverse", &Op::has_inverse)
                .def("apply_inverse", &Op::apply_inverse, py::arg("dst"), py::arg("rhs"))
                .def("has_adjoint", &Op::has_adjoint)
                .def("apply_adjoint", &Op::apply_adjoint, py::arg("dst"), py::arg("src"))
                .def("apply_inverse_adjoint", &Op::apply_inverse_adjoint, py::arg("dst"), py::arg("rhs"));
    }
}

#endif /* BINDINGS_DEALII_OPERATOR_H */
