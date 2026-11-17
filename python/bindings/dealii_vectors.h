#ifndef BINDINGS_DEALII_VECTORS_H
#define BINDINGS_DEALII_VECTORS_H


#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include <memory>
#include <string>


namespace bindings
{
    namespace py = pybind11;

    template <typename VectorType>
    void bind_vector(py::module_& module, std::string const& name)
    {
        using value_type = typename VectorType::value_type;
        using size_type  = typename VectorType::size_type;

        // Element access takes global indices but only reaches the locally owned range; for a
        // distributed vector everything else lives on another rank (or is a possibly stale ghost).
        auto const check_index = [](const VectorType& self, size_type i)
        {
            if (i >= self.size())
                throw py::index_error("index " + std::to_string(i) + " out of range for size " +
                                      std::to_string(self.size()));
            if (!self.in_local_range(i))
                throw py::index_error("index " + std::to_string(i) +
                                      " is not locally owned on this rank");
        };

        py::class_<VectorType, std::shared_ptr<VectorType>>(module, name.c_str(), py::buffer_protocol())
                .def(py::init<>())
                .def(py::init<const VectorType&>())
                .def(py::init<const size_type>(), py::arg("size"))
                .def("swap", &VectorType::swap)
                .def(py::self * py::self)
                .def(py::self += py::self)
                .def(py::self -= py::self)
                .def(py::self *= value_type())
                .def(py::self /= value_type())
                //----core for pyMOR-----
                .def("copy", [](const VectorType& self)
                    { return std::make_shared<VectorType>(self); })
                .def("scal", [](VectorType& self, value_type a)
                    { self *= a; }, py::arg("a"))
                .def("axpy",
                    [](VectorType& self, value_type a, const VectorType& x)
                    { self.add(a, x); }, py::arg("a"), py::arg("x"))
                .def("inner",
                    [](const VectorType& self, const VectorType& x)
                    { return self * x; }, py::arg("x"))
                //----core for pyMOR-----
                .def("mean_value", &VectorType::mean_value)
                .def("norm_sqr", &VectorType::norm_sqr)
                .def("l1_norm", &VectorType::l1_norm)
                .def("l2_norm", &VectorType::l2_norm)
                .def("lp_norm", &VectorType::lp_norm, py::arg("p"))
                .def("linfty_norm", &VectorType::linfty_norm)
                .def("all_zero", &VectorType::all_zero)
                .def("size", &VectorType::size)
                .def("locally_owned_size", &VectorType::locally_owned_size)
                .def("__len__", &VectorType::size)
                .def("__repr__", [name](const VectorType& self)
                    { return "<dealii_core_module." + name + " of size " +
                    std::to_string(self.size()) + ">";})
                .def("__getitem__",
                    [check_index](const VectorType& self, size_type i)
                    { check_index(self, i); return self[i]; })
                .def("__setitem__",
                    [check_index](VectorType& self, size_type i, value_type v)
                    { check_index(self, i); self[i] = v; })
                // Exposes the locally owned entries only (ghosts excluded); for a serial vector
                // that is the whole vector.
                .def_buffer([](VectorType& self) -> py::buffer_info {
                    return py::buffer_info(
                        self.begin(),                               /* Pointer to buffer */
                        sizeof(value_type),                         /* Size of one scalar */
                        py::format_descriptor<value_type>::format(), /* Python struct-style format descriptor */
                        1,                                          /* Number of dimensions */
                        { self.locally_owned_size() },              /* Buffer dimensions */
                        { sizeof(value_type) }                      /* Strides (in bytes) */
                    );
                });
    }
}

#endif /* BINDINGS_DEALII_VECTORS_H */
