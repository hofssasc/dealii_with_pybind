#ifndef BINDINGS_DEALII_VECTOR_SPACE_H
#define BINDINGS_DEALII_VECTOR_SPACE_H


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <interfaces/vector_space.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>


namespace bindings
{
    namespace py = pybind11;

    // Conversions between the vectors of a VectorSpace and NumPy arrays over all of its entries.
    // They are collective: every rank calls them with the same arguments and gets the same result.
    namespace internal
    {
        template <typename VectorType>
        using Array = py::array_t<typename VectorType::value_type, py::array::c_style | py::array::forcecast>;

        using IndexArray = py::array_t<dealii::types::global_dof_index, py::array::c_style | py::array::forcecast>;

        // Agreed on by all ranks, so that either all of them throw or none enters the collective
        // operation that follows.
        template <typename VectorType>
        void check_compatible(const VectorSpace<VectorType>& space, const VectorType& vector)
        {
            const bool compatible_here = vector.size() == space.n_dofs() &&
                vector.locally_owned_size() == space.locally_owned_dofs().n_elements();

            if (dealii::Utilities::MPI::logical_or(!compatible_here, space.communicator()))
                throw py::value_error("The vector does not belong to this space.");
        }

        template <typename VectorType>
        Array<VectorType> to_numpy(const VectorSpace<VectorType>& space, const VectorType& vector)
        {
            check_compatible(space, vector);

            Array<VectorType> result(static_cast<py::ssize_t>(space.n_dofs()));
            dealii::ArrayView<typename VectorType::value_type> values(result.mutable_data(), result.size());

            // Every rank fills in what it owns, the sum over ranks assembles the whole vector.
            std::fill(values.begin(), values.end(), 0);
            for (const auto i : space.locally_owned_dofs())
                values[i] = vector(i);
            dealii::Utilities::MPI::sum<typename VectorType::value_type>(values, space.communicator(), values);

            return result;
        }

        template <typename VectorType>
        std::shared_ptr<VectorType> from_numpy(const VectorSpace<VectorType>& space, const Array<VectorType>& data)
        {
            if (data.ndim() != 1 || static_cast<dealii::types::global_dof_index>(data.shape(0)) != space.n_dofs())
                throw py::value_error("Expected a one-dimensional array of length " +
                                      std::to_string(space.n_dofs()) + ".");

            // All ranks hold the whole array, each keeps the entries it owns.
            auto vector = space.create_vector();
            const auto values = data.template unchecked<1>();
            for (const auto i : space.locally_owned_dofs())
                (*vector)(i) = values(i);

            return vector;
        }

        template <typename VectorType>
        Array<VectorType> dofs(const VectorSpace<VectorType>& space, const VectorType& vector, const IndexArray& indices)
        {
            check_compatible(space, vector);

            const auto requested = indices.template unchecked<1>();
            Array<VectorType> result(requested.shape(0));
            dealii::ArrayView<typename VectorType::value_type> values(result.mutable_data(), result.size());

            std::fill(values.begin(), values.end(), 0);
            for (py::ssize_t k = 0; k < requested.shape(0); ++k)
            {
                const auto i = requested(k);
                if (i >= space.n_dofs())
                    throw py::index_error("index " + std::to_string(i) + " out of range for dimension " +
                                          std::to_string(space.n_dofs()));
                if (space.locally_owned_dofs().is_element(i))
                    values[k] = vector(i);
            }
            dealii::Utilities::MPI::sum<typename VectorType::value_type>(values, space.communicator(), values);

            return result;
        }

        // Index and magnitude of the entry with the largest magnitude. Ties go to the smallest
        // index, as with numpy.argmax.
        template <typename VectorType>
        std::pair<dealii::types::global_dof_index, typename VectorType::real_type>
        amax(const VectorSpace<VectorType>& space, const VectorType& vector)
        {
            check_compatible(space, vector);

            // Starts below every magnitude, so ranks that own nothing never win.
            typename VectorType::real_type local_max = -1;
            dealii::types::global_dof_index local_index = dealii::numbers::invalid_dof_index;
            for (const auto i : space.locally_owned_dofs())
                if (std::abs(vector(i)) > local_max)
                {
                    local_max = std::abs(vector(i));
                    local_index = i;
                }

            const auto global_max = dealii::Utilities::MPI::max(local_max, space.communicator());
            const auto global_index = dealii::Utilities::MPI::min(
                local_max == global_max ? local_index : dealii::numbers::invalid_dof_index,
                space.communicator());

            return {global_index, global_max};
        }
    }

    template <typename VectorType>
    void bind_vector_space(py::module_& module, const std::string& name)
    {
        using Space = VectorSpace<VectorType>;

        py::class_<Space, std::shared_ptr<Space>>(module, name.c_str())
                .def_property_readonly("dim", &Space::n_dofs)
                .def("create_vector", &Space::create_vector)
                .def("make_admissible", &Space::make_admissible, py::arg("vector"))
                .def("to_numpy", &internal::to_numpy<VectorType>, py::arg("vector"))
                .def("from_numpy", &internal::from_numpy<VectorType>, py::arg("data"))
                .def("dofs", &internal::dofs<VectorType>, py::arg("vector"), py::arg("indices"))
                .def("amax", &internal::amax<VectorType>, py::arg("vector"))
                // A space is its C++ object: two spaces of the same dimension can still distribute
                // their entries differently.
                .def("__eq__", [](const Space& self, const Space& other)
                    { return &self == &other; }, py::is_operator())
                .def("__hash__", [](const Space& self)
                    { return std::hash<const Space*>{}(&self); })
                .def("__repr__", [name](const Space& self)
                    { return "<dealii_core_module." + name + " of dimension " +
                    std::to_string(self.n_dofs()) + ">";});
    }
}

#endif /* BINDINGS_DEALII_VECTOR_SPACE_H */
