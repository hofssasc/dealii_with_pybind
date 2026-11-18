#include <pybind11/pybind11.h>
#include <implementations/code.h>
#include <interfaces/vector_space.h>

#include <deal.II/base/mpi.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

namespace py = pybind11;

using SerialVector      = dealii::Vector<double>;
using DistributedVector = dealii::LinearAlgebra::distributed::Vector<double>;

PYBIND11_MODULE(example, m, py::mod_gil_not_used()) {
    m.doc() = "pybind11 example plugin";

    // The VectorSpace classes are bound in dealii_core_module, spaces returned from here are
    // handed to Python as those.
    py::module_::import("dealii_core_module");

    // function using the costum library
    m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b"));

    // function using dealii
    m.def("make_uniform_space",
          [](dealii::types::global_dof_index n_dofs) -> std::shared_ptr<VectorSpace<DistributedVector>>
          {
              return std::make_shared<PartitionedVectorSpace<DistributedVector>>(
                  dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD, n_dofs),
                  MPI_COMM_WORLD);
          },
          "A space of n_dofs entries split evenly over the ranks of MPI_COMM_WORLD.",
          py::arg("n_dofs"));

    m.def("make_serial_space",
          [](dealii::types::global_dof_index n_dofs) -> std::shared_ptr<VectorSpace<SerialVector>>
          {
              return std::make_shared<PartitionedVectorSpace<SerialVector>>(dealii::complete_index_set(n_dofs),
                                                                             MPI_COMM_SELF);
          },
          "A space of n_dofs entries, all of them on this rank.",
          py::arg("n_dofs"));
}
