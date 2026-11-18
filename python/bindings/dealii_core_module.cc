#include <pybind11/pybind11.h>

#include <deal.II/base/mpi.h>

// Functions to register a dealii vector class, its VectorSpace, Operator and StationaryModel
// with pybind11
#include "dealii_vectors.h"
#include "dealii_vector_space.h"
#include "dealii_operator.h"
#include "dealii_stationary_model.h"
// Currently registered vector types
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace py = pybind11;


namespace
{
#ifdef DEAL_II_WITH_MPI
    void finalize_mpi()
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
            MPI_Finalize();
    }
#endif

    // Every rank runs the same script (SPMD), so MPI has to be initialized before the first
    // collective operation. Something else may have done it already, e.g. mpi4py.
    //
    // Finalizing has to wait until Python has destroyed the deal.II objects it holds: a
    // parallel::distributed::Triangulation, for instance, frees its communicator in its destructor.
    // Python's atexit handlers run before that, Py_AtExit() functions at the very end of the
    // interpreter shutdown. mpi4py must therefore not finalize itself (dealii_pymor makes sure of
    // that for pyMOR, which otherwise asks mpi4py to).
    void ensure_mpi_initialized()
    {
#ifdef DEAL_II_WITH_MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int provided = 0;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
        }

        Py_AtExit(&finalize_mpi);
#endif
    }
}


PYBIND11_MODULE(dealii_core_module, m, py::mod_gil_not_used()) {
    m.doc() = "core bindings to use the dealii library with pymor";

    ensure_mpi_initialized();

    m.def("this_mpi_process", []() { return dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD); },
          "Rank of this process in MPI_COMM_WORLD.");
    m.def("n_mpi_processes", []() { return dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); },
          "Number of processes in MPI_COMM_WORLD.");

    bindings::bind_vector<dealii::Vector<double>>(m, "Vector");
    bindings::bind_vector<dealii::LinearAlgebra::distributed::Vector<double>>(m, "DistributedVector");

    bindings::bind_vector_space<dealii::Vector<double>>(m, "VectorSpace");
    bindings::bind_vector_space<dealii::LinearAlgebra::distributed::Vector<double>>(m, "DistributedVectorSpace");

    bindings::bind_operator<dealii::Vector<double>>(m, "Operator");
    bindings::bind_operator<dealii::LinearAlgebra::distributed::Vector<double>>(m, "DistributedOperator");

    bindings::bind_stationary_model<dealii::Vector<double>>(m, "StationaryModel");
    bindings::bind_stationary_model<dealii::LinearAlgebra::distributed::Vector<double>>(m, "DistributedStationaryModel");
}
