# Recomended use (in source dir):

cmake -S . -B build

cmake --build build

cmake --install build


ctest --test-dir build

Runs only apps tests and outputs on failure

ctest --test-dir build -R apps --output-on-failure


# Project
## 1. CPP Side
Applications are written in cpp, they need to expose the correct features such that pyMOR bindings for intrusive ROMs are feasible.\
The interface is split into Vector, VectorSpace and Operator.
### 1.1. Implementations
Implementations are in the apps folder and are compiled as shared library. They are dependent on the dealii:dealii target. They are installed into the folder /lib.\

Vectors (serial, distributed or block) are implemented in dealii natively, all needed functionality is there.\
The VectorSpace concept is not native and thus in vector_space.h a abstract class is defined, it defines all functionality that is needed as virtual functions. A application has to either directly implement a class derived from VectorSpace or derived from PartitionedVectorSpace (later gives you some impementations for standard cases).\
Operators TODO\

### 1.2. Bindings
Bindings are written using pybind11 and can be found in python/bindings. The modules are linked and installed with a custom macro.\

Through the abstract Vectorspace class, both Vector and Vectorspace modules only need to be created once for a specific VectorType.\

There are some additional bindings in example.cc that only serve testing purposes.

## 2. Python Side
PyMOR gives the abstract classes CopyOnWriteVector and ComplexifiedListVectorSpace. With the functionality that the pybind11 modules expose, derived classes DealIIVector and DealIIVectorSpace can be easily implemented. These are the classes pyMOR wants to see and import dealii_pymor runs __init__.py which imports them.

Currently the python side is running on each rank. This means while the apps are distributing their work and are communicating over mpi, pyMOR recieves same information on each rank and does same computations on each rank. The scripts are started with mpirun -n N python sript.py this sets up MPI_COMM_WORLD such that when the app is called from each ranks identical pyMOR script it will know which rank is responsible for what part of the Simulation.

## 3. Ctest
