# deal.II applications for pyMOR

This project makes deal.II simulations usable for model order reduction with
[pyMOR](https://pymor.org). The simulation stays in C++: assembly, solvers and
MPI parallelism are deal.II's. pyMOR runs in Python and only sees vectors,
operators and models that it can add, scale, apply and invert. pybind11
connects the two.

The design goal is that a new application only implements a few small C++
interfaces. Everything between those interfaces and pyMOR is written once and
shared by all applications.


## 1. The big picture

```
  pyMOR script               mpirun -n N python script.py
       |                     pyMOR's VectorArray, Operator, StationaryModel
  dealii_pymor               python/dealii_pymor         pure Python, once
       |                     adapts the bound C++ interfaces to pyMOR's classes
  dealii_core_module         python/bindings             pybind11, once per vector type
       |                     makes the C++ interfaces callable from Python
  interfaces                 apps/include/interfaces     abstract C++ classes
       |                     VectorSpace, Operator, StationaryModel
  implementations            apps/include/implementations, apps/source
       |                     building blocks and applications (libapplications.so)
  deal.II, ExaDG
```

Read from the bottom up:

1. An **application** is a C++ class that implements the abstract
   `StationaryModel` interface, e.g. `GaussianSource`. It hands out its vector
   space and its operators through the `VectorSpace` and `Operator`
   interfaces.
2. **`dealii_core_module`** binds these three interfaces to Python, once for
   each deal.II vector type. A new application needs no new bindings for
   them. It only needs a small module with a `make_model()` function that
   creates the application.
3. **`dealii_pymor`** wraps the bound objects into the classes pyMOR expects.
   `stationary_model()` turns a C++ model into a pyMOR `StationaryModel`.
4. The **pyMOR script** then uses it like any other pyMOR model: solve,
   build a reduced basis, reduce.

The same four concepts appear on every layer:

| concept      | C++ (`apps/include`)                | Python binding (`dealii_core_module`)     | pyMOR adapter (`dealii_pymor`)       | pyMOR base class                |
|--------------|-------------------------------------|-------------------------------------------|--------------------------------------|---------------------------------|
| vector       | a deal.II vector, e.g. `LinearAlgebra::distributed::Vector<double>` | `Vector`, `DistributedVector` | `DealIIVector`                       | `CopyOnWriteVector`             |
| vector space | `VectorSpace<VectorType>`           | `VectorSpace`, `DistributedVectorSpace`   | `DealIIVectorSpace`                  | `ListVectorSpace`               |
| operator     | `Operator<VectorType>`              | `Operator`, `DistributedOperator`         | `DealIIOperator`, `DealIISolver`     | `ListVectorArrayOperatorBase`   |
| model        | `StationaryModel<VectorType>`       | `StationaryModel`, `DistributedStationaryModel` | `stationary_model()`, `DealIIRhsOperator`, `DealIIVisualizer` | `StationaryModel` |
| restricted right-hand side | `RestrictedRhs`           | `RestrictedRhs`                           | `DealIIRestrictedRhsOperator`        | `Operator` (from `restricted()`) |

pyMOR's *list vector arrays* are lists of single vectors. Each entry here is
one deal.II vector, and pyMOR's linear algebra on the array calls deal.II's
vector operations entry by entry.

### What happens in `fom.solve(mu)`

Following one call through all layers shows how they fit together:

1. pyMOR's `StationaryModel` asks its right-hand side for `f(mu)`.
   `DealIIRhsOperator` calls the C++ `assemble_rhs(mu)`, which assembles a new
   deal.II vector. pyMOR's parameter values `mu` arrive in C++ as
   `ParameterValues`, a map from names to `std::vector<double>`.
2. pyMOR calls `apply_inverse` of its system operator. `DealIIOperator` passes
   it to its `DealIISolver`, which calls the C++ `Operator::apply_inverse`. For
   `GaussianSource`, that is CG with algebraic multigrid on a Trilinos matrix.
3. The result is a deal.II vector that is wrapped as a `DealIIVector`. It
   never leaves C++ memory and stays distributed over the MPI ranks.

A reduced basis method only adds operations of the same kind: inner products,
linear combinations and `apply`. Reduced matrices and vectors are small NumPy
arrays. Every rank holds the same copy of them (see
[section 5](#5-parallel-execution-spmd)).


## 2. C++ interfaces (`apps/include/interfaces`)

Each interface is a class template in the vector type. Most applications
override only the pure virtual functions.

### `VectorSpace<VectorType>` (`vector_space.h`)

A vector space describes the vectors of one discretization: their size, how
they are distributed over the MPI ranks, and which constraints they satisfy.
deal.II vectors do the arithmetic themselves. The space adds what the Python
layer needs in addition:

| member                     | meaning |
|----------------------------|---------|
| `n_dofs()`                 | dimension of the space |
| `communicator()`, `locally_owned_dofs()` | the parallel distribution; the bindings use it to gather vectors into NumPy arrays and to pick single entries |
| `create_vector()`          | a new zero vector of the space |
| `make_admissible(vector)`  | project a vector onto the space, e.g. set constrained entries to zero; a no-op by default |

Two implementations cover the usual cases:

- **`PartitionedVectorSpace`** is built from the locally owned (and
  optionally ghost) dofs, or from an existing deal.II `Partitioner`, e.g. the
  one of a `MatrixFree` object, whose operators require vectors with exactly
  that partitioner. All its vectors share the partitioner, so creating a
  vector needs no MPI communication. It supports `dealii::Vector` (serial) and
  `LinearAlgebra::distributed::Vector`.
- **`ConstrainedVectorSpace`** adds `AffineConstraints`. Vectors are stored
  *condensed*: constrained entries (Dirichlet values, hanging nodes) are zero.
  `make_admissible` restores that, for example after a vector was created
  from NumPy data. `constraints.distribute()` is applied only for output.

Spaces are compared by object identity. Two vectors are compatible if they
come from the same space object.

### `Operator<VectorType>` (`operator.h`)

An operator is a linear map between two vector spaces: a system matrix, but
also an inner product (pyMOR "product") such as the H¹ seminorm.

| member                     | meaning |
|----------------------------|---------|
| `source()`, `range()`      | the spaces; must return the same object on every call |
| `apply(dst, src)`          | `dst = A src` |
| `has_inverse()`            | whether `apply_inverse` is implemented; false by default |
| `apply_inverse(dst, rhs)`  | solve `A dst = rhs`; `dst` holds the initial guess on entry, as in deal.II's solvers |

### `StationaryModel<VectorType>` (`stationary_model.h`)

A model collects the parts of a parameterized stationary problem

    A u(mu) = f(mu)

with a fixed system operator `A` and a right-hand side that may depend on the
parameters in any way. Each application implements this once.

| member                         | meaning |
|--------------------------------|---------|
| `solution_space()`             | the space of `u` |
| `parameters()`                 | parameter names and their number of components, e.g. `{"center": 2}` |
| `system_operator()`            | `A`; must have an inverse |
| `assemble_rhs(mu)`             | `f(mu)` as a new vector |
| `restricted_rhs(dofs)`         | `f(mu)` at a few dofs only, for empirical interpolation (below); `nullptr` by default |
| `products()`                   | named inner products; pyMOR adds `<name>_product` and `<name>_norm` to the model; none by default |
| `write_vtu(directory, basename, fields, names)` | write vectors for ParaView; used by pyMOR's `visualize`; throws by default |

### Empirical interpolation of the right-hand side

pyMOR cannot see how `f` depends on `mu`. Without help, every reduced solve
assembles the full `f(mu)` and projects it, and pyMOR logs a warning
("inefficient generic projection"). This is correct, but costs an assembly on
the full mesh per reduced solve.

Empirical interpolation (EI) removes this cost:

- **offline**, `ei_greedy` (or `deim`) picks M interpolation dofs and M
  collateral basis vectors from snapshots of `f(mu)`. `f(mu)` is then
  approximated by the combination of basis vectors that matches it at the
  interpolation dofs;
- **online**, only the M entries of `f(mu)` at the interpolation dofs are
  computed. Nothing depends on the size of the mesh any more.

`restricted_rhs(dofs)` provides these entries. It returns a `RestrictedRhs`,
set up once for the dofs, whose `evaluate(mu)` computes `f(mu)` at them, e.g.
by integrating only over the cells around them. This is pyMOR's
`Operator.restricted()` on the C++ side: `DealIIRhsOperator.restricted()`
calls it, and pyMOR's `EmpiricalInterpolatedOperator` uses the result. A model
that returns `nullptr` can still be interpolated, but every evaluation then
assembles the full `f(mu)`.

```python
from pymor.algorithms.ei import ei_greedy
from pymor.operators.ei import EmpiricalInterpolatedOperator

dofs, collateral_basis, _ = ei_greedy(rhs_snapshots, max_interpolation_dofs=64)
rhs_ei = EmpiricalInterpolatedOperator(fom.rhs, dofs, collateral_basis, triangular=True)
reductor = dealii_pymor.DealIIStationaryRBReductor(fom.with_(rhs=rhs_ei), RB=RB, product=h1)
```

Reduce with `DealIIStationaryRBReductor` instead of pyMOR's
`StationaryRBReductor`. pyMOR's reductor projects the right-hand side, an
operator from R¹, without a basis for R¹. pyMOR's efficient projection of an
interpolated operator needs one, so it falls back to a generic projection that
works in the full space again. `DealIIStationaryRBReductor` projects with the
identity of R¹ instead.

## 3. Implementations (`apps/include/implementations`, `apps/source`)

These are compiled into the shared library `libapplications.so`:

- **`TrilinosMatrixOperator`**: a building block. It is a symmetric positive
  definite `TrilinosWrappers::SparseMatrix` as an `Operator`, inverted by CG
  with an algebraic multigrid preconditioner (set up once, in the
  constructor). CG logs its iteration count to `deallog`, which stays silent
  unless a stream is attached, as in the tests.
- **`GaussianSource<dim>`**: the template application. It solves Poisson's
  equation `-Δu = exp(-|x - center|² / width²)` on the unit square, with
  `u = 0` on the boundary and the parameter `center`. It uses a distributed
  mesh (`parallel::distributed::Triangulation`), Q_k elements, a
  `ConstrainedVectorSpace` and a `TrilinosMatrixOperator`, and has the product
  `h1_0_semi`. Its restricted right-hand side, `GaussianSourceRestrictedRhs`,
  integrates the source only over the locally owned cells around the given
  dofs and sums over the ranks. It requires a mesh without hanging nodes.
- **`ExaDGPoissonOperator<dim>`**: a building block. It wraps the
  `Poisson::Operator` of ExaDG as an `Operator`: `apply` is its matrix-free
  `vmult`, and `apply_inverse` is its `solve` (CG with the preconditioner set
  in its parameters). Like `TrilinosMatrixOperator`, it logs the iteration
  count to `deallog`.
- **`GaussianSourceMatrixFree<dim>`**: the same problem with ExaDG's
  matrix-free solver, CG preconditioned by polynomial and geometric multigrid.
  The discretization is the same (continuous Q_k elements, same mesh), so both
  models have the same discrete solution up to the solver tolerance. Only the
  operator and the solver differ. ExaDG has no restricted assembly, so it uses
  `GaussianSourceRestrictedRhs` on ExaDG's `DoFHandler` (same quadrature).
- `code.h`: a placeholder function for the testsuite setup.

### Which part of ExaDG is used

ExaDG's Poisson module has three layers:

- the `Driver`, which sets up, solves once and post-processes;
- the `ApplicationBase`, which describes the problem and reads its parameters
  from a file;
- the PDE operator `Poisson::Operator`.

Here pyMOR takes the place of the driver, and `make_model(...)` takes the
place of the parameter file. Only `Poisson::Operator` is used. It maps directly
onto the interfaces:

| interface | `Poisson::Operator` |
|-----------|---------------------|
| `Operator::apply`, `apply_inverse` | `vmult`, `solve` |
| `assemble_rhs(mu)` | `rhs`, after setting the parameter on the right-hand side function |
| `VectorSpace` | the vector partitioner of `get_matrix_free()` |
| `restricted_rhs(dofs)` | none, `GaussianSourceRestrictedRhs` on `get_dof_handler()` |

`GaussianSourceMatrixFree` builds the operator's inputs directly in its
constructor, with ExaDG's helpers in `GridUtilities`: `Parameters`, grid,
mapping, boundary descriptor and field functions. Two things to keep in mind:

- `Poisson::Operator` keeps a reference to its `Parameters`, so the model
  stores them.
- ExaDG writes setup information to `std::cout` on rank 0.

### Adding an application

1. Implement `StationaryModel<VectorType>` in
   `apps/include/implementations/<name>.h` and `apps/source/<name>.cc`. Add
   both to `apps/CMakeLists.txt`.
2. Add a module `python/bindings/<name>.cc` with a `make_model()` function
   (copy `gaussian_source.cc`), and register it in `python/CMakeLists.txt`
   with `dealii_pybind_add_module(<name> SOURCES ... LINK applications)`.
3. In Python, call `dealii_pymor.stationary_model(<name>.make_model(...))`.

New bindings for vector spaces, operators or models are only needed for a new
vector type (see below).


## 4. Python side

### Bindings (`python/bindings`)

| file | content |
|------|---------|
| `dealii_vectors.h`, `dealii_vector_space.h`, `dealii_operator.h`, `dealii_stationary_model.h` | one template function `bind_<interface><VectorType>(module, name)` each |
| `dealii_core_module.cc` | the module `dealii_core_module`: calls these for `dealii::Vector<double>` (`Vector`, `VectorSpace`, ...) and `LinearAlgebra::distributed::Vector<double>` (`DistributedVector`, ...). It also initializes MPI and provides `this_mpi_process()` and `n_mpi_processes()`. |
| `gaussian_source.cc` | the application module `gaussian_source`, with `make_model(n_refinements, degree, width)` and `make_matrix_free_model(...)` |
| `example.cc` | helpers for the tests |

Application modules import `dealii_core_module` first, so the classes they
return are already known to Python. Supporting a new vector type means adding
one `bind_...` call per interface in `dealii_core_module.cc`, and a `reinit`
overload in `PartitionedVectorSpace` if needed.

All objects are held by `std::shared_ptr`. C++ returns
`std::shared_ptr<const T>` where the caller must not modify an object. Python
does not enforce `const`, so it only documents intent.

### pyMOR adapter (`python/dealii_pymor`)

| module           | classes |
|------------------|---------|
| `vectorarray.py` | `DealIIVector` (one deal.II vector), `DealIIVectorSpace` (a bound `VectorSpace`) |
| `operators.py`   | `DealIIOperator` (a bound `Operator`), `DealIISolver` (its `apply_inverse`), `DealIIRhsOperator` (`assemble_rhs` as a pyMOR operator), `DealIIRestrictedRhsOperator` (a bound `RestrictedRhs`, from `DealIIRhsOperator.restricted()`) |
| `models.py`      | `stationary_model()` (builds pyMOR's `StationaryModel`), `DealIIVisualizer` (`write_vtu`) |
| `reductors.py`   | `DealIIStationaryRBReductor` (pyMOR's `StationaryRBReductor`, efficient for an interpolated right-hand side) |

Import `dealii_pymor` before `pymor` (see [section 5](#5-parallel-execution-spmd)):

```python
import dealii_pymor  # first
import gaussian_source

fom = dealii_pymor.stationary_model(gaussian_source.make_model(n_refinements=5))
u = fom.solve(fom.parameters.parse([0.3, 0.6]))
fom.visualize(u, filename='solution')
```

`tests/pymor/gaussian_source_rb.py` is a complete example with POD and
reduction, `tests/pymor/gaussian_source_ei.py` adds empirical interpolation.


## 5. Parallel execution (SPMD)

Scripts run as `mpirun -n N python script.py`. Every rank runs the whole
script. The deal.II objects are distributed: each rank owns part of every
vector and of the mesh. Everything pyMOR computes from them (inner products,
norms, reduced matrices) comes from collective reductions, so it is identical
on all ranks. pyMOR therefore runs unchanged and in step on every rank.

Rules for scripts and adapters:

- Every rank has to make the same calls. Never put a call that communicates
  (solve, norm, inner product, `to_numpy`, ...) under `if rank == 0`.
- Print or write files only on rank 0 (`dealii_core_module.this_mpi_process()`).
  `write_vtu` handles this itself.
- Random data needs the same seed on every rank. pyMOR's default does this.

pyMOR has its own MPI mode, the opposite one: rank 0 runs the script and the
other ranks wait for commands. `import pymor` starts it automatically under
`mpirun`. `dealii_pymor` switches it off (`PYMOR_CONFIG_DISABLE=MPI`), which
is why it has to be imported first.

`dealii_core_module` initializes MPI when it is imported. It finalizes MPI at
the very end of the interpreter shutdown, after Python has destroyed all
deal.II objects, because some of them (e.g. a distributed triangulation) still
use MPI in their destructors. `dealii_pymor` therefore stops `mpi4py` from
finalizing earlier.


## 6. Project structure and CMake

```
CMakeLists.txt              top level: dependencies, then apps, python, tests
cmake/
  DealiiPybindModule.cmake  dealii_pybind_add_module(): add, link and install a Python module
  PybindPickupTests.cmake   pybind_pickup_tests(): Python tests with deal.II's test conventions
  run_python_test.cmake     runs one Python test (used by the above)
extern/pybind11/            pybind11, git submodule
apps/                       library "applications" (libapplications.so)
  include/interfaces/       abstract interfaces
  include/implementations/  building blocks and applications
  source/                   their .cc files
python/
  bindings/                 pybind11 modules
  dealii_pymor/             pyMOR adapter package
  scripts/                  Python scripts installed with the modules
tests/
  tests.h                   initlog() and mpi_initlog() for C++ tests
  apps/                     C++ tests of the library
  python/                   tests of the bindings
  pymor/                    tests of the pyMOR adapter (only if pyMOR is found)
```

### Configure, build, install

```sh
git submodule update --init       # pybind11

cmake -S . -B build \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DEXADG_DIR=/path/to/exadg/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE=$(which python)
cmake --build build
cmake --install build              # into <source>/install unless CMAKE_INSTALL_PREFIX is set
```

- **`DEAL_II_DIR`, `EXADG_DIR`**: where to find deal.II and ExaDG. deal.II
  has to be found before `project()`, because it sets the compiler and its
  flags.
- **`CMAKE_BUILD_TYPE`**: `Debug` or `Release`. It has to match the build type
  of ExaDG. Otherwise the debug and release libraries of deal.II would be
  loaded into the same process, and configuring stops with an error.
- **`Python_EXECUTABLE`**: the modules are built for this interpreter. Use the
  environment that has pyMOR installed, otherwise the pyMOR tests are
  skipped.

Top-level `CMakeLists.txt`, in order: find deal.II, `project()`, load
deal.II's targets, find ExaDG and check the build type, add pybind11, include
`DealiiPybindModule`, then `add_subdirectory` for `apps`, `python` and
`tests`.

### Targets

- **`applications`** (`apps/CMakeLists.txt`): a shared library linked
  against `dealii::dealii` and `EXADG::exadg`. Its headers are a `FILE_SET`, so targets that link
  it get the include path `apps/include`. Includes are written
  `#include <interfaces/operator.h>`.
- **Python modules** (`python/CMakeLists.txt`): each is added with

  ```cmake
  dealii_pybind_add_module(<name>
    SOURCES bindings/<name>.cc
    LINK    applications          # and/or dealii::dealii, EXADG::exadg
    )
  ```

  This builds the module with `pybind11_add_module()`, links the targets,
  sets the RPATH, installs it to `python/`, and registers it for the
  testsuite.

### Install layout

```
<prefix>/
  include/interfaces/, include/implementations/   headers
  lib/libapplications.so
  python/dealii_core_module.*.so, gaussian_source.*.so, ...
  python/dealii_pymor/                            pyMOR adapter
```

The modules find `libapplications.so` through their RPATH `$ORIGIN/../lib`,
and deal.II, Trilinos and MPI through the paths they were linked against.
To use an installation, put `<prefix>/python` on `PYTHONPATH`.


## 7. Tests

```sh
ctest --test-dir build                       # all
ctest --test-dir build -R apps --output-on-failure
ctest --test-dir build -L pymor              # by label: python (all Python tests) or pymor
ctest --test-dir build -E mpirun             # serial tests only
```

All test directories follow deal.II's conventions. A test is a source file
plus the output it has to produce:

| file                          | meaning |
|-------------------------------|---------|
| `<name>.cc` / `<name>.py`     | the test |
| `<name>.output`               | expected output of a serial run |
| `<name>.mpirun=N.output`      | expected output with N ranks; the same test can have several |
| `<name>.run_only`             | the test only has to exit successfully |

Outputs are compared with `numdiff`, using deal.II's tolerances.

- **`tests/apps`** (C++, `deal_ii_pickup_tests()`): every test is linked
  against `applications`. A test writes to `deallog` after `initlog()`, or
  `mpi_initlog()` if it also runs in parallel; both are in `tests/tests.h`.
- **`tests/python`**, **`tests/pymor`** (`pybind_pickup_tests()`): the output
  is everything the script prints. The modules from the build tree are on
  `PYTHONPATH`, and for `tests/pymor` also `python/` from the source tree, so
  `dealii_pymor` is tested without installing it.

| test                          | checks |
|-------------------------------|--------|
| `apps/gaussian_source`        | Q1–Q3 on refined meshes: convergence order of the energy ‖∇u‖² (2k for Q_k), CG iterations about constant under refinement (AMG works); on 2 ranks the same dofs and energies |
| `python/vector`               | the bound deal.II vectors |
| `python/vector_space`         | the bound `VectorSpace` on a distributed space: NumPy conversion, single entries, `amax` |
| `pymor/vectorarray`           | pyMOR's vector array operations on deal.II vectors, compared with pyMOR's NumPy arrays |
| `apps/gaussian_source_matrix_free` | the same for `GaussianSourceMatrixFree`: dofs and energies as in `apps/gaussian_source`, multigrid iterations |
| `pymor/gaussian_source_rb`    | full-order solve, visualization, POD, and reduced basis errors |
| `pymor/gaussian_source_matrix_free_rb` | the same for the matrix-free model (reuses `run()` of `gaussian_source_rb.py`) |
| `apps/gaussian_source_restricted_rhs` | the restricted right-hand side of both models equals the assembled one at all interior dofs, Q1–Q3, also on 2 ranks |
| `pymor/gaussian_source_ei` | reduced basis with an interpolated right-hand side: errors for 16, 32, 64 interpolation dofs, efficient projection |

**`mpirun` tests:** they are started with deal.II's `DEAL_II_MPIEXEC`. If
deal.II was configured with the `mpiexec` of a different MPI than the one it
links (e.g. ParaView's MPICH instead of the system's Open MPI), every rank
starts as a separate serial run and these tests fail. To fix this, reconfigure
deal.II with `-DMPIEXEC_EXECUTABLE=/usr/bin/mpiexec`. Until then, run them by
hand:

```sh
mpirun -n 2 build/tests/apps/gaussian_source.debug/gaussian_source.debug   # writes ./output
PYTHONPATH=build/python:python mpirun -n 2 $(which python) tests/pymor/gaussian_source_rb.py
```


## 8. Next steps

- System operators that depend on the parameters in a non-affine way, or
  nonlinear ones. They need a parameter-dependent operator interface. Its
  restriction will have the same shape as `RestrictedRhs`, plus the source
  dofs the evaluation reads.
- `GaussianSourceRestrictedRhs` on adaptive meshes: with hanging nodes, it
  also needs the cells of the dofs that are constrained to an interpolation
  dof.
- The projection problem behind `DealIIStationaryRBReductor` could be
  reported to pyMOR: its rule could treat a missing source basis of R¹ as the
  identity.
