// Minimal counterpart to deal.II's tests/tests.h (which is not installed).
// C++ tests include it via "../tests.h" and write their results to the
// file "output", which the testsuite compares against <test>.output.

#ifndef DEALII_WITH_PYBIND_TESTS_H
#define DEALII_WITH_PYBIND_TESTS_H

#include <deal.II/base/logstream.h>

#include <fstream>
#include <iomanip>

using namespace dealii;

inline std::ofstream deallogfile;

// Attach deallog to the file "output" in the current (test run) directory.
inline void
initlog(const bool console = false)
{
  deallogfile.open("output");
  deallogfile << std::setprecision(9);
  deallog.attach(deallogfile);
  deallog.depth_console(console ? 10 : 0);
}

#endif // DEALII_WITH_PYBIND_TESTS_H
