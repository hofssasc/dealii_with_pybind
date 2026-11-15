// Dummy test to check the C++ testsuite setup: links against the "code"
// library and compares its log against dummy.output.

#include "../tests.h"

#include "code.h"

int
main()
{
  initlog();

  deallog << "add(1.5, 2.25) = " << add(1.5, 2.25) << std::endl;
}
