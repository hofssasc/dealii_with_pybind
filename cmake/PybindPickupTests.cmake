# pybind_pickup_tests([category])
#
# Python counterpart to deal_ii_pickup_tests(). It picks up all python tests
# in the current source directory with the same file conventions, and uses
# deal.II's run_test.sh, so running and diffing (numdiff tolerances,
# comparison file variants) behave the same as for the C++ tests.
#
# A test is a pair of
#   <name>.py                          the test script
#   <name>[.mpirun=N][.threads=N].output   compared against the test output
#   or <name>[...].run_only            the script only has to exit with 0
#
# The test's output is everything the script prints to stdout/stderr, or the
# file "output" if the script writes one into its working directory.
# Several comparison files for the same script define several tests, e.g.
# foo.output and foo.mpirun=2.output.
#
# Test names: <category>/<name>[.mpirun=N][.threads=N][.<build>], so
#   ctest -R python/vector    or    ctest -L python
#
# Variables read (same names/semantics as deal.II where they overlap):
#   TEST_PYTHON_MODULES  - module targets (pybind11 extension libraries) the
#                          tests import. Their directories are prepended to
#                          PYTHONPATH and a setup test builds them first.
#   TEST_PYTHON_PATH     - further directories prepended to PYTHONPATH, e.g.
#                          those of pure python packages in the source tree
#   TEST_PICKUP_REGEX    - only set up tests whose "<category>/<name>"
#                          matches (also read from the environment)
#   TEST_TIME_LIMIT      - per-test timeout in seconds (default 600)
#   TEST_MPI_RANK_LIMIT  - skip .mpirun=N tests with N above this (0: no limit)
#   TEST_THREAD_LIMIT    - skip .threads=N tests with N above this (0: no limit)
#   NUMDIFF_DIR          - hint for finding numdiff
#
# The category defaults to the name of the current source directory.

# Location of run_python_test.cmake; captured at include time because
# CMAKE_CURRENT_LIST_DIR points to the caller inside the function.
set(_PYBIND_PICKUP_TESTS_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(_pybind_default _variable)
  if("${${_variable}}" STREQUAL "")
    set(${_variable} "$ENV{${_variable}}")
  endif()
  if("${${_variable}}" STREQUAL "")
    set(${_variable} "${ARGN}")
  endif()
  set(${_variable} "${${_variable}}" PARENT_SCOPE)
endfunction()

function(pybind_pickup_tests)
  if(ARGC GREATER 0)
    set(_category "${ARGV0}")
  else()
    get_filename_component(_category "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  endif()

  #
  # Tools:
  #
  if(NOT Python_EXECUTABLE)
    find_package(Python REQUIRED COMPONENTS Interpreter)
  endif()

  find_program(BASH NAMES bash)
  mark_as_advanced(BASH)
  if(NOT BASH)
    message(FATAL_ERROR "pybind_pickup_tests: bash is required.")
  endif()

  find_program(NUMDIFF_EXECUTABLE NAMES numdiff
    HINTS ${NUMDIFF_DIR} $ENV{NUMDIFF_DIR} PATH_SUFFIXES bin)
  mark_as_advanced(NUMDIFF_EXECUTABLE)

  set(_run_test_sh "${DEAL_II_PATH}/${DEAL_II_SHARE_RELDIR}/scripts/run_test.sh")
  if(NOT EXISTS "${_run_test_sh}")
    message(FATAL_ERROR
      "pybind_pickup_tests: deal.II's run_test.sh not found at\n"
      "  ${_run_test_sh}\n"
      "Call find_package(deal.II) before using this function.")
  endif()

  set(_runner "${_PYBIND_PICKUP_TESTS_DIR}/run_python_test.cmake")

  _pybind_default(TEST_PICKUP_REGEX "")
  _pybind_default(TEST_TIME_LIMIT 600)
  _pybind_default(TEST_MPI_RANK_LIMIT 0)
  _pybind_default(TEST_THREAD_LIMIT 0)

  if(CMAKE_BUILD_TYPE)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_suffix)
    set(_build_suffix ".${_build_suffix}")
  else()
    set(_build_suffix "")
  endif()

  #
  # Setup test: ctest does not build anything itself, so build the python
  # modules before any test of this category runs (like deal.II does for
  # its test executables).
  #
  set(_fixture "")
  if(TEST_PYTHON_MODULES)
    foreach(_module ${TEST_PYTHON_MODULES})
      if(NOT TARGET ${_module})
        message(FATAL_ERROR
          "pybind_pickup_tests: TEST_PYTHON_MODULES entry \"${_module}\" "
          "is not a target.")
      endif()
    endforeach()

    set(_fixture "${_category}.python_modules")
    add_test(NAME ${_category}/build_python_modules
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
              --config $<CONFIG> --target ${TEST_PYTHON_MODULES}
      )
    set_tests_properties(${_category}/build_python_modules PROPERTIES
      LABELS "test_dependency"
      FIXTURES_SETUP ${_fixture}
      )
  endif()

  #
  # Pick up tests:
  #
  file(GLOB _comparison_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.output"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.run_only"
    )
  list(SORT _comparison_files)

  set(_numdiff_warned FALSE)

  foreach(_comparison ${_comparison_files})
    string(REGEX REPLACE "\\..*$" "" _name "${_comparison}")

    if(NOT "${TEST_PICKUP_REGEX}" STREQUAL "" AND
       NOT "${_category}/${_name}" MATCHES "${TEST_PICKUP_REGEX}")
      continue()
    endif()

    set(_script "${CMAKE_CURRENT_SOURCE_DIR}/${_name}.py")
    if(NOT EXISTS "${_script}")
      message(FATAL_ERROR "\n${CMAKE_CURRENT_SOURCE_DIR}/${_comparison}:\n"
        "A comparison file (ending in .output or .run_only) has been picked "
        "up but the python script \"${_script}\" does not exist.\n")
    endif()

    #
    # MPI ranks and threads from the comparison file name:
    #
    set(_n_cpu 0)
    if("${_comparison}" MATCHES "\\.mpirun=([0-9]+)\\.")
      set(_n_cpu ${CMAKE_MATCH_1})
    endif()
    set(_n_threads 0)
    if("${_comparison}" MATCHES "\\.threads=([0-9]+)\\.")
      set(_n_threads ${CMAKE_MATCH_1})
    endif()

    if(_n_cpu GREATER 0 AND NOT DEAL_II_WITH_MPI)
      continue()
    endif()
    if(TEST_MPI_RANK_LIMIT GREATER 0 AND _n_cpu GREATER TEST_MPI_RANK_LIMIT)
      continue()
    endif()
    if(TEST_THREAD_LIMIT GREATER 0 AND _n_threads GREATER TEST_THREAD_LIMIT)
      continue()
    endif()

    #
    # Test name and run directory (mirrors deal.II's layout):
    #
    set(_test_full "${_category}/${_name}")
    set(_test_dir "${CMAKE_CURRENT_BINARY_DIR}/${_name}${_build_suffix}")
    if(_n_cpu EQUAL 0 AND _n_threads EQUAL 0)
      string(APPEND _test_dir "/serial")
    endif()
    if(_n_cpu GREATER 0)
      string(APPEND _test_full ".mpirun=${_n_cpu}")
      string(APPEND _test_dir "/mpirun=${_n_cpu}")
    endif()
    if(_n_threads GREATER 0)
      string(APPEND _test_full ".threads=${_n_threads}")
      string(APPEND _test_dir "/threads=${_n_threads}")
    endif()
    string(APPEND _test_full "${_build_suffix}")

    #
    # Command:
    #
    set(_command "${Python_EXECUTABLE}" "${_script}")
    if(_n_cpu GREATER 0)
      set(_command
        "${DEAL_II_MPIEXEC}" ${DEAL_II_MPIEXEC_NUMPROC_FLAG} ${_n_cpu}
        ${DEAL_II_MPIEXEC_PREFLAGS} ${_command} ${DEAL_II_MPIEXEC_POSTFLAGS})
    endif()

    #
    # Comparison (only for .output files and if numdiff is available):
    #
    set(_diff_args "")
    if("${_comparison}" MATCHES "\\.output$")
      if(NUMDIFF_EXECUTABLE)
        set(_diff_args
          "-DNUMDIFF=${NUMDIFF_EXECUTABLE}"
          "-DCOMPARISON=${CMAKE_CURRENT_SOURCE_DIR}/${_comparison}")
      elseif(NOT _numdiff_warned)
        message(WARNING
          "pybind_pickup_tests: numdiff not found, the output of the tests "
          "in category \"${_category}\" is not compared. Set NUMDIFF_DIR.")
        set(_numdiff_warned TRUE)
      endif()
    endif()

    add_test(NAME ${_test_full}
      COMMAND ${CMAKE_COMMAND}
        "-DTEST=${_test_full}"
        "-DBASH=${BASH}"
        "-DRUN_TEST_SH=${_run_test_sh}"
        "-DWORKDIR=${_test_dir}"
        "-DN_THREADS=${_n_threads}"
        ${_diff_args}
        -P "${_runner}"
        -- ${_command}
      )

    set(_env_modification "")
    foreach(_module ${TEST_PYTHON_MODULES})
      list(APPEND _env_modification
        "PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:${_module}>")
    endforeach()
    foreach(_path ${TEST_PYTHON_PATH})
      list(APPEND _env_modification "PYTHONPATH=path_list_prepend:${_path}")
    endforeach()

    set_tests_properties(${_test_full} PROPERTIES
      LABELS "${_category};python"
      TIMEOUT ${TEST_TIME_LIMIT}
      ENVIRONMENT "PYTHONUNBUFFERED=1;PYTHONDONTWRITEBYTECODE=1"
      )
    if(_env_modification)
      set_tests_properties(${_test_full} PROPERTIES
        ENVIRONMENT_MODIFICATION "${_env_modification}")
    endif()
    if(_fixture)
      set_tests_properties(${_test_full} PROPERTIES
        FIXTURES_REQUIRED ${_fixture})
    endif()
    if(_n_cpu GREATER 0)
      set_tests_properties(${_test_full} PROPERTIES PROCESSORS ${_n_cpu})
    endif()
  endforeach()
endfunction()
