# Runs a single python test; set up by pybind_pickup_tests(). Invoked as
#
#   cmake -DTEST=<name> -DBASH=<bash> -DRUN_TEST_SH=<deal.II run_test.sh>
#         -DWORKDIR=<run directory> -DN_THREADS=<n>
#         [-DNUMDIFF=<numdiff> -DCOMPARISON=<file.output>]
#         -P run_python_test.cmake -- <command...>
#
# Runs deal.II's run_test.sh "run" stage (writes <WORKDIR>/output) and, if a
# comparison file is given, its "diff" stage. Prints the same
# "<TEST>: <STAGE> successful/failed." lines as the C++ testsuite.

foreach(_var TEST BASH RUN_TEST_SH WORKDIR N_THREADS)
  if("${${_var}}" STREQUAL "")
    message(FATAL_ERROR "run_python_test.cmake: ${_var} not set")
  endif()
endforeach()

# Collect the test command: everything after "--".
set(_command "")
set(_collect FALSE)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE ${_last})
  if(_collect)
    list(APPEND _command "${CMAKE_ARGV${_i}}")
  elseif("${CMAKE_ARGV${_i}}" STREQUAL "--")
    set(_collect TRUE)
  endif()
endforeach()
if("${_command}" STREQUAL "")
  message(FATAL_ERROR "run_python_test.cmake: no command given after --")
endif()

file(MAKE_DIRECTORY "${WORKDIR}")

# RUN stage
execute_process(
  COMMAND "${BASH}" "${RUN_TEST_SH}" run "${TEST}"
          "TEST_N_THREADS=${N_THREADS}" ${_command}
  WORKING_DIRECTORY "${WORKDIR}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _output
  )
if(NOT _result EQUAL 0)
  message("${_output}")
  message(FATAL_ERROR "${TEST}: ******    RUN failed    *******")
endif()
message("${TEST}: RUN successful.")

# DIFF stage
if(NOT "${COMPARISON}" STREQUAL "")
  execute_process(
    COMMAND "${BASH}" "${RUN_TEST_SH}" diff "${TEST}"
            "${NUMDIFF}" "${COMPARISON}"
    WORKING_DIRECTORY "${WORKDIR}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _output
    )
  if(NOT _result EQUAL 0)
    message("${_output}")
    message(FATAL_ERROR "${TEST}: ******    DIFF failed    *******")
  endif()
  if(EXISTS "${WORKDIR}/diff")
    file(READ "${WORKDIR}/diff" _variant)
    if(NOT "${_variant}" STREQUAL "")
      message("${_variant}")
    endif()
  endif()
  message("${TEST}: DIFF successful.")
else()
  message("${TEST}: DIFF skipped.")
endif()

message("${TEST}: PASSED.")
