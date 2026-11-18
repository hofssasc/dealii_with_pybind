add_subdirectory(apps)
add_subdirectory(python)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pymor"
  RESULT_VARIABLE _pymor_missing
  OUTPUT_QUIET ERROR_QUIET
  )
if(_pymor_missing)
  message(STATUS "pyMOR not found for ${Python_EXECUTABLE}, skipping tests/pymor")
else()
  add_subdirectory(pymor)
endif()
