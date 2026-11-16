# dealii_pybind_add_module(<name>
#   SOURCES <src>...
#   [LINK <target>...]
#   [DESTINATION <dir>]
#   )
#
# Adds a python extension module <name> via pybind11_add_module() (which
# handles extension suffix, hidden visibility, LTO and stripping) and links
# it against the given targets, e.g.
#
#   LINK dealii::dealii           deal.II (flags, includes, Debug/Release lib)
#   LINK EXADG::exadg             ExaDG (brings its deal.II along)
#   LINK applications             a library of this project
#
# All links use the keyword signature, so do not call deal_ii_setup_target()
# on the module; dealii::dealii carries the same compile/link setup.
#
# The module is installed to <DESTINATION> (default: python) with an RPATH
# to <prefix>/lib (this project's shared libraries) and to all linked
# libraries outside of the build tree, and its name is
# appended to the global property DEALII_PYBIND_MODULES (used by the python
# testsuite to put all modules on PYTHONPATH).

function(dealii_pybind_add_module name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "DESTINATION" "SOURCES;LINK")

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "dealii_pybind_add_module(${name}): unknown arguments "
      "\"${ARG_UNPARSED_ARGUMENTS}\"")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "dealii_pybind_add_module(${name}): no SOURCES given")
  endif()
  foreach(_link ${ARG_LINK})
    if(NOT TARGET ${_link})
      message(FATAL_ERROR
        "dealii_pybind_add_module(${name}): LINK entry \"${_link}\" is not a "
        "target. Is the corresponding find_package() missing?")
    endif()
  endforeach()
  if(NOT ARG_DESTINATION)
    set(ARG_DESTINATION python)
  endif()

  pybind11_add_module(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE ${ARG_LINK})

  # Installed modules find this project's shared libraries in
  # <prefix>/lib relative to their own location, and external libraries
  # (deal.II, MPI, ...) via the paths they were linked from.
  file(RELATIVE_PATH _lib_rel
    "${CMAKE_INSTALL_PREFIX}/${ARG_DESTINATION}" "${CMAKE_INSTALL_PREFIX}/lib")
  set_target_properties(${name} PROPERTIES
    INSTALL_RPATH "$ORIGIN/${_lib_rel}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    )

  install(TARGETS ${name} LIBRARY DESTINATION ${ARG_DESTINATION})

  set_property(GLOBAL APPEND PROPERTY DEALII_PYBIND_MODULES ${name})
endfunction()
