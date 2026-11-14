set(PYBIND11_FINDPYTHON ON)

# ---- general example module ----
# easier but conflicts with plain of deal_ii_setup_target
#pybind11_add_module(example
#    bindings/example.cc
#)

#manual version of pybind_add_module
add_library(example
    MODULE bindings/example.cc)

deal_ii_setup_target(example)

target_link_libraries(example pybind11::module pybind11::lto code)

# creates extension suffix for the target library
# example.so -> example.cpython-.....so
pybind11_extension(example)


# avoiding symbol collision
set_target_properties(example
    PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
	pybind11_strip(example)
endif()

install(
    TARGETS example
    LIBRARY DESTINATION python
)

# --- dealii core bingings ---
add_library(dealii_core_module
	MODULE bindings/dealii_core_module.cc)

deal_ii_setup_target(dealii_core_module)

target_link_libraries(dealii_core_module pybind11::module pybind11::lto)

pybind11_extension(dealii_core_module)

set_target_properties(dealii_core_module
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
        pybind11_strip(dealii_core_module)
endif()


install(
    TARGETS dealii_core_module
    LIBRARY DESTINATION python
)

# install python scripts
install(
    DIRECTORY scripts/
    DESTINATION python/scripts
    FILES_MATCHING PATTERN "*.py"
)
