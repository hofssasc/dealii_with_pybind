add_library(applications SHARED)

target_sources(applications
	PRIVATE
	    source/code.cc
	    source/exadg_poisson_operator.cc
	    source/gaussian_source.cc
	    source/gaussian_source_matrix_free.cc
	    source/trilinos_matrix_operator.cc

	PUBLIC
	    FILE_SET HEADERS
	    BASE_DIRS
	        include
	    FILES
	        # abstract interfaces, bound to python once per vector type
	        include/interfaces/operator.h
	        include/interfaces/stationary_model.h
	        include/interfaces/vector_space.h
	        # their implementations: building blocks and applications
	        include/implementations/code.h
	        include/implementations/exadg_poisson_operator.h
	        include/implementations/gaussian_source.h
	        include/implementations/gaussian_source_matrix_free.h
	        include/implementations/trilinos_matrix_operator.h
  )

# ExaDG brings the deal.II library of its own build type, see the check in the top-level
# CMakeLists.txt.
target_link_libraries(applications PUBLIC dealii::dealii EXADG::exadg)

# keep the paths to deal.II etc. in the installed library
set_target_properties(applications PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Installed next to the python modules' folder; the modules find it via
# their RPATH $ORIGIN/../lib (see cmake/DealiiPybindModule.cmake).
install(TARGETS applications
  LIBRARY DESTINATION lib
  FILE_SET HEADERS DESTINATION include
  )
