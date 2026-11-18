add_library(applications SHARED)

target_sources(applications
	PRIVATE
	    source/code.cc

	PUBLIC
	    FILE_SET HEADERS
	    BASE_DIRS
	        include
	    FILES
	        include/code.h
	        include/vector_space.h
  )

target_link_libraries(applications PUBLIC dealii::dealii)

# keep the paths to deal.II etc. in the installed library
set_target_properties(applications PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Installed next to the python modules' folder; the modules find it via
# their RPATH $ORIGIN/../lib (see cmake/DealiiPybindModule.cmake).
install(TARGETS applications
  LIBRARY DESTINATION lib
  FILE_SET HEADERS DESTINATION include
  )
