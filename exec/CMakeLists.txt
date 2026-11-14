
add_library(code)

target_sources(code
	PRIVATE
	    source/code.cc

	PUBLIC
	    FILE_SET HEADERS
	    BASE_DIRS 
	        include
	    FILES
	        include/code.h
  )

#target_include_directories(code
#    PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}/include
#)
 
deal_ii_setup_target(code)

