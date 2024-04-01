list(APPEND SOURCES src/*.cpp)
list(APPEND HEADERS include/*.hpp)

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
# include_directories(include BEFORE ${PROJECT_BINARY_DIR})

message("Inside mhd")
# Add one executable per cpp file

convert_filenames_to_full_paths(SOURCES)
convert_filenames_to_full_paths(HEADERS)

set(MHD_COMMON_FILES
EXTRA_SOURCES ${SOURCES}
EXTRA_HEADERS ${HEADERS})

# add_mfem_miniapp(mhd
# MAIN mhd.cpp
# ${MHD_COMMON_FILES}
# LIBRARIES mfem)

# list(APPEND ALL_EXE_SRCS
# src/mhd.cpp 
# ${SOURCES})


# add_mfem_examples(ALL_EXE_SRCS)
