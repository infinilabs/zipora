# CMakeLists.txt for C++ zipora benchmark wrapper
cmake_minimum_required(VERSION 3.12)
project(zipora_benchmark_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for fair comparison
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Try to find the reference library
find_path(REFERENCE_LIB_INCLUDE_DIR
    NAMES terark/valvec.hpp
    PATHS
        /usr/local/include
        /usr/include
        ${CMAKE_SOURCE_DIR}/../reference-lib/src
        ${CMAKE_SOURCE_DIR}/../../reference/reference-lib/src
        /usr/local/google/home/binwu/workspace/reference/reference-lib/src
    DOC "reference library include directory"
)

find_library(REFERENCE_LIB_LIBRARY
    NAMES terark-core terark
    PATHS
        /usr/local/lib
        /usr/lib
        ${CMAKE_SOURCE_DIR}/../reference-lib/build
        ${CMAKE_SOURCE_DIR}/../../reference/reference-lib/build
    DOC "reference library"
)

# Configure based on whether we found reference library
if(REFERENCE_LIB_INCLUDE_DIR AND REFERENCE_LIB_LIBRARY)
    message(STATUS "Found reference library at ${REFERENCE_LIB_INCLUDE_DIR}")
    add_definitions(-DHAVE_REFERENCE_LIB)
    include_directories(${REFERENCE_LIB_INCLUDE_DIR})
    set(REFERENCE_LIB_LIBS ${REFERENCE_LIB_LIBRARY})
else()
    message(WARNING "reference library not found, using stub implementations")
    set(REFERENCE_LIB_LIBS)
endif()

# Create the benchmark wrapper library
add_library(zipora_wrapper SHARED
    wrapper.cpp
    enhanced_wrapper.cpp
)

target_include_directories(zipora_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${REFERENCE_LIB_INCLUDE_DIR}
)

target_link_libraries(zipora_wrapper
    ${REFERENCE_LIB_LIBS}
)

# Set library properties
set_target_properties(zipora_wrapper PROPERTIES
    VERSION 1.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Create a static version as well for easier linking
add_library(zipora_wrapper_static STATIC
    wrapper.cpp
    enhanced_wrapper.cpp
)

target_include_directories(zipora_wrapper_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${REFERENCE_LIB_INCLUDE_DIR}
)

target_link_libraries(zipora_wrapper_static
    ${REFERENCE_LIB_LIBS}
)

set_target_properties(zipora_wrapper_static PROPERTIES
    OUTPUT_NAME zipora_wrapper
    POSITION_INDEPENDENT_CODE ON
)

# Install targets
install(TARGETS zipora_wrapper zipora_wrapper_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES wrapper.hpp
    DESTINATION include
)

# Create a test executable to verify the wrapper works
add_executable(wrapper_test test_wrapper.cpp)
target_link_libraries(wrapper_test zipora_wrapper_static)

# Enable testing
enable_testing()
add_test(NAME wrapper_functionality_test COMMAND wrapper_test)