# CMakeLists.txt for C++ topling-zip benchmark wrapper
cmake_minimum_required(VERSION 3.12)
project(topling_zip_benchmark_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for fair comparison
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Try to find the original topling-zip library
find_path(TOPLING_ZIP_INCLUDE_DIR
    NAMES terark/valvec.hpp
    PATHS
        /usr/local/include
        /usr/include
        ${CMAKE_SOURCE_DIR}/../topling-zip/src
        ${CMAKE_SOURCE_DIR}/../../topling/topling-zip/src
        /usr/local/google/home/binwu/workspace/topling/topling-zip/src
    DOC "topling-zip include directory"
)

find_library(TOPLING_ZIP_LIBRARY
    NAMES terark-core terark
    PATHS
        /usr/local/lib
        /usr/lib
        ${CMAKE_SOURCE_DIR}/../topling-zip/build
        ${CMAKE_SOURCE_DIR}/../../topling/topling-zip/build
    DOC "topling-zip library"
)

# Configure based on whether we found topling-zip
if(TOPLING_ZIP_INCLUDE_DIR AND TOPLING_ZIP_LIBRARY)
    message(STATUS "Found topling-zip at ${TOPLING_ZIP_INCLUDE_DIR}")
    add_definitions(-DHAVE_TOPLING_ZIP)
    include_directories(${TOPLING_ZIP_INCLUDE_DIR})
    set(TOPLING_ZIP_LIBS ${TOPLING_ZIP_LIBRARY})
else()
    message(WARNING "topling-zip not found, using stub implementations")
    set(TOPLING_ZIP_LIBS)
endif()

# Create the benchmark wrapper library
add_library(topling_zip_wrapper SHARED
    wrapper.cpp
)

target_include_directories(topling_zip_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TOPLING_ZIP_INCLUDE_DIR}
)

target_link_libraries(topling_zip_wrapper
    ${TOPLING_ZIP_LIBS}
)

# Set library properties
set_target_properties(topling_zip_wrapper PROPERTIES
    VERSION 1.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Create a static version as well for easier linking
add_library(topling_zip_wrapper_static STATIC
    wrapper.cpp
)

target_include_directories(topling_zip_wrapper_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TOPLING_ZIP_INCLUDE_DIR}
)

target_link_libraries(topling_zip_wrapper_static
    ${TOPLING_ZIP_LIBS}
)

set_target_properties(topling_zip_wrapper_static PROPERTIES
    OUTPUT_NAME topling_zip_wrapper
    POSITION_INDEPENDENT_CODE ON
)

# Install targets
install(TARGETS topling_zip_wrapper topling_zip_wrapper_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES wrapper.hpp
    DESTINATION include
)

# Create a test executable to verify the wrapper works
add_executable(wrapper_test test_wrapper.cpp)
target_link_libraries(wrapper_test topling_zip_wrapper_static)

# Enable testing
enable_testing()
add_test(NAME wrapper_functionality_test COMMAND wrapper_test)