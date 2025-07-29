#!/bin/bash

# Build script for C++ benchmark wrapper
set -e

echo "Building C++ benchmark wrapper for topling-zip comparison..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc

# Build the wrapper library
make -j$(nproc)

# Run tests to verify functionality
echo "Running wrapper tests..."
./wrapper_test

echo "Build completed successfully!"
echo ""
echo "To use the wrapper in Rust benchmarks:"
echo "1. Copy libtopling_zip_wrapper.so to a library path"
echo "2. Set LD_LIBRARY_PATH or install the library"
echo "3. Run 'cargo bench --bench cpp_comparison'"