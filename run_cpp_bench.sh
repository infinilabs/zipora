#!/bin/bash
# Simple script to run C++ comparison benchmark with proper library path

cd /usr/local/google/home/binwu/go/src/infini.sh/zipora

# Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/cpp_benchmark

# Also tell cargo/rustc where to find the library at link time
export RUSTFLAGS="-L $(pwd)/cpp_benchmark"

echo "Running C++ vs Rust benchmark comparison..."
echo "Library path: $(pwd)/cpp_benchmark"
echo ""

cargo bench --bench cpp_comparison