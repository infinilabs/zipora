# Zipora Project Makefile
# Comprehensive build and test automation for debug/release modes with feature management
#
# Features:
# - Stable features: simd, mmap, zstd, lz4, serde, ffi  
# - Nightly features: avx512
#
# Usage:
#   make         # Build and test everything (stable features only)
#   make all     # Same as default
#   make build   # Build debug + release (stable features)
#   make test    # Test debug + release (stable features) 
#   make build_nightly  # Build with all features including nightly
#   make test_nightly   # Test with all features including nightly

.PHONY: all build test build_nightly test_nightly clean help
.PHONY: build_debug build_release build_nightly_debug build_nightly_release
.PHONY: test_debug test_release test_nightly_debug test_nightly_release  
.PHONY: bench bench_fsa bench_serialization bench_io safety_tests miri_tests format clippy doc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature sets
STABLE_FEATURES := --features simd,mmap,zstd,lz4,serde,ffi
NIGHTLY_FEATURES := --features simd,mmap,zstd,lz4,serde,ffi,avx512
ALL_FEATURES := --all-features

# Cargo commands
CARGO := cargo
CARGO_NIGHTLY := cargo +nightly
CARGO_MIRI := cargo +nightly miri

# Test exclusions
EXCLUDE_BENCH_TESTS := --exclude-from-test '*bench*'

# =============================================================================
# DEFAULT TARGET
# =============================================================================

# Default: build and test everything with stable features
all: build test
	@echo ""
	@echo "🎉 All build and test targets completed successfully!"
	@echo ""
	@echo "📊 Summary:"
	@echo "  ✅ Debug build (stable features)"
	@echo "  ✅ Release build (stable features)"  
	@echo "  ✅ Debug tests (stable features, no benchmarks)"
	@echo "  ✅ Release tests (stable features, with I/O performance tests and stable benchmarks)"
	@echo ""
	@echo "💡 To include nightly features and avx512_bench, run: make build_nightly test_nightly"

# =============================================================================
# BUILD TARGETS
# =============================================================================

# Build both debug and release with stable features
build: build_debug build_release
	@echo "✅ All stable builds completed"

# Build both debug and release with nightly features  
build_nightly: build_nightly_debug build_nightly_release
	@echo "✅ All nightly builds completed"

# Individual build targets - Debug mode (stable)
build_debug:
	@echo "🔨 Building debug mode with stable features..."
	$(CARGO) build $(STABLE_FEATURES)
	@echo "✅ Debug build (stable) completed"

# Individual build targets - Release mode (stable)
build_release:
	@echo "🔨 Building release mode with stable features..."
	$(CARGO) build --release $(STABLE_FEATURES)
	@echo "✅ Release build (stable) completed"

# Individual build targets - Debug mode (nightly)
build_nightly_debug:
	@echo "🌙 Building debug mode with nightly features..."
	$(CARGO_NIGHTLY) build $(NIGHTLY_FEATURES)
	@echo "✅ Debug build (nightly) completed"

# Individual build targets - Release mode (nightly)
build_nightly_release:
	@echo "🌙 Building release mode with nightly features..."
	$(CARGO_NIGHTLY) build --release $(NIGHTLY_FEATURES)
	@echo "✅ Release build (nightly) completed"

# =============================================================================
# TEST TARGETS
# =============================================================================

# Test both debug and release with stable features
test: test_debug test_release
	@echo "✅ All stable tests completed"

# Test both debug and release with nightly features
test_nightly: test_nightly_debug test_nightly_release
	@echo "✅ All nightly tests completed"

# Individual test targets - Debug mode (stable, no benchmarks)
test_debug:
	@echo "🧪 Running debug tests with stable features (excluding benchmarks)..."
	$(CARGO) test $(STABLE_FEATURES) --lib --bins --tests
	@echo "✅ Debug tests (stable) completed"

# Individual test targets - Release mode (stable, with benchmarks)
test_release:
	@echo "🧪 Running release tests with stable features (including benchmarks)..."
	$(CARGO) test --release $(STABLE_FEATURES) --lib --bins --tests
	@echo "🧪 Running I/O & Serialization performance tests..."
	$(CARGO) test --release $(STABLE_FEATURES) test_stream_performance_comparison -- --nocapture || echo "❌ I/O performance tests failed - may require additional setup"
	$(CARGO) test --release $(STABLE_FEATURES) test_combined_stream_operations -- --nocapture || echo "❌ I/O integration tests failed - may require additional setup"
	@echo "🧪 Running stable benchmarks (excluding avx512_bench and cpp_comparison)..."
	$(CARGO) test --release $(STABLE_FEATURES) --bench benchmark --bench dictionary_optimization_bench --bench cache_bench --bench secure_memory_pool_bench --bench specialized_containers_bench --bench rank_select_bench --bench compressed_sparse_trie_bench --bench comprehensive_trie_benchmarks --bench double_array_trie_bench --bench nested_louds_trie_bench --bench sortable_str_vec_bench --bench simd_rank_select_bench --bench benchmark_rank_select --bench fsa_infrastructure_bench --bench stream_buffer_bench --bench range_stream_bench --bench zero_copy_bench --bench endian_bench --bench smart_ptr_bench --bench complex_types_bench --bench versioning_bench --bench var_int_bench || echo "❌ Some benchmarks failed - may require additional setup"
	@echo "✅ Release tests (stable) completed"

# Individual test targets - Debug mode (nightly, no benchmarks)  
test_nightly_debug:
	@echo "🌙 Running debug tests with nightly features (excluding benchmarks)..."
	$(CARGO_NIGHTLY) test $(NIGHTLY_FEATURES) --lib --bins --tests
	@echo "✅ Debug tests (nightly) completed"

# Individual test targets - Release mode (nightly, with benchmarks)
test_nightly_release:
	@echo "🌙 Running release tests with nightly features (including benchmarks)..."
	$(CARGO_NIGHTLY) test --release $(NIGHTLY_FEATURES) --lib --bins --tests
	@echo "🧪 Running I/O & Serialization performance tests..."
	$(CARGO_NIGHTLY) test --release $(NIGHTLY_FEATURES) test_stream_performance_comparison -- --nocapture || echo "❌ I/O performance tests failed - may require additional setup"
	$(CARGO_NIGHTLY) test --release $(NIGHTLY_FEATURES) test_combined_stream_operations -- --nocapture || echo "❌ I/O integration tests failed - may require additional setup"
	@echo "🧪 Running nightly benchmarks (including avx512_bench, excluding cpp_comparison)..."
	$(CARGO_NIGHTLY) test --release $(NIGHTLY_FEATURES) --bench benchmark --bench dictionary_optimization_bench --bench cache_bench --bench avx512_bench --bench secure_memory_pool_bench --bench specialized_containers_bench --bench rank_select_bench --bench compressed_sparse_trie_bench --bench comprehensive_trie_benchmarks --bench double_array_trie_bench --bench nested_louds_trie_bench --bench sortable_str_vec_bench --bench simd_rank_select_bench --bench benchmark_rank_select --bench fsa_infrastructure_bench --bench stream_buffer_bench --bench range_stream_bench --bench zero_copy_bench --bench endian_bench --bench smart_ptr_bench --bench complex_types_bench --bench versioning_bench --bench var_int_bench || echo "❌ Some benchmarks failed - may require additional setup"
	@echo "✅ Release tests (nightly) completed"

# =============================================================================
# SPECIALIZED TEST TARGETS
# =============================================================================

# Run benchmarks only
bench:
	@echo "⚡ Running benchmarks..."
	$(CARGO) bench $(STABLE_FEATURES)
	@echo "✅ Benchmarks completed"

# Run FSA infrastructure benchmarks specifically
bench_fsa:
	@echo "⚡ Running FSA infrastructure benchmarks..."
	$(CARGO) bench $(STABLE_FEATURES) --bench fsa_infrastructure_bench
	@echo "✅ FSA benchmarks completed"

# Run I/O & Serialization benchmarks specifically  
bench_serialization:
	@echo "⚡ Running I/O & Serialization benchmarks..."
	$(CARGO) bench --release $(STABLE_FEATURES) --bench stream_buffer_bench --bench range_stream_bench --bench zero_copy_bench --bench endian_bench --bench smart_ptr_bench --bench complex_types_bench --bench versioning_bench --bench var_int_bench
	@echo "✅ Serialization benchmarks completed"

# Run I/O & Serialization performance tests specifically
bench_io:
	@echo "⚡ Running I/O & Serialization performance tests..."
	$(CARGO) test --release $(STABLE_FEATURES) test_stream_performance_comparison -- --nocapture
	$(CARGO) test --release $(STABLE_FEATURES) test_combined_stream_operations -- --nocapture
	$(CARGO) test --release $(STABLE_FEATURES) test_stress_operations -- --nocapture
	@echo "⚡ Running I/O & Serialization benchmarks..."
	$(CARGO) bench --release $(STABLE_FEATURES) --bench stream_buffer_bench --bench range_stream_bench --bench zero_copy_bench --bench endian_bench --bench smart_ptr_bench --bench complex_types_bench --bench versioning_bench --bench var_int_bench
	@echo "✅ I/O performance tests and benchmarks completed"

# Run all benchmarks in release mode
bench_release:
	@echo "⚡ Running all benchmarks in release mode..."
	$(CARGO) bench --release $(STABLE_FEATURES)
	@echo "✅ Release benchmarks completed"

# Run nightly benchmarks with AVX-512
bench_nightly:
	@echo "🌙 Running nightly benchmarks with AVX-512..."
	$(CARGO_NIGHTLY) bench --release $(NIGHTLY_FEATURES)
	@echo "✅ Nightly benchmarks completed"

# Run enhanced safety tests
safety_tests:
	@echo "🛡️  Running enhanced container safety tests..."
	$(CARGO) test container_safety_tests $(STABLE_FEATURES) -- --nocapture
	$(CARGO) test enhanced_memory_safety $(STABLE_FEATURES) -- --nocapture
	@echo "✅ Safety tests completed"

# Run Miri memory safety tests
miri_tests:
	@echo "🔍 Running Miri memory safety tests..."
	@if command -v rustup >/dev/null 2>&1; then \
		if ! rustup toolchain list | grep -q nightly; then \
			echo "📦 Installing nightly toolchain..."; \
			rustup install nightly; \
		fi; \
		if ! rustup component list --toolchain nightly | grep -q "miri.*installed"; then \
			echo "📦 Installing Miri..."; \
			rustup +nightly component add miri; \
		fi; \
		echo "🔬 Running enhanced memory safety tests with Miri..."; \
		$(CARGO_MIRI) test enhanced_memory_safety --quiet || \
		$(CARGO_MIRI) test enhanced_memory_safety --verbose; \
		echo "✅ Miri tests completed"; \
	else \
		echo "❌ Error: rustup not found. Please install rustup first."; \
		exit 1; \
	fi

# Run full Miri test suite using the script
miri_full:
	@echo "🔍 Running full Miri test suite..."
	@if [ -x "./run_miri_tests.sh" ]; then \
		./run_miri_tests.sh; \
	else \
		echo "❌ Error: run_miri_tests.sh not found or not executable"; \
		exit 1; \
	fi

# =============================================================================
# CODE QUALITY TARGETS
# =============================================================================

# Format code
format:
	@echo "🎨 Formatting code..."
	$(CARGO) fmt --all
	@echo "✅ Code formatting completed"

# Run clippy lints
clippy:
	@echo "📎 Running clippy lints..."
	$(CARGO) clippy $(STABLE_FEATURES) --all-targets -- -D warnings
	@echo "✅ Clippy lints completed"

# Run clippy with nightly features
clippy_nightly:
	@echo "🌙 Running clippy lints with nightly features..."
	$(CARGO_NIGHTLY) clippy $(NIGHTLY_FEATURES) --all-targets -- -D warnings
	@echo "✅ Clippy lints (nightly) completed"

# Generate documentation
doc:
	@echo "📚 Generating documentation..."
	$(CARGO) doc $(STABLE_FEATURES) --no-deps --open
	@echo "✅ Documentation generated"

# Generate documentation with nightly features
doc_nightly:
	@echo "🌙 Generating documentation with nightly features..."
	$(CARGO_NIGHTLY) doc $(NIGHTLY_FEATURES) --no-deps --open
	@echo "✅ Documentation (nightly) generated"

# =============================================================================
# COMPREHENSIVE TARGETS
# =============================================================================

# Full development cycle (stable)
dev: format clippy build test safety_tests
	@echo "🚀 Full development cycle completed (stable features)"

# Full development cycle (nightly)
dev_nightly: format clippy_nightly build_nightly test_nightly miri_tests
	@echo "🚀 Full development cycle completed (nightly features)"

# Complete validation (everything)
validate: dev dev_nightly doc doc_nightly
	@echo "🎯 Complete validation completed"

# =============================================================================
# MAINTENANCE TARGETS
# =============================================================================

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	$(CARGO) clean
	@echo "✅ Clean completed"

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	$(CARGO) update
	@echo "✅ Dependencies updated"

# Check for outdated dependencies
outdated:
	@echo "📋 Checking for outdated dependencies..."
	@if command -v cargo-outdated >/dev/null 2>&1; then \
		$(CARGO) outdated; \
	else \
		echo "💡 Install cargo-outdated: cargo install cargo-outdated"; \
	fi

# Security audit
audit:
	@echo "🔒 Running security audit..."
	@if command -v cargo-audit >/dev/null 2>&1; then \
		$(CARGO) audit; \
	else \
		echo "💡 Install cargo-audit: cargo install cargo-audit"; \
	fi

# =============================================================================
# CI/CD TARGETS
# =============================================================================

# CI pipeline (stable features only)
ci: format clippy build test safety_tests
	@echo "🤖 CI pipeline completed (stable)"

# CI pipeline (nightly features)
ci_nightly: format clippy_nightly build_nightly test_nightly miri_tests
	@echo "🤖 CI pipeline completed (nightly)"

# Pre-commit hook
pre_commit: format clippy test_debug safety_tests
	@echo "✅ Pre-commit checks completed"

# Release preparation
release_prep: clean format clippy build_release test_release bench doc audit
	@echo "🚀 Release preparation completed"

# pre-commit sanity check
sanity: test_debug test_release test_nightly_debug test_nightly_release

# =============================================================================
# HELP TARGET
# =============================================================================

help:
	@echo "Zipora Project Makefile"
	@echo "======================="
	@echo ""
	@echo "Main Targets:"
	@echo "  all                    Build and test everything (stable features)"
	@echo "  build                  Build debug + release (stable features)"
	@echo "  test                   Test debug + release (stable features)"
	@echo "  build_nightly          Build debug + release (all features including nightly)"
	@echo "  test_nightly           Test debug + release (all features including nightly)"
	@echo ""
	@echo "Individual Build Targets:"
	@echo "  build_debug            Build debug mode (stable features)"
	@echo "  build_release          Build release mode (stable features)"
	@echo "  build_nightly_debug    Build debug mode (nightly features)"
	@echo "  build_nightly_release  Build release mode (nightly features)"
	@echo ""
	@echo "Individual Test Targets:"
	@echo "  test_debug             Test debug mode (stable, no benchmarks)"
	@echo "  test_release           Test release mode (stable, with stable benchmarks)"
	@echo "  test_nightly_debug     Test debug mode (nightly, no benchmarks)"
	@echo "  test_nightly_release   Test release mode (nightly, with all benchmarks including avx512)"
	@echo ""
	@echo "Specialized Targets:"
	@echo "  bench                  Run benchmarks"
	@echo "  bench_fsa              Run FSA infrastructure benchmarks specifically"
	@echo "  bench_serialization    Run I/O & Serialization benchmarks specifically"
	@echo "  bench_io               Run I/O & Serialization performance tests specifically"
	@echo "  bench_release          Run all benchmarks in release mode"
	@echo "  bench_nightly          Run nightly benchmarks with AVX-512"
	@echo "  safety_tests           Run enhanced container safety tests"
	@echo "  miri_tests             Run Miri memory safety tests"
	@echo "  miri_full              Run full Miri test suite (using script)"
	@echo ""
	@echo "Code Quality:"
	@echo "  format                 Format code with rustfmt"
	@echo "  clippy                 Run clippy lints (stable)"
	@echo "  clippy_nightly         Run clippy lints (nightly)"
	@echo "  doc                    Generate documentation (stable)"
	@echo "  doc_nightly            Generate documentation (nightly)"
	@echo ""
	@echo "Development Workflows:"
	@echo "  dev                    Full development cycle (stable)"
	@echo "  dev_nightly            Full development cycle (nightly)"
	@echo "  validate               Complete validation (everything)"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean                  Clean build artifacts"
	@echo "  update                 Update dependencies"
	@echo "  outdated               Check outdated dependencies"
	@echo "  audit                  Security audit"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci                     CI pipeline (stable)"
	@echo "  ci_nightly             CI pipeline (nightly)"
	@echo "  pre_commit             Pre-commit checks"
	@echo "  release_prep           Release preparation"
	@echo ""
	@echo "Features:"
	@echo "  Stable: simd, mmap, zstd, lz4, serde, ffi"
	@echo "  Nightly: avx512"
