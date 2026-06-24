# Zipora Project Makefile
#
# Default features: simd, mmap, zstd, serde, lz4, async, avx512
# Optional: ffi, criterion
#
# Usage:
#   make             # Build and test (debug + release)
#   make sanity      # Quick build+test in 3 feature configs
#   make bench_all   # Run all benchmarks

.PHONY: all build build_debug build_release
.PHONY: test test_debug test_release test_simd_base64
.PHONY: bench bench_all bench_avx512 bench_fsa bench_io bench_serialization
.PHONY: safety_tests miri_tests miri_full
.PHONY: format clippy doc
.PHONY: dev validate ci pre_commit release_prep sanity
.PHONY: clean update outdated audit help

CARGO := cargo
CARGO_MIRI := cargo +nightly miri

# =============================================================================
# BUILD
# =============================================================================

all: build test

build: build_debug build_release

build_debug:
	$(CARGO) build

build_release:
	$(CARGO) build --release

# =============================================================================
# TEST
# =============================================================================

test: test_debug test_release

test_debug:
	$(CARGO) test --lib --bins --tests

test_release:
	$(CARGO) test --release --lib --bins --tests

test_simd_base64:
	$(CARGO) test --release --test simd_base64_tests -- --nocapture

# =============================================================================
# BENCHMARK
# =============================================================================

bench:
	$(CARGO) bench

bench_all:
	$(CARGO) bench --release

bench_avx512:
	$(CARGO) bench --release --bench avx512_bench

bench_fsa:
	$(CARGO) bench --bench fsa_infrastructure_bench

bench_serialization:
	$(CARGO) bench --release --bench memory_performance --bench memory_pools_bench --bench adaptive_mmap_bench

bench_io:
	$(CARGO) test --release test_stream_performance_comparison -- --nocapture
	$(CARGO) test --release test_combined_stream_operations -- --nocapture
	$(CARGO) test --release test_stress_operations -- --nocapture
	$(CARGO) bench --release --bench memory_performance --bench memory_pools_bench --bench adaptive_mmap_bench

# =============================================================================
# SAFETY & MIRI
# =============================================================================

safety_tests:
	$(CARGO) test container_safety_tests -- --nocapture
	$(CARGO) test enhanced_memory_safety -- --nocapture

miri_tests:
	@if command -v rustup >/dev/null 2>&1; then \
		if ! rustup toolchain list | grep -q nightly; then \
			rustup install nightly; \
		fi; \
		if ! rustup component list --toolchain nightly | grep -q "miri.*installed"; then \
			rustup +nightly component add miri; \
		fi; \
		$(CARGO_MIRI) test enhanced_memory_safety --quiet || \
		$(CARGO_MIRI) test enhanced_memory_safety --verbose; \
	else \
		echo "Error: rustup not found"; \
		exit 1; \
	fi

miri_full:
	@if [ -x "./run_miri_tests.sh" ]; then \
		./run_miri_tests.sh full; \
	else \
		echo "Error: run_miri_tests.sh not found or not executable"; \
		exit 1; \
	fi

# =============================================================================
# CODE QUALITY
# =============================================================================

format:
	$(CARGO) fmt --all

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

doc:
	$(CARGO) doc --no-deps --open

# =============================================================================
# WORKFLOWS
# =============================================================================

dev: format clippy build test safety_tests

validate: dev doc

ci: format clippy build test safety_tests

pre_commit: format clippy test_debug safety_tests

release_prep: clean format clippy build_release test_release bench doc audit

# Sanity check: clippy gate + default features (debug + release) + all features (release)
sanity:
	@echo "=== Clippy (all targets, all features, deny warnings) ==="
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	@echo "=== Default (debug) ==="
	$(CARGO) build
	$(CARGO) test --lib
	@echo "=== Default (release) ==="
	$(CARGO) build --release
	$(CARGO) test --release --lib
	@echo "=== All features (release) ==="
	$(CARGO) build --release --all-features
	$(CARGO) test --release --lib --all-features
	@echo "=== No default features (guard feature-gated cfg attrs) ==="
	$(CARGO) clippy --no-default-features -- -D unused_variables -D unused_imports
	@echo "=== Sanity: PASS ==="

# =============================================================================
# MAINTENANCE
# =============================================================================

clean:
	$(CARGO) clean
	@rm -f bench_results.txt benchmark_output.txt benchmark_results.txt
	@rm -f benchmark_summary.txt final_bench_results.txt cpp_impl_bench_results.txt
	@rm -f *_bench_results.txt *_benchmark_*.txt
	@rm -rf target/criterion

update:
	$(CARGO) update

outdated:
	@if command -v cargo-outdated >/dev/null 2>&1; then \
		$(CARGO) outdated; \
	else \
		echo "Install: cargo install cargo-outdated"; \
	fi

audit:
	@if command -v cargo-audit >/dev/null 2>&1; then \
		$(CARGO) audit; \
	else \
		echo "Install: cargo install cargo-audit"; \
	fi

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "Zipora Makefile"
	@echo ""
	@echo "  all              Build and test (debug + release)"
	@echo "  build            Build debug + release"
	@echo "  test             Test debug + release"
	@echo "  sanity           Clippy gate + 3 feature configs x debug+release"
	@echo ""
	@echo "  bench            Run all benchmarks"
	@echo "  bench_all        Run all benchmarks (release)"
	@echo "  bench_avx512     Run AVX-512 benchmarks only"
	@echo ""
	@echo "  safety_tests     Container safety tests"
	@echo "  miri_tests       Miri memory safety (needs nightly)"
	@echo ""
	@echo "  format           rustfmt"
	@echo "  clippy           Clippy lints"
	@echo "  doc              Generate docs"
	@echo ""
	@echo "  dev              format + clippy + build + test + safety"
	@echo "  ci               Same as dev"
	@echo "  release_prep     Full release pipeline"
	@echo "  clean            Remove build artifacts"
	@echo ""
	@echo "Features: simd mmap zstd serde lz4 async avx512 (default)"
	@echo "Optional: ffi criterion"
