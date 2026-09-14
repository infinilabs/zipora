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
.PHONY: unsafe_audit api_honesty index_audit
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

# ConcurrentCsppTrie multi-writer soundness campaign (plan.md Phase 4.3).
# Run periodically (CI is build-only by policy): Miri catches data races/UB
# at small scale; TSAN stresses the full contended tests. crossbeam_sanitize
# switches crossbeam-epoch to its sanitizer-friendly paths (its fence-based
# protocol otherwise triggers a known TSAN false positive in Local::drop).
# Tree Borrows: crossbeam-epoch's int-to-pointer casts (atomic.rs Pointable)
# are not Stacked-Borrows-compatible (violation in ITS internal.rs Local
# deref); the crate is fine under the newer Tree Borrows aliasing model.
# ignore-leaks: crossbeam-epoch's static global collector queue is never
# torn down at process exit (known upstream; their Miri CI does the same).
miri_cspp:
	MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" \
	$(CARGO_MIRI) test --lib fsa::cspp_trie_concurrent -- --test-threads=1

tsan_cspp:
	CARGO_TARGET_DIR=target-tsan \
	RUSTFLAGS="-Zsanitizer=thread --cfg crossbeam_sanitize" \
	TSAN_OPTIONS="suppressions=$(CURDIR)/tsan.supp" \
	$(CARGO) +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu \
		--release --lib fsa::cspp_trie_concurrent

# LockFreeMemoryPool TSAN stress (plan.md 6.3). Same recipe as tsan_cspp;
# drives the module's multi-threaded alloc/free stress tests under the
# thread sanitizer. Run periodically — CI is build-only by policy.
tsan_pool:
	CARGO_TARGET_DIR=target-tsan \
	RUSTFLAGS="-Zsanitizer=thread --cfg crossbeam_sanitize" \
	TSAN_OPTIONS="suppressions=$(CURDIR)/tsan.supp" \
	$(CARGO) +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu \
		--release --lib memory::lockfree

# Periodic Miri job (plan.md 6.2). CI is build-only by policy, so this is
# the manual/weekly equivalent: unsafe-heavy container + hash-map cores
# under Miri (uint_vec tail reads, hash-map probe/migration, circular queue).
miri_core:
	$(CARGO_MIRI) test --lib containers::uint_vec_min0
	$(CARGO_MIRI) test --lib hash_map::zipora_hash_map
	$(CARGO_MIRI) test --lib containers::specialized::circular_queue
	$(CARGO_MIRI) test --lib containers::fast_vec

# SIMD-adjacent modules under Miri: all dispatch (macros, cached has_* bools,
# ifunc resolvers) routes to scalar under cfg(miri), so the surrounding index
# arithmetic, table lookups, and buffer handling get real Miri coverage.
miri_simd:
	$(CARGO_MIRI) test --lib algorithms::bit_ops
	$(CARGO_MIRI) test --lib compression::stream_vbyte
	$(CARGO_MIRI) test --lib algorithms::simd_search

# =============================================================================
# FUZZING (plan.md 6.1) — requires: cargo install cargo-fuzz; nightly toolchain
# =============================================================================

FUZZ_TARGETS = fuzz_zip_offset_load fuzz_simple_zip_build fuzz_mixed_len_build \
	fuzz_huffman_decode fuzz_rans_decode fuzz_fse_decompress \
	fuzz_double_array_trie fuzz_uint_vec

# 1 GiB, not the libFuzzer default of 2 GiB and not the 4 GiB this used to
# pass: a decoder that turns a few bytes of input into a multi-gigabyte
# allocation is the bug we want reported, and a limit at or above the output
# ceiling of the format hides exactly that class. The FSE decoder could be
# driven to a 4 GiB allocation from 26 bytes of input and fuzzing never
# flagged it.
FUZZ_RSS_LIMIT_MB = 1024

# Quick smoke: 60s per target, sequential.
fuzz_smoke:
	@for t in $(FUZZ_TARGETS); do \
		echo "=== $$t (60s) ==="; \
		cargo +nightly fuzz run $$t -- -max_total_time=60 -rss_limit_mb=$(FUZZ_RSS_LIMIT_MB) || exit 1; \
	done

# Soak: 1 hour per target, all in parallel (needs ~8 cores).
fuzz_soak:
	@for t in $(FUZZ_TARGETS); do \
		cargo +nightly fuzz run $$t -- -max_total_time=3600 -rss_limit_mb=$(FUZZ_RSS_LIMIT_MB) & \
	done; wait

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

unsafe_audit:
	python3 scripts/unsafe_audit.py

api_honesty:
	python3 scripts/api_honesty.py

index_audit:
	python3 scripts/index_audit.py

# Sanity check: clippy gate + audits + tests + drift guard (Rule B8)
sanity:
	@echo "=== Clippy (all targets, all features, deny warnings) ==="
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	@echo "=== Unsafe Code Audit (B7) ==="
	python3 scripts/unsafe_audit.py
	@echo "=== API Honesty Audit (D1) ==="
	python3 scripts/api_honesty.py
	@echo "=== Decoder Indexing Safety Audit (D10.1) ==="
	python3 scripts/index_audit.py
	@echo "=== No default features (guard feature-gated cfg attrs) ==="
	$(CARGO) clippy --no-default-features -- -D unused_variables -D unused_imports
	@mkdir -p target/sanity
	@echo "=== Tests: debug (--all-features --tests) ==="
	$(CARGO) test --all-features --tests 2>&1 | tee target/sanity/debug_tests.log
	@echo "=== Tests: release lib (--all-features --release) ==="
	$(CARGO) test --release --lib --all-features 2>&1 | tee target/sanity/release_lib.log
	@echo "=== Tests: doctests (--doc) ==="
	$(CARGO) test --doc 2>&1 | tee target/sanity/doctests.log
	@echo "=== Test counts and drift guard (B8) ==="
	python3 scripts/check_gate_drift.py target/sanity
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
	@echo "  miri_cspp        Miri on ConcurrentCsppTrie (races/UB, small scale)"
	@echo "  tsan_cspp        TSAN stress on ConcurrentCsppTrie (needs nightly + rust-src)"
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
