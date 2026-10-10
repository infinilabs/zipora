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
.PHONY: safety_tests miri_tests miri_full miri_pool miri_algorithms
.PHONY: miri_cspp miri_core miri_simd miri_string miri_io miri_thread miri_ffi miri_scalar
.PHONY: tsan_cspp tsan_pool tsan_concurrency
.PHONY: format clippy doc
.PHONY: dev validate ci pre_commit release_prep sanity
.PHONY: unsafe_audit api_honesty index_audit unwrap_audit hygiene_audit
.PHONY: clean update outdated audit help

CARGO := cargo

# Miri target features and target dirs (S21-R4).
#
# Under Miri, is_x86_feature_detected! reports exactly the target features the
# crate was compiled with, so the build flags decide which SIMD kernels Miri
# executes. On x86_64 hosts every miri_* target therefore builds with the
# flags below. They are the rustflags of the maintainer's gitignored
# .cargo/config.toml, so a clean checkout gets the same Miri coverage as a
# checkout that has that file. MIRI_SCALAR=1 builds with no target flags
# instead, so the scalar fallbacks are Miri-checked too (see miri_scalar);
# SSE2 is part of the x86_64 baseline, so SSE2 kernels still run in that
# build. A RUSTFLAGS variable, even an empty one, overrides rustflags from any
# .cargo/config.toml, so neither build depends on that file. Each build has
# its own target dir, so switching between them does not force a full
# rebuild. Other hosts build both with no target flags.
#
# Neither build runs the SIMD arms of simd_dispatch!/simd_feature_check! (both
# macros compile them out under cfg(miri)), code selected through CpuFeatures
# (its detection returns all-false under cfg(miri)), code behind a has_*()
# helper that returns false under cfg!(miri), or any AVX-512 code (haswell has
# none). The miri_cspp, miri_simd, miri_string, miri_io and miri_algorithms
# comments say which #[target_feature] kernels their suites run; the miri_core
# and miri_pool suites run none in either build.
ifeq ($(shell uname -m),x86_64)
MIRI_SIMD_RUSTFLAGS := -C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt
endif
ifdef MIRI_SCALAR
MIRI_RUSTFLAGS :=
MIRI_TARGET_DIR := target/miri-scalar
else
MIRI_RUSTFLAGS := $(MIRI_SIMD_RUSTFLAGS)
MIRI_TARGET_DIR := target/miri-simd
endif
MIRI_ENV := RUSTFLAGS="$(MIRI_RUSTFLAGS)" CARGO_TARGET_DIR=$(MIRI_TARGET_DIR)
CARGO_MIRI := $(MIRI_ENV) cargo +nightly miri

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

# run_miri_tests.sh calls cargo itself; MIRI_ENV hands it the target flags
# and target dir.
miri_full:
	@if [ -x "./run_miri_tests.sh" ]; then \
		$(MIRI_ENV) ./run_miri_tests.sh full; \
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
# The default build runs one #[target_feature] kernel, sse4_2_search_byte in
# fsa/fast_search.rs; MIRI_SCALAR=1 runs its scalar code instead.
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

# Thread and concurrency TSAN stress (plan.md C10). Runs src/thread and
# src/concurrency (atomic_ext, linux_futex, instance_tls, work_stealing,
# fiber_pool, parallel_trie, pipeline) under the thread sanitizer.
tsan_concurrency:
	CARGO_TARGET_DIR=target-tsan \
	RUSTFLAGS="-Zsanitizer=thread --cfg crossbeam_sanitize" \
	TSAN_OPTIONS="suppressions=$(CURDIR)/tsan.supp" \
	$(CARGO) +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu \
		--release --lib thread::
	CARGO_TARGET_DIR=target-tsan \
	RUSTFLAGS="-Zsanitizer=thread --cfg crossbeam_sanitize" \
	TSAN_OPTIONS="suppressions=$(CURDIR)/tsan.supp" \
	$(CARGO) +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu \
		--release --lib concurrency::

# Periodic Miri job (plan.md 6.2). CI is build-only by policy, so this is
# the manual/weekly equivalent: unsafe-heavy container + hash-map cores
# under Miri (uint_vec tail reads, hash-map probe/migration, circular queue).
miri_core:
	$(CARGO_MIRI) test --lib containers::uint_vec_min0
	$(CARGO_MIRI) test --lib hash_map::zipora_hash_map
	$(CARGO_MIRI) test --lib containers::specialized::circular_queue
	$(CARGO_MIRI) test --lib containers::fast_vec

# Memory-pool allocators under Miri (plan.md C3 / D8). These modules are the
# crate's largest unsafe surface and their defects are layout and provenance
# bugs that no assertion in the default suite can see: C3.1 (a block header
# written past the end of the arena) and C3.2 (dealloc with an alignment the
# allocation never had) were both found this way.
#
# The two UB-by-design tests that used to sit #[ignore]d in
# tests/security_memory_pool.rs are NOT here: they demonstrate use after
# return to the pool, and a pool does not hand recycled memory back to the
# system, so there is no dead allocation for Miri to flag. They are deleted;
# see the header of tests/memory_pool_contract.rs.
#
# `memory::cache` needs -Zmiri-disable-isolation: detect_numa_nodes() reads
# /sys/devices/system/node, and Miri's isolation blocks `open`.
#
# The five lines are five separate Miri runs and the whole target takes the
# better part of an hour. make stops at the first failing line, so a failure in
# an early suite means the later ones did not run at all -- read the output for
# five `test result:` lines before calling the target green.
miri_pool:
	$(CARGO_MIRI) test --lib memory::fixed_capacity_pool
	$(CARGO_MIRI) test --lib memory::five_level_pool
	$(CARGO_MIRI) test --lib memory::lockfree_pool -- --test-threads=1
	$(CARGO_MIRI) test --lib memory::bump
	MIRIFLAGS="-Zmiri-disable-isolation" $(CARGO_MIRI) test --lib memory::cache -- --test-threads=1
	$(CARGO_MIRI) test --lib memory::simd_ops
	$(CARGO_MIRI) test --lib memory::prefetch

# SIMD-adjacent modules under Miri. These three modules pick their kernels
# through has_popcnt() (bit_ops.rs), has_avx2()/has_sse2() (simd_search.rs,
# directly or through the simd::isa tokens built on them) and
# resolve_decode_into() (stream_vbyte.rs), which all return the scalar choice
# under cfg!(miri). So in both builds none of their #[target_feature] kernels
# runs; Miri checks the scalar tiers and the index arithmetic, table lookups
# and buffer handling around them. That is also why miri_scalar does not
# repeat this target.
miri_simd:
	$(CARGO_MIRI) test --lib algorithms::bit_ops
	$(CARGO_MIRI) test --lib compression::stream_vbyte
	$(CARGO_MIRI) test --lib algorithms::simd_search

# src/algorithms under Miri (plan.md S20-R10, S21-R4). Two Miri runs, and make
# stops at the first failing one: read the output for two `test result:` lines
# before calling the target green. Tests with large inputs run smaller under
# cfg(miri).
#
# Run 1, Stacked Borrows with the leak check on: every algorithms:: test except
# the MIRI_RADIX_CROSSBEAM tests. -Zmiri-disable-isolation: the external_sort
# tests write their runs to a temp directory named after SystemTime::now(), and
# isolation blocks both the clock (the first error is clock_gettime with
# REALTIME) and the file system.
#
# Run 2, Tree Borrows: the MIRI_RADIX_CROSSBEAM tests, selected by exact name.
# Nine of them drop a SecureMemoryPool (AdvancedRadixSort's default config owns
# one, and its Drop pins an epoch) and four run a rayon pool. Under Stacked
# Borrows each stops inside crossbeam-epoch (internal.rs, Local::element_of),
# the violation miri_cspp also avoids with Tree Borrows. Under Tree Borrows
# each passes and then fails Miri's leak check. Every leaked allocation is made
# inside crossbeam-epoch and belongs to its static global collector, which is
# never dropped: each thread's Local, and the nodes of the collector's bag
# queue (its first node, and one per sealed bag that an exiting thread hands
# over). So -Zmiri-ignore-leaks turns the leak check off for these 13 tests
# only. The `integer-to-pointer cast` warnings point into crossbeam-epoch's
# atomic.rs. A new radix_sort test that drops a SecureMemoryPool or runs a
# rayon pool stops in run 1 with that Stacked Borrows violation: add it to
# MIRI_RADIX_CROSSBEAM.
#
# The default build runs the three AVX2 kernels in simd_merge and the SSE4.1
# kernels in simd_set_intersect and simd_set_union in run 1, and
# count_digits_avx2_bmi2 (radix_sort/advanced.rs) in run 2, so that one is
# checked under Tree Borrows only. MIRI_SCALAR=1 runs their scalar code. The
# simd_search and bit_ops kernels take their scalar code in both builds (see
# miri_simd).
#
# Measured (S21-R4, AMD EPYC 7B13, clean checkout, busy shared host): real
# 21m34s including the build. Run 1: 284 passed, 2 ignored, finished in
# 808 s. Run 2: 13 passed; its "finished in 166s" is Miri's virtual clock,
# because run 2 keeps isolation on. MIRI_SCALAR=1: run 1 finished in 686 s,
# run 2 passed 13.
MIRI_RADIX_CROSSBEAM := \
	test_advanced_algorithm_trait \
	test_advanced_radix_sort_forced_strategy \
	test_advanced_radix_sort_parallel \
	test_advanced_radix_sort_strings \
	test_advanced_radix_sort_u32_large \
	test_advanced_radix_sort_u32_simple \
	test_advanced_radix_sort_u64 \
	test_advanced_radix_sort_u64_random_simd \
	test_memory_pool_integration \
	test_phase_times_tracking \
	test_s20_r3_parallel_merge_tree_stable_and_matches_std \
	test_s20_r3b_radix_sort_parallel_path_and_used_parallel \
	test_s21_r1_advanced_radix_sort_reports_the_path_that_ran
miri_algorithms:
	MIRIFLAGS="-Zmiri-disable-isolation" \
	$(CARGO_MIRI) test --lib algorithms:: -- --test-threads=1 \
		$(foreach t,$(MIRI_RADIX_CROSSBEAM),--skip algorithms::radix_sort::tests::$(t))
	MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks" \
	$(CARGO_MIRI) test --lib -- --test-threads=1 --exact \
		$(foreach t,$(MIRI_RADIX_CROSSBEAM),algorithms::radix_sort::tests::$(t))

# src/string under Miri (S21-R2, S21-R4). One run with the default MIRIFLAGS:
# Stacked Borrows, with isolation and the leak check on. The default build runs
# eleven #[target_feature] kernels: hash_avx2 and hash_sse2 (fast_str.rs), and
# the SSE4.2 and AVX2 strchr/strstr/multi-search/strcmp kernels
# (simd_search.rs). MIRI_SCALAR=1 runs only hash_sse2. Neither build runs the
# bmi2_string_ops kernels or unicode.rs's is_ascii_simd: both are selected
# through CpuFeatures.
miri_string:
	$(CARGO_MIRI) test --lib string:: -- --test-threads=1

# src/io and src/memory/mmap_vec under Miri (S22-R7). Two runs, both with
# Stacked Borrows, the leak check on, and -Zmiri-disable-isolation (data_output,
# range_stream, mmap's BufferedIO tests, and mmap_vec's
# test_c9_f9_create_rejects_read_only_config create temporary files; tests that
# reach OS mmap/mprotect are marked #[cfg_attr(miri, ignore = ...)] or
# #[cfg(not(miri))]). In the default build, the direct sse42_* tests in
# io::simd_memory::search run sse42_strchr_impl, sse42_strstr_impl,
# sse42_multi_search_impl and sse42_strcmp_impl, and the direct kernel test in
# io::simd_validation::checksum runs crc32c_sse42; MIRI_SCALAR=1 runs their
# scalar fallbacks instead.
miri_io:
	MIRIFLAGS="-Zmiri-disable-isolation" $(CARGO_MIRI) test --lib io:: -- --test-threads=1
	MIRIFLAGS="-Zmiri-disable-isolation" $(CARGO_MIRI) test --lib memory::mmap_vec:: -- --test-threads=1

# src/thread under Miri (plan.md C10). Checks atomic_ext (AsAtomic, lock-free
# RMW helpers), linux_futex (FutexMutex, FutexCondvar, FutexRwLock via Miri's
# emulated SYS_futex), and instance_tls (InstanceTls, TlsPool, lifecycle/drop
# cleanup). Inline x86_64 assembly is skipped under cfg(miri).
miri_thread:
	$(CARGO_MIRI) test --lib thread:: -- --test-threads=1

# src/ffi under Miri (plan.md C11). Checks all 30 FFI unit tests under Stacked
# Borrows with leak checking enabled (null, misaligned, overflow length,
# embedded-NUL error strings, re-entrant callback unregistration, CResult mapping,
# CBuffer/CString lifecycle, FastVec/MemoryPool/BlobStore/SuffixArray/RadixSort).
miri_ffi:
	$(CARGO_MIRI) test --lib --features ffi ffi:: -- --test-threads=1

# The suites whose #[target_feature] kernels differ between the two builds,
# rebuilt with no target features (MIRI_SCALAR=1, target/miri-scalar), so the
# scalar code those kernels replace is Miri-checked too. miri_core, miri_pool
# and miri_simd run no #[target_feature] kernel in either build, so they are
# not repeated. Which kernels miri_tests and miri_full run in each build has
# not been fully measured.
miri_scalar:
	$(MAKE) --no-print-directory MIRI_SCALAR=1 miri_cspp miri_string miri_io miri_algorithms

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

unwrap_audit:
	python3 scripts/unwrap_audit.py

hygiene_audit:
	python3 scripts/hygiene_audit.py

# Sanity check: clippy gate + audits + tests + drift guard (Rule B8)
# Every cargo command uses --locked: the gate runs against the committed
# Cargo.lock and fails if Cargo.toml no longer matches it, instead of
# re-resolving dependencies.
sanity:
	@echo "=== Clippy (all targets, all features, deny warnings) ==="
	$(CARGO) clippy --locked --all-targets --all-features -- -D warnings
	@echo "=== Unsafe Code Audit (B7) ==="
	python3 scripts/unsafe_audit.py
	@echo "=== API Honesty Audit (D1) ==="
	python3 scripts/api_honesty.py
	@echo "=== Decoder Indexing Safety Audit (D10.1) ==="
	python3 scripts/index_audit.py
	@echo "=== Production Unwrap Audit (Rule D10 / S2-R1) ==="
	python3 scripts/unwrap_audit.py
	@echo "=== Phase D Hygiene Audit (Rules D6 / D7 / D11) ==="
	python3 scripts/hygiene_audit.py
	@echo "=== No default features (guard feature-gated cfg attrs) ==="
	$(CARGO) clippy --locked --no-default-features -- -D unused_variables -D unused_imports
	@mkdir -p target/sanity
	@echo "=== Tests: debug (--all-features --tests) ==="
	$(CARGO) test --locked --all-features --tests 2>&1 | tee target/sanity/debug_tests.log
	@echo "=== Tests: release lib (--all-features --release) ==="
	$(CARGO) test --locked --release --lib --all-features 2>&1 | tee target/sanity/release_lib.log
	@echo "=== Tests: doctests (--doc) ==="
	$(CARGO) test --locked --doc 2>&1 | tee target/sanity/doctests.log
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

# Cargo.lock is committed, so dependency versions change only in a commit.
# Refresh policy: monthly and before each release, run `make update`, then
# `make sanity` (which uses --locked, so it tests the new lock), and commit
# Cargo.lock on its own, with the sanity result in the message. Between
# refreshes, the CI job latest_deps builds and tests without the lock.
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
	@echo "  miri_full        Miri via run_miri_tests.sh full"
	@echo "  miri_core        Miri on the container and hash-map cores"
	@echo "  miri_cspp        Miri on ConcurrentCsppTrie (races/UB, small scale)"
	@echo "  miri_pool        Miri on the memory pools (layout/provenance UB)"
	@echo "  miri_simd        Miri on bit_ops, stream_vbyte, simd_search"
	@echo "  miri_string      Miri on src/string"
	@echo "  miri_algorithms  Miri on src/algorithms (Tree Borrows for 13 radix tests)"
	@echo "  miri_scalar      miri_cspp/string/algorithms with no target features"
	@echo "                   (miri_* build with haswell features on x86_64)"
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
