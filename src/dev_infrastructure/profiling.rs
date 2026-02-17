//! Profiling integration - minimal stub.
//!
//! Previously 4,491 LOC of profiling framework (Profiler trait, ProfilerScope,
//! ProfilerRegistry, HardwareProfiler, MemoryProfiler, CacheProfiler,
//! ProfilerReporter, etc.). Gutted to match topling-zip's ~150 LOC approach.
//!
//! The actual timing/benchmarking utilities live in `debug.rs` (ScopedTimer,
//! HighPrecisionTimer, BenchmarkSuite, format_duration). This file previously
//! duplicated that functionality with excessive abstraction layers.
//!
//! For profiling, use:
//! - `debug::ScopedTimer` for RAII-based timing
//! - `debug::HighPrecisionTimer` for manual timing
//! - `debug::BenchmarkSuite` for running benchmarks
//! - `debug::format_duration` for human-readable durations
