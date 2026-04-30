//! Profiling integration - minimal stub.
//!
//! Previously 4,491 LOC of profiling framework (Profiler trait, ProfilerScope,
//! ProfilerRegistry, HardwareProfiler, MemoryProfiler, CacheProfiler,
//! ProfilerReporter, etc.).
//!
//! The actual timing/benchmarking utilities live in `debug.rs` (ScopedTimer,
//! DevHighPrecisionTimer, DevBenchmarkSuite, format_duration). This file previously
//! duplicated that functionality with excessive abstraction layers.
//!
//! For profiling, use:
//! - `debug::ScopedTimer` for RAII-based timing
//! - `debug::DevHighPrecisionTimer` for manual timing
//! - `debug::DevBenchmarkSuite` for running benchmarks
//! - `debug::format_duration` for human-readable durations
