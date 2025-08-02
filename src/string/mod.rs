//! Zero-copy string operations with SIMD optimization
//!
//! This module provides high-performance string types optimized for minimal copying
//! and maximum throughput using SIMD instructions where available.

mod fast_str;

pub use fast_str::FastStr;
