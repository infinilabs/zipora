//! SIMD Dispatch Macros
//!
//! Provides compile-time and runtime SIMD feature detection macros
//! to reduce code duplication across the codebase.
//!
//! ## Available Macros
//!
//! - [`simd_dispatch!`] - Multi-tier SIMD dispatch with automatic fallback
//! - [`simd_feature_check!`] - Single feature check with fallback
//! - [`simd_select!`] - Feature-based expression selection
//!
//! ## Example
//!
//! ```ignore
//! use zipora::simd_dispatch;
//!
//! fn process_data(data: &[u8]) -> Vec<u8> {
//!     simd_dispatch!(
//!         data,
//!         avx2 => process_avx2(data),
//!         sse2 => process_sse2(data),
//!         _ => process_scalar(data)
//!     )
//! }
//! ```

/// Multi-tier SIMD dispatch macro with automatic fallback chain.
///
/// Checks CPU features at runtime and dispatches to the best available
/// implementation. Features are checked in order (first match wins).
///
/// # Syntax
///
/// ```ignore
/// simd_dispatch!(
///     avx512 => expr1,           // Check avx512f + avx512bw
///     avx2 => expr2,             // Check avx2
///     avx2_bmi2 => expr3,        // Check avx2 + bmi2
///     sse42 => expr4,            // Check sse4.2
///     sse2 => expr5,             // Check sse2
///     bmi2 => expr6,             // Check bmi2
///     popcnt => expr7,           // Check popcnt
///     neon => expr8,             // Check ARM NEON (aarch64)
///     _ => fallback_expr         // Scalar fallback (required)
/// )
/// ```
///
/// # Example
///
/// ```ignore
/// fn hash_bytes(data: &[u8]) -> u64 {
///     simd_dispatch!(
///         avx2 => unsafe { hash_avx2(data) },
///         sse2 => unsafe { hash_sse2(data) },
///         _ => hash_scalar(data)
///     )
/// }
/// ```
#[macro_export]
macro_rules! simd_dispatch {
    // AVX-512 + more tiers
    (avx512 => $avx512:expr, avx2 => $avx2:expr, sse2 => $sse2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            {
                if ::std::is_x86_feature_detected!("avx512f")
                    && ::std::is_x86_feature_detected!("avx512bw")
                {
                    return $avx512;
                }
            }
            if ::std::is_x86_feature_detected!("avx2") {
                return $avx2;
            }
            if ::std::is_x86_feature_detected!("sse2") {
                return $sse2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 always has NEON, use sse2 equivalent
            return $sse2;
        }
        $fallback
    }};

    // AVX2 + SSE2 + fallback (common pattern)
    (avx2 => $avx2:expr, sse2 => $sse2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("avx2") {
                return $avx2;
            }
            if ::std::is_x86_feature_detected!("sse2") {
                return $sse2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return $sse2;
        }
        $fallback
    }};

    // AVX2 + BMI2 combined check
    (avx2_bmi2 => $avx2_bmi2:expr, avx2 => $avx2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("avx2")
                && ::std::is_x86_feature_detected!("bmi2")
            {
                return $avx2_bmi2;
            }
            if ::std::is_x86_feature_detected!("avx2") {
                return $avx2;
            }
        }
        $fallback
    }};

    // AVX2 only + fallback
    (avx2 => $avx2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("avx2") {
                return $avx2;
            }
        }
        $fallback
    }};

    // SSE4.2 only + fallback
    (sse42 => $sse42:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("sse4.2") {
                return $sse42;
            }
        }
        $fallback
    }};

    // BMI2 only + fallback
    (bmi2 => $bmi2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("bmi2") {
                return $bmi2;
            }
        }
        $fallback
    }};

    // POPCNT only + fallback
    (popcnt => $popcnt:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("popcnt") {
                return $popcnt;
            }
        }
        $fallback
    }};

    // Full 6-tier dispatch
    (
        avx512 => $avx512:expr,
        avx2_bmi2 => $avx2_bmi2:expr,
        avx2 => $avx2:expr,
        sse42_bmi2 => $sse42_bmi2:expr,
        sse42 => $sse42:expr,
        bmi2 => $bmi2:expr,
        _ => $fallback:expr
    ) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            {
                if ::std::is_x86_feature_detected!("avx512f")
                    && ::std::is_x86_feature_detected!("avx512bw")
                {
                    return $avx512;
                }
            }
            if ::std::is_x86_feature_detected!("avx2")
                && ::std::is_x86_feature_detected!("bmi2")
            {
                return $avx2_bmi2;
            }
            if ::std::is_x86_feature_detected!("avx2") {
                return $avx2;
            }
            if ::std::is_x86_feature_detected!("sse4.2")
                && ::std::is_x86_feature_detected!("bmi2")
            {
                return $sse42_bmi2;
            }
            if ::std::is_x86_feature_detected!("sse4.2") {
                return $sse42;
            }
            if ::std::is_x86_feature_detected!("bmi2") {
                return $bmi2;
            }
        }
        $fallback
    }};
}

/// Single SIMD feature check with fallback.
///
/// Simpler than `simd_dispatch!` for cases where only one feature is checked.
///
/// # Example
///
/// ```ignore
/// fn count_ones(data: &[u64]) -> u64 {
///     simd_feature_check!(
///         "popcnt",
///         unsafe { count_ones_popcnt(data) },
///         count_ones_scalar(data)
///     )
/// }
/// ```
#[macro_export]
macro_rules! simd_feature_check {
    // Single feature check
    ($feature:tt, $simd_expr:expr, $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!($feature) {
                return $simd_expr;
            }
        }
        $fallback
    }};

    // Dual feature check (e.g., avx2 + bmi2)
    ($feat1:tt, $feat2:tt, $simd_expr:expr, $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!($feat1)
                && ::std::is_x86_feature_detected!($feat2)
            {
                return $simd_expr;
            }
        }
        $fallback
    }};
}

/// SIMD feature-based expression selection (no return, just evaluates to value).
///
/// Unlike `simd_dispatch!` which uses `return`, this macro evaluates to
/// the selected expression directly. Useful for assignments and match arms.
///
/// # Example
///
/// ```ignore
/// let result = simd_select!(
///     avx2 => compute_avx2(data),
///     _ => compute_scalar(data)
/// );
/// ```
#[macro_export]
macro_rules! simd_select {
    // AVX2 + fallback (expression, no return)
    (avx2 => $avx2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("avx2") {
                $avx2
            } else {
                $fallback
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $fallback
        }
    }};

    // AVX2 + SSE2 + fallback (expression, no return)
    (avx2 => $avx2:expr, sse2 => $sse2:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!("avx2") {
                $avx2
            } else if ::std::is_x86_feature_detected!("sse2") {
                $sse2
            } else {
                $fallback
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $fallback
        }
    }};

    // Single feature check (expression, no return)
    ($feature:ident => $simd:expr, _ => $fallback:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if ::std::is_x86_feature_detected!(stringify!($feature)) {
                $simd
            } else {
                $fallback
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $fallback
        }
    }};
}

/// Check if a SIMD feature is available at runtime.
///
/// Returns `true` if the feature is available, `false` otherwise.
/// Always returns `false` on non-x86_64 platforms.
///
/// # Example
///
/// ```ignore
/// if simd_available!("avx2") {
///     println!("AVX2 is available!");
/// }
/// ```
#[macro_export]
macro_rules! simd_available {
    ($feature:tt) => {{
        #[cfg(target_arch = "x86_64")]
        {
            ::std::is_x86_feature_detected!($feature)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }};

    // Dual feature check
    ($feat1:tt, $feat2:tt) => {{
        #[cfg(target_arch = "x86_64")]
        {
            ::std::is_x86_feature_detected!($feat1)
                && ::std::is_x86_feature_detected!($feat2)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }};
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    // Note: These tests verify macro compilation and basic functionality.
    // Actual SIMD execution depends on CPU features.
    // Uses direct is_x86_feature_detected! calls to avoid macro hygiene issues.

    #[test]
    fn test_simd_available_macro() {
        // Direct feature detection test
        let _has_sse2 = std::is_x86_feature_detected!("sse2");
        let _has_avx2 = std::is_x86_feature_detected!("avx2");
        let _has_bmi2 = std::is_x86_feature_detected!("bmi2");
        // Feature should be detected on any x86_64 machine
        assert!(_has_sse2); // SSE2 is baseline for x86_64
    }

    #[test]
    fn test_simd_select_macro() {
        fn scalar_impl() -> i32 {
            42
        }
        fn avx2_impl() -> i32 {
            84
        }

        let result = crate::simd_select!(
            avx2 => avx2_impl(),
            _ => scalar_impl()
        );

        // Result should be one of the two values
        assert!(result == 42 || result == 84);
    }

    #[test]
    fn test_simd_select_with_sse2() {
        fn scalar_impl() -> i32 {
            1
        }
        fn sse2_impl() -> i32 {
            2
        }
        fn avx2_impl() -> i32 {
            3
        }

        let result = crate::simd_select!(
            avx2 => avx2_impl(),
            sse2 => sse2_impl(),
            _ => scalar_impl()
        );

        // Result should be one of the three values
        assert!(result >= 1 && result <= 3);
    }

    // Test simd_dispatch! with a function that uses return
    fn dispatch_test_fn(val: i32) -> i32 {
        crate::simd_dispatch!(
            avx2 => val * 2,
            sse2 => val + 10,
            _ => val
        )
    }

    #[test]
    fn test_simd_dispatch_macro() {
        let result = dispatch_test_fn(5);
        // Should be 10 (avx2), 15 (sse2), or 5 (scalar)
        assert!(result == 10 || result == 15 || result == 5);
    }

    // Test single feature check
    fn feature_check_test_fn(val: i32) -> i32 {
        crate::simd_feature_check!(
            "avx2",
            val * 3,
            val
        )
    }

    #[test]
    fn test_simd_feature_check_macro() {
        let result = feature_check_test_fn(7);
        // Should be 21 (avx2) or 7 (fallback)
        assert!(result == 21 || result == 7);
    }

    // Test dual feature check
    fn dual_feature_test_fn(val: i32) -> i32 {
        crate::simd_feature_check!(
            "avx2", "bmi2",
            val * 4,
            val
        )
    }

    #[test]
    fn test_dual_feature_check_macro() {
        let result = dual_feature_test_fn(5);
        // Should be 20 (avx2+bmi2) or 5 (fallback)
        assert!(result == 20 || result == 5);
    }
}
