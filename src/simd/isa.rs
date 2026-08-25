//! ISA capability tokens: typestate proof that CPU feature detection ran.
//!
//! A token value can only be obtained through [`detect`], which runs the
//! feature check first. Kernel wrappers take a token parameter and are safe
//! functions — the SAFETY argument for their internal `unsafe` block is
//! carried by the token instead of being re-proven at every call site.
//!
//! Policy: **new** SIMD modules use tokens; existing modules keep their
//! current dispatch (`simd_dispatch!` / cached `has_*` bools) — migrating
//! them is churn without behavior change.
//!
//! All `is_enabled` implementations return `false` under Miri (directly or
//! via the cached `has_*` helpers), so token-guarded SIMD paths are never
//! taken there.

/// Proof-of-detection marker. Implementors are zero-sized tokens.
///
/// # Safety
///
/// Implementor contract: a value may only be constructed after
/// `is_enabled()` returned `true`. [`detect`] enforces this; implementors
/// must not offer any other public constructor.
pub unsafe trait Isa: Copy + 'static {
    fn is_enabled() -> bool;

    /// # Safety
    ///
    /// Caller must have verified `is_enabled()` returned `true`.
    unsafe fn new_unchecked() -> Self;
}

/// Returns a token iff the ISA is available on this machine.
#[inline]
pub fn detect<S: Isa>() -> Option<S> {
    // SAFETY: `is_enabled()` verified true on this same call.
    S::is_enabled().then(|| unsafe { S::new_unchecked() })
}

macro_rules! isa_token {
    ($(#[$doc:meta])* $name:ident => $enabled:expr) => {
        $(#[$doc])*
        // dead_code: tokens are the sanctioned dispatch pattern for future
        // SIMD modules; only some have production consumers yet.
        #[allow(dead_code)]
        #[derive(Debug, Clone, Copy)]
        pub struct $name(());

        // SAFETY: the private unit field makes `detect` (via `new_unchecked`)
        // the only constructor; `is_enabled` gates it per the trait contract.
        unsafe impl Isa for $name {
            #[inline]
            fn is_enabled() -> bool {
                $enabled
            }

            #[inline]
            unsafe fn new_unchecked() -> Self {
                Self(())
            }
        }
    };
}

isa_token!(
    /// Always-available scalar fallback.
    Scalar => true
);

isa_token!(
    /// x86_64 AVX2 (256-bit integer/float lanes).
    Avx2 => crate::algorithms::simd_search::has_avx2()
);

isa_token!(
    /// x86_64 SSSE3 (`pshufb` shuffle tables).
    Ssse3 => {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            std::arch::is_x86_feature_detected!("ssse3")
        }
        #[cfg(any(not(target_arch = "x86_64"), miri))]
        {
            false
        }
    }
);

isa_token!(
    /// aarch64 NEON (always present on aarch64).
    Neon => cfg!(all(target_arch = "aarch64", not(miri)))
);

isa_token!(
    /// BMI2 with *fast* PDEP/PEXT: excludes AMD Zen 1/2, whose microcoded
    /// PDEP (250-300 cycles) makes CPUID's BMI2 bit an insufficient guard.
    FastBmi2 => crate::algorithms::bit_ops::has_fast_bmi2()
);

// AVX-512 is NOT one token — zipora kernels need three distinct feature
// sets. A monolithic token gated on avx512f alone would SIGILL
// VPOPCNTDQ kernels on Skylake-X/Cascade Lake; gated on vpopcntdq it
// would wrongly disable f-only kernels there.

#[cfg(feature = "avx512")]
isa_token!(
    /// AVX-512 Foundation only (512-bit lanes, mask registers).
    Avx512F => crate::algorithms::simd_search::has_avx512f()
);

#[cfg(feature = "avx512")]
isa_token!(
    /// avx512f + avx512vpopcntdq (Ice Lake / Zen 4+): per-lane popcount.
    Avx512Vpopcntdq => {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        }
        #[cfg(any(not(target_arch = "x86_64"), miri))]
        {
            false
        }
    }
);

#[cfg(feature = "avx512")]
isa_token!(
    /// avx512f + avx512bw (byte/word ops) — the set `simd_dispatch!` checks.
    Avx512Bw => {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
        }
        #[cfg(any(not(target_arch = "x86_64"), miri))]
        {
            false
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_token_always_available() {
        assert!(detect::<Scalar>().is_some());
    }

    #[test]
    fn test_tokens_agree_with_cached_helpers() {
        assert_eq!(
            detect::<Avx2>().is_some(),
            crate::algorithms::simd_search::has_avx2()
        );
        assert_eq!(
            detect::<FastBmi2>().is_some(),
            crate::algorithms::bit_ops::has_fast_bmi2()
        );
    }

    #[test]
    #[cfg(feature = "avx512")]
    fn test_avx512_subset_implications() {
        // vpopcntdq/bw tokens imply the f-only token.
        if detect::<Avx512Vpopcntdq>().is_some() {
            assert!(detect::<Avx512F>().is_some());
        }
        if detect::<Avx512Bw>().is_some() {
            assert!(detect::<Avx512F>().is_some());
        }
    }

    #[test]
    fn test_token_as_kernel_parameter() {
        // A kernel taking a token is an ordinary safe fn.
        fn kernel(_proof: Scalar, x: u32) -> u32 {
            x + 1
        }
        let tok = detect::<Scalar>().unwrap();
        assert_eq!(kernel(tok, 41), 42);
    }
}
