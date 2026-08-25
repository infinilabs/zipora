//! One-shot resolved SIMD dispatch (IFUNC pattern).
//!
//! Instead of re-checking CPU features on every call, a dispatch static
//! resolves the best kernel for this machine exactly once (thread-safe) and
//! caches the function pointer; every later call pays one initialized-check
//! plus an indirect call.
//!
//! Usage rules (see the resolver requirements in the module users):
//! - The resolver must publish only **safe** wrapper fns; each wrapper's
//!   internal `unsafe` block cites the feature check the resolver performed.
//! - Outer guards (small-input early exits, argument validation) stay in the
//!   public entry function so they run before the indirect call.
//! - Keep tiny always-inline leaves (e.g. `select_in_word`) on cached-bool
//!   dispatch instead — an indirect call would block inlining.

/// Declares a lazily-resolved dispatch static.
///
/// ```ignore
/// ifunc_dispatch!(static POPCOUNT_IMPL: fn(&[u64]) -> usize = resolve_popcount;);
/// // call site:
/// (POPCOUNT_IMPL)(words)
/// ```
///
/// The resolver runs at most once; `std::sync::LazyLock` guarantees all
/// threads observe the same resolved pointer.
#[macro_export]
macro_rules! ifunc_dispatch {
    ($vis:vis static $name:ident : fn($($arg:ty),* $(,)?) -> $ret:ty = $resolver:expr;) => {
        $vis static $name: std::sync::LazyLock<fn($($arg),*) -> $ret> =
            std::sync::LazyLock::new($resolver);
    };
}

#[cfg(test)]
mod tests {
    fn double(x: u64) -> u64 {
        x * 2
    }

    fn resolve_double() -> fn(u64) -> u64 {
        double
    }

    ifunc_dispatch!(static DOUBLE_IMPL: fn(u64) -> u64 = resolve_double;);

    #[test]
    fn test_resolved_pointer_identical_across_threads() {
        use std::sync::Barrier;

        let barrier = Barrier::new(8);
        let addrs: Vec<usize> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        *DOUBLE_IMPL as usize
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert!(addrs.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn test_resolved_fn_result_matches_direct_call() {
        for x in [0u64, 1, 7, u64::MAX / 2] {
            assert_eq!((DOUBLE_IMPL)(x), double(x));
        }
    }
}
