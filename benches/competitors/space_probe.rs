//! Fresh-process space probes (Task 3.0). The parent spawns
//! `current_exe --space-probe <key> <n>`; the child measures and prints
//! `SPACE <key> <value>`. Child processes skip `init_bench_allocator_policy()`.

use std::process::Command;

const PAGE: usize = 4096;

/// Resident set size in bytes. Reads `Rss:` from `/proc/self/smaps_rollup`, which walks the
/// page tables and is exact; `/proc/self/statm` reads the kernel's batched per-CPU RSS counters
/// (Linux >= 6.2), which can lag by hundreds of KiB.
pub fn rss_bytes() -> usize {
    let exact = std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Rss:"))
                .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<usize>().ok())
        });
    exact.map(|kb| kb * 1024).unwrap_or_else(|| {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
            .unwrap_or(0)
            * PAGE
    })
}

/// Faults in every page of `v`'s spare capacity so a probe does not count the vector's own
/// pages. A zero fill is not enough: LLVM may turn `alloc` + `memset(0)` into `calloc`, whose
/// fresh pages stay untouched.
pub fn touch_spare_capacity<T>(v: &mut Vec<T>) {
    let spare = v.spare_capacity_mut();
    let bytes = std::mem::size_of_val(spare);
    let base = spare.as_mut_ptr().cast::<u8>();
    for off in (0..bytes).step_by(PAGE) {
        // SAFETY: `off < bytes`, so the write stays inside the vector's spare capacity, which is
        // uninitialized memory the vector owns.
        unsafe { base.add(off).write_volatile(0) };
    }
}

/// Resets `VmHWM` to the current RSS (Linux >= 4.0, `clear_refs` value 5).
pub fn reset_peak_rss() {
    std::fs::write("/proc/self/clear_refs", "5")
        .expect("failed to write 5 to /proc/self/clear_refs to reset VmHWM");
}

/// Peak resident set size in bytes (`VmHWM` from `/proc/self/status`).
pub fn peak_rss_bytes() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmHWM:"))
                .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<usize>().ok())
        })
        .unwrap_or(0)
        * 1024
}

/// Parent side: spawn the probe and parse its single `SPACE` line.
pub fn probe(key: &str, n: usize) -> f64 {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .args(["--space-probe", key, &n.to_string()])
        .output()
        .expect("spawn space probe");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|l| {
            let mut it = l.strip_prefix("SPACE ")?.split_whitespace();
            if it.next()? != key {
                return None;
            }
            it.next()?.parse::<f64>().ok()
        })
        .unwrap_or_else(|| {
            panic!(
                "space probe {key} failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
        })
}

/// Child side: returns `true` when this process was a probe and must exit.
pub fn run_if_requested(args: &[String]) -> bool {
    let Some(i) = args.iter().position(|a| a == "--space-probe") else {
        return false;
    };
    let key = args.get(i + 1).expect("--space-probe <key> <n>");
    let n: usize = args
        .get(i + 2)
        .and_then(|v| v.parse().ok())
        .expect("--space-probe <key> <n>");
    // Same 4 KiB page backing as the timed runs (the parent may run with `--thp-on`).
    crate::common::disable_transparent_hugepages_for_self();
    let value = super::memory_pools::space_probe_memory_pools(key)
        .or_else(|| super::radix_sort::space_probe_radix_sort(key, n))
        .unwrap_or_else(|| panic!("unknown or failed space probe {key}"));
    println!("SPACE {key} {value:.4}");
    true
}
