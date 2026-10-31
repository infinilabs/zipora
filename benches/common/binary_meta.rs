//! Pure-Rust streaming SHA-256 and executable binary path/digest helpers (`S41-R2` + `S42-R7` + `S53-R3`).

#![allow(dead_code)]

use std::fs::File;
use std::io::Read;
use std::process::Command;

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline]
fn sha256_compress_block(h: &mut [u32; 8], chunk: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            chunk[i * 4],
            chunk[i * 4 + 1],
            chunk[i * 4 + 2],
            chunk[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

/// Computes the lower-case 64-character hex SHA-256 digest of a stream using an 8 KiB stack
/// buffer (zero heap allocation for file contents).
pub fn sha256_reader<R: Read>(mut reader: R) -> std::io::Result<String> {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut buf = [0u8; 8192];
    let mut tail = [0u8; 64];
    let mut tail_len = 0usize;
    let mut total_bytes = 0u64;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total_bytes = total_bytes.wrapping_add(n as u64);
        let mut slice = &buf[..n];
        if tail_len > 0 {
            let need = 64 - tail_len;
            if slice.len() >= need {
                tail[tail_len..64].copy_from_slice(&slice[..need]);
                sha256_compress_block(&mut h, &tail);
                tail_len = 0;
                slice = &slice[need..];
            } else {
                tail[tail_len..tail_len + slice.len()].copy_from_slice(slice);
                tail_len += slice.len();
                slice = &[];
            }
        }
        let (chunks, rem) = slice.as_chunks::<64>();
        for block in chunks {
            sha256_compress_block(&mut h, block);
        }
        if !rem.is_empty() {
            tail[..rem.len()].copy_from_slice(rem);
            tail_len = rem.len();
        }
    }

    tail[tail_len] = 0x80;
    tail_len += 1;
    if tail_len > 56 {
        tail[tail_len..64].fill(0);
        sha256_compress_block(&mut h, &tail);
        tail[..56].fill(0);
    } else {
        tail[tail_len..56].fill(0);
    }
    let bit_len = total_bytes.wrapping_mul(8);
    tail[56..64].copy_from_slice(&bit_len.to_be_bytes());
    sha256_compress_block(&mut h, &tail);

    let mut out = String::with_capacity(64);
    for word in h {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{word:08x}");
    }
    Ok(out)
}

/// Computes the lower-case 64-character hex SHA-256 digest of `data` in pure Rust.
pub fn sha256_bytes(data: &[u8]) -> String {
    sha256_reader(data).unwrap_or_default()
}

/// Resolves the short git commit SHA from `ZIPORA_BENCH_GIT_COMMIT` (set by outer runner)
/// or once post-benchmark via `git rev-parse --short=7 HEAD` (appending `-dirty` if `src/`
/// or `benches/` source files differ from `HEAD`).
pub fn detect_git_commit() -> String {
    if let Ok(c) = std::env::var("ZIPORA_BENCH_GIT_COMMIT") {
        let trimmed = c.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    if sha == "unknown" {
        return sha;
    }
    let code_clean = Command::new("git")
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            "Cargo.toml",
            "Cargo.lock",
            "benches/core_bench.rs",
            "benches/competitor_bench.rs",
            "benches/e2e_search_bench.rs",
            "benches/common",
            "benches/competitors",
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(true);
    if code_clean {
        sha
    } else {
        format!("{sha}-dirty")
    }
}

/// Computes the lower-case 64-character hex SHA-256 digest of `Cargo.lock` in the workspace root (`S54-R1`).
pub fn detect_cargo_lock_sha256() -> String {
    let lock_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    File::open(&lock_path)
        .or_else(|_| File::open("Cargo.lock"))
        .and_then(sha256_reader)
        .unwrap_or_default()
}

/// Resolves the canonical path and SHA-256 digest of the running benchmark binary (`current_exe`)
/// post-benchmark via an 8 KiB stack buffer (`S53-R3` + `S54-R1`).
pub fn detect_self_binary_path_and_sha256() -> (String, String) {
    let Ok(exe) = std::env::current_exe() else {
        return (String::new(), String::new());
    };
    let canonical = std::fs::canonicalize(&exe).unwrap_or(exe);
    let path_str = canonical.to_string_lossy().into_owned();
    let sha = File::open(&canonical)
        .and_then(sha256_reader)
        .unwrap_or_default();
    (path_str, sha)
}
