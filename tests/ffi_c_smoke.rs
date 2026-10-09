#![cfg(feature = "ffi")]

//! C11 / F3 verification suite:
//! 1. Verifies `export_count == ffi_guard_count == 24` across `src/ffi/*.rs` and every symbol
//!    is declared in `include/zipora.h`.
//! 2. Verifies byte-for-byte parity between `cbindgen`'s build output (`OUT_DIR/zipora.h`) and
//!    the committed `include/zipora.h`.
//! 3. Verifies ISO C11 (`gcc -std=c11 -Wall -Wextra -Wpedantic -Werror`) and ISO C++17
//!    (`g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror`) header compilation and compiles + runs
//!    a real standalone C smoke caller linked against `libzipora.a`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct CleanupDir(PathBuf);

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn build_fresh_staticlib(root: &Path, out_archive: &Path) {
    let build_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ffi_staticlib_target");
    let _ = fs::remove_dir_all(&build_dir);
    let _cleanup = CleanupDir(build_dir.clone());

    let output = Command::new("cargo")
        .current_dir(root)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .arg("rustc")
        .args([
            "--locked",
            "--lib",
            "--features",
            "ffi",
            "--crate-type",
            "staticlib",
            "--target-dir",
        ])
        .arg(&build_dir)
        .output()
        .expect("failed to invoke cargo rustc --lib --features ffi --crate-type staticlib");
    assert!(
        output.status.success(),
        "cargo rustc --lib --features ffi --crate-type staticlib failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let built = build_dir.join("debug/libzipora.a");
    assert!(
        built.exists(),
        "expected fresh staticlib at {}",
        built.display()
    );
    fs::copy(&built, out_archive).expect("failed to copy fresh libzipora.a to smoke dir");
}

#[test]
fn test_c11_export_and_ffi_guard_parity_and_header_coverage() {
    let root = repo_root();
    let ffi_dir = root.join("src/ffi");
    let header = fs::read_to_string(root.join("include/zipora.h"))
        .expect("include/zipora.h must exist in repository");

    let mut exported_names = Vec::new();
    let mut ffi_guard_calls = 0usize;

    for entry in fs::read_dir(&ffi_dir).expect("failed to read src/ffi") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let content = fs::read_to_string(&path).expect("failed to read FFI source file");
        let prod_code = content
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .unwrap_or(&content);

        let lines: Vec<&str> = prod_code.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim() == "#[unsafe(no_mangle)]" {
                // Find the following `pub unsafe extern "C" fn <name>`
                let sig_line = lines[i + 1..]
                    .iter()
                    .find(|l| l.contains("pub unsafe extern \"C\" fn "))
                    .unwrap_or_else(|| {
                        panic!("missing `pub unsafe extern \"C\" fn` after #[unsafe(no_mangle)] in {}", path.display())
                    });
                let after_fn = sig_line
                    .split("pub unsafe extern \"C\" fn ")
                    .nth(1)
                    .expect("expected function identifier");
                let fn_name = after_fn
                    .split('(')
                    .next()
                    .expect("expected `(` after function name")
                    .trim();
                exported_names.push(fn_name.to_string());
            }
            if line.contains("ffi_guard(") && !line.trim_start().starts_with("pub(crate) fn ffi_guard") {
                ffi_guard_calls += 1;
            }
        }
    }

    exported_names.sort();
    assert_eq!(
        exported_names.len(),
        24,
        "expected exactly 24 #[unsafe(no_mangle)] C exports, found {}: {:?}",
        exported_names.len(),
        exported_names
    );
    assert_eq!(
        ffi_guard_calls,
        exported_names.len(),
        "every exported C FFI function must wrap its body in `ffi_guard`"
    );

    for sym in &exported_names {
        assert!(
            header.contains(&format!("{}(", sym)),
            "exported C symbol `{}` missing from include/zipora.h",
            sym
        );
    }
}

#[test]
fn test_c11_cbindgen_header_matches_committed_include_header() {
    let root = repo_root();
    let committed = fs::read_to_string(root.join("include/zipora.h"))
        .expect("include/zipora.h must exist in repository");
    let generated_path = PathBuf::from(env!("OUT_DIR")).join("zipora.h");
    let generated = fs::read_to_string(&generated_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", generated_path.display(), e));

    assert_eq!(
        committed, generated,
        "include/zipora.h is out of sync with cbindgen output in {}",
        generated_path.display()
    );
}

#[test]
fn test_f3_c11_and_cpp17_header_compile_and_linked_c_smoke_binary_runs() {
    let root = repo_root();
    let header_path = root.join("include/zipora.h");

    // 1. Strict ISO C11 header syntax check
    let c_check = Command::new("gcc")
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsyntax-only",
            "-x",
            "c",
        ])
        .arg(&header_path)
        .output()
        .expect("failed to invoke gcc for C11 header check");
    assert!(
        c_check.status.success(),
        "gcc C11 header check failed:\n{}",
        String::from_utf8_lossy(&c_check.stderr)
    );

    // 2. Strict ISO C++17 header syntax check
    let cpp_check = Command::new("g++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fsyntax-only",
            "-x",
            "c++",
        ])
        .arg(&header_path)
        .output()
        .expect("failed to invoke g++ for C++17 header check");
    assert!(
        cpp_check.status.success(),
        "g++ C++17 header check failed:\n{}",
        String::from_utf8_lossy(&cpp_check.stderr)
    );

    // 3. Compile and link an end-to-end C11 smoke caller exercising all C API groups
    let tmp = tempfile::tempdir().expect("failed to create tempdir for C smoke build");
    let c_src = tmp.path().join("ffi_smoke.c");
    let bin_path = tmp.path().join("ffi_smoke_bin");
    fs::write(
        &c_src,
        r#"#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "zipora.h"

static int g_cb_count = 0;
static void smoke_error_callback(const char *msg) {
    if (msg != NULL && strlen(msg) > 0) {
        g_cb_count += 1;
    }
}

int main(void) {
    assert(zipora_init() == Success);
    const char *ver = zipora_version();
    assert(ver != NULL && strlen(ver) > 0);
    int simd = zipora_has_simd();
    assert(simd == 0 || simd == 1);

    /* FastVec lifecycle */
    CFastVec *vec = fast_vec_new();
    assert(vec != NULL);
    assert(fast_vec_len(vec) == 0);
    assert(fast_vec_push(vec, 11) == Success);
    assert(fast_vec_push(vec, 22) == Success);
    assert(fast_vec_len(vec) == 2);
    const uint8_t *vdata = fast_vec_data(vec);
    assert(vdata != NULL && vdata[0] == 11 && vdata[1] == 22);
    fast_vec_free(vec);

    /* MemoryPool lifecycle */
    CMemoryPool *pool = memory_pool_new(256, 8);
    assert(pool != NULL);
    void *chunk = memory_pool_allocate(pool);
    assert(chunk != NULL);
    assert(memory_pool_deallocate(pool, chunk) == Success);
    memory_pool_free(pool);

    /* BlobStore lifecycle + missing record NotFound check */
    CBlobStore *store = blob_store_new();
    assert(store != NULL);
    const uint8_t payload[] = "c-smoke-blob";
    uint32_t rec_id = 0;
    assert(blob_store_put(store, payload, sizeof(payload) - 1, &rec_id) == Success);

    const uint8_t *got_ptr = NULL;
    uintptr_t got_len = 0;
    assert(blob_store_get(store, rec_id, &got_ptr, &got_len) == Success);
    assert(got_len == sizeof(payload) - 1);
    assert(memcmp(got_ptr, payload, got_len) == 0);
    zipora_free_blob_data((uint8_t *)got_ptr, got_len);

    assert(blob_store_get(store, 999999u, &got_ptr, &got_len) == NotFound);
    blob_store_free(store);

    /* SuffixArray + RadixSort */
    const uint8_t text[] = "abracadabra";
    const uint8_t pat[] = "abra";
    CSuffixArray *sa = suffix_array_new(text, sizeof(text) - 1);
    assert(sa != NULL);
    assert(suffix_array_len(sa) == sizeof(text) - 1);
    uintptr_t sa_start = 0, sa_count = 0;
    assert(suffix_array_search(sa, text, sizeof(text) - 1, pat, sizeof(pat) - 1, &sa_start, &sa_count) == Success);
    assert(sa_count == 2);
    suffix_array_free(sa);

    uint32_t nums[5] = {42, 7, 19, 1, 99};
    assert(radix_sort_u32(nums, 5) == Success);
    assert(nums[0] == 1 && nums[1] == 7 && nums[2] == 19 && nums[3] == 42 && nums[4] == 99);

    /* Error callback + last error */
    zipora_set_error_callback(smoke_error_callback);
    assert(fast_vec_push(NULL, 1) == InvalidInput);
    assert(g_cb_count == 1);
    assert(strlen(zipora_last_error()) > 0);
    zipora_set_error_callback(NULL);

    return 0;
}
"#,
    )
    .expect("failed to write C smoke source");

    let libzipora = tmp.path().join("libzipora.a");
    build_fresh_staticlib(&root, &libzipora);
    let compile = Command::new("gcc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(root.join("include"))
        .arg(&c_src)
        .arg(&libzipora)
        .args(["-lpthread", "-ldl", "-lm", "-o"])
        .arg(&bin_path)
        .output()
        .expect("failed to invoke gcc to link C smoke test");
    assert!(
        compile.status.success(),
        "failed to compile/link C smoke test against {}:\n{}",
        libzipora.display(),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin_path)
        .output()
        .expect("failed to execute compiled C smoke binary");
    assert!(
        run.status.success(),
        "C smoke binary exited with non-zero status {:?}:\nstdout: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn test_s31_r2_cargo_toml_lib_crate_type_is_rlib_only() {
    let manifest = fs::read_to_string(repo_root().join("Cargo.toml"))
        .expect("Cargo.toml must be readable");
    let lib_section = manifest
        .split("\n[lib]\n")
        .nth(1)
        .and_then(|s| s.split("\n[").next())
        .expect("[lib] section must exist in Cargo.toml");
    let crate_type_line = lib_section
        .lines()
        .find(|line| line.trim_start().starts_with("crate-type"))
        .expect("crate-type key must exist in [lib]");
    assert_eq!(
        crate_type_line.trim(),
        "crate-type = [\"rlib\"]",
        "Cargo.toml [lib] crate-type must default to rlib only so non-FFI builds do not emit staticlib/cdylib"
    );
}
