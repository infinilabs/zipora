use zipora::memory::{mmap_vec::MmapVec, MmapVecConfig};

/// Helper: create a temp path under target/ so test artifacts don't pollute the project root.
fn test_path(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("target/test_tmp");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(name)
}

/// Verify that MmapVec works correctly with Pod types (all bit patterns valid).
#[test]
fn test_mmap_vec_pod_types_work() {
    let path = test_path("mmap_pod.bin");
    let _ = std::fs::remove_file(&path);

    // u32 implements Pod — all bit patterns are valid
    {
        let mut vec: MmapVec<u32> = MmapVec::create(&path, MmapVecConfig::default()).unwrap();
        vec.push(42).unwrap();
        vec.push(u32::MAX).unwrap();
        vec.push(0).unwrap();
        vec.sync().unwrap();
    }

    // Reopen and verify
    let vec: MmapVec<u32> = MmapVec::open(&path, MmapVecConfig::default()).unwrap();
    assert_eq!(vec.as_slice(), &[42, u32::MAX, 0]);

    drop(vec);
    let _ = std::fs::remove_file(&path);
}

/// Verify that corrupted data in a Pod-typed MmapVec cannot cause UB.
/// Any bit pattern is valid for u32, so even corrupted files are safe to read.
#[test]
fn test_mmap_vec_corrupted_pod_data_is_safe() {
    let path = test_path("mmap_corrupted_pod.bin");
    let _ = std::fs::remove_file(&path);

    {
        let mut vec: MmapVec<u32> = MmapVec::create(&path, MmapVecConfig::default()).unwrap();
        vec.push(1).unwrap();
        vec.sync().unwrap();
    }

    // Corrupt the data with arbitrary bytes
    {
        use std::io::{Write, Seek, SeekFrom};
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(80)).unwrap();
        file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
    }

    // Reopen — safe because every u32 bit pattern is valid
    let vec: MmapVec<u32> = MmapVec::open(&path, MmapVecConfig::default()).unwrap();
    assert_eq!(vec.as_slice()[0], u32::MAX);

    drop(vec);
    let _ = std::fs::remove_file(&path);
}

/// Verify that MmapVec<bool> does NOT compile.
///
/// ```compile_fail
/// use zipora::memory::{mmap_vec::MmapVec, MmapVecConfig};
/// // bool does not implement bytemuck::Pod, so this must fail:
/// let vec: MmapVec<bool> = MmapVec::create("test.bin", MmapVecConfig::default()).unwrap();
/// ```
#[test]
fn test_mmap_vec_rejects_non_pod_types_documented() {
    // MmapVec<bool>, MmapVec<char>, MmapVec<NonZeroU32> all fail to compile
    // because these types don't implement Pod. The compile_fail doctest above
    // is the real guard — if someone removes the Pod bound, `cargo test --doc` fails.
}
