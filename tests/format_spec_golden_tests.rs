//! Golden-file byte layout and round-trip tests for all persisted formats (`C2.1` / `G13` / `S4-R2`).
//!
//! Asserts every documented header/entry field at its exact byte range in `docs/format_spec.md`
//! and verifies lossless round-trip decoding across all 8 persisted formats.

use std::io::Cursor;
#[cfg(any(feature = "mmap", feature = "zstd"))]
use tempfile::tempdir;
use zipora::blob_store::file_header::ChecksumType;
#[cfg(feature = "mmap")]
use zipora::blob_store::reorder_map::{ZReorderMap, ZReorderMapBuilder};
use zipora::blob_store::{
    BlobStore, BlobStoreFileFooter, FileHeaderBase, SortedUintVec, SortedUintVecBuilder,
    SortedUintVecConfig,
};
#[cfg(feature = "zstd")]
use zipora::blob_store::{ZipOffsetBlobStore, ZipOffsetBlobStoreBuilder, ZipOffsetBlobStoreConfig};
use zipora::containers::ZoSortedStrVec;
use zipora::entropy::dictionary::{Dictionary, DictionaryEntry};
use zipora::error::Result;
use zipora::io::{SignedVarInt, VarInt};
#[cfg(feature = "mmap")]
use zipora::memory::{MmapVec, MmapVecConfig};

#[test]
fn test_golden_1_file_header_base_and_footer_every_field() -> Result<()> {
    let mut header = FileHeaderBase::new();
    header.set_class_name("ZipOffsetBlobStore");
    header.set_file_size(4096);
    header.set_unzip_size(8192);
    header.set_records(42);
    header.set_checksum_type(ChecksumType::Crc32c);
    header.set_format_version(1);
    header.set_global_dict_size(1024);

    let bytes = header.as_bytes();
    assert_eq!(bytes.len(), 80);

    // Every field of FileHeaderBase (80 B):
    // 0..1: magic_len = 17
    assert_eq!(bytes[0], 17);
    // 1..20: magic = b"terark-blob-store\0\0"
    assert_eq!(&bytes[1..20], b"terark-blob-store\0\0");
    // 20..40: class_name (20 B NUL-padded)
    assert_eq!(&bytes[20..40], b"ZipOffsetBlobStore\0\0");
    // 40..48: file_size = 4096 (u64 LE)
    assert_eq!(&bytes[40..48], &4096u64.to_le_bytes());
    // 48..56: unzip_size = 8192 (u64 LE)
    assert_eq!(&bytes[48..56], &8192u64.to_le_bytes());
    // 56..64: packed_records_field = 42 | ((ChecksumType::Crc32c as u64) << 40) | (1 << 48)
    let expected_packed_records = 42u64 | ((ChecksumType::Crc32c as u64) << 40) | (1u64 << 48);
    assert_eq!(&bytes[56..64], &expected_packed_records.to_le_bytes());
    // 64..72: packed_dict_field = 1024 (u64 LE)
    assert_eq!(&bytes[64..72], &1024u64.to_le_bytes());
    // 72..80: padding = 0
    assert_eq!(&bytes[72..80], &[0u8; 8]);

    let decoded = FileHeaderBase::from_bytes(bytes)
        .ok_or_else(|| zipora::error::ZiporaError::invalid_data("Invalid FileHeaderBase magic"))?;
    assert_eq!(decoded.magic_len(), 17);
    assert_eq!(decoded.class_name(), "ZipOffsetBlobStore");
    assert_eq!(decoded.file_size(), 4096);
    assert_eq!(decoded.unzip_size(), 8192);
    assert_eq!(decoded.records(), 42);
    assert_eq!(decoded.checksum_type(), ChecksumType::Crc32c);
    assert_eq!(decoded.format_version(), 1);
    assert_eq!(decoded.global_dict_size(), 1024);

    // Every field of BlobStoreFileFooter (64 B):
    let mut footer = BlobStoreFileFooter::new();
    footer.set_zip_data_xxhash(0x1122_3344_5566_7788u64);
    footer.set_file_xxhash(0xDEAD_BEEF_CAFE_BABEu64);
    let fbytes = footer.as_bytes();
    assert_eq!(fbytes.len(), 64);
    // 0..8: zip_data_xxhash
    assert_eq!(&fbytes[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    // 8..16: file_xxhash
    assert_eq!(&fbytes[8..16], &0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    // 16..56: reserved (40 zero bytes)
    assert_eq!(&fbytes[16..56], &[0u8; 40]);
    // 56..60: padding (4 zero bytes)
    assert_eq!(&fbytes[56..60], &[0u8; 4]);
    // 60..64: footer_length = 64 (u32 LE)
    assert_eq!(&fbytes[60..64], &64u32.to_le_bytes());

    let decoded_footer = BlobStoreFileFooter::from_bytes(fbytes);
    assert_eq!(decoded_footer.zip_data_xxhash(), 0x1122_3344_5566_7788u64);
    assert_eq!(decoded_footer.file_xxhash(), 0xDEAD_BEEF_CAFE_BABEu64);
    assert_eq!(decoded_footer.footer_length(), 64);
    Ok(())
}

#[test]
fn test_golden_2_sorted_uint_vec_zsuv_every_field() -> Result<()> {
    let values = vec![0u64, 15, 42, 100, 255];
    let cfg = SortedUintVecConfig::default();
    let mut builder = SortedUintVecBuilder::with_config(cfg);
    for &v in &values {
        builder.push(v)?;
    }
    let suv = builder.finish()?;
    let mut bytes = Vec::new();
    suv.write_to(&mut bytes)?;

    // Every field of the 32-byte ZSUV header:
    // 0..4: magic = b"ZSUV"
    assert_eq!(&bytes[0..4], b"ZSUV");
    // 4..5: version = 1
    assert_eq!(bytes[4], 1);
    // 5..6: log2_block_units
    assert_eq!(bytes[5], cfg.log2_block_units);
    // 6..7: offset_width
    assert_eq!(bytes[6], cfg.offset_width);
    // 7..8: sample_width
    assert_eq!(bytes[7], cfg.sample_width);
    // 8..16: size = 5 (u64 LE)
    assert_eq!(&bytes[8..16], &5u64.to_le_bytes());
    // 16..24: index_len (u64 LE)
    let index_len = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]) as usize;
    // 24..32: data_len (u64 LE)
    let data_len = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]) as usize;
    // Total serialized length = 32 + index_len + data_len
    assert_eq!(bytes.len(), 32 + index_len + data_len);

    let mut cursor = Cursor::new(&bytes);
    let decoded = SortedUintVec::read_from(&mut cursor)?;
    assert_eq!(decoded.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(decoded.get(i)?, v);
    }
    Ok(())
}

#[cfg(feature = "zstd")]
#[test]
fn test_golden_3_zip_offset_blob_store_every_field() -> Result<()> {
    let dir = tempdir().map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    let path = dir.path().join("golden_zip_offset.zbs");

    let cfg = ZipOffsetBlobStoreConfig::default();
    let mut builder = ZipOffsetBlobStoreBuilder::with_config(cfg.clone())?;
    builder.add_record(b"alpha_record")?;
    builder.add_record(b"beta_record_payload")?;
    builder.add_record(b"gamma")?;
    let store = builder.finish()?;
    store.save_to_file(&path)?;

    let raw =
        std::fs::read(&path).map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;

    // Every field of the 128-byte ZipOffsetBlobStore FileHeader:
    // 0..20: magic = b"zipora-blob-store\0\0\0"
    assert_eq!(&raw[0..20], b"zipora-blob-store\0\0\0");
    // 20..40: class_name = b"ZipOffsetBlobStore\0\0"
    assert_eq!(&raw[20..40], b"ZipOffsetBlobStore\0\0");
    // 40..48: file_size (u64 LE) == raw.len()
    let file_size = u64::from_le_bytes([
        raw[40], raw[41], raw[42], raw[43], raw[44], raw[45], raw[46], raw[47],
    ]);
    assert_eq!(file_size as usize, raw.len());
    // 48..56: unzip_size = 12 + 19 + 5 = 36 (u64 LE)
    let unzip_size = u64::from_le_bytes([
        raw[48], raw[49], raw[50], raw[51], raw[52], raw[53], raw[54], raw[55],
    ]);
    assert_eq!(unzip_size, 36);
    // 56..64: records_checksum_version (records = offsets.len() = 4, version = 1 at bits 48..64)
    let rcv = u64::from_le_bytes([
        raw[56], raw[57], raw[58], raw[59], raw[60], raw[61], raw[62], raw[63],
    ]);
    assert_eq!(rcv & 0xFF_FFFF_FFFF, 4); // 3 records + 1 sentinel offset
    assert_eq!((rcv >> 48) as u16, 1); // FORMAT_VERSION = 1
    // 64..72: content_bytes (u64 LE)
    let content_bytes = u64::from_le_bytes([
        raw[64], raw[65], raw[66], raw[67], raw[68], raw[69], raw[70], raw[71],
    ]);
    // 72..80: offsets_bytes (u64 LE)
    let offsets_bytes = u64::from_le_bytes([
        raw[72], raw[73], raw[74], raw[75], raw[76], raw[77], raw[78], raw[79],
    ]);
    let content_padding = (16 - (content_bytes % 16)) % 16;
    assert_eq!(
        128 + content_bytes + content_padding + offsets_bytes + 64,
        file_size
    );
    // 80..83: offsets_log2_block_units, checksum_level, compress_level
    assert_eq!(raw[80], cfg.offset_config.log2_block_units);
    assert_eq!(raw[81], cfg.checksum_level);
    assert_eq!(raw[82], cfg.compress_level);
    // 83..128: zero padding
    assert_eq!(&raw[83..128], &[0u8; 45]);
    // Embedded ZSUV magic at 128 + content_bytes + content_padding
    let zsuv_start = (128 + content_bytes + content_padding) as usize;
    assert_eq!(&raw[zsuv_start..zsuv_start + 4], b"ZSUV");
    // Footer (last 64 bytes): trailing 60 bytes are 0
    let footer_start = raw.len() - 64;
    assert_eq!(&raw[footer_start + 4..raw.len()], &[0u8; 60]);

    let loaded = ZipOffsetBlobStore::load_from_file(&path)?;
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.get(0)?, b"alpha_record");
    assert_eq!(loaded.get(1)?, b"beta_record_payload");
    assert_eq!(loaded.get(2)?, b"gamma");
    Ok(())
}

#[test]
fn test_golden_4_entropy_dictionary_every_field() -> Result<()> {
    let mut dict = Dictionary::new();
    dict.insert(b"abc".to_vec(), DictionaryEntry::new(12, 3));
    dict.insert(b"wxyz".to_vec(), DictionaryEntry::new(40, 4));
    let bytes = dict.serialize();

    // Exact expected wire layout (sorted by sequence key: "abc" then "wxyz"):
    // [0..4]: entry_count = 2u32 LE
    // Entry 0 ("abc"): [seq_len: 3u16 LE][b"abc"][offset: 12u32 LE][length: 3u32 LE] (13 B)
    // Entry 1 ("wxyz"): [seq_len: 4u16 LE][b"wxyz"][offset: 40u32 LE][length: 4u32 LE] (14 B)
    let mut expected = Vec::new();
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&3u16.to_le_bytes());
    expected.extend_from_slice(b"abc");
    expected.extend_from_slice(&12u32.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    expected.extend_from_slice(&4u16.to_le_bytes());
    expected.extend_from_slice(b"wxyz");
    expected.extend_from_slice(&40u32.to_le_bytes());
    expected.extend_from_slice(&4u32.to_le_bytes());

    assert_eq!(bytes, expected);

    let decoded = Dictionary::deserialize(&bytes)?;
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded.get(b"abc"), Some(&DictionaryEntry::new(12, 3)));
    assert_eq!(decoded.get(b"wxyz"), Some(&DictionaryEntry::new(40, 4)));
    assert_eq!(decoded.serialize(), expected);
    Ok(())
}

#[test]
fn test_golden_5_zo_sorted_str_vec_zosv_every_field() -> Result<()> {
    let strings = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "delta".to_string(),
        "omega".to_string(),
    ];
    let zosv = ZoSortedStrVec::from_sorted_strings(strings.clone())?;
    let bytes = zosv.to_bytes()?;

    // Every field of the 16-byte ZOSV header + payload:
    // 0..4: magic = b"ZOSV"
    assert_eq!(&bytes[0..4], b"ZOSV");
    // 4..6: version = 1 (u16 LE)
    assert_eq!(&bytes[4..6], &1u16.to_le_bytes());
    // 6..8: flags = 0x0011 (u16 LE)
    assert_eq!(&bytes[6..8], &0x0011u16.to_le_bytes());
    // 8..12: count = 4 (u32 LE)
    assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
    // 12..16: payload_len = (4 + 5) + (4 + 4) + (4 + 5) + (4 + 5) = 35 (u32 LE)
    assert_eq!(&bytes[12..16], &35u32.to_le_bytes());
    assert_eq!(bytes.len(), 16 + 35);
    // First entry at 16..25: [5u32 LE][b"alpha"]
    assert_eq!(&bytes[16..20], &5u32.to_le_bytes());
    assert_eq!(&bytes[20..25], b"alpha");

    let decoded = ZoSortedStrVec::from_bytes(&bytes)?;
    assert_eq!(decoded.len(), 4);
    for (i, expected) in strings.iter().enumerate() {
        assert_eq!(decoded.get(i), Some(expected.as_str()));
    }
    Ok(())
}

#[test]
fn test_golden_6_varint_leb128_and_zigzag_every_field() -> Result<()> {
    // Unsigned LEB128 for 300 (0b00000010_00101100 -> [0xAC, 0x02])
    let buf = VarInt::encode(300);
    assert_eq!(buf, &[0xAC, 0x02]);
    let (decoded, consumed) = VarInt::decode(&buf)?;
    assert_eq!(decoded, 300);
    assert_eq!(consumed, 2);

    // Signed ZigZag + LEB128: -1 -> 1 ([0x01]), -2 -> 3 ([0x03]), 1 -> 2 ([0x02])
    assert_eq!(VarInt::encode_signed(-1), &[0x01]);
    assert_eq!(VarInt::encode_signed(1), &[0x02]);
    assert_eq!(VarInt::encode_signed(-2), &[0x03]);
    assert_eq!(VarInt::decode_signed(&[0x01])?, (-1, 1));
    assert_eq!(VarInt::decode_signed(&[0x03])?, (-2, 1));
    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
fn test_golden_7_mmap_vec_header_v2_every_field() -> Result<()> {
    let dir = tempdir().map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    let path = dir.path().join("golden_mmap_vec.bin");
    {
        let mut vec = MmapVec::<u64>::create(&path, MmapVecConfig::default())?;
        vec.push(0x0102_0304_0506_0708u64)?;
        vec.push(0x1122_3344_5566_7788u64)?;
        vec.sync()?;
    }

    let bytes =
        std::fs::read(&path).map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    // Every field of the 80-byte MmapVecHeader v2:
    // 0..8: magic = 0x4D4D_4150_5F56_4543 (u64 LE, wire bytes b"CEV_PAMM")
    assert_eq!(&bytes[0..8], &0x4D4D_4150_5F56_4543u64.to_le_bytes());
    assert_eq!(&bytes[0..8], b"CEV_PAMM");
    // 8..10: version = 2 (u16 LE)
    assert_eq!(&bytes[8..10], &2u16.to_le_bytes());
    // 10..12: flags = 0x0011 (u16 LE)
    assert_eq!(&bytes[10..12], &0x0011u16.to_le_bytes());
    // 12..16: element_size = 8 (u32 LE)
    assert_eq!(&bytes[12..16], &8u32.to_le_bytes());
    // 16..24: length = 2 (u64 LE)
    assert_eq!(&bytes[16..24], &2u64.to_le_bytes());
    // 24..32: capacity = 1024 (u64 LE)
    assert_eq!(&bytes[24..32], &1024u64.to_le_bytes());
    // 32..80: reserved (48 zero bytes)
    assert_eq!(&bytes[32..80], &[0u8; 48]);
    // 80..96: first two u64 elements in LE
    assert_eq!(&bytes[80..88], &0x0102_0304_0506_0708u64.to_le_bytes());
    assert_eq!(&bytes[88..96], &0x1122_3344_5566_7788u64.to_le_bytes());

    let reopened = MmapVec::<u64>::open(&path, MmapVecConfig::default())?;
    assert_eq!(reopened.len(), 2);
    assert_eq!(reopened.get(0), Some(&0x0102_0304_0506_0708u64));
    assert_eq!(reopened.get(1), Some(&0x1122_3344_5566_7788u64));
    Ok(())
}

#[cfg(feature = "mmap")]
#[test]
fn test_golden_8_zreorder_map_every_field() -> Result<()> {
    let dir = tempdir().map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    let path = dir.path().join("golden_reorder.zrm");

    let values = [3usize, 2, 1, 0, 4, 5, 6, 7];
    let mut builder = ZReorderMapBuilder::new(&path, values.len(), -1)?;
    for &v in &values {
        builder.push(v)?;
    }
    builder.finish()?;

    let raw =
        std::fs::read(&path).map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    // Total wire size: 16 B header + 6 B descending run [3,2,1,0] + 4 * 5 B singles [4],[5],[6],[7] = 42 B
    assert_eq!(raw.len(), 42);
    // 0..8: size = 8 (u64 LE)
    assert_eq!(&raw[0..8], &8u64.to_le_bytes());
    // 8..16: sign = -1i64 (i64 LE, [0xFF; 8])
    assert_eq!(&raw[8..16], &(-1i64).to_le_bytes());
    // 16..22: Entry 0 (descending run 3, 2, 1, 0: base=3, is_single=0 -> encoded = (3 << 1) | 0 = 6 in 5-byte LE, followed by unsigned var_uint seq_length = 4)
    assert_eq!(&raw[16..21], &[6, 0, 0, 0, 0]);
    assert_eq!(raw[21], 4);
    // 22..27: Entry 1 (single 4: base=4, is_single=1 -> encoded = (4 << 1) | 1 = 9 in 5-byte LE)
    assert_eq!(&raw[22..27], &[9, 0, 0, 0, 0]);
    // 27..32: Entry 2 (single 5: base=5, is_single=1 -> encoded = (5 << 1) | 1 = 11 in 5-byte LE)
    assert_eq!(&raw[27..32], &[11, 0, 0, 0, 0]);
    // 32..37: Entry 3 (single 6: base=6, is_single=1 -> encoded = (6 << 1) | 1 = 13 in 5-byte LE)
    assert_eq!(&raw[32..37], &[13, 0, 0, 0, 0]);
    // 37..42: Entry 4 (single 7: base=7, is_single=1 -> encoded = (7 << 1) | 1 = 15 in 5-byte LE)
    assert_eq!(&raw[37..42], &[15, 0, 0, 0, 0]);

    let loaded = ZReorderMap::open(&path)?;
    assert_eq!(loaded.size(), 8);
    let collected: Vec<usize> = loaded.collect();
    assert_eq!(collected, values);
    Ok(())
}

#[cfg(not(miri))]
#[test]
fn test_b4_feature_matrix_and_msrv_1_95_gates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let (_, dev_deps_section) = cargo_toml
        .split_once("[dev-dependencies]")
        .expect("Cargo.toml must have [dev-dependencies]");
    let dev_deps_only = dev_deps_section
        .split("\n[")
        .next()
        .expect("dev-dependencies block");
    assert!(
        dev_deps_only.contains("serde_json"),
        "B4 requires serde_json in [dev-dependencies] so integration tests compile with --no-default-features"
    );

    let compressed_rs = std::fs::read_to_string(root.join("src/blob_store/compressed.rs")).unwrap();
    assert!(
        compressed_rs.contains("#[cfg(any(feature = \"zstd\", feature = \"lz4\"))]\n    use crate::blob_store::MemoryBlobStore;"),
        "B4 requires MemoryBlobStore test import in src/blob_store/compressed.rs to be gated on any(feature = \"zstd\", feature = \"lz4\")"
    );

    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    assert!(
        makefile.contains("feature_matrix:"),
        "Makefile must define the B4 feature_matrix target"
    );
    for feature in [
        "simd", "mmap", "zstd", "lz4", "serde", "async", "ffi", "avx512",
    ] {
        assert!(
            makefile.contains(feature),
            "Makefile feature_matrix must include single-feature check for {feature}"
        );
    }
    assert!(
        makefile.contains("+1.95 check --locked --all-features"),
        "Makefile feature_matrix must verify MSRV 1.95 check"
    );

    let ci_yml = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    assert!(
        ci_yml.contains("feature_matrix:"),
        ".github/workflows/ci.yml must define the B4 feature_matrix job"
    );
    assert!(
        ci_yml.contains("1.95"),
        ".github/workflows/ci.yml must verify MSRV 1.95"
    );
}

#[cfg(not(miri))]
#[test]
fn test_b5_supply_chain_audit_and_deny_gates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let deny_toml_path = root.join("deny.toml");
    assert!(
        deny_toml_path.exists(),
        "B5 requires committed deny.toml at repository root"
    );
    let deny_toml = std::fs::read_to_string(&deny_toml_path).unwrap();
    for section in ["[advisories]", "[licenses]", "[bans]", "[sources]"] {
        assert!(
            deny_toml.contains(section),
            "deny.toml must configure {section}"
        );
    }

    let cargo_lock = std::fs::read_to_string(root.join("Cargo.lock")).unwrap();
    assert!(
        !cargo_lock.contains("name = \"bytes\"\nversion = \"1.10.1\""),
        "Cargo.lock must upgrade bytes past 1.10.1 (RUSTSEC-2026-0007)"
    );
    assert!(
        !cargo_lock.contains("name = \"crossbeam-epoch\"\nversion = \"0.9.18\""),
        "Cargo.lock must upgrade crossbeam-epoch past 0.9.18 (RUSTSEC-2026-0204)"
    );

    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    assert!(
        makefile.contains("$(CARGO) audit") && makefile.contains("$(CARGO) deny check"),
        "Makefile audit target must run both cargo audit and cargo deny check"
    );

    let ci_yml = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    assert!(
        ci_yml.contains("supply_chain:")
            && ci_yml.contains("cargo audit")
            && ci_yml.contains("cargo deny check"),
        ".github/workflows/ci.yml must define supply_chain job running cargo audit and cargo deny check"
    );
}

#[cfg(not(miri))]
#[test]
fn test_b6_rustdoc_warnings_gate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    assert!(
        makefile.contains(
            "RUSTDOCFLAGS=\"-D warnings\" $(CARGO) doc --locked --no-deps --all-features"
        ),
        "Makefile doc and sanity targets must enforce RUSTDOCFLAGS=\"-D warnings\" cargo doc --locked --no-deps --all-features"
    );

    let ci_yml = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    assert!(
        ci_yml.contains("RUSTDOCFLAGS=\"-D warnings\" cargo doc --locked --no-deps --all-features"),
        ".github/workflows/ci.yml must enforce RUSTDOCFLAGS=\"-D warnings\" cargo doc --locked --no-deps --all-features"
    );

    let five_level = std::fs::read_to_string(root.join("src/memory/five_level_pool.rs")).unwrap();
    assert!(
        !five_level.contains("[`reject_zero_size`]"),
        "src/memory/five_level_pool.rs must not link to private item [`reject_zero_size`] in public rustdoc"
    );
}

#[cfg(not(miri))]
#[test]
fn test_b2_2_weak_memory_and_aarch64_gates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    assert!(
        makefile.contains("miri_weak:")
            && makefile.contains("-Zmiri-many-seeds")
            && makefile.contains("--target aarch64-unknown-linux-gnu"),
        "Makefile must define miri_weak running Miri weak-memory many-seeds stress and aarch64 clippy"
    );

    let adv_yml_path = root.join(".github/workflows/adversarial.yml");
    assert!(
        adv_yml_path.exists(),
        "B2/B2.2 requires committed .github/workflows/adversarial.yml"
    );
    let adv_yml = std::fs::read_to_string(&adv_yml_path).unwrap();
    for item in [
        "fuzz_smoke",
        "weekly_miri_tsan",
        "weekly_fuzz_soak",
        "weekly_aarch64",
        "make miri_weak",
        "cross test --target aarch64-unknown-linux-gnu --all-features --lib",
    ] {
        assert!(
            adv_yml.contains(item),
            ".github/workflows/adversarial.yml must include '{item}'"
        );
    }

    let cache_mod = std::fs::read_to_string(root.join("src/cache/mod.rs")).unwrap();
    assert!(
        !cache_mod.contains("__pldl1keep"),
        "src/cache/mod.rs must not call non-existent std::arch::aarch64::__pldl1keep"
    );
    let bm25 = std::fs::read_to_string(root.join("src/scoring/bm25.rs")).unwrap();
    assert!(
        !bm25.contains("std::arch::aarch64::_prefetch"),
        "src/scoring/bm25.rs must not call unstable std::arch::aarch64::_prefetch"
    );
}

#[cfg(not(miri))]
#[test]
fn test_b2_1_fuzz_seed_corpora_and_structure_aware_targets() {
    use arbitrary::{Arbitrary, Unstructured};
    use std::collections::BTreeMap;
    use zipora::blob_store::{MixedLenBlobStore, SimpleZipBlobStore, SimpleZipConfig};
    use zipora::containers::UintVecMin0;
    use zipora::entropy::{
        FseConfig, FseDecoder, FseEncoder, HuffmanDecoder, HuffmanEncoder, HuffmanTree, ParallelX1,
        Rans64Encoder, RansDecoder,
    };
    use zipora::fsa::{DoubleArrayTrie, DoubleArrayTrieMap};
    use zipora::succinct::elias_fano::HybridPostingList;

    #[derive(Arbitrary, Debug)]
    struct PostingInput {
        list_a: Vec<u32>,
        list_b: Vec<u32>,
        queries: Vec<u32>,
        encoding_sel: u8,
    }

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let write_seeds = std::env::var("ZIPORA_WRITE_FUZZ_SEEDS").as_deref() == Ok("1");

    let all_targets = [
        "fuzz_zip_offset_load",
        "fuzz_simple_zip_build",
        "fuzz_mixed_len_build",
        "fuzz_huffman_decode",
        "fuzz_rans_decode",
        "fuzz_fse_decompress",
        "fuzz_double_array_trie",
        "fuzz_uint_vec",
        "fuzz_hybrid_posting_list",
        "fuzz_double_array_trie_map",
        "fuzz_blob_store_roundtrip",
    ];

    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    assert!(
        makefile.contains("fuzz_seed:"),
        "B2.1 requires fuzz_seed target in Makefile"
    );
    let fuzz_toml = std::fs::read_to_string(root.join("fuzz/Cargo.toml")).unwrap();
    assert!(
        fuzz_toml.contains("arbitrary ="),
        "B2.1 requires arbitrary dependency in fuzz/Cargo.toml"
    );
    for t in all_targets {
        assert!(
            makefile.contains(t),
            "Makefile FUZZ_TARGETS must include {t}"
        );
        assert!(
            fuzz_toml.contains(&format!("name = \"{t}\"")),
            "fuzz/Cargo.toml must register binary {t}"
        );
        assert!(
            root.join(format!("fuzz/fuzz_targets/{t}.rs")).exists(),
            "fuzz/fuzz_targets/{t}.rs must exist"
        );
    }

    let write_or_load_seed = |target: &str, generated: Option<Vec<u8>>| -> Vec<u8> {
        let dir = root.join("fuzz/corpus").join(target);
        let seed_path = dir.join("seed_0001.bin");
        if write_seeds && let Some(ref bytes) = generated {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&seed_path, bytes).unwrap();
        }
        assert!(
            seed_path.exists(),
            "B2.1 seed file missing: {} (run `make fuzz_seed`)",
            seed_path.display()
        );
        let on_disk = std::fs::read(&seed_path).unwrap();
        assert!(!on_disk.is_empty(), "seed file must not be empty");
        if let Some(expected) = generated {
            assert_eq!(
                on_disk, expected,
                "committed seed for {target} drifted from encoder output"
            );
        }
        on_disk
    };

    // 1. fuzz_zip_offset_load: real ZipOffsetBlobStore wire image
    #[cfg(feature = "zstd")]
    let zip_offset_generated = {
        let mut builder = ZipOffsetBlobStoreBuilder::new().unwrap();
        for rec in [
            &b"alpha-record-0"[..],
            &b"beta-record-1"[..],
            &b"gamma-record-2"[..],
        ] {
            builder.add_record(rec).unwrap();
        }
        let store = builder.finish().unwrap();
        let mut buf = Vec::new();
        store.save_to_writer(&mut buf).unwrap();
        Some(buf)
    };
    #[cfg(not(feature = "zstd"))]
    let zip_offset_generated: Option<Vec<u8>> = None;

    let seed_zip_offset = write_or_load_seed("fuzz_zip_offset_load", zip_offset_generated);
    assert!(seed_zip_offset.len() > 80);
    #[cfg(feature = "zstd")]
    {
        let mut cursor = Cursor::new(&seed_zip_offset);
        let loaded = ZipOffsetBlobStore::load_from_reader(&mut cursor)
            .expect("fuzz_zip_offset_load seed must pass header+checksum validation");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.get(0).unwrap(), b"alpha-record-0");
    }

    // 2. fuzz_simple_zip_build & 3. fuzz_mixed_len_build: length-prefixed record stream
    let len_prefixed_seed = vec![
        5, b'a', b'l', b'p', b'h', b'a', 4, b'b', b'e', b't', b'a', 5, b'g', b'a', b'm', b'm', b'a',
    ];
    let seed_simple_zip =
        write_or_load_seed("fuzz_simple_zip_build", Some(len_prefixed_seed.clone()));
    {
        let records = vec![b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()];
        let store = SimpleZipBlobStore::build_from(&records, &SimpleZipConfig::default()).unwrap();
        assert_eq!(store.get(0).unwrap(), b"alpha");
        assert!(!seed_simple_zip.is_empty());
    }
    let seed_mixed_len = write_or_load_seed("fuzz_mixed_len_build", Some(len_prefixed_seed));
    {
        let records = vec![b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()];
        let store = MixedLenBlobStore::build_from(&records).unwrap();
        assert_eq!(store.get_ref(1).unwrap(), b"beta");
        assert!(!seed_mixed_len.is_empty());
    }

    // 4. fuzz_huffman_decode: 8-byte frequency/len header + real HuffmanEncoder bitstream
    let huff_msg = b"abcaabcbabcc";
    let huff_header = [2u8, b'a', b'b', b'c', (huff_msg.len() as u8), 0, 0, 0];
    let mut huff_freqs = [0u32; 256];
    let n_syms = (huff_header[0] as usize % 8) + 2;
    for i in 0..n_syms {
        huff_freqs[huff_header[1 + (i % 4)] as usize % 256] += 1 + huff_header[i] as u32;
    }
    let huff_enc = HuffmanEncoder::from_frequencies(&huff_freqs).unwrap();
    let huff_payload = huff_enc.encode(huff_msg).unwrap();
    let mut huff_seed = Vec::from(huff_header);
    huff_seed.extend_from_slice(&huff_payload);
    let seed_huff = write_or_load_seed("fuzz_huffman_decode", Some(huff_seed));
    {
        let tree = HuffmanTree::from_frequencies(&huff_freqs).unwrap();
        let dec = HuffmanDecoder::new(tree);
        let decoded = dec
            .decode(&seed_huff[8..], huff_msg.len())
            .expect("fuzz_huffman_decode seed must decode cleanly");
        assert_eq!(decoded, huff_msg);
    }

    // 5. fuzz_rans_decode: 8-byte frequency/len header + real Rans64Encoder<ParallelX1> stream
    let rans_msg = b"abcdabcdabcd";
    let rans_header = [b'a', b'b', b'c', b'd', 10, 8, (rans_msg.len() as u8), 0];
    let mut rans_freqs = [0u32; 256];
    for i in 0..4 {
        rans_freqs[rans_header[i] as usize] += 1 + rans_header[i + 4] as u32;
    }
    let rans_enc = Rans64Encoder::<ParallelX1>::new(&rans_freqs).unwrap();
    let rans_payload = rans_enc.encode(rans_msg).unwrap();
    let mut rans_seed = Vec::from(rans_header);
    rans_seed.extend_from_slice(&rans_payload);
    let seed_rans = write_or_load_seed("fuzz_rans_decode", Some(rans_seed));
    {
        let dec = RansDecoder::new(&rans_enc);
        let decoded = dec
            .decode(&seed_rans[8..], rans_msg.len())
            .expect("fuzz_rans_decode seed must decode cleanly");
        assert_eq!(decoded, rans_msg);
    }

    // 6. fuzz_fse_decompress: real FseEncoder frame
    let fse_msg = b"zipora-fse-seed-corpus-payload-zipora-fse-seed-corpus-payload";
    let fse_frame = FseEncoder::new(FseConfig::default())
        .unwrap()
        .compress(fse_msg)
        .unwrap();
    let seed_fse = write_or_load_seed("fuzz_fse_decompress", Some(fse_frame));
    {
        let mut dec = FseDecoder::new();
        let decoded = dec
            .decompress(&seed_fse)
            .expect("fuzz_fse_decompress seed must pass header check and decompress");
        assert_eq!(decoded, fse_msg);
    }

    // 7. fuzz_double_array_trie: 0xFF-delimited keys
    let dat_seed = b"alpha\xffbeta\xffgamma\xffdelta".to_vec();
    let seed_dat = write_or_load_seed("fuzz_double_array_trie", Some(dat_seed));
    {
        let mut trie = DoubleArrayTrie::new();
        for k in seed_dat.split(|&b| b == 0xFF) {
            trie.insert(k).unwrap();
        }
        assert!(trie.contains(b"alpha") && trie.contains(b"delta"));
    }

    // 8. fuzz_uint_vec: 10-byte header + three 10-byte (idx, val) operations
    let mut uint_seed = Vec::with_capacity(40);
    uint_seed.extend_from_slice(&8u16.to_le_bytes());
    uint_seed.extend_from_slice(&1023u64.to_le_bytes());
    for (idx, val) in [(0u16, 42u64), (3u16, 511u64), (7u16, 1023u64)] {
        uint_seed.extend_from_slice(&idx.to_le_bytes());
        uint_seed.extend_from_slice(&val.to_le_bytes());
    }
    let seed_uint = write_or_load_seed("fuzz_uint_vec", Some(uint_seed));
    {
        let mut v = UintVecMin0::new(8, 1023);
        v.set(3, 511);
        assert_eq!(v.get(3), 511);
        assert_eq!(seed_uint.len(), 40);
    }

    // 9..11. Structure-aware arbitrary targets: 128 payload bytes + 64 tail control bytes
    // (`arbitrary::Unstructured` pops bool continuation bytes from the tail: odd = true, 0 = false).
    let mut arb_bytes: Vec<u8> = (0u8..128u8)
        .map(|i| i.wrapping_mul(37).wrapping_add(11) | 1)
        .collect();
    for i in 0u8..64u8 {
        arb_bytes.push(if i % 3 == 1 { 0 } else { 3 });
    }

    let seed_posting = write_or_load_seed("fuzz_hybrid_posting_list", Some(arb_bytes.clone()));
    {
        let mut u = Unstructured::new(&seed_posting);
        let input = PostingInput::arbitrary(&mut u).unwrap();
        let mut a = input.list_a;
        a.sort_unstable();
        a.dedup();
        let mut b = input.list_b;
        b.sort_unstable();
        b.dedup();
        assert!(!a.is_empty() && !b.is_empty() && !input.queries.is_empty());
        let _ = input.encoding_sel;
        let pl_a = HybridPostingList::from_sorted(&a);
        let pl_b = HybridPostingList::from_sorted(&b);
        assert_eq!(pl_a.len(), a.len());
        assert_eq!(pl_a.next_geq(a[0] as u64), Some((0, a[0] as u64)));
        let _ = pl_a.intersect_count(&pl_b);
    }

    let seed_dat_map = write_or_load_seed("fuzz_double_array_trie_map", Some(arb_bytes.clone()));
    {
        let mut u = Unstructured::new(&seed_dat_map);
        let keys = Vec::<Vec<u8>>::arbitrary(&mut u).unwrap();
        let mut oracle = BTreeMap::new();
        let mut trie = DoubleArrayTrieMap::<u32>::new();
        for (idx, k) in keys.into_iter().filter(|k| !k.is_empty()).enumerate() {
            let val = idx as u32;
            oracle.insert(k.clone(), val);
            trie.insert(&k, val).unwrap();
        }
        assert!(!oracle.is_empty());
        for (k, &v) in &oracle {
            assert_eq!(trie.get(k), Some(v));
        }
    }

    let seed_blob_rt = write_or_load_seed("fuzz_blob_store_roundtrip", Some(arb_bytes));
    {
        let mut u = Unstructured::new(&seed_blob_rt);
        let pairs = Vec::<(Vec<u8>, Vec<u8>)>::arbitrary(&mut u).unwrap();
        assert!(!pairs.is_empty());
    }
}
