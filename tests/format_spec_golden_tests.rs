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
    assert!(
        !cargo_lock.contains("name = \"tokio\"\nversion = \"1.50.0\""),
        "S52-R4: Cargo.lock must upgrade tokio to >= 1.51.6 (e.g., 1.54.1) instead of ignoring RUSTSEC-2026-0386"
    );
    assert!(
        !deny_toml.contains("RUSTSEC-2026-0386"),
        "S52-R4: deny.toml must drop the RUSTSEC-2026-0386 ignore once tokio is >= 1.51.6"
    );
    assert!(
        deny_toml.contains("ignored (no fix exists)"),
        "S52-R4: deny.toml must describe bincode RUSTSEC-2025-0141 as \"ignored (no fix exists)\""
    );
    assert!(
        deny_toml.contains("unsound = \"all\"") && deny_toml.contains("RUSTSEC-2026-0253"),
        "S52-R4: deny.toml must set `unsound = \"all\"` so lru (RUSTSEC-2026-0253) is in cargo-deny's unsound scope"
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
            && makefile.contains("-Zmiri-many-seeds=0..64")
            && makefile
                .contains("test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory")
            && makefile.contains("--target aarch64-unknown-linux-gnu")
            && makefile.contains("compiled only; never executed"),
        "Makefile must define miri_weak with 0..64 pool seeds, concurrent WorkStealingQueue steal test, and explicit AArch64 compiled-only note"
    );

    let ws_src = std::fs::read_to_string(root.join("src/concurrency/work_stealing.rs")).unwrap();
    assert!(
        ws_src.contains("fn test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory"),
        "src/concurrency/work_stealing.rs must define test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory"
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
        "compiled only; never executed",
    ] {
        assert!(
            adv_yml.contains(item),
            ".github/workflows/adversarial.yml must include '{item}'"
        );
    }
    assert!(
        !adv_yml.contains("cross test --target aarch64-unknown-linux-gnu"),
        ".github/workflows/adversarial.yml must not claim unverified cross test execution on AArch64"
    );

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

#[cfg(not(miri))]
#[test]
fn test_e7_f2_perf_table_and_readme_claims_audit() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_path = root.join("scripts/perf_table.py");
    assert!(
        script_path.exists(),
        "E7 requires committed scripts/perf_table.py"
    );

    let output = std::process::Command::new("python3")
        .arg(&script_path)
        .arg("--check")
        .current_dir(root)
        .output()
        .expect("python3 scripts/perf_table.py --check must execute");
    assert!(
        output.status.success(),
        "scripts/perf_table.py --check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for doc_rel in ["README.md", "README_cn.md", "docs/PERFORMANCE.md"] {
        let doc = std::fs::read_to_string(root.join(doc_rel)).unwrap();
        assert!(
            doc.contains("<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->")
                && doc.contains("<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->"),
            "{doc_rel} must embed generated performance table from scripts/perf_table.py"
        );
        for stale in [
            "1,122",
            "10.7 bytes/key",
            "10.7 字节/键",
            "20.6 ns",
            "5.2 Gwords/s",
            "2.46 ns",
            "CacheOptimizedHashMap",
            "NestTrieDawg",
            "8.0 B/state",
        ] {
            assert!(
                !doc.contains(stale),
                "{doc_rel} must not contain stale or nonexistent claim `{stale}`"
            );
        }
    }

    let readme = std::fs::read_to_string(root.join("README.md")).unwrap();
    assert!(
        readme.contains("`DoubleArrayTrie`, `12.0 B/state`")
            && readme.contains("`DoubleArrayTrieMap<u32>`, `12.0–16.0 B/state`"),
        "README.md must match E6 test figures for DoubleArrayTrie (12.0 B/state) and DoubleArrayTrieMap<u32> (12.0–16.0 B/state)"
    );

    let e4_md = std::fs::read_to_string(root.join("docs/review/e4.md")).unwrap();
    for invented in [
        "test_s51_r1_deserialize_rejects_non_prefix_code_sets_regardless_of_map_order",
        "test_s51_r1_deserialize_rejects_incomplete_code_sets_and_enforces_1bit_single_symbol",
        "test_s51_r1_decode_order0_single_symbol_verifies_bits_and_decode_xn_empty_data_errors",
    ] {
        assert!(
            !e4_md.contains(invented),
            "docs/review/e4.md must not cite invented test name `{invented}`"
        );
    }
    for real_test in [
        "test_s51_r1_prefix_violating_codes_rejected_all_orders_and_interleaved",
        "test_s51_r1_incomplete_code_sets_rejected_except_single_symbol_1bit",
        "test_s51_r1_decode_order0_single_symbol_checks_stream_bits_and_decode_xn_empty_data",
        "test_s51_r1_trained_model_wire_format_matrix",
    ] {
        assert!(
            e4_md.contains(real_test),
            "docs/review/e4.md must cite real test name `{real_test}`"
        );
    }

    // Verify scripts/perf_table.py audit_readme_claims rejects numeric mutations and appended unmeasured claims.
    let mut_check = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys; sys.path.insert(0, 'scripts'); import perf_table as pt; \
             core, comp, e2e = pt.load_baselines(); \
             readme = (pt.ROOT / 'README.md').read_text(); \
             e1 = pt.audit_readme_claims(readme.replace('35.65 B/entry', '99.99 B/entry'), core, comp, e2e); \
             assert e1, 'audit_readme_claims must reject mutated 99.99 B/entry'; \
             e2 = pt.audit_readme_claims(readme + '\\nExtra claim: 5.2 Gops/s, 26x faster\\n', core, comp, e2e); \
             assert e2, 'audit_readme_claims must reject appended unmeasured claims'; \
             e3 = pt.audit_readme_claims(readme.replace('GoldHashMap', 'GoldHashMap, `CacheOptimizedHashMap`'), core, comp, e2e); \
             assert e3, 'audit_readme_claims must reject nonexistent type CacheOptimizedHashMap'",
        )
        .current_dir(root)
        .output()
        .expect("python3 mutation check must execute");
    assert!(
        mut_check.status.success(),
        "scripts/perf_table.py mutation checks failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&mut_check.stdout),
        String::from_utf8_lossy(&mut_check.stderr)
    );
}

#[test]
fn test_f4_release_prep_gate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();

    let release_prep_line = makefile
        .lines()
        .find(|line| line.starts_with("release_prep:"))
        .expect("Makefile must define a release_prep target");
    for required_dep in [
        "sanity",
        "feature_matrix",
        "audit",
        "miri_core",
        "miri_simd",
        "miri_cspp",
        "miri_weak",
        "tsan_cspp",
        "tsan_pool",
        "fuzz_smoke",
    ] {
        assert!(
            release_prep_line
                .split_whitespace()
                .any(|tok| tok == required_dep),
            "Makefile release_prep must depend on `{required_dep}` (got: {release_prep_line})"
        );
    }

    assert!(
        makefile.contains("publish --dry-run --locked"),
        "Makefile release_prep must run `cargo publish --dry-run --locked`"
    );
    assert!(
        makefile.contains("publish: release_prep"),
        "Makefile `publish` target must depend on `release_prep` (Rule F4)"
    );

    let release_yml = std::fs::read_to_string(root.join(".github/workflows/release.yml")).unwrap();
    assert!(
        release_yml.contains("make release_prep"),
        ".github/workflows/release.yml must gate releases on `make release_prep`"
    );
}

#[test]
fn test_s52_r6_simd_isa_module_is_pub_crate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let simd_mod = std::fs::read_to_string(root.join("src/simd/mod.rs")).unwrap();
    assert!(
        simd_mod.contains("pub(crate) mod isa;"),
        "src/simd/mod.rs must keep `isa` as `pub(crate) mod isa;` rather than exposing `pub mod isa;`"
    );
    assert!(
        !simd_mod.contains("pub mod isa;") && !simd_mod.contains("pub use isa::"),
        "src/simd/mod.rs must not publicly export `simd::isa`"
    );
}

#[test]
fn test_s52_r9_package_exclude_single_feature_tests_ci_toolchain_and_fuzz_doc() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    // 1. Cargo.toml [package] must specify `exclude` covering plan.md and cpp_benchmark/wrapper_test
    let cargo_toml = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(
        cargo_toml.contains("exclude = [")
            && cargo_toml.contains("\"plan.md\"")
            && cargo_toml.contains("\"cpp_benchmark/"),
        "Cargo.toml [package] must define `exclude` covering `plan.md` and `cpp_benchmark/`"
    );

    // 2. Makefile feature_matrix must run `cargo test` on each single-feature build, not just clippy
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    let fm_idx = makefile
        .find("feature_matrix:")
        .expect("Makefile must have feature_matrix target");
    let fm_block = &makefile[fm_idx..];
    let fm_end = fm_block.find("\nsanity:").unwrap_or(fm_block.len());
    let fm_section = &fm_block[..fm_end];
    assert!(
        fm_section.contains("test --locked --no-default-features --features $$f --lib"),
        "Makefile feature_matrix must run unit tests (`test --locked --no-default-features --features $$f --lib`) for each single feature"
    );

    // 3. .github/workflows/ci.yml must not pass comma-separated `toolchain: stable, 1.95.0`
    let ci_yml = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    assert!(
        !ci_yml.contains("toolchain: stable, 1.95.0"),
        ".github/workflows/ci.yml must not pass comma-separated toolchains to dtolnay/rust-toolchain"
    );
    assert!(
        ci_yml.contains("toolchain: 1.95.0"),
        ".github/workflows/ci.yml must install toolchain 1.95.0 in its own step"
    );

    // 4. fuzz/fuzz_targets/fuzz_double_array_trie_map.rs doc comment must accurately describe operations
    let fuzz_dat =
        std::fs::read_to_string(root.join("fuzz/fuzz_targets/fuzz_double_array_trie_map.rs"))
            .unwrap();
    assert!(
        !fuzz_dat.contains("and `remove`"),
        "fuzz_double_array_trie_map.rs doc comment must not claim `remove` when it only tests insert, get, and shrink_to_fit"
    );
}

#[test]
fn test_s52_r2_fuzz_soak_checks_per_job_exit_codes_and_records_1h_summary() {
    // 1. Regression for the SortedUintVec::read_from multiplication overflow found by
    // fuzz_zip_offset_load during the 1-hour soak (crash-2950c6e3b040e890a337af10eedb41f09adc3ffd).
    let crash_bytes: [u8; 316] = [
        122, 105, 112, 111, 114, 97, 45, 98, 108, 111, 98, 45, 115, 116, 111, 114, 101, 0, 0, 0,
        90, 105, 112, 79, 102, 102, 115, 101, 116, 66, 108, 111, 98, 83, 116, 111, 114, 101, 0, 0,
        60, 1, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 1, 0, 80, 0, 0, 0, 0,
        0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40,
        181, 47, 253, 32, 14, 113, 0, 0, 97, 108, 112, 104, 97, 45, 114, 101, 99, 111, 114, 100,
        45, 48, 127, 81, 139, 120, 40, 181, 47, 253, 32, 13, 105, 0, 0, 98, 101, 116, 97, 220, 114,
        101, 99, 111, 114, 100, 45, 49, 18, 213, 72, 77, 40, 181, 47, 253, 32, 14, 113, 0, 0, 103,
        97, 109, 109, 97, 45, 114, 101, 99, 111, 114, 100, 45, 50, 112, 138, 59, 182, 90, 83, 85,
        86, 1, 6, 16, 32, 4, 0, 0, 0, 0, 0, 0, 120, 40, 181, 47, 253, 32, 13, 105, 0, 0, 98, 101,
        116, 97, 220, 114, 101, 50, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 119, 133, 209, 38, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut cursor = std::io::Cursor::new(&crash_bytes[..]);
    assert!(
        zipora::blob_store::ZipOffsetBlobStore::load_from_reader(&mut cursor).is_err(),
        "Crafted ZipOffsetBlobStore with overflowing SortedUintVec size must return Err, not panic"
    );

    // 2. Makefile fuzz_soak must collect background PIDs and check every exit status.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    let soak_idx = makefile
        .find("fuzz_soak:")
        .expect("Makefile must define fuzz_soak target");
    let soak_block = &makefile[soak_idx..];
    let soak_end = soak_block.find("\nformat:").unwrap_or(soak_block.len());
    let soak_section = &soak_block[..soak_end];
    assert!(
        soak_section.contains("pids=")
            && soak_section.contains("wait $$p || fail=1")
            && soak_section.contains("exit $$fail"),
        "Makefile fuzz_soak must collect PIDs and fail when any background job fails (got: {soak_section})"
    );

    // 3. docs/review/fuzz_soak.md must exist and record the 1-hour soak across all 11 targets.
    let soak_doc_path = root.join("docs/review/fuzz_soak.md");
    assert!(
        soak_doc_path.exists(),
        "docs/review/fuzz_soak.md must exist and record the 1-hour fuzz soak results"
    );
    let soak_doc = std::fs::read_to_string(&soak_doc_path).unwrap();
    for target in [
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
    ] {
        assert!(
            soak_doc.contains(target),
            "docs/review/fuzz_soak.md must record 1-hour soak stats for `{target}`"
        );
    }
    assert!(
        soak_doc.contains("max_total_time=3600")
            && soak_doc.contains("rss_limit_mb=1024")
            && soak_doc.contains("oom=0, timeout=0, crash=0")
            && !soak_doc.contains("PENDING_1H"),
        "docs/review/fuzz_soak.md must document max_total_time=3600, rss_limit_mb=1024, oom=0, timeout=0, crash=0, and all 11 completed 1h rows"
    );
}

#[test]
fn test_s53_r1_sorted_uint_vec_and_zip_offset_bound_allocations_by_remaining_input() {
    use zipora::blob_store::{SortedUintVec, ZipOffsetBlobStore, ZipOffsetBlobStoreBuilder};

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    // 1. Exact s53/rev_code/blob `revblob huge 8796093022208` reproducer:
    // ~300-byte ZipOffsetBlobStore with internally consistent SortedUintVec
    // size = 2^43 (8,796,093,022,208), index_len = 512 GiB, data_len = 16 TiB.
    let mut builder = ZipOffsetBlobStoreBuilder::new().unwrap();
    builder.add_record(b"a").unwrap();
    builder.add_record(b"bb").unwrap();
    let built = builder.finish().unwrap();
    let mut base = Vec::new();
    built.save_to_writer(&mut base).unwrap();

    let z = base
        .windows(4)
        .position(|w| w == b"ZSUV")
        .expect("ZSUV header must be present");
    let l2 = base[z + 5] as u64;
    let ow = base[z + 6] as u64;
    let sw = base[z + 7] as u64;
    let size: u64 = 8_796_093_022_208; // 1u64 << 43
    let nb = size.div_ceil(1u64 << l2);
    let il = (nb * sw).div_ceil(8);
    let dl = (size * ow).div_ceil(8);
    assert_eq!(il, 549_755_813_888); // 512 GiB

    let mut crafted_inner = base.clone();
    crafted_inner[z + 8..z + 16].copy_from_slice(&size.to_le_bytes());
    crafted_inner[z + 16..z + 24].copy_from_slice(&il.to_le_bytes());
    crafted_inner[z + 24..z + 32].copy_from_slice(&dl.to_le_bytes());

    assert!(
        ZipOffsetBlobStore::load_from_reader(&mut std::io::Cursor::new(&crafted_inner)).is_err(),
        "Crafted ~300B ZipOffsetBlobStore with 512 GiB ZSUV header must return Err without allocating 512 GiB"
    );

    // 2. Direct SortedUintVec::read_from on the short slice must also return Err without allocating 512 GiB.
    assert!(
        SortedUintVec::read_from(&mut std::io::Cursor::new(&crafted_inner[z..])).is_err(),
        "Direct SortedUintVec::read_from on truncated 512 GiB header must return Err without allocating 512 GiB"
    );

    // 3. Outer FileHeader also forged to match huge offsets_bytes / content_bytes on a ~300B input.
    let mut crafted_outer = crafted_inner.clone();
    let huge_offsets_bytes = 32u64 + il + dl;
    let content_bytes = u64::from_le_bytes(crafted_outer[64..72].try_into().unwrap());
    let content_pad = (16 - (content_bytes % 16)) % 16;
    let huge_file_size = 128u64 + content_bytes + content_pad + huge_offsets_bytes + 64u64;
    crafted_outer[40..48].copy_from_slice(&huge_file_size.to_le_bytes());
    crafted_outer[72..80].copy_from_slice(&huge_offsets_bytes.to_le_bytes());
    assert!(
        ZipOffsetBlobStore::load_from_reader(&mut std::io::Cursor::new(&crafted_outer)).is_err(),
        "Crafted ~300B ZipOffsetBlobStore with huge outer offsets_bytes + file_size must return Err without large allocation"
    );

    let mut crafted_content = base.clone();
    let huge_content: u64 = 1u64 << 40; // 1 TiB
    let huge_total = 128u64 + huge_content + 64u64 + 64u64;
    crafted_content[40..48].copy_from_slice(&huge_total.to_le_bytes());
    crafted_content[64..72].copy_from_slice(&huge_content.to_le_bytes());
    assert!(
        ZipOffsetBlobStore::load_from_reader(&mut std::io::Cursor::new(&crafted_content)).is_err(),
        "Crafted ~300B ZipOffsetBlobStore with 1 TiB content_bytes must return Err without pre-allocating"
    );

    // 4. False comment must be removed from src/blob_store/sorted_uint_vec.rs.
    let suv_src = std::fs::read_to_string(root.join("src/blob_store/sorted_uint_vec.rs")).unwrap();
    assert!(
        !suv_src.contains("a crafted section cannot make us allocate more than it can fill"),
        "src/blob_store/sorted_uint_vec.rs must not contain the false allocation claim comment"
    );

    // 5. Makefile fuzz_smoke and fuzz_soak must pass -malloc_limit_mb alongside -rss_limit_mb.
    let makefile = std::fs::read_to_string(root.join("Makefile")).unwrap();
    let smoke_idx = makefile.find("fuzz_smoke:").expect("fuzz_smoke target");
    let soak_idx = makefile.find("fuzz_soak:").expect("fuzz_soak target");
    let fmt_idx = makefile.find("\nformat:").unwrap_or(makefile.len());
    let smoke_sec = &makefile[smoke_idx..soak_idx];
    let soak_sec = &makefile[soak_idx..fmt_idx];
    assert!(
        smoke_sec.contains("-malloc_limit_mb=$(FUZZ_RSS_LIMIT_MB)"),
        "Makefile fuzz_smoke must pass -malloc_limit_mb=$(FUZZ_RSS_LIMIT_MB)"
    );
    assert!(
        soak_sec.contains("-malloc_limit_mb=$(FUZZ_RSS_LIMIT_MB)"),
        "Makefile fuzz_soak must pass -malloc_limit_mb=$(FUZZ_RSS_LIMIT_MB)"
    );
}

#[test]
fn test_s53_r2_r4_get2_inlining_steal_assertion_soak_sum_and_ledger_refs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    // 1. S53-R2: SortedUintVec::get2 and get_unchecked must carry #[inline] / #[inline(always)].
    let suv_src = std::fs::read_to_string(root.join("src/blob_store/sorted_uint_vec.rs")).unwrap();
    assert!(
        suv_src.contains("#[inline]\n    pub fn get2("),
        "SortedUintVec::get2 must be marked #[inline] (S53-R2)"
    );
    assert!(
        suv_src.contains("#[inline(always)]\n    fn get_unchecked("),
        "SortedUintVec::get_unchecked must be marked #[inline(always)] (S53-R2)"
    );

    // 2. S52-R3: test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory must assert steal_count > 0.
    let ws_src = std::fs::read_to_string(root.join("src/concurrency/work_stealing.rs")).unwrap();
    let fn_idx = ws_src
        .find("fn test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory")
        .expect("work_stealing.rs must define test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory");
    let fn_body = &ws_src[fn_idx..];
    assert!(
        fn_body.contains("stolen_count")
            && fn_body.contains("stolen_count.fetch_add(1")
            && fn_body.contains("assert!(\n            stolen_count.load("),
        "test_s52_r3_concurrent_owner_push_pop_and_thief_steal_weak_memory must count steals and assert stolen_count > 0 (S52-R3)"
    );

    // 3. S52-R2: docs/review/fuzz_soak.md must have exact arithmetic sums (including pre-fix 229,033,298 / 360,359,994)
    // and post-fix 1-hour re-soak table with -malloc_limit_mb=1024.
    let soak_md = std::fs::read_to_string(root.join("docs/review/fuzz_soak.md")).unwrap();
    assert!(
        !soak_md.contains("229,033,300") && !soak_md.contains("360,359,996"),
        "docs/review/fuzz_soak.md must not contain the off-by-2 sums 229,033,300 or 360,359,996"
    );
    assert!(
        soak_md.contains("229,033,298") && soak_md.contains("360,359,994"),
        "docs/review/fuzz_soak.md must record the exact pre-fix sums 229,033,298 and 360,359,994"
    );
    assert!(
        soak_md.contains("-malloc_limit_mb=1024"),
        "docs/review/fuzz_soak.md must document -malloc_limit_mb=1024"
    );

    // Verify table row sum matches the section header in docs/review/fuzz_soak.md
    let mut row_count = 0usize;
    let mut total_execs = 0u64;
    for line in soak_md.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("| `fuzz_") && trimmed.contains("| `DONE` |") {
            let cols: Vec<&str> = trimmed.split('|').map(|s| s.trim()).collect();
            assert!(
                cols.len() >= 5,
                "Unexpected table row format in fuzz_soak.md: {trimmed}"
            );
            let dur_str = cols[3].trim_matches('`').replace(',', "");
            let exec_str = cols[4].trim_matches('`').replace(',', "");
            let dur: u64 = dur_str.parse().expect("valid duration integer");
            let execs: u64 = exec_str.parse().expect("valid executions integer");
            assert!(dur >= 3600, "Soak duration must be >= 3600s, got {dur}");
            row_count += 1;
            total_execs += execs;
        }
    }
    assert_eq!(row_count, 11, "fuzz_soak.md must have 11 DONE table rows");
    let formatted_sum = {
        let s = total_execs.to_string();
        let mut out = String::new();
        for (i, ch) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                out.push(',');
            }
            out.push(ch);
        }
        out.chars().rev().collect::<String>()
    };
    assert!(
        soak_md.contains(&format!("`{formatted_sum}` Total Executions")),
        "fuzz_soak.md header must match exact row sum `{formatted_sum}` Total Executions"
    );

    // 4. S53-R4: All file:line references and test names in docs/review/{e4,e2_e3,e1,fuzz_soak}.md must be valid.
    let e4_md = std::fs::read_to_string(root.join("docs/review/e4.md")).unwrap();
    assert!(
        !e4_md.contains("src/entropy/huffman/tree.rs:284-489"),
        "docs/review/e4.md must not contain stale tree.rs:284-489 reference"
    );
    assert!(
        e4_md.contains("src/entropy/huffman/tree.rs:267-550"),
        "docs/review/e4.md must reference src/entropy/huffman/tree.rs:267-550"
    );
    let e1_md = std::fs::read_to_string(root.join("docs/review/e1.md")).unwrap();
    assert!(
        !e1_md.contains("test_c9_f8_create_temp_in_cleans_up_backing_file_when_mmap_mut_fails"),
        "docs/review/e1.md must not reference nonexistent test name"
    );
    assert!(
        e1_md.contains("test_c9_f8_with_capacity_simd_refuses_symlinks_and_cleans_up_on_error"),
        "docs/review/e1.md must reference real test name test_c9_f8_with_capacity_simd_refuses_symlinks_and_cleans_up_on_error"
    );

    for doc_rel in [
        "docs/review/e4.md",
        "docs/review/e2_e3.md",
        "docs/review/e1.md",
        "docs/review/fuzz_soak.md",
    ] {
        let txt = std::fs::read_to_string(root.join(doc_rel)).unwrap();
        for seg in txt.split('`').skip(1).step_by(2) {
            if let Some((path_part, line_part)) = seg.rsplit_once(':') {
                let is_tracked_file = path_part == "Makefile"
                    || path_part.ends_with(".rs")
                    || path_part.ends_with(".py")
                    || path_part.ends_with(".yml")
                    || path_part.ends_with(".toml")
                    || path_part.ends_with(".md");
                if !is_tracked_file || path_part.contains(' ') {
                    continue;
                }
                let (l1_str, l2_str) = line_part.split_once('-').unwrap_or((line_part, line_part));
                if let (Ok(l1), Ok(l2)) = (l1_str.parse::<usize>(), l2_str.parse::<usize>()) {
                    let target_path = root.join(path_part);
                    assert!(
                        target_path.is_file(),
                        "{doc_rel}: referenced file `{path_part}` does not exist (`{seg}`)"
                    );
                    let nlines = std::fs::read_to_string(&target_path)
                        .unwrap()
                        .lines()
                        .count();
                    assert!(
                        l1 >= 1 && l1 <= l2 && l2 <= nlines,
                        "{doc_rel}: out-of-range reference `{seg}` (file `{path_part}` has {nlines} lines)"
                    );
                }
            }
        }
    }
}
