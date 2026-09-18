//! Golden-file byte layout and round-trip tests for all persisted formats (`C2.1` / `G13`).
//!
//! Verifies that every persisted format documented in `docs/format_spec.md` encodes
//! deterministic little-endian bytes and round-trips cleanly across architectures.

use std::io::Cursor;
use tempfile::tempdir;
use zipora::blob_store::reorder_map::{ZReorderMap, ZReorderMapBuilder};
use zipora::blob_store::{
    BlobStore, BlobStoreFileFooter, FileHeaderBase, SortedUintVec, SortedUintVecBuilder,
    SortedUintVecConfig, ZipOffsetBlobStore, ZipOffsetBlobStoreBuilder,
};
use zipora::containers::ZoSortedStrVec;
use zipora::entropy::dictionary::Dictionary;
use zipora::entropy::DictionaryBuilder;
use zipora::error::Result;
use zipora::io::VarInt;
use zipora::memory::{MmapVec, MmapVecConfig};

#[test]
fn test_golden_file_header_base_and_footer_le_bytes() -> Result<()> {
    let mut header = FileHeaderBase::new();
    header.set_class_name("ZipOffsetBlobStore");
    header.set_file_size(4096);
    header.set_unzip_size(8192);
    header.set_records(42);
    header.set_format_version(1);

    let bytes = header.as_bytes();
    assert_eq!(bytes.len(), 80);

    // Check magic_len (17) and magic ("terark-blob-store\0")
    assert_eq!(bytes[0], 17);
    assert_eq!(&bytes[1..19], b"terark-blob-store\0");
    // Check file_size = 4096 (u64 LE at 40..48)
    assert_eq!(&bytes[40..48], &4096u64.to_le_bytes());
    // Check unzip_size = 8192 (u64 LE at 48..56)
    assert_eq!(&bytes[48..56], &8192u64.to_le_bytes());

    let decoded = FileHeaderBase::from_bytes(bytes)
        .ok_or_else(|| zipora::error::ZiporaError::invalid_data("Invalid FileHeaderBase magic"))?;
    assert_eq!(decoded.class_name(), "ZipOffsetBlobStore");
    assert_eq!(decoded.records(), 42);
    assert_eq!(decoded.file_size(), 4096);
    assert_eq!(decoded.unzip_size(), 8192);
    assert_eq!(decoded.format_version(), 1);

    let mut footer = BlobStoreFileFooter::new();
    footer.set_zip_data_xxhash(0x1122_3344_5566_7788u64);
    footer.set_file_xxhash(0xDEAD_BEEF_CAFE_BABEu64);
    let fbytes = footer.as_bytes();
    assert_eq!(fbytes.len(), 64);
    assert_eq!(&fbytes[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&fbytes[8..16], &0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes());
    assert_eq!(&fbytes[60..64], &64u32.to_le_bytes());

    let decoded_footer = BlobStoreFileFooter::from_bytes(fbytes);
    assert_eq!(decoded_footer.zip_data_xxhash(), 0x1122_3344_5566_7788u64);
    assert_eq!(decoded_footer.file_xxhash(), 0xDEAD_BEEF_CAFE_BABEu64);
    assert_eq!(decoded_footer.footer_length(), 64);
    Ok(())
}

#[test]
fn test_golden_sorted_uint_vec_zsuv_le_bytes() -> Result<()> {
    let values = vec![0u64, 15, 42, 100, 255];
    let mut builder = SortedUintVecBuilder::with_config(SortedUintVecConfig::default());
    for &v in &values {
        builder.push(v)?;
    }
    let suv = builder.finish()?;
    let mut bytes = Vec::new();
    suv.write_to(&mut bytes)?;

    // Check ZSUV header: [b"ZSUV"][version: 1u8][log2_block_units][offset_width][sample_width][len: u64 LE]
    assert_eq!(&bytes[0..4], b"ZSUV");
    assert_eq!(bytes[4], 1);
    assert_eq!(&bytes[8..16], &(values.len() as u64).to_le_bytes());

    let mut cursor = Cursor::new(&bytes);
    let decoded = SortedUintVec::read_from(&mut cursor)?;
    assert_eq!(decoded.len(), values.len());
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(decoded.get(i)?, v);
    }
    Ok(())
}

#[test]
fn test_golden_zip_offset_blob_store_file_roundtrip() -> Result<()> {
    let dir = tempdir().map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    let path = dir.path().join("golden_zip_offset.zbs");

    let mut builder = ZipOffsetBlobStoreBuilder::new()?;
    builder.add_record(b"alpha_record")?;
    builder.add_record(b"beta_record_payload")?;
    builder.add_record(b"gamma")?;
    let store = builder.finish()?;
    store.save_to_file(&path)?;

    let raw =
        std::fs::read(&path).map_err(|e| zipora::error::ZiporaError::io_error(e.to_string()))?;
    assert_eq!(&raw[0..18], b"zipora-blob-store\0");

    let loaded = ZipOffsetBlobStore::load_from_file(&path)?;
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.get(0)?, b"alpha_record");
    assert_eq!(loaded.get(1)?, b"beta_record_payload");
    assert_eq!(loaded.get(2)?, b"gamma");
    Ok(())
}

#[test]
fn test_golden_entropy_dictionary_le_bytes() -> Result<()> {
    let dict = DictionaryBuilder::new()
        .min_match_length(3)
        .max_match_length(32)
        .build(b"the quick brown fox jumps over the quick brown dog");
    let bytes = dict.serialize();

    // First 4 bytes: entry_count as u32 LE
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    assert_eq!(count, dict.len());

    let decoded = Dictionary::deserialize(&bytes)?;
    assert_eq!(decoded.len(), dict.len());
    assert_eq!(decoded.serialize(), bytes);
    Ok(())
}

#[test]
fn test_golden_zo_sorted_str_vec_zosv_le_bytes() -> Result<()> {
    let strings = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "delta".to_string(),
        "omega".to_string(),
    ];
    let zosv = ZoSortedStrVec::from_sorted_strings(strings.clone())?;
    let bytes = zosv.to_bytes();

    // Verify 16-byte ZOSV header
    assert_eq!(&bytes[0..4], b"ZOSV");
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 1);
    assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 0x0011);
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        4
    );

    let decoded = ZoSortedStrVec::from_bytes(&bytes)?;
    assert_eq!(decoded.len(), 4);
    for (i, expected) in strings.iter().enumerate() {
        assert_eq!(decoded.get(i), Some(expected.as_str()));
    }
    Ok(())
}

#[test]
fn test_golden_varint_leb128_bytes() -> Result<()> {
    // Golden bytes for 300 (0b00000010_00101100 -> [0xAC, 0x02])
    let buf = VarInt::encode(300);
    assert_eq!(buf, &[0xAC, 0x02]);
    let (decoded, consumed) = VarInt::decode(&buf)?;
    assert_eq!(decoded, 300);
    assert_eq!(consumed, 2);
    Ok(())
}

#[test]
fn test_golden_mmap_vec_header_v2_le_bytes() -> Result<()> {
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
    assert_eq!(
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]
        ]),
        0x4D4D_4150_5F56_4543u64
    );
    assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 2);
    assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 0x0011);
    assert_eq!(
        u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        8
    );
    assert_eq!(
        u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23]
        ]),
        2
    );
    assert_eq!(
        u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31]
        ]),
        1024
    );

    let reopened = MmapVec::<u64>::open(&path, MmapVecConfig::default())?;
    assert_eq!(reopened.len(), 2);
    assert_eq!(reopened.get(0), Some(&0x0102_0304_0506_0708u64));
    assert_eq!(reopened.get(1), Some(&0x1122_3344_5566_7788u64));
    Ok(())
}

#[test]
fn test_golden_zreorder_map_le_bytes() -> Result<()> {
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
    assert_eq!(
        u64::from_le_bytes([raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]),
        8
    );

    let loaded = ZReorderMap::open(&path)?;
    assert_eq!(loaded.size(), 8);
    let collected: Vec<usize> = loaded.collect();
    assert_eq!(collected, values);
    Ok(())
}
