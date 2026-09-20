//! Golden-file byte layout and round-trip tests for all persisted formats (`C2.1` / `G13` / `S4-R2`).
//!
//! Asserts every documented header/entry field at its exact byte range in `docs/format_spec.md`
//! and verifies lossless round-trip decoding across all 8 persisted formats.

use std::io::Cursor;
use tempfile::tempdir;
use zipora::blob_store::file_header::ChecksumType;
use zipora::blob_store::reorder_map::{ZReorderMap, ZReorderMapBuilder};
use zipora::blob_store::{
    BlobStore, BlobStoreFileFooter, FileHeaderBase, SortedUintVec, SortedUintVecBuilder,
    SortedUintVecConfig, ZipOffsetBlobStore, ZipOffsetBlobStoreBuilder,
    ZipOffsetBlobStoreConfig,
};
use zipora::containers::ZoSortedStrVec;
use zipora::entropy::dictionary::{Dictionary, DictionaryEntry};
use zipora::error::Result;
use zipora::io::{SignedVarInt, VarInt};
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
    let expected_packed_records =
        42u64 | ((ChecksumType::Crc32c as u64) << 40) | (1u64 << 48);
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
