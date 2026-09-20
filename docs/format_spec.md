# Zipora Persisted Binary Format Specification (`C2.1` / `G13`)

Every persisted binary format in `zipora` obeys two mandatory invariants:

1. **Explicit Little-Endian Wire Layout**: All multi-byte integers on disk or in serialized byte slices are encoded and decoded exclusively via `to_le_bytes()` / `from_le_bytes()`. Native struct pointer casting (`*const Header as *const u8`) is forbidden on persisted bytes.
2. **Bounded Header & Payload Validation**: Every persisted container validates magic/version/range fields and bounds all lengths and counts against remaining input bytes before allocation.

---

## 1. Summary Inventory of Persisted Formats

| # | Format / Type | Source File | Magic / Signature | Version | Flags / Config Bytes | Header Length |
|---|---------------|-------------|-------------------|---------|----------------------|---------------|
| 1 | `FileHeaderBase` + `BlobStoreFileFooter` | `src/blob_store/file_header.rs` | `magic_len = 17` (`@0`), `"terark-blob-store\0\0"` (`@1..20`) | `format_version: u16` LE (`@62..64`) | `checksum_type: u8` (`@61`), `footer_length = 64u32` LE (`@60..64`) | `80 B` header + `64 B` footer |
| 2 | `SortedUintVec` (`ZSUV`) | `src/blob_store/sorted_uint_vec.rs` | `"ZSUV"` (`@0..4`, `0x5655_535A`) | `1` (`u8` `@4`) | `[log2_block_units: u8 @5][offset_width: u8 @6][sample_width: u8 @7]` | `32 B` |
| 3 | `ZipOffsetBlobStore` | `src/blob_store/zip_offset.rs` | `"zipora-blob-store\0\0\0"` (`@0..20`) | `1` (`u16` LE `@62..64`) | `[log2_block_units: u8 @80][checksum_level: u8 @81][compress_level: u8 @82]` | `128 B` header + `64 B` footer |
| 4 | `Dictionary` | `src/entropy/dictionary.rs` | `entry_count: u32` LE (`@0..4`) | `1` (stream param `min/max_match_length`) | Sorted by `sequence`; per-entry `[seq_len: u16 LE][seq][offset: u32 LE][length: u32 LE]` | `4 B` + `10 B + seq_len` per entry |
| 5 | `ZoSortedStrVec` (`ZOSV`) | `src/containers/specialized/zo_sorted_str_vec.rs` | `"ZOSV"` (`@0..4`, `0x5653_4F5A`) | `1` (`u16` LE `@4..6`) | `0x0011` (`u16` LE `@6..8`: bit 0 = LE, bit 4 = 64-bit width) | `16 B` |
| 6 | `VarInt` (`LEB128` / `ZigZag`) | `src/io/var_int.rs`, `src/io/var_int_variants.rs` | Self-delimiting (`0x80` MSB continuation bit) | `1` | Little-endian 7-bit groups (`u32`/`u64`), ZigZag `(n << 1) ^ (n >> 63)` for signed | `1–5 B` (`u32`) / `1–10 B` (`u64`) |
| 7 | `MmapVecHeader` (`MMAP_VEC`) | `src/memory/mmap_vec.rs` | `0x4D4D_4150_5F56_4543` (`u64` LE `@0..8`, bytes `b"CEV_PAMM"`) | `2` (`u16` LE `@8..10`) | `0x0011` (`u16` LE `@10..12`: bit 0 = LE, bit 4 = 64-bit header field width) | `80 B` (padded to `align_of::<T>()`) |
| 8 | `ZReorderMap` | `src/blob_store/reorder_map.rs` | `sign` validated in `{-1i64, 1i64}` (`@8..16`) | `1` | `[size: u64 LE @0..8][sign: i64 LE @8..16]` + `VarInt` signed-run stream | `16 B` |

---

## 2. Exact Byte Layouts (Verified Against Serializer Code)

### 2.1 `FileHeaderBase` (80 B) and `BlobStoreFileFooter` (64 B)
**Source**: [`src/blob_store/file_header.rs`](../src/blob_store/file_header.rs)

#### `FileHeaderBase` (80 bytes, little-endian)
| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..1` | 1 B | `u8` | `magic_len` | `17` (`MAGIC_STR_LEN`) |
| `1..20` | 19 B | `[u8; 19]` | `magic` | `b"terark-blob-store\0\0"` |
| `20..40` | 20 B | `[u8; 20]` | `class_name` | NUL-padded ASCII class name (max 19 chars + NUL) |
| `40..48` | 8 B | `u64` LE | `file_size` | `file_size.to_le_bytes()` |
| `48..56` | 8 B | `u64` LE | `unzip_size` | `unzip_size.to_le_bytes()` |
| `56..64` | 8 B | `u64` LE | `packed_records_field` | `(records & 0xFF_FFFF_FFFF) \| ((checksum_type as u64) << 40) \| ((format_version as u64) << 48)` (`56..61` = 40-bit `records`, `61` = `checksum_type`, `62..64` = `format_version: u16` LE) |
| `64..72` | 8 B | `u64` LE | `packed_dict_field` | `global_dict_size & 0xFF_FFFF_FFFF` |
| `72..80` | 8 B | `[u8; 8]` | `padding` | `[0u8; 8]` |

#### `BlobStoreFileFooter` (64 bytes, little-endian)
| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..8` | 8 B | `u64` LE | `zip_data_xxhash` | `zip_data_xxhash.to_le_bytes()` |
| `8..16` | 8 B | `u64` LE | `file_xxhash` | `file_xxhash.to_le_bytes()` |
| `16..56` | 40 B | `[u8; 40]` | `reserved` | `[0u8; 40]` |
| `56..60` | 4 B | `u32` LE | `padding` | `0u32.to_le_bytes()` |
| `60..64` | 4 B | `u32` LE | `footer_length` | `64u32.to_le_bytes()` (`[0x40, 0x00, 0x00, 0x00]`) |

---

### 2.2 `SortedUintVec` (`ZSUV`, 32 B Header)
**Source**: [`src/blob_store/sorted_uint_vec.rs`](../src/blob_store/sorted_uint_vec.rs) (`write_to` / `read_from`)

| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..4` | 4 B | `[u8; 4]` | `magic` | `b"ZSUV"` (`SERIAL_MAGIC`) |
| `4..5` | 1 B | `u8` | `version` | `1` (`SERIAL_VERSION`) |
| `5..6` | 1 B | `u8` | `log2_block_units` | `self.config.log2_block_units` (`6` or `7`) |
| `6..7` | 1 B | `u8` | `offset_width` | `self.config.offset_width` (`1..=32`) |
| `7..8` | 1 B | `u8` | `sample_width` | `self.config.sample_width` (`1..=64`) |
| `8..16` | 8 B | `u64` LE | `size` | `(self.size as u64).to_le_bytes()` |
| `16..24` | 8 B | `u64` LE | `index_len` | `(self.index.len() as u64).to_le_bytes()` |
| `24..32` | 8 B | `u64` LE | `data_len` | `(self.data.len() as u64).to_le_bytes()` |
| `32..32+index_len` | `index_len` | `[u8]` | `index` | Packed block-minimum samples (`num_blocks * sample_width` bits) |
| `32+index_len..32+index_len+data_len` | `data_len` | `[u8]` | `data` | Packed per-element block deltas (`size * offset_width` bits) |

---

### 2.3 `ZipOffsetBlobStore` (128 B Header + Content + Padding + `ZSUV` + 64 B Footer)
**Source**: [`src/blob_store/zip_offset.rs`](../src/blob_store/zip_offset.rs) (`FileHeader::to_bytes` / `from_bytes`, `save_to_writer`)

#### `FileHeader` (128 bytes, `HEADER_SIZE = 128`)
| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..20` | 20 B | `[u8; 20]` | `magic` | `*b"zipora-blob-store\0\0\0"` (`MAGIC_SIGNATURE`) |
| `20..40` | 20 B | `[u8; 20]` | `class_name` | `*b"ZipOffsetBlobStore\0\0"` (`CLASS_NAME`) |
| `40..48` | 8 B | `u64` LE | `file_size` | `file_size.to_le_bytes()` |
| `48..56` | 8 B | `u64` LE | `unzip_size` | `unzip_size.to_le_bytes()` |
| `56..64` | 8 B | `u64` LE | `records_checksum_version` | `(records & 0xFF_FFFF_FFFF) \| ((checksum_type as u64) << 40) \| ((1u64) << 48)` (`records = offsets.len()`) |
| `64..72` | 8 B | `u64` LE | `content_bytes` | `content_bytes.to_le_bytes()` |
| `72..80` | 8 B | `u64` LE | `offsets_bytes` | `offsets_bytes.to_le_bytes()` |
| `80..81` | 1 B | `u8` | `offsets_log2_block_units` | `config.offset_config.log2_block_units` |
| `81..82` | 1 B | `u8` | `checksum_level` | `config.checksum_level` (`0..=3`) |
| `82..83` | 1 B | `u8` | `compress_level` | `config.compress_level` |
| `83..128` | 45 B | `[u8; 45]` | `_padding` | `[0u8; 45]` |

#### File Body & 64 B Footer (`FOOTER_SIZE = 64`)
- `128..128 + content_bytes`: Record payload bytes (`content`).
- `128 + content_bytes..128 + content_bytes + content_padding`: `content_padding = (16 - (content_bytes % 16)) % 16` zero bytes.
- `.. + offsets_bytes`: Serialized `SortedUintVec` (`ZSUV`) section when `offsets_bytes > 0`.
- Final `64 B`: `[content_crc32c: u32 LE @0..4][reserved: [u8; 60] = 0 @4..64]`.

---

### 2.4 `Dictionary` (`src/entropy/dictionary.rs`)
**Source**: [`src/entropy/dictionary.rs`](../src/entropy/dictionary.rs) (`Dictionary::serialize` / `deserialize`)

| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..4` | 4 B | `u32` LE | `entry_count` | `(self.entries.len() as u32).to_le_bytes()` |
| Per entry (`0..2`) | 2 B | `u16` LE | `seq_len` | `(sequence.len() as u16).to_le_bytes()` |
| Per entry (`2..2+seq_len`) | `seq_len` B | `[u8]` | `sequence` | Raw sequence bytes (entries ordered lexicographically by `sequence`) |
| Per entry (`2+seq_len..6+seq_len`) | 4 B | `u32` LE | `offset` | `entry.offset.to_le_bytes()` |
| Per entry (`6+seq_len..10+seq_len`) | 4 B | `u32` LE | `length` | `entry.length.to_le_bytes()` |

---

### 2.5 `ZoSortedStrVec` (`ZOSV` v1, 16 B Header)
**Source**: [`src/containers/specialized/zo_sorted_str_vec.rs`](../src/containers/specialized/zo_sorted_str_vec.rs) (`to_bytes` / `from_bytes`)

| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..4` | 4 B | `[u8; 4]` | `magic` | `*b"ZOSV"` (`0x5653_4F5A`) |
| `4..6` | 2 B | `u16` LE | `version` | `1u16.to_le_bytes()` |
| `6..8` | 2 B | `u16` LE | `flags` | `0x0011u16.to_le_bytes()` (bit 0 = LE, bit 4 = 64-bit header field width) |
| `8..12` | 4 B | `u32` LE | `count` | `u32::try_from(self.len)?.to_le_bytes()` |
| `12..16` | 4 B | `u32` LE | `payload_len` | `u32::try_from(payload.len())?.to_le_bytes()` |
| `16..16+payload_len` | `payload_len` B | `[u32 LE, utf8]*` | `entries` | `count` entries of `[str_len: u32 LE][utf8_bytes: str_len B]` |

---

### 2.6 `VarInt` (`LEB128` / `ZigZag`)
**Source**: [`src/io/var_int.rs`](../src/io/var_int.rs), [`src/io/var_int_variants.rs`](../src/io/var_int_variants.rs)

| Variant | Wire Encoding |
|---------|---------------|
| `VarInt::encode(u64)` / `encode_u32(u32)` | 7-bit little-endian groups (`byte & 0x7F`); bit 7 (`0x80`) set on all bytes except the final byte (`1..=10 B` for `u64`, `1..=5 B` for `u32`). |
| `VarInt::encode_signed(i64)` | ZigZag mapping `((value << 1) ^ (value >> 63)) as u64` encoded as unsigned `LEB128`. |
| `encode_u32_sequence` / `encode_u64_sequence` | `[count: VarInt u64][v_0: VarInt]...[v_{count-1}: VarInt]`. |
| `encode_delta_sequence` | `[count: VarInt u64][first: VarInt u64][ZigZag(v_i - v_{i-1}): VarInt u64]...` with wrapping arithmetic. |

---

### 2.7 `MmapVecHeader` (`MMAP_VEC` v2, 80 B Header)
**Source**: [`src/memory/mmap_vec.rs`](../src/memory/mmap_vec.rs) (`MmapVecHeader`)

| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..8` | 8 B | `u64` LE | `magic` | `0x4D4D_4150_5F56_4543u64.to_le_bytes()` (wire bytes `b"CEV_PAMM"`) |
| `8..10` | 2 B | `u16` LE | `version` | `2u16.to_le_bytes()` (`MMAP_VEC_VERSION`) |
| `10..12` | 2 B | `u16` LE | `flags` | `0x0011u16.to_le_bytes()` (`MMAP_VEC_FLAGS`: bit 0 = LE, bit 4 = 64-bit header field width) |
| `12..16` | 4 B | `u32` LE | `element_size` | `(size_of::<T>() as u32).to_le_bytes()` |
| `16..24` | 8 B | `u64` LE | `length` | `length.to_le_bytes()` (`length <= capacity`) |
| `24..32` | 8 B | `u64` LE | `capacity` | `capacity.to_le_bytes()` (validated via `capacity.checked_mul(size_of::<T>())`) |
| `32..80` | 48 B | `[u8; 48]` | `reserved` | `[0u8; 48]` |

Element payload starts at `data_offset::<T>() = 80.div_ceil(align_of::<T>().max(1)) * align_of::<T>().max(1)`.

---

### 2.8 `ZReorderMap` (16 B Header + Signed-Run `VarInt` Stream)
**Source**: [`src/blob_store/reorder_map.rs`](../src/blob_store/reorder_map.rs) (`ZReorderMapBuilder::new`, `ZReorderMap::open`)

| Offset | Size | Type | Field | Exact Encoding |
|--------|------|------|-------|----------------|
| `0..8` | 8 B | `u64` LE | `size` | `(size as u64).to_le_bytes()` (`size <= 0x7FFF_FFFF_FFFF`) |
| `8..16` | 8 B | `i64` LE | `sign` | `sign.to_le_bytes()` (`1i64` for ascending runs or `-1i64` for descending runs) |
| `16..` | variable | `VarInt` runs | `entries` | Pairs of `[first_value: VarInt u64][signed_len: ZigZag VarInt i64]` terminated by `signed_len == 0` (`EOF`), with `\|signed_len\| <= remaining_elements` |

---

## 3. Format & Stream Parameter Ledger (`S4-R5`)

The following wire/stream format updates were introduced in Stage 2–3 (pre-stable 4.2.0):

1. **FSE (`src/entropy/fse.rs`, `d6ad3e3`)**:
   - Encoder initial state changed from `1` to `INITIAL_STATE = 4096` (`TOTFREQ`) to eliminate zero-bit fixed points (`state < freq_0` with `start == 0`).
   - Single-symbol FSE headers store `original_size` in the normalized frequency slot and emit a 0-byte bitstream payload (`O(1)` decode).
2. **`MmapVecHeader` (`src/memory/mmap_vec.rs`, `62540be`)**:
   - Replaced native-struct pointer cast with explicit 80-byte little-endian byte serialization, bumped `MMAP_VEC_VERSION` from `1` to `2`, and set portable `MMAP_VEC_FLAGS = 0x0011`.
3. **Entropy Blob Stores (`src/blob_store/entropy.rs`, `53d05d2`)**:
   - `HuffmanBlobStore`, `RansBlobStore`, and `DictionaryBlobStore` wrap each stored record in a 5-byte frame `[flag: u8][uncompressed_len: u32 LE][payload]` (`flag = 0` uncompressed, `flag = 1` compressed) and reject records exceeding `u32::MAX` bytes at `put`.
4. **Dictionary & PA-Zip Stream Format Parameters (`src/entropy/dictionary.rs`, `src/compression/dict_zip/compressor.rs`)**:
   - `DictionaryCompressor` and `OptimizedDictionaryCompressor`: `min_match_length` and `max_match_length` are stream format parameters enforced on `decompress`; `OptimizedDictionaryCompressor` uses `flag = 0` (literal), `flag = 1` (local sliding-window back-reference in `data`), and `flag = 2` (dictionary offset in `self.text`).
   - `PaZipCompressor`: `local_config.max_match_length` is a stream format parameter enforced on `decompress` (`length <= local_config.max_match_length.max(256)`).
