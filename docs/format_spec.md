# Zipora Persisted Binary Format Specification (`C2.1` / `G13`)

Every persisted binary format in `zipora` obeys two mandatory invariants:

1. **Explicit Little-Endian Wire Layout**: All multi-byte integers on disk or in serialized byte slices are encoded and decoded exclusively via `to_le_bytes()` / `from_le_bytes()`. Native struct pointer casting (`*const Header as *const u8`) is forbidden on persisted bytes.
2. **Self-Describing Header Envelope**: Every persisted container or file format carries a magic signature, format version, endianness/word-size flags (`0x0011` = little-endian (`0x0001`) | 64-bit word size (`0x0010`) where applicable), and explicit header/payload lengths validated before allocation.

---

## 1. Summary Inventory of Persisted Formats

| # | Format / Type | Source File | Magic (`4 B+`) | Version | Flags (`Endian / Word`) | Header Length |
|---|---------------|-------------|----------------|---------|-------------------------|---------------|
| 1 | `FileHeaderBase` + `BlobStoreFileFooter` | `src/blob_store/file_header.rs` | `"terark-blob-store\0"` (`18 B` at offset 1) | `format_version` (`u16` LE at `62..64`) | Little-endian fixed fields + XXHash64 (`u64` LE) | `80 B` header + `64 B` footer |
| 2 | `SortedUintVec` (`ZSUV`) | `src/blob_store/sorted_uint_vec.rs` | `"ZSUV"` (`0x5655_535A`) | `1` (`u8`) | Little-endian (`u32`/`u64` LE fields) | `17 B` (`[magic:4][ver:1][block_shift:4][len:8]`) |
| 3 | `ZipOffsetBlobStore` | `src/blob_store/zip_offset.rs` | `"terark-blob-store\0"` | `1` (`u16` LE) | `checksum_level`/`checksum_type` + LE `ZSUV` index | `80 B` `FileHeaderBase` + `ZSUV` offset index + `64 B` footer |
| 4 | `Dictionary` / `SuffixArrayDictionary` | `src/entropy/dictionary.rs`, `src/compression/dict_zip/dictionary.rs` | Entry-count framed / `bincode` LE | `1` | Little-endian (`u32`/`u16` LE fields; `min/max_match_length` stream params) | `4 B` (`entry_count: u32 LE`) + `10 B + len` per entry |
| 5 | `ZoSortedStrVec` (`ZOSV`) | `src/containers/specialized/zo_sorted_str_vec.rs` | `"ZOSV"` (`0x5653_4F5A`) | `1` (`u16` LE) | `0x0011` (`u16` LE: LE + 64-bit word) | `16 B` |
| 6 | `VarInt` (`LEB128` / `ZigZag` / `GroupVarint`) | `src/io/var_int.rs`, `src/io/var_int_variants.rs` | Self-delimiting (`MSB` continuation bit) | `1` | Little-endian 7-bit payload groups (`LEB128`) | `1–5 B` (`u32`) / `1–10 B` (`u64`) |
| 7 | `MmapVecHeader` (`MMAP`) | `src/memory/mmap_vec.rs` | `"MMAP\0\0\0\0"` (`0x0000_0000_5041_4D4D`) | `2` (`u16` LE) | `0x0011` (`u16` LE: bit 0 = LE, bit 4 = 64-bit word) | `80 B` (padded to `align_of::<T>()`) |
| 8 | `ZReorderMap` | `src/blob_store/reorder_map.rs` | `0x5A52_4D50_3030_3000` (`"ZRMP000"` in upper 56 bits) | `1` | Little-endian (`u64` LE header words + LE `VarInt` run stream) | `16 B` |

---

## 2. Detailed Byte Layouts

### 2.1 `FileHeaderBase` (80 B) and `BlobStoreFileFooter` (64 B)
**Source**: [`src/blob_store/file_header.rs`](../src/blob_store/file_header.rs)

#### `FileHeaderBase` (80 bytes, little-endian)
| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..1` | 1 B | `u8` | `magic_len` | `17` (`MAGIC_STR_LEN`) |
| `1..20` | 19 B | `[u8; 19]` | `magic` | `"terark-blob-store\0\0"` |
| `20..40` | 20 B | `[u8; 20]` | `class_name` | NUL-padded store class name (e.g. `"ZipOffsetBlobStore"`) |
| `40..48` | 8 B | `u64` LE | `file_size` | Total file size in bytes (`header + payload + footer`) |
| `48..56` | 8 B | `u64` LE | `unzip_size` | Total uncompressed payload size in bytes |
| `56..64` | 8 B | `u64` LE | `packed_records` | `records(40 bits) \| (checksum_type(8 bits) << 40) \| (format_version(16 bits) << 48)` |
| `64..72` | 8 B | `u64` LE | `global_dict_size` | `global_dict_size(40 bits) \| pad(24 bits)` |
| `72..80` | 8 B | `u64` LE | `padding` | Zero-filled |

#### `BlobStoreFileFooter` (64 bytes, little-endian)
| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..8` | 8 B | `u64` LE | `zip_data_xxhash` | XXHash64 over compressed data payload |
| `8..16` | 8 B | `u64` LE | `file_xxhash` | XXHash64 over `[0..header+payload]` |
| `16..56` | 40 B | `[u64; 5]` LE | `reserved` | Store-specific metadata (index offset, data size, flags, checksum level) |
| `56..60` | 4 B | `u32` LE | `padding` | `0` |
| `60..64` | 4 B | `u32` LE | `footer_length` | Always `64` (`0x40, 0x00, 0x00, 0x00`) |

---

### 2.2 `SortedUintVec` (`ZSUV`)
**Source**: [`src/blob_store/sorted_uint_vec.rs`](../src/blob_store/sorted_uint_vec.rs)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..4` | 4 B | `[u8; 4]` | `magic` | `b"ZSUV"` |
| `4..5` | 1 B | `u8` | `version` | `1` |
| `5..9` | 4 B | `u32` LE | `block_size_shift` | Log2 of samples per block (`6`..=`9`) |
| `9..17` | 8 B | `u64` LE | `len` | Number of stored `u64` values |
| `17..` | `4 B + N×21 B` | LE blocks | `blocks` | `num_blocks: u32 LE` followed by per-block `[base_value: u64 LE][delta_bits: u8][data_offset: u32 LE][first_delta: i64 LE]` and `bit_data` (`u64` LE words) |

---

### 2.3 `MmapVecHeader` (`MMAP` v2, 80 B)
**Source**: [`src/memory/mmap_vec.rs`](../src/memory/mmap_vec.rs)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..8` | 8 B | `u64` LE | `magic` | `0x4D4D_4150_5F56_4543` (`"MMAP_VEC"`) |
| `8..10` | 2 B | `u16` LE | `version` | `2` (`MMAP_VEC_VERSION`) |
| `10..12` | 2 B | `u16` LE | `flags` | `0x0011` (`MMAP_VEC_FLAGS`: bit 0 = LE, bit 4 = 64-bit word size) |
| `12..16` | 4 B | `u32` LE | `element_size` | `size_of::<T>() as u32` |
| `16..24` | 8 B | `u64` LE | `len` | Current initialized element count (`len <= capacity`) |
| `24..32` | 8 B | `u64` LE | `capacity` | Allocated element capacity (validated via `capacity.checked_mul(size_of::<T>())`) |
| `32..40` | 8 B | `u64` LE | `checksum` | Header integrity checksum |
| `40..80` | 40 B | `[u8; 40]` | `reserved` | Zero-filled reserved bytes |

Data starts at `data_offset::<T>() = 80.div_ceil(align_of::<T>().max(1)) * align_of::<T>().max(1)`.

---

### 2.4 `ZoSortedStrVec` (`ZOSV` v1, 16 B Header)
**Source**: [`src/containers/specialized/zo_sorted_str_vec.rs`](../src/containers/specialized/zo_sorted_str_vec.rs)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..4` | 4 B | `[u8; 4]` | `magic` | `b"ZOSV"` |
| `4..6` | 2 B | `u16` LE | `version` | `1` |
| `6..8` | 2 B | `u16` LE | `flags` | `0x0011` (LE + 64-bit word size) |
| `8..12` | 4 B | `u32` LE | `count` | Number of lexicographically sorted strings |
| `12..16` | 4 B | `u32` LE | `payload_len` | Total byte length of string payload |
| `16..` | `payload_len` | `[u32 LE, utf8]*` | `entries` | `count` length-prefixed UTF-8 strings |

---

### 2.5 `ZReorderMap` (16 B Header)
**Source**: [`src/blob_store/reorder_map.rs`](../src/blob_store/reorder_map.rs)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| `0..8` | 8 B | `u64` LE | `size` | Total mapping elements (`<= 0x7FFF_FFFF_FFFF`) |
| `8..9` | 1 B | `u8` | `bits_per_element` | Bit width per element (`0`..=`40`) |
| `9..16` | 7 B | `[u8; 7]` | `magic` | Upper 56 bits of `0x5A52_4D50_3030_3000` (`"ZRMP000"`) |
| `16..` | variable | `VarInt` runs | `entries` | Signed-length run-encoded `u32` sequences bounded by `size` |

---

### 2.6 `Dictionary` (`src/entropy/dictionary.rs`) & `VarInt` (`src/io/var_int.rs`)
- **`Dictionary::serialize` / `deserialize`**:
  - `[0..4]`: `entry_count: u32 LE` (bounded against `remaining_bytes / 10` before allocation).
  - Per entry: `[id: u16 LE][frequency: u32 LE][data_len: u32 LE][data: data_len bytes]`.
  - Stream compression/decompression requires matching `min_match_length` and `max_match_length` format parameters.
- **`VarInt`**:
  - Unsigned `u32`/`u64` encoded in 7-bit little-endian groups with bit `0x80` continuation flag; signed `i32`/`i64` mapped via ZigZag `(n << 1) ^ (n >> 63)` prior to `LEB128`.
  - Sequence decoders (`decode_u32_sequence`, `decode_u64_sequence`, `decode_delta_sequence`) bound `Vec::with_capacity(count)` against remaining input bytes (`data.len() - header_len`) before allocation.
