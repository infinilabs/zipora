//! On-disk file header structures for blob stores
//!
//! All structures are packed to exact byte boundaries for binary compatibility.

/// Magic string identifying blob store files.
pub const MAGIC_STRING: &[u8; 18] = b"terark-blob-store\0";

/// Length of the magic string (excluding null terminator in comparison).
pub const MAGIC_STR_LEN: usize = 17;

/// Checksum type for record-level integrity verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChecksumType {
    /// CRC-32C (Castagnoli) - 4 bytes per record
    Crc32c = 0,
    /// CRC-16C - 2 bytes per record
    Crc16c = 1,
}

impl ChecksumType {
    /// Size of the checksum value in bytes.
    #[inline]
    pub fn size(self) -> usize {
        match self {
            ChecksumType::Crc32c => 4,
            ChecksumType::Crc16c => 2,
        }
    }

    /// Create from raw u8 value.
    #[inline]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => ChecksumType::Crc16c,
            _ => ChecksumType::Crc32c,
        }
    }
}

/// Base file header for all blob store types.
///
/// Exactly 80 bytes. All blob store file headers start with this structure,
/// followed by per-type fields to reach 128 bytes total.
///
/// # Layout (80 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       1     magic_len
/// 1       19    magic (null-padded string)
/// 20      20    class_name (null-padded string)
/// 40      8     file_size
/// 48      8     unzip_size
/// 56      8     records(40) | checksum_type(8) | format_version(16)
/// 64      8     global_dict_size(40) | pad(24)
/// 72      8     padding
/// ```
#[derive(Clone)]
pub struct FileHeaderBase {
    data: [u8; 80],
}

impl FileHeaderBase {
    /// Create a new zeroed header.
    pub fn new() -> Self {
        let mut h = Self { data: [0u8; 80] };
        h.set_magic_len(MAGIC_STR_LEN as u8);
        h.data[1..1 + MAGIC_STR_LEN].copy_from_slice(&MAGIC_STRING[..MAGIC_STR_LEN]);
        h
    }

    /// Create from raw bytes. Returns None if magic is invalid.
    pub fn from_bytes(bytes: &[u8; 80]) -> Option<Self> {
        let h = Self { data: *bytes };
        if h.validate_magic() { Some(h) } else { None }
    }

    /// Get raw bytes for writing.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 80] {
        &self.data
    }

    /// Validate the magic string.
    #[inline]
    pub fn validate_magic(&self) -> bool {
        self.data[1..1 + MAGIC_STR_LEN] == MAGIC_STRING[..MAGIC_STR_LEN]
    }

    // --- Field accessors ---

    #[inline]
    pub fn magic_len(&self) -> u8 {
        self.data[0]
    }

    #[inline]
    pub fn set_magic_len(&mut self, v: u8) {
        self.data[0] = v;
    }

    /// Get magic string as bytes (19 bytes starting at offset 1).
    #[inline]
    pub fn magic(&self) -> &[u8] {
        &self.data[1..20]
    }

    /// Get class name as a trimmed string.
    pub fn class_name(&self) -> &str {
        let bytes = &self.data[20..40];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(20);
        std::str::from_utf8(&bytes[..end]).unwrap_or("")
    }

    /// Set class name (max 19 chars + null).
    pub fn set_class_name(&mut self, name: &str) {
        let bytes = name.as_bytes();
        let len = bytes.len().min(19);
        self.data[20..20 + len].copy_from_slice(&bytes[..len]);
        // Zero-fill remaining
        for b in &mut self.data[20 + len..40] {
            *b = 0;
        }
    }

    #[inline]
    pub fn file_size(&self) -> u64 {
        u64::from_le_bytes(self.data[40..48].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    pub fn set_file_size(&mut self, v: u64) {
        self.data[40..48].copy_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn unzip_size(&self) -> u64 {
        u64::from_le_bytes(self.data[48..56].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    pub fn set_unzip_size(&mut self, v: u64) {
        self.data[48..56].copy_from_slice(&v.to_le_bytes());
    }

    /// Packed field at offset 56: records(40) | checksum_type(8) | format_version(16).
    #[inline]
    fn packed_records_field(&self) -> u64 {
        u64::from_le_bytes(self.data[56..64].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    fn set_packed_records_field(&mut self, v: u64) {
        self.data[56..64].copy_from_slice(&v.to_le_bytes());
    }

    /// Number of records (40-bit field, max ~1 trillion).
    #[inline]
    pub fn records(&self) -> u64 {
        self.packed_records_field() & 0xFF_FFFF_FFFF
    }

    /// Set number of records.
    #[inline]
    pub fn set_records(&mut self, records: u64) {
        let old = self.packed_records_field();
        let new = (old & !0xFF_FFFF_FFFF) | (records & 0xFF_FFFF_FFFF);
        self.set_packed_records_field(new);
    }

    /// Checksum type (8-bit field).
    #[inline]
    pub fn checksum_type(&self) -> ChecksumType {
        let v = ((self.packed_records_field() >> 40) & 0xFF) as u8;
        ChecksumType::from_u8(v)
    }

    /// Set checksum type.
    #[inline]
    pub fn set_checksum_type(&mut self, ct: ChecksumType) {
        let old = self.packed_records_field();
        let new = (old & !(0xFF << 40)) | ((ct as u64) << 40);
        self.set_packed_records_field(new);
    }

    /// Format version (16-bit field).
    #[inline]
    pub fn format_version(&self) -> u16 {
        ((self.packed_records_field() >> 48) & 0xFFFF) as u16
    }

    /// Set format version.
    #[inline]
    pub fn set_format_version(&mut self, v: u16) {
        let old = self.packed_records_field();
        let new = (old & !(0xFFFF << 48)) | ((v as u64) << 48);
        self.set_packed_records_field(new);
    }

    /// Global dictionary size (40-bit field at offset 64).
    #[inline]
    pub fn global_dict_size(&self) -> u64 {
        let packed = u64::from_le_bytes(self.data[64..72].try_into().expect("slice is 8 bytes"));
        packed & 0xFF_FFFF_FFFF
    }

    /// Set global dictionary size.
    #[inline]
    pub fn set_global_dict_size(&mut self, v: u64) {
        let old = u64::from_le_bytes(self.data[64..72].try_into().expect("slice is 8 bytes"));
        let new = (old & !0xFF_FFFF_FFFF) | (v & 0xFF_FFFF_FFFF);
        self.data[64..72].copy_from_slice(&new.to_le_bytes());
    }
}

impl Default for FileHeaderBase {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FileHeaderBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHeaderBase")
            .field("class_name", &self.class_name())
            .field("file_size", &self.file_size())
            .field("unzip_size", &self.unzip_size())
            .field("records", &self.records())
            .field("checksum_type", &self.checksum_type())
            .field("format_version", &self.format_version())
            .finish()
    }
}

/// File footer for blob store files.
///
/// Exactly 64 bytes for binary compatibility.
/// Placed at the end of every blob store file for integrity verification.
///
/// # Layout (64 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       8     zip_data_xxhash
/// 8       8     file_xxhash
/// 16      40    reserved (5 x u64)
/// 56      4     padding
/// 60      4     footer_length (always 64)
/// ```
#[derive(Clone)]
pub struct BlobStoreFileFooter {
    data: [u8; 64],
}

impl BlobStoreFileFooter {
    /// Create a new zeroed footer with correct footer_length.
    pub fn new() -> Self {
        let mut f = Self { data: [0u8; 64] };
        f.set_footer_length(64);
        f
    }

    /// Create from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        Self { data: *bytes }
    }

    /// Get raw bytes for writing.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.data
    }

    /// XXHash64 of compressed/zipped data blocks.
    #[inline]
    pub fn zip_data_xxhash(&self) -> u64 {
        u64::from_le_bytes(self.data[0..8].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    pub fn set_zip_data_xxhash(&mut self, v: u64) {
        self.data[0..8].copy_from_slice(&v.to_le_bytes());
    }

    /// XXHash64 of the entire file (header + data, excluding footer).
    #[inline]
    pub fn file_xxhash(&self) -> u64 {
        u64::from_le_bytes(self.data[8..16].try_into().expect("slice is 8 bytes"))
    }

    #[inline]
    pub fn set_file_xxhash(&mut self, v: u64) {
        self.data[8..16].copy_from_slice(&v.to_le_bytes());
    }

    /// Footer length field (always 64).
    #[inline]
    pub fn footer_length(&self) -> u32 {
        u32::from_le_bytes(self.data[60..64].try_into().expect("slice is 4 bytes"))
    }

    #[inline]
    fn set_footer_length(&mut self, v: u32) {
        self.data[60..64].copy_from_slice(&v.to_le_bytes());
    }
}

impl Default for BlobStoreFileFooter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BlobStoreFileFooter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobStoreFileFooter")
            .field(
                "zip_data_xxhash",
                &format!("0x{:016x}", self.zip_data_xxhash()),
            )
            .field("file_xxhash", &format!("0x{:016x}", self.file_xxhash()))
            .field("footer_length", &self.footer_length())
            .finish()
    }
}

/// Calculate padding needed to align `offset` to `align` boundary.
#[inline]
pub fn align_padding(offset: usize, align: usize) -> usize {
    let rem = offset % align;
    if rem == 0 { 0 } else { align - rem }
}

/// Round up `offset` to next `align` boundary.
#[inline]
pub fn align_up(offset: usize, align: usize) -> usize {
    offset + align_padding(offset, align)
}

/// Size constants.
pub const FILE_HEADER_BASE_SIZE: usize = 80;
pub const FILE_HEADER_FULL_SIZE: usize = 128;
pub const FILE_FOOTER_SIZE: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_header_base_size() {
        assert_eq!(std::mem::size_of::<[u8; FILE_HEADER_BASE_SIZE]>(), 80);
        assert_eq!(FILE_HEADER_FULL_SIZE, 128);
        assert_eq!(FILE_FOOTER_SIZE, 64);
    }

    #[test]
    fn test_file_header_base_new() {
        let h = FileHeaderBase::new();
        assert!(h.validate_magic());
        assert_eq!(h.magic_len(), MAGIC_STR_LEN as u8);
        assert_eq!(h.file_size(), 0);
        assert_eq!(h.unzip_size(), 0);
        assert_eq!(h.records(), 0);
        assert_eq!(h.format_version(), 0);
        assert_eq!(h.global_dict_size(), 0);
    }

    #[test]
    fn test_file_header_base_class_name() {
        let mut h = FileHeaderBase::new();

        h.set_class_name("MixedLenBlobStore");
        assert_eq!(h.class_name(), "MixedLenBlobStore");

        h.set_class_name("ZeroLengthBlobStore");
        assert_eq!(h.class_name(), "ZeroLengthBlobStore");

        // Truncation at 19 chars
        h.set_class_name("ThisIsAVeryLongClassName");
        assert_eq!(h.class_name().len(), 19);
    }

    #[test]
    fn test_file_header_base_records_packed() {
        let mut h = FileHeaderBase::new();

        // Set records (40-bit)
        h.set_records(0xFF_FFFF_FFFF);
        assert_eq!(h.records(), 0xFF_FFFF_FFFF);

        // Set checksum type
        h.set_checksum_type(ChecksumType::Crc16c);
        assert_eq!(h.checksum_type(), ChecksumType::Crc16c);
        assert_eq!(h.records(), 0xFF_FFFF_FFFF); // records unchanged

        // Set format version
        h.set_format_version(42);
        assert_eq!(h.format_version(), 42);
        assert_eq!(h.checksum_type(), ChecksumType::Crc16c); // unchanged
        assert_eq!(h.records(), 0xFF_FFFF_FFFF); // unchanged
    }

    #[test]
    fn test_file_header_base_roundtrip() {
        let mut h = FileHeaderBase::new();
        h.set_class_name("SimpleZipBlobStore");
        h.set_file_size(4096);
        h.set_unzip_size(8192);
        h.set_records(1000);
        h.set_checksum_type(ChecksumType::Crc32c);
        h.set_format_version(1);
        h.set_global_dict_size(512);

        let bytes = h.as_bytes();
        let h2 = FileHeaderBase::from_bytes(bytes).unwrap();

        assert_eq!(h2.class_name(), "SimpleZipBlobStore");
        assert_eq!(h2.file_size(), 4096);
        assert_eq!(h2.unzip_size(), 8192);
        assert_eq!(h2.records(), 1000);
        assert_eq!(h2.checksum_type(), ChecksumType::Crc32c);
        assert_eq!(h2.format_version(), 1);
        assert_eq!(h2.global_dict_size(), 512);
    }

    #[test]
    fn test_file_header_base_invalid_magic() {
        let mut bytes = [0u8; 80];
        bytes[0] = 17;
        bytes[1..18].copy_from_slice(b"invalid-magic-str");
        assert!(FileHeaderBase::from_bytes(&bytes).is_none());
    }

    #[test]
    fn test_blob_store_file_footer() {
        let f = BlobStoreFileFooter::new();
        assert_eq!(f.footer_length(), 64);
        assert_eq!(f.zip_data_xxhash(), 0);
        assert_eq!(f.file_xxhash(), 0);
    }

    #[test]
    fn test_blob_store_file_footer_roundtrip() {
        let mut f = BlobStoreFileFooter::new();
        f.set_zip_data_xxhash(0xDEADBEEF_CAFEBABE);
        f.set_file_xxhash(0x12345678_9ABCDEF0);

        let bytes = f.as_bytes();
        let f2 = BlobStoreFileFooter::from_bytes(bytes);

        assert_eq!(f2.zip_data_xxhash(), 0xDEADBEEF_CAFEBABE);
        assert_eq!(f2.file_xxhash(), 0x12345678_9ABCDEF0);
        assert_eq!(f2.footer_length(), 64);
    }

    #[test]
    fn test_checksum_type() {
        assert_eq!(ChecksumType::Crc32c.size(), 4);
        assert_eq!(ChecksumType::Crc16c.size(), 2);
        assert_eq!(ChecksumType::from_u8(0), ChecksumType::Crc32c);
        assert_eq!(ChecksumType::from_u8(1), ChecksumType::Crc16c);
        assert_eq!(ChecksumType::from_u8(255), ChecksumType::Crc32c); // default
    }

    #[test]
    fn test_align_padding() {
        assert_eq!(align_padding(0, 16), 0);
        assert_eq!(align_padding(1, 16), 15);
        assert_eq!(align_padding(15, 16), 1);
        assert_eq!(align_padding(16, 16), 0);
        assert_eq!(align_padding(17, 16), 15);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(100, 8), 104);
    }

    #[test]
    fn test_file_sizes_match_reference() {
        // Verify struct sizes match expected layout
        assert_eq!(FILE_HEADER_BASE_SIZE, 80);
        assert_eq!(FILE_FOOTER_SIZE, 64);
        // Total file header should be 128 bytes for all stores
        assert_eq!(FILE_HEADER_FULL_SIZE, 128);
    }

    #[test]
    fn test_header_debug_format() {
        let h = FileHeaderBase::new();
        let s = format!("{:?}", h);
        assert!(s.contains("FileHeaderBase"));
        assert!(s.contains("records"));

        let f = BlobStoreFileFooter::new();
        let s = format!("{:?}", f);
        assert!(s.contains("BlobStoreFileFooter"));
        assert!(s.contains("footer_length"));
    }

    #[test]
    fn test_large_record_count() {
        let mut h = FileHeaderBase::new();
        // Test with large record count (near 40-bit max)
        let large_count = (1u64 << 39) + 12345;
        h.set_records(large_count);
        assert_eq!(h.records(), large_count);

        // Ensure other packed fields unaffected
        h.set_checksum_type(ChecksumType::Crc16c);
        h.set_format_version(3);
        assert_eq!(h.records(), large_count);
        assert_eq!(h.checksum_type(), ChecksumType::Crc16c);
        assert_eq!(h.format_version(), 3);
    }
}
