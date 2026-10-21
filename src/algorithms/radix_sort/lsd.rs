use super::config::RadixSortConfig;
use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::Result;
use rayon::prelude::*;

// AVX2/BMI2 intrinsics for advanced SIMD acceleration

// AVX-512 intrinsics (avx512 feature)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use std::arch::x86_64::{
    __m512i, _mm512_and_si512, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_srlv_epi32,
    _mm512_storeu_si512,
};

/// Maximum value range for which counting sort is selected.
///
/// `counting_sort_u32` allocates `(max_val + 1)` buckets, so its memory cost is
/// O(value range), not O(len). Selecting it on element count alone allows a
/// handful of large-valued elements (e.g. a single `u32::MAX`) to request 32 GiB.
/// Above this range LSD radix sort is used instead (bounded O(len) memory).
const COUNTING_SORT_MAX_RANGE: usize = 1 << 16; // 65_536 buckets (512 KiB)

/// High-performance radix sort implementation
pub struct RadixSort {
    config: RadixSortConfig,
    stats: AlgorithmStats,
}

impl RadixSort {
    /// Create a new radix sort instance
    pub fn new() -> Self {
        Self::with_config(RadixSortConfig::default())
    }

    /// Create a radix sort instance with custom configuration
    pub fn with_config(config: RadixSortConfig) -> Self {
        Self {
            config,
            stats: AlgorithmStats {
                items_processed: 0,
                processing_time_us: 0,
                memory_used: 0,
                used_parallel: false,
                used_simd: false,
            },
        }
    }

    /// Sort a slice of unsigned 32-bit integers
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<()> {
        if self.config.radix_bits == 0 || self.config.radix_bits > 16 {
            return Err(crate::error::ZiporaError::invalid_parameter(
                "radix_bits must be between 1 and 16",
            ));
        }

        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        let used_parallel =
            if self.config.use_parallel && data.len() >= self.config.parallel_threshold {
                self.sort_u32_parallel(data)?
            } else {
                self.sort_u32_sequential(data)?;
                false
            };

        let elapsed = start_time.elapsed();
        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: self.estimate_memory_u32(data.len()),
            used_parallel,
            used_simd: self.config.use_simd,
        };

        Ok(())
    }

    /// Sort a slice of unsigned 64-bit integers
    pub fn sort_u64(&mut self, data: &mut [u64]) -> Result<()> {
        if self.config.radix_bits == 0 || self.config.radix_bits > 16 {
            return Err(crate::error::ZiporaError::invalid_parameter(
                "radix_bits must be between 1 and 16",
            ));
        }

        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        let used_parallel =
            if self.config.use_parallel && data.len() >= self.config.parallel_threshold {
                self.sort_u64_parallel(data)?
            } else {
                self.sort_u64_sequential(data)?;
                false
            };

        let elapsed = start_time.elapsed();
        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: self.estimate_memory_u64(data.len()),
            used_parallel,
            used_simd: self.config.use_simd,
        };

        Ok(())
    }

    /// Get algorithm statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
    }

    /// Sort a slice of byte arrays by their content
    pub fn sort_bytes(&mut self, data: &mut Vec<Vec<u8>>) -> Result<()> {
        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        self.sort_bytes_msd(data.as_mut_slice(), 0)?;

        let elapsed = start_time.elapsed();
        let total_bytes: usize = data.iter().map(|v| v.len()).sum();

        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: total_bytes + data.len() * std::mem::size_of::<Vec<u8>>(),
            used_parallel: false, // MSD radix sort is inherently sequential for strings
            used_simd: false,
        };

        Ok(())
    }

    fn sort_u32_sequential(&self, data: &mut [u32]) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        // Counting sort is chosen only when the dataset is small AND its value
        // range is bounded — its bucket array is sized by the max value, so a
        // large range would allocate excessive memory regardless of len.
        if data.len() <= self.config.use_counting_sort_threshold {
            let max_val = data.iter().copied().max().unwrap_or(0) as usize;
            if max_val <= COUNTING_SORT_MAX_RANGE {
                self.counting_sort_u32(data);
                return Ok(());
            }
        }

        if self.config.radix_bits == 8 {
            self.sort_u32_radix8(data);
            return Ok(());
        }

        let radix = 1usize << self.config.radix_bits;
        let mask = (radix - 1) as u32;
        let mut buffer = vec![0u32; data.len()];
        let mut counts = vec![0usize; radix];

        let max_passes = 32_usize.div_ceil(self.config.radix_bits);

        // Ping-pong: each pass scatters src into dst and the roles swap,
        // instead of copying the buffer back after every pass.
        let mut in_data = true;
        for pass in 0..max_passes {
            // shift < 32, so the u32 shifts below cannot overflow: sort_u32
            // (the only caller, directly or through sort_chunks_parallel)
            // rejects radix_bits outside 1..=16, and the last pass has
            // shift = (ceil(32 / r) - 1) * r < 32.
            let shift = pass * self.config.radix_bits;
            let (src, dst): (&[u32], &mut [u32]) = if in_data {
                (&*data, &mut buffer[..])
            } else {
                (&buffer[..], &mut *data)
            };

            // Count occurrences with AVX-512 acceleration when available
            counts.fill(0);

            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            {
                if self.config.use_simd && !cfg!(miri) && src.len() >= 16 && shift < 24 {
                    if std::arch::is_x86_feature_detected!("avx512f")
                        && std::arch::is_x86_feature_detected!("avx512bw")
                    {
                        // SAFETY: avx512f and avx512bw features detected at runtime
                        unsafe {
                            self.count_digits_avx512(src, shift, mask, &mut counts);
                        }
                    } else {
                        for &value in src.iter() {
                            let digit = ((value >> shift) & mask) as usize;
                            counts[digit] += 1;
                        }
                    }
                } else {
                    for &value in src.iter() {
                        let digit = ((value >> shift) & mask) as usize;
                        counts[digit] += 1;
                    }
                }
            }

            #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
            {
                for &value in src.iter() {
                    let digit = ((value >> shift) & mask) as usize;
                    counts[digit] += 1;
                }
            }

            // Convert counts to positions
            let mut pos = 0;
            for count in counts.iter_mut() {
                let old_count = *count;
                *count = pos;
                pos += old_count;
            }

            // Distribute elements
            for &value in src.iter() {
                let digit = ((value >> shift) & mask) as usize;
                dst[counts[digit]] = value;
                counts[digit] += 1;
            }

            in_data = !in_data;
        }

        // An odd number of passes leaves the result in the buffer.
        if !in_data {
            data.copy_from_slice(&buffer);
        }

        Ok(())
    }

    /// Cache-conscious 8-bit radix sort for `u32`:
    /// - For inputs fitting in L2 cache (`<= 65,536` elements), computes all 4 byte
    ///   histograms in a single pass and runs 8-bit LSD ping-pong while skipping
    ///   trivial single-bucket passes.
    /// - For larger inputs (`> 65,536` elements), scatters once by the highest
    ///   non-trivial byte into 256 L2-resident buckets (`~156 KiB` at `10^7` elements)
    ///   and finishes the remaining lower bytes inside L2 cache.
    fn sort_u32_radix8(&self, data: &mut [u32]) {
        const L2_THRESHOLD: usize = 65_536;
        let n = data.len();

        if n <= L2_THRESHOLD {
            let mut buffer = Vec::<std::mem::MaybeUninit<u32>>::with_capacity(n);
            // SAFETY: `MaybeUninit<u32>` does not require initialization.
            unsafe {
                buffer.set_len(n);
            }
            // SAFETY: `buffer` has length `n == data.len()`.
            let buf_slice =
                unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u32, n) };
            Self::lsd_u32_in_cache(data, buf_slice, 4);
            return;
        }

        let mut msb_counts = [0usize; 256];
        let mut prev = 0u32;
        let mut sorted = true;
        let mut or_diff = 0u32;
        let first = data[0];
        for &v in data.iter() {
            sorted &= v >= prev;
            prev = v;
            or_diff |= v ^ first;
            msb_counts[(v >> 24) as usize] += 1;
        }
        if sorted || or_diff == 0 {
            return;
        }

        let top_pass = ((31 - or_diff.leading_zeros()) / 8) as usize;
        if top_pass < 3 {
            msb_counts = [0usize; 256];
            let shift = top_pass * 8;
            for &v in data.iter() {
                msb_counts[((v >> shift) & 0xFF) as usize] += 1;
            }
        }

        let mut buffer = Vec::<std::mem::MaybeUninit<u32>>::with_capacity(n);
        // SAFETY: `MaybeUninit<u32>` does not require initialization.
        unsafe {
            buffer.set_len(n);
        }
        let buf_ptr = buffer.as_mut_ptr() as *mut u32;

        if top_pass == 0 {
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += msb_counts[b];
            }
            let src_ptr = data.as_ptr();
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from prefix sums of `msb_counts`.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = (v & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    *buf_ptr.add(out_pos) = v;
                }
            }
            // SAFETY: all `n` elements in `buf_ptr` were initialized by the scatter loop above.
            unsafe {
                std::ptr::copy_nonoverlapping(buf_ptr, data.as_mut_ptr(), n);
            }
            return;
        }

        let top_shift = top_pass * 8;
        let mut bucket_starts = [0usize; 256];
        let mut offsets = [0usize; 256];
        let mut pos = 0usize;
        for b in 0..256 {
            bucket_starts[b] = pos;
            offsets[b] = pos;
            pos += msb_counts[b];
        }

        let src_ptr = data.as_ptr();
        for i in 0..n {
            // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from prefix sums of `msb_counts`.
            unsafe {
                let v = *src_ptr.add(i);
                let digit = ((v >> top_shift) & 0xFF) as usize;
                let out_pos = *offsets.get_unchecked_mut(digit);
                *offsets.get_unchecked_mut(digit) = out_pos + 1;
                *buf_ptr.add(out_pos) = v;
            }
        }

        for b in 0..256 {
            let count = msb_counts[b];
            if count == 0 {
                continue;
            }
            let start = bucket_starts[b];
            // SAFETY: `start + count <= n` and `buf_ptr` was fully initialized above.
            let buf_bucket = unsafe { std::slice::from_raw_parts_mut(buf_ptr.add(start), count) };
            let data_bucket = &mut data[start..start + count];
            if count == 1 {
                data_bucket[0] = buf_bucket[0];
                continue;
            }
            Self::lsd_u32_from_buf_to_data(buf_bucket, data_bucket, top_pass);
        }
    }

    #[inline]
    fn lsd_u32_in_cache(data: &mut [u32], buf: &mut [u32], num_passes: usize) {
        let n = data.len();
        let mut hist = [[0usize; 256]; 4];
        let mut prev = 0u32;
        let mut sorted = true;
        for &v in data.iter() {
            sorted &= v >= prev;
            prev = v;
            hist[0][(v & 0xFF) as usize] += 1;
            hist[1][((v >> 8) & 0xFF) as usize] += 1;
            hist[2][((v >> 16) & 0xFF) as usize] += 1;
            hist[3][(v >> 24) as usize] += 1;
        }
        if sorted {
            return;
        }

        let mut in_data = true;
        for (pass, counts) in hist.iter().enumerate().take(num_passes) {
            if counts.contains(&n) {
                continue;
            }
            let shift = pass * 8;
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += counts[b];
            }
            let (src_ptr, dst_ptr) = if in_data {
                (data.as_ptr(), buf.as_mut_ptr())
            } else {
                (buf.as_ptr(), data.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> shift) & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_data = !in_data;
        }
        if !in_data {
            data.copy_from_slice(buf);
        }
    }

    #[inline]
    fn lsd_u32_from_buf_to_data(buf: &mut [u32], data: &mut [u32], num_passes: usize) {
        let n = buf.len();
        let mut hist = [[0usize; 256]; 3];
        let mut prev = 0u32;
        let mut sorted = true;
        for &v in buf.iter() {
            sorted &= v >= prev;
            prev = v;
            hist[0][(v & 0xFF) as usize] += 1;
            if num_passes > 1 {
                hist[1][((v >> 8) & 0xFF) as usize] += 1;
            }
            if num_passes > 2 {
                hist[2][((v >> 16) & 0xFF) as usize] += 1;
            }
        }
        if sorted {
            data.copy_from_slice(buf);
            return;
        }

        let mut in_buf = true;
        for (pass, counts) in hist.iter().enumerate().take(num_passes) {
            if counts.contains(&n) {
                continue;
            }
            let shift = pass * 8;
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += counts[b];
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf.as_ptr(), data.as_mut_ptr())
            } else {
                (data.as_ptr(), buf.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> shift) & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = !in_buf;
        }
        if in_buf {
            data.copy_from_slice(buf);
        }
    }

    fn sort_u32_parallel(&self, data: &mut [u32]) -> Result<bool> {
        self.sort_chunks_parallel(data, Self::sort_u32_sequential)
    }

    /// Chunked parallel driver shared by `sort_u32` and `sort_u64`: sorts
    /// `rayon::current_num_threads()` contiguous chunks with `sort_seq` in
    /// parallel, then merges the sorted runs with the parallel merge tree in
    /// `super::merge`.
    ///
    /// Returns whether that chunked path ran. Below `2 * parallel_threshold`
    /// elements, or when there would be a single chunk (one rayon thread), the
    /// whole slice is sorted by `sort_seq` instead and `Ok(false)` is returned,
    /// so `stats().used_parallel` reports the path that actually ran.
    ///
    /// Until S20-R3b the runs were combined by a single-threaded k-way heap
    /// merge (`MultiWayMerge`) followed by a full copy back, which made this
    /// path slower than the sequential sort.
    fn sort_chunks_parallel<T: Copy + Ord + Send + Sync>(
        &self,
        data: &mut [T],
        sort_seq: fn(&Self, &mut [T]) -> Result<()>,
    ) -> Result<bool> {
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = data.len().div_ceil(num_threads);
        if chunk_size >= data.len() || data.len() < self.config.parallel_threshold.saturating_mul(2)
        {
            sort_seq(self, data)?;
            return Ok(false);
        }

        data.par_chunks_mut(chunk_size)
            .try_for_each(|chunk| sort_seq(self, chunk))?;
        super::merge::merge_sorted_runs(data, chunk_size);
        Ok(true)
    }

    fn sort_u64_sequential(&self, data: &mut [u64]) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        if self.config.radix_bits == 8 {
            self.sort_u64_radix8(data);
            return Ok(());
        }

        let radix = 1usize << self.config.radix_bits;
        let mask = (radix - 1) as u64;
        let mut buffer = vec![0u64; data.len()];
        let mut counts = vec![0usize; radix];

        let max_passes = 64_usize.div_ceil(self.config.radix_bits);

        // Ping-pong: each pass scatters src into dst and the roles swap,
        // instead of copying the buffer back after every pass.
        let mut in_data = true;
        for pass in 0..max_passes {
            let shift = pass * self.config.radix_bits;
            let (src, dst): (&[u64], &mut [u64]) = if in_data {
                (&*data, &mut buffer[..])
            } else {
                (&buffer[..], &mut *data)
            };

            // Count occurrences
            counts.fill(0);
            for &value in src.iter() {
                let digit = ((value >> shift) & mask) as usize;
                counts[digit] += 1;
            }

            // Convert counts to positions
            let mut pos = 0;
            for count in counts.iter_mut() {
                let old_count = *count;
                *count = pos;
                pos += old_count;
            }

            // Distribute elements
            for &value in src.iter() {
                let digit = ((value >> shift) & mask) as usize;
                dst[counts[digit]] = value;
                counts[digit] += 1;
            }

            in_data = !in_data;
        }

        // An odd number of passes leaves the result in the buffer.
        if !in_data {
            data.copy_from_slice(&buffer);
        }

        Ok(())
    }

    /// Cache-conscious hybrid MSB-bucket + L2-resident LSD radix sort for `u64`:
    /// - For inputs `<= 32,768` elements, computes all 8 byte histograms in one pass
    ///   and runs 8-bit LSD ping-pong in L2 cache.
    /// - For larger inputs (`> 32,768` elements), scatters once by the top 8 bits
    ///   (`256` DTLB-friendly buckets, `~312 KiB` per bucket at `10^7` elements) into `buffer`,
    ///   then finishes the remaining 56 bits in 5 L2-resident passes
    ///   (`11 + 11 + 11 + 11 + 12 = 56` bits) landing directly in `data`
    ///   (6 total passes instead of 8).
    fn sort_u64_radix8(&self, data: &mut [u64]) {
        const L2_THRESHOLD: usize = 32_768;
        let n = data.len();

        if n <= L2_THRESHOLD {
            let mut buffer = Vec::<std::mem::MaybeUninit<u64>>::with_capacity(n);
            // SAFETY: `MaybeUninit<u64>` does not require initialization.
            unsafe {
                buffer.set_len(n);
            }
            // SAFETY: `buffer` has length `n == data.len()`.
            let buf_slice =
                unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u64, n) };
            Self::lsd_u64_in_cache(data, buf_slice, 8);
            return;
        }

        let mut msb_counts = [0usize; 256];
        let mut prev = 0u64;
        let mut sorted = true;
        let mut or_diff = 0u64;
        let first = data[0];
        for &v in data.iter() {
            sorted &= v >= prev;
            prev = v;
            or_diff |= v ^ first;
            msb_counts[(v >> 56) as usize] += 1;
        }
        if sorted || or_diff == 0 {
            return;
        }

        let mut buffer = Vec::<std::mem::MaybeUninit<u64>>::with_capacity(n);
        // SAFETY: `MaybeUninit<u64>` does not require initialization.
        unsafe {
            buffer.set_len(n);
        }
        let buf_ptr = buffer.as_mut_ptr() as *mut u64;

        if (or_diff >> 56) == 0 {
            let num_passes = (((63 - or_diff.leading_zeros()) / 8) + 1) as usize;
            // SAFETY: `buf_ptr` has capacity `n == data.len()`.
            let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf_ptr, n) };
            Self::lsd_u64_in_cache(data, buf_slice, num_passes);
            return;
        }

        let mut bucket_starts = [0usize; 256];
        let mut offsets = [0usize; 256];
        let mut pos = 0usize;
        for b in 0..256 {
            bucket_starts[b] = pos;
            offsets[b] = pos;
            pos += msb_counts[b];
        }

        let src_ptr = data.as_ptr();
        for i in 0..n {
            // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums.
            unsafe {
                let v = *src_ptr.add(i);
                let digit = (v >> 56) as usize;
                let out_pos = *offsets.get_unchecked_mut(digit);
                *offsets.get_unchecked_mut(digit) = out_pos + 1;
                *buf_ptr.add(out_pos) = v;
            }
        }

        let mut hists = Box::new(Low56Hists {
            h0: [0usize; 2048],
            h1: [0usize; 2048],
            h2: [0usize; 2048],
            h3: [0usize; 2048],
            h4: [0usize; 4096],
        });

        for b in 0..256 {
            let count = msb_counts[b];
            if count == 0 {
                continue;
            }
            let start = bucket_starts[b];
            // SAFETY: `start + count <= n` and `buf_ptr` was fully initialized above.
            let buf_bucket = unsafe { std::slice::from_raw_parts_mut(buf_ptr.add(start), count) };
            let data_bucket = &mut data[start..start + count];
            if count <= 64 {
                data_bucket.copy_from_slice(buf_bucket);
                data_bucket.sort_unstable();
                continue;
            }
            Self::lsd_u64_low56_from_buf_to_data(buf_bucket, data_bucket, &mut hists);
        }
    }

    #[inline]
    fn lsd_u64_in_cache(data: &mut [u64], buf: &mut [u64], num_passes: usize) {
        let n = data.len();
        let mut hist = [[0usize; 256]; 8];
        let mut prev = 0u64;
        let mut sorted = true;
        for &v in data.iter() {
            sorted &= v >= prev;
            prev = v;
            hist[0][(v & 0xFF) as usize] += 1;
            hist[1][((v >> 8) & 0xFF) as usize] += 1;
            hist[2][((v >> 16) & 0xFF) as usize] += 1;
            hist[3][((v >> 24) & 0xFF) as usize] += 1;
            hist[4][((v >> 32) & 0xFF) as usize] += 1;
            hist[5][((v >> 40) & 0xFF) as usize] += 1;
            hist[6][((v >> 48) & 0xFF) as usize] += 1;
            hist[7][(v >> 56) as usize] += 1;
        }
        if sorted {
            return;
        }

        let mut in_data = true;
        for (pass, counts) in hist.iter().enumerate().take(num_passes) {
            if counts.contains(&n) {
                continue;
            }
            let shift = pass * 8;
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += counts[b];
            }
            let (src_ptr, dst_ptr) = if in_data {
                (data.as_ptr(), buf.as_mut_ptr())
            } else {
                (buf.as_ptr(), data.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> shift) & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_data = !in_data;
        }
        if !in_data {
            data.copy_from_slice(buf);
        }
    }

    /// Sort the lower 56 bits of `buf` into `data` using 5 passes (`11 + 11 + 11 + 11 + 12 = 56` bits).
    #[inline]
    fn lsd_u64_low56_from_buf_to_data(buf: &mut [u64], data: &mut [u64], hists: &mut Low56Hists) {
        let n = buf.len();
        let Low56Hists { h0, h1, h2, h3, h4 } = hists;
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        h3.fill(0);
        h4.fill(0);
        let mut prev = 0u64;
        let mut sorted = true;
        for &v in buf.iter() {
            sorted &= v >= prev;
            prev = v;
            h0[(v & 0x7FF) as usize] += 1;
            h1[((v >> 11) & 0x7FF) as usize] += 1;
            h2[((v >> 22) & 0x7FF) as usize] += 1;
            h3[((v >> 33) & 0x7FF) as usize] += 1;
            h4[((v >> 44) & 0xFFF) as usize] += 1;
        }
        if sorted {
            data.copy_from_slice(buf);
            return;
        }

        let mut in_buf = true;

        // Pass 0: bits 0..11 (2048 buckets)
        if !h0.contains(&n) {
            let mut pos = 0usize;
            for count in h0.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let src_ptr = buf.as_ptr();
            let dst_ptr = data.as_mut_ptr();
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = (v & 0x7FF) as usize;
                    let out_pos = *h0.get_unchecked_mut(digit);
                    *h0.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = false;
        }

        // Pass 1: bits 11..22 (2048 buckets)
        if !h1.contains(&n) {
            let mut pos = 0usize;
            for count in h1.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf.as_ptr(), data.as_mut_ptr())
            } else {
                (data.as_ptr(), buf.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> 11) & 0x7FF) as usize;
                    let out_pos = *h1.get_unchecked_mut(digit);
                    *h1.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = !in_buf;
        }

        // Pass 2: bits 22..33 (2048 buckets)
        if !h2.contains(&n) {
            let mut pos = 0usize;
            for count in h2.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf.as_ptr(), data.as_mut_ptr())
            } else {
                (data.as_ptr(), buf.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> 22) & 0x7FF) as usize;
                    let out_pos = *h2.get_unchecked_mut(digit);
                    *h2.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = !in_buf;
        }

        // Pass 3: bits 33..44 (2048 buckets)
        if !h3.contains(&n) {
            let mut pos = 0usize;
            for count in h3.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf.as_ptr(), data.as_mut_ptr())
            } else {
                (data.as_ptr(), buf.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> 33) & 0x7FF) as usize;
                    let out_pos = *h3.get_unchecked_mut(digit);
                    *h3.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = !in_buf;
        }

        // Pass 4: bits 44..56 (4096 buckets)
        if !h4.contains(&n) {
            let mut pos = 0usize;
            for count in h4.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf.as_ptr(), data.as_mut_ptr())
            } else {
                (data.as_ptr(), buf.as_mut_ptr())
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 4096`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let v = *src_ptr.add(i);
                    let digit = ((v >> 44) & 0xFFF) as usize;
                    let out_pos = *h4.get_unchecked_mut(digit);
                    *h4.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_ptr.add(out_pos) = v;
                }
            }
            in_buf = !in_buf;
        }

        if in_buf {
            data.copy_from_slice(buf);
        }
    }

    fn sort_u64_parallel(&self, data: &mut [u64]) -> Result<bool> {
        self.sort_chunks_parallel(data, Self::sort_u64_sequential)
    }

    fn counting_sort_u32(&self, data: &mut [u32]) {
        let Some(&max_ref) = data.iter().max() else {
            return;
        };

        // Find maximum value to determine count buffer size
        let max_val = max_ref as usize;
        let mut counts = vec![0usize; max_val + 1];

        // Count occurrences
        for &value in data.iter() {
            counts[value as usize] += 1;
        }

        // Reconstruct sorted array
        let mut index = 0;
        for (value, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                data[index] = value as u32;
                index += 1;
            }
        }
    }

    fn sort_bytes_msd(&self, data: &mut [Vec<u8>], depth: usize) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        let mut counts = [0usize; 257];

        for item in data.iter() {
            let b = if depth < item.len() {
                item[depth] as usize + 1
            } else {
                0
            };
            counts[b] += 1;
        }

        let mut offsets = [0usize; 257];
        let mut current_pos = 0;
        for i in 0..257 {
            offsets[i] = current_pos;
            current_pos += counts[i];
        }

        let mut next_free = offsets;

        for b in 0..257 {
            let end = if b == 256 { data.len() } else { offsets[b + 1] };
            while next_free[b] < end {
                let pos = next_free[b];
                let item_b = if depth < data[pos].len() {
                    data[pos][depth] as usize + 1
                } else {
                    0
                };

                if item_b == b {
                    next_free[b] += 1;
                } else {
                    data.swap(pos, next_free[item_b]);
                    next_free[item_b] += 1;
                }
            }
        }

        // Recursively sort each bucket (skip bucket 0 as it contains strings that have ended)
        for b in 1..257 {
            let start = offsets[b];
            let end = if b == 256 { data.len() } else { offsets[b + 1] };
            if end - start > 1 {
                self.sort_bytes_msd(&mut data[start..end], depth + 1)?;
            }
        }

        Ok(())
    }

    fn estimate_memory_u32(&self, len: usize) -> usize {
        let radix = 1usize
            .checked_shl(self.config.radix_bits.min(16) as u32)
            .unwrap_or(256);
        len * std::mem::size_of::<u32>() + // buffer
        radix * std::mem::size_of::<usize>() // counts
    }

    fn estimate_memory_u64(&self, len: usize) -> usize {
        let radix = 1usize
            .checked_shl(self.config.radix_bits.min(16) as u32)
            .unwrap_or(256);
        len * std::mem::size_of::<u64>() + // buffer
        radix * std::mem::size_of::<usize>() // counts
    }

    /// AVX-512 digit extraction and 4-bank histogram accumulation for the
    /// radix-sort counting pass (`E4(a)`).
    ///
    /// Extracts 16 digits per ZMM register via `_mm512_srlv_epi32` + `_mm512_and_si512`
    /// and accumulates into 4 independent histogram banks when `mask < 256` to break
    /// store-to-load forwarding stalls on repeated digits.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx512f` and that `counts.len() > mask as usize`.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn count_digits_avx512(
        &self,
        data: &[u32],
        shift: usize,
        mask: u32,
        counts: &mut [usize],
    ) {
        let mut i = 0;
        let shift_vec = _mm512_set1_epi32(shift as i32);
        let mask_vec = _mm512_set1_epi32(mask as i32);

        if mask <= 0xFF && counts.len() >= 256 {
            let mut bank1 = [0usize; 256];
            let mut bank2 = [0usize; 256];
            let mut bank3 = [0usize; 256];
            while i + 16 <= data.len() {
                // SAFETY: #[target_feature] ensures avx512f, i+16 <= data.len() ensures 64 bytes available
                let values = unsafe { _mm512_loadu_si512(data[i..].as_ptr() as *const __m512i) };
                let shifted = if shift > 0 {
                    _mm512_srlv_epi32(values, shift_vec)
                } else {
                    values
                };
                let digits = _mm512_and_si512(shifted, mask_vec);
                let mut d = [0u32; 16];
                // SAFETY: d is 16 u32s = 64 bytes, matches __m512i size
                unsafe { _mm512_storeu_si512(d.as_mut_ptr() as *mut __m512i, digits) };
                counts[d[0] as usize] += 1;
                bank1[d[1] as usize] += 1;
                bank2[d[2] as usize] += 1;
                bank3[d[3] as usize] += 1;
                counts[d[4] as usize] += 1;
                bank1[d[5] as usize] += 1;
                bank2[d[6] as usize] += 1;
                bank3[d[7] as usize] += 1;
                counts[d[8] as usize] += 1;
                bank1[d[9] as usize] += 1;
                bank2[d[10] as usize] += 1;
                bank3[d[11] as usize] += 1;
                counts[d[12] as usize] += 1;
                bank1[d[13] as usize] += 1;
                bank2[d[14] as usize] += 1;
                bank3[d[15] as usize] += 1;
                i += 16;
            }
            for b in 0..256 {
                counts[b] += bank1[b] + bank2[b] + bank3[b];
            }
        } else {
            while i + 16 <= data.len() {
                // SAFETY: #[target_feature] ensures avx512f, i+16 <= data.len() ensures 64 bytes available
                let values = unsafe { _mm512_loadu_si512(data[i..].as_ptr() as *const __m512i) };
                let shifted = if shift > 0 {
                    _mm512_srlv_epi32(values, shift_vec)
                } else {
                    values
                };
                let digits = _mm512_and_si512(shifted, mask_vec);
                let mut digit_array = [0u32; 16];
                // SAFETY: digit_array is 16 u32s = 64 bytes, matches __m512i size
                unsafe { _mm512_storeu_si512(digit_array.as_mut_ptr() as *mut __m512i, digits) };
                for digit in digit_array.iter() {
                    counts[*digit as usize] += 1;
                }
                i += 16;
            }
        }

        // Handle remaining elements sequentially
        for &value in &data[i..] {
            let digit = ((value >> shift) & mask) as usize;
            counts[digit] += 1;
        }
    }
}

impl Default for RadixSort {
    fn default() -> Self {
        Self::new()
    }
}

impl Algorithm for RadixSort {
    type Config = RadixSortConfig;
    type Input = Vec<u32>;
    type Output = Vec<u32>;

    fn execute(&self, config: &Self::Config, mut input: Self::Input) -> Result<Self::Output> {
        let mut sorter = Self::with_config(config.clone());
        sorter.sort_u32(&mut input)?;
        Ok(input)
    }

    fn stats(&self) -> AlgorithmStats {
        self.stats.clone()
    }

    fn estimate_memory(&self, input_size: usize) -> usize {
        self.estimate_memory_u32(input_size)
    }

    fn supports_parallel(&self) -> bool {
        true
    }

    fn supports_simd(&self) -> bool {
        cfg!(feature = "simd")
    }
}

struct Low56Hists {
    h0: [usize; 2048],
    h1: [usize; 2048],
    h2: [usize; 2048],
    h3: [usize; 2048],
    h4: [usize; 4096],
}

/// Specialized radix sort for key-value pairs
pub struct KeyValueRadixSort<K, V> {
    _phantom: std::marker::PhantomData<(K, V)>,
}

impl<K, V> KeyValueRadixSort<K, V>
where
    K: Copy + Into<u64>,
    V: Clone,
{
    /// Create a new key-value radix sort
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sort key-value pairs by key (`E3.8`).
    ///
    /// Uses a single scratch buffer of `(K, V)` (`32 B/elem` for `(u64, u64)`, down from
    /// `56 B/elem` in the former 3-allocation index+cycle permutation) and cache-conscious
    /// 8-bit MSB-bucket (`256` DTLB-friendly buckets) + 5-pass L2-resident LSD ping-pong
    /// (`11 + 11 + 11 + 11 + 12 = 56` bits).
    ///
    /// Calls `K::into()` at most once per element before any unsafe pointer moves, so an
    /// impure or panicking `Into<u64>` implementation can never cause out-of-bounds writes
    /// or double-drops (`S47-R2`, `S47-R3`).
    pub fn sort_by_key(&self, data: &mut [(K, V)]) -> Result<()> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        const L2_THRESHOLD: usize = 16_384;
        let can_use_raw_u64 = n > L2_THRESHOLD
            && std::mem::size_of::<K>() == std::mem::size_of::<u64>()
            && std::mem::align_of::<K>() == std::mem::align_of::<u64>()
            && !std::mem::needs_drop::<(K, V)>()
            && std::ptr::fn_addr_eq(
                <K as Into<u64>>::into as fn(K) -> u64,
                <u64 as Into<u64>>::into as fn(u64) -> u64,
            );

        if !can_use_raw_u64 {
            return Self::sort_by_key_materialized(data);
        }

        let mut msb_counts = [0usize; 256];
        let mut prev = 0u64;
        let mut sorted = true;
        let mut or_diff = 0u64;
        let first: u64 = data[0].0.into();
        for (idx, &(k, _)) in data.iter().enumerate() {
            let key: u64 = if idx == 0 { first } else { k.into() };
            // SAFETY: `size_of::<K>() == 8`, `align_of::<K>() == 8`, and `fn_addr_eq` verified
            // that `K::into` is the 64-bit identity conversion.
            let raw_key: u64 = unsafe { std::ptr::read_unaligned(&k as *const K as *const u64) };
            if key != raw_key {
                return Self::sort_by_key_materialized_from(data, idx, key);
            }
            sorted &= key >= prev;
            prev = key;
            or_diff |= key ^ first;
            msb_counts[(key >> 56) as usize] += 1;
        }
        if sorted || or_diff == 0 {
            return Ok(());
        }

        let mut buffer = Vec::<std::mem::MaybeUninit<(K, V)>>::with_capacity(n);
        // SAFETY: `MaybeUninit<(K, V)>` does not require initialization; drops are not run on
        // `MaybeUninit<(K, V)>`, so bitwise moves between `data` and `buffer` never double-drop `V`.
        unsafe {
            buffer.set_len(n);
        }
        let buf_ptr = buffer.as_mut_ptr() as *mut (K, V);
        let data_ptr = data.as_mut_ptr();

        if (or_diff >> 56) == 0 {
            let num_passes = (((63 - or_diff.leading_zeros()) / 8) + 1) as usize;
            // SAFETY: Every element's raw 64-bit key was verified identical to `K::into()`.
            unsafe {
                Self::lsd_kv_in_cache_raw_u64(data, buf_ptr, num_passes);
            }
            return Ok(());
        }

        let mut bucket_starts = [0usize; 256];
        let mut offsets = [0usize; 256];
        let mut pos = 0usize;
        for b in 0..256 {
            bucket_starts[b] = pos;
            offsets[b] = pos;
            pos += msb_counts[b];
        }

        for i in 0..n {
            // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums over
            // immutable raw `u64` keys.
            unsafe {
                let key: u64 =
                    std::ptr::read_unaligned(&(*data_ptr.add(i)).0 as *const K as *const u64);
                let digit = (key >> 56) as usize;
                let out_pos = *offsets.get_unchecked_mut(digit);
                *offsets.get_unchecked_mut(digit) = out_pos + 1;
                std::ptr::copy_nonoverlapping(data_ptr.add(i), buf_ptr.add(out_pos), 1);
            }
        }

        let mut hists = Box::new(Low56Hists {
            h0: [0usize; 2048],
            h1: [0usize; 2048],
            h2: [0usize; 2048],
            h3: [0usize; 2048],
            h4: [0usize; 4096],
        });

        for b in 0..256 {
            let count = msb_counts[b];
            if count == 0 {
                continue;
            }
            let start = bucket_starts[b];
            // SAFETY: `start + count <= n` and `buf_ptr` was initialized with a permutation of `data`.
            unsafe {
                if count == 1 {
                    std::ptr::copy_nonoverlapping(buf_ptr.add(start), data_ptr.add(start), 1);
                } else {
                    Self::lsd_kv_low56_from_buf_to_data(
                        buf_ptr.add(start),
                        data_ptr.add(start),
                        count,
                        &mut hists,
                    );
                }
            }
        }

        Ok(())
    }

    /// Sort `data` by calling `K::into()` exactly once per element upfront before any
    /// unsafe pointer copies.
    fn sort_by_key_materialized(data: &mut [(K, V)]) -> Result<()> {
        let n = data.len();
        let mut keys = Vec::with_capacity(n);
        let mut prev = 0u64;
        let mut sorted = true;
        let mut or_diff = 0u64;
        let mut first = 0u64;
        for (i, &(k, _)) in data.iter().enumerate() {
            let key: u64 = k.into();
            if i == 0 {
                first = key;
            }
            sorted &= key >= prev;
            prev = key;
            or_diff |= key ^ first;
            keys.push(key);
        }
        if sorted || or_diff == 0 {
            return Ok(());
        }
        Self::lsd_kv_with_materialized_keys(data, &mut keys, or_diff);
        Ok(())
    }

    fn sort_by_key_materialized_from(
        data: &mut [(K, V)],
        mismatch_idx: usize,
        mismatch_key: u64,
    ) -> Result<()> {
        let n = data.len();
        let mut keys = Vec::with_capacity(n);
        for &(k, _) in data[..mismatch_idx].iter() {
            // SAFETY: `data[..mismatch_idx]` already verified `k.into() == raw_u64(k)`.
            let raw_key: u64 = unsafe { std::ptr::read_unaligned(&k as *const K as *const u64) };
            keys.push(raw_key);
        }
        keys.push(mismatch_key);
        for &(k, _) in data[mismatch_idx + 1..].iter() {
            keys.push(k.into());
        }
        let mut or_diff = 0u64;
        let first = keys[0];
        for &k in &keys {
            or_diff |= k ^ first;
        }
        if or_diff == 0 {
            return Ok(());
        }
        Self::lsd_kv_with_materialized_keys(data, &mut keys, or_diff);
        Ok(())
    }

    fn lsd_kv_with_materialized_keys(data: &mut [(K, V)], keys: &mut [u64], or_diff: u64) {
        let n = data.len();
        let mut hist = [[0usize; 256]; 8];
        for &key in keys.iter() {
            hist[0][(key & 0xFF) as usize] += 1;
            hist[1][((key >> 8) & 0xFF) as usize] += 1;
            hist[2][((key >> 16) & 0xFF) as usize] += 1;
            hist[3][((key >> 24) & 0xFF) as usize] += 1;
            hist[4][((key >> 32) & 0xFF) as usize] += 1;
            hist[5][((key >> 40) & 0xFF) as usize] += 1;
            hist[6][((key >> 48) & 0xFF) as usize] += 1;
            hist[7][(key >> 56) as usize] += 1;
        }

        let num_passes = (((63 - or_diff.leading_zeros()) / 8) + 1) as usize;
        let mut val_buf = Vec::<std::mem::MaybeUninit<(K, V)>>::with_capacity(n);
        // SAFETY: `MaybeUninit<(K, V)>` does not require initialization.
        unsafe {
            val_buf.set_len(n);
        }
        let mut key_buf = vec![0u64; n];

        let data_ptr = data.as_mut_ptr();
        let val_buf_ptr = val_buf.as_mut_ptr() as *mut (K, V);
        let keys_ptr = keys.as_mut_ptr();
        let key_buf_ptr = key_buf.as_mut_ptr();

        let mut in_data = true;
        for (pass, counts) in hist.iter().enumerate().take(num_passes) {
            if counts.contains(&n) {
                continue;
            }
            let shift = pass * 8;
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += counts[b];
            }
            let (src_val, dst_val, src_key, dst_key) = if in_data {
                (
                    data_ptr as *const (K, V),
                    val_buf_ptr,
                    keys_ptr as *const u64,
                    key_buf_ptr,
                )
            } else {
                (
                    val_buf_ptr as *const (K, V),
                    data_ptr,
                    key_buf_ptr as *const u64,
                    keys_ptr,
                )
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums
                // over the immutable `u64` array `keys`; no user code runs in this loop.
                unsafe {
                    let key = *src_key.add(i);
                    let digit = ((key >> shift) & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    *dst_key.add(out_pos) = key;
                    std::ptr::copy_nonoverlapping(src_val.add(i), dst_val.add(out_pos), 1);
                }
            }
            in_data = !in_data;
        }
        if !in_data {
            // SAFETY: `val_buf_ptr` holds a complete permutation of `data`'s `n` elements.
            unsafe {
                std::ptr::copy_nonoverlapping(val_buf_ptr, data_ptr, n);
            }
        }
    }

    /// # Safety
    ///
    /// `K` must have `size_of::<K>() == 8` and `align_of::<K>() == 8`, and `buf_ptr` must
    /// point to a valid allocation of at least `data.len()` `(K, V)` elements.
    #[inline]
    unsafe fn lsd_kv_in_cache_raw_u64(
        data: &mut [(K, V)],
        buf_ptr: *mut (K, V),
        num_passes: usize,
    ) {
        let n = data.len();
        let mut hist = [[0usize; 256]; 8];
        let mut prev = 0u64;
        let mut sorted = true;
        for &(k, _) in data.iter() {
            // SAFETY: Caller guarantees `K` is layout-identical to `u64`.
            let key: u64 = unsafe { std::ptr::read_unaligned(&k as *const K as *const u64) };
            sorted &= key >= prev;
            prev = key;
            hist[0][(key & 0xFF) as usize] += 1;
            hist[1][((key >> 8) & 0xFF) as usize] += 1;
            hist[2][((key >> 16) & 0xFF) as usize] += 1;
            hist[3][((key >> 24) & 0xFF) as usize] += 1;
            hist[4][((key >> 32) & 0xFF) as usize] += 1;
            hist[5][((key >> 40) & 0xFF) as usize] += 1;
            hist[6][((key >> 48) & 0xFF) as usize] += 1;
            hist[7][(key >> 56) as usize] += 1;
        }
        if sorted {
            return;
        }

        let data_ptr = data.as_mut_ptr();
        let mut in_data = true;
        for (pass, counts) in hist.iter().enumerate().take(num_passes) {
            if counts.contains(&n) {
                continue;
            }
            let shift = pass * 8;
            let mut offsets = [0usize; 256];
            let mut pos = 0usize;
            for b in 0..256 {
                offsets[b] = pos;
                pos += counts[b];
            }
            let (src_ptr, dst_ptr) = if in_data {
                (data_ptr as *const (K, V), buf_ptr)
            } else {
                (buf_ptr as *const (K, V), data_ptr)
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 256`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*src_ptr.add(i)).0 as *const K as *const u64);
                    let digit = ((key >> shift) & 0xFF) as usize;
                    let out_pos = *offsets.get_unchecked_mut(digit);
                    *offsets.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(src_ptr.add(i), dst_ptr.add(out_pos), 1);
                }
            }
            in_data = !in_data;
        }
        if !in_data {
            // SAFETY: `buf_ptr` holds a complete permutation of `data`'s `n` elements.
            unsafe {
                std::ptr::copy_nonoverlapping(buf_ptr, data_ptr, n);
            }
        }
    }

    /// # Safety
    ///
    /// `K` must have `size_of::<K>() == 8` and `align_of::<K>() == 8`, and `buf_ptr` and
    /// `data_ptr` must be valid, non-overlapping allocations of at least `n` `(K, V)`
    /// elements, with `buf_ptr[0..n]` initialized.
    #[inline]
    unsafe fn lsd_kv_low56_from_buf_to_data(
        buf_ptr: *mut (K, V),
        data_ptr: *mut (K, V),
        n: usize,
        hists: &mut Low56Hists,
    ) {
        let Low56Hists { h0, h1, h2, h3, h4 } = hists;
        h0.fill(0);
        h1.fill(0);
        h2.fill(0);
        h3.fill(0);
        h4.fill(0);
        let mut prev = 0u64;
        let mut sorted = true;
        for i in 0..n {
            // SAFETY: `i < n` and `buf_ptr[0..n]` is initialized with layout-verified `u64` keys.
            let key: u64 =
                unsafe { std::ptr::read_unaligned(&(*buf_ptr.add(i)).0 as *const K as *const u64) };
            sorted &= key >= prev;
            prev = key;
            h0[(key & 0x7FF) as usize] += 1;
            h1[((key >> 11) & 0x7FF) as usize] += 1;
            h2[((key >> 22) & 0x7FF) as usize] += 1;
            h3[((key >> 33) & 0x7FF) as usize] += 1;
            h4[((key >> 44) & 0xFFF) as usize] += 1;
        }
        if sorted {
            // SAFETY: `buf_ptr` and `data_ptr` do not overlap and have capacity `n`.
            unsafe {
                std::ptr::copy_nonoverlapping(buf_ptr, data_ptr, n);
            }
            return;
        }

        let mut in_buf = true;

        if !h0.contains(&n) {
            let mut pos = 0usize;
            for count in h0.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*buf_ptr.add(i)).0 as *const K as *const u64);
                    let digit = (key & 0x7FF) as usize;
                    let out_pos = *h0.get_unchecked_mut(digit);
                    *h0.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(buf_ptr.add(i), data_ptr.add(out_pos), 1);
                }
            }
            in_buf = false;
        }

        if !h1.contains(&n) {
            let mut pos = 0usize;
            for count in h1.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf_ptr as *const (K, V), data_ptr)
            } else {
                (data_ptr as *const (K, V), buf_ptr)
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*src_ptr.add(i)).0 as *const K as *const u64);
                    let digit = ((key >> 11) & 0x7FF) as usize;
                    let out_pos = *h1.get_unchecked_mut(digit);
                    *h1.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(src_ptr.add(i), dst_ptr.add(out_pos), 1);
                }
            }
            in_buf = !in_buf;
        }

        if !h2.contains(&n) {
            let mut pos = 0usize;
            for count in h2.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf_ptr as *const (K, V), data_ptr)
            } else {
                (data_ptr as *const (K, V), buf_ptr)
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*src_ptr.add(i)).0 as *const K as *const u64);
                    let digit = ((key >> 22) & 0x7FF) as usize;
                    let out_pos = *h2.get_unchecked_mut(digit);
                    *h2.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(src_ptr.add(i), dst_ptr.add(out_pos), 1);
                }
            }
            in_buf = !in_buf;
        }

        if !h3.contains(&n) {
            let mut pos = 0usize;
            for count in h3.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf_ptr as *const (K, V), data_ptr)
            } else {
                (data_ptr as *const (K, V), buf_ptr)
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 2048`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*src_ptr.add(i)).0 as *const K as *const u64);
                    let digit = ((key >> 33) & 0x7FF) as usize;
                    let out_pos = *h3.get_unchecked_mut(digit);
                    *h3.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(src_ptr.add(i), dst_ptr.add(out_pos), 1);
                }
            }
            in_buf = !in_buf;
        }

        if !h4.contains(&n) {
            let mut pos = 0usize;
            for count in h4.iter_mut() {
                let c = *count;
                *count = pos;
                pos += c;
            }
            let (src_ptr, dst_ptr) = if in_buf {
                (buf_ptr as *const (K, V), data_ptr)
            } else {
                (data_ptr as *const (K, V), buf_ptr)
            };
            for i in 0..n {
                // SAFETY: `i < n`, `digit < 4096`, and `out_pos < n` from exact prefix sums.
                unsafe {
                    let key: u64 =
                        std::ptr::read_unaligned(&(*src_ptr.add(i)).0 as *const K as *const u64);
                    let digit = ((key >> 44) & 0xFFF) as usize;
                    let out_pos = *h4.get_unchecked_mut(digit);
                    *h4.get_unchecked_mut(digit) = out_pos + 1;
                    std::ptr::copy_nonoverlapping(src_ptr.add(i), dst_ptr.add(out_pos), 1);
                }
            }
            in_buf = !in_buf;
        }

        if in_buf {
            // SAFETY: `buf_ptr` holds the current permutation of the `n` bucket elements.
            unsafe {
                std::ptr::copy_nonoverlapping(buf_ptr, data_ptr, n);
            }
        }
    }
}

impl<K, V> Default for KeyValueRadixSort<K, V>
where
    K: Copy + Into<u64>,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}
