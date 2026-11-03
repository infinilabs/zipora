//! Fuzz rANS decode with arbitrary streams (plan.md 6.1, H11; plan v2 C4).
//!
//! Header (seed-compatible with `fuzz/corpus/fuzz_rans_decode/seed_0001.bin`):
//! - `data[0..4]` symbols and `data[4..8]` weights of the frequency table;
//! - low 12 bits of `u16::from_le_bytes([data[6], data[7]])`: small output length;
//! - `data[7]` bits 4-5: stream variant (x1, x2, x4, x8);
//! - `data[7]` bits 6-7: caller-supplied huge `output_length` mode, which the
//!   decoders must reject or bound instead of preallocating (C4).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::entropy::rans::ParallelVariant;
use zipora::entropy::{ParallelX1, ParallelX2, ParallelX4, ParallelX8, Rans64Encoder, RansDecoder};

fn decode_with<P: ParallelVariant>(freqs: &[u32; 256], payload: &[u8], output_length: usize) {
    if let Ok(encoder) = Rans64Encoder::<P>::new(freqs) {
        let decoder = RansDecoder::new(&encoder);
        let _ = decoder.decode(payload, output_length);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut freqs = [0u32; 256];
    for i in 0..4 {
        freqs[data[i] as usize] += 1 + data[i + 4] as u32;
    }
    let small_len = u16::from_le_bytes([data[6], data[7]]) as usize % 4096;
    let huge_mode = data[7] >> 6;
    let output_length = match huge_mode {
        0 => small_len,
        // Above the fuzzer's 1 GiB malloc limit: must never be reserved up front.
        1 => (1usize << 31) + small_len,
        // Not representable by `Vec` at all ("capacity overflow").
        2 => (isize::MAX as usize).wrapping_add(1 + small_len),
        _ => usize::MAX - small_len,
    };
    // A single-symbol alphabet codes every symbol in zero bits, so any length
    // is a valid decode of it; keep the huge modes on multi-symbol tables so
    // each iteration stays bounded by the payload size.
    if huge_mode != 0 && freqs.iter().filter(|&&f| f > 0).count() < 2 {
        return;
    }
    let payload = &data[8..];
    match (data[7] >> 4) & 3 {
        0 => decode_with::<ParallelX1>(&freqs, payload, output_length),
        1 => decode_with::<ParallelX2>(&freqs, payload, output_length),
        2 => decode_with::<ParallelX4>(&freqs, payload, output_length),
        _ => decode_with::<ParallelX8>(&freqs, payload, output_length),
    }
});
