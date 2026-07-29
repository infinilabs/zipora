//! Fuzz rANS decode with arbitrary streams (plan.md 6.1, H11).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::entropy::{ParallelX1, Rans64Encoder, RansDecoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut freqs = [0u32; 256];
    for i in 0..4 {
        freqs[data[i] as usize] += 1 + data[i + 4] as u32;
    }
    let output_length = u16::from_le_bytes([data[6], data[7]]) as usize % 4096;
    let payload = &data[8..];
    if let Ok(encoder) = Rans64Encoder::<ParallelX1>::new(&freqs) {
        let decoder = RansDecoder::new(&encoder);
        let _ = decoder.decode(payload, output_length);
    }
});
