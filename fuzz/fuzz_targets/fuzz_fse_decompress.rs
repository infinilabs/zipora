//! Fuzz FseDecoder::decompress with fully arbitrary bytes (plan.md 6.1, H11/C4).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::entropy::FseDecoder;

fuzz_target!(|data: &[u8]| {
    let mut decoder = FseDecoder::new();
    let _ = decoder.decompress(data);
});
