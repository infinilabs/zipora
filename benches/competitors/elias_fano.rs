//! Family 2: Elias-Fano Posting Lists (100,000 elements, 100,000 distinct queries).
//!
//! Zipora: `EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`
//! Competitors: `sucds::mii_sequences::EliasFano`, `vers_vecs::EliasFanoVec`
//!
//! Fairness invariant (`S42-R1`): 1 instance per implementation, 1 shared query slice
//! (`targets`, `cursor_streams`, `indices`), zero multi-clone rings or heap spacers.

use std::collections::BTreeMap;
use std::hint::black_box;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sucds::Serializable as SucdsSerializable;
use sucds::mii_sequences::EliasFanoBuilder as SucdsEliasFanoBuilder;
use vers_vecs::EliasFanoVec as VersEliasFanoVec;
use zipora::succinct::elias_fano::{
    ClusteredEliasFano, EliasFano, HybridPostingList, OptimalPartitionedEliasFano,
    PartitionedEliasFano,
};

use crate::common::*;

pub fn bench_elias_fano(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.elias_fano_elements;
    let num_queries = n;
    let mut params = BenchParams::medium(n as u64, num_queries as u64);
    params.warmup_calls = 2;
    params.trials = 7;
    params.bursts_per_trial = 3;

    let uniform = gen_uniform_sparse(n, 50, SEED_BASE ^ 0x02);
    let clustered = gen_zipf_clustered(n, SEED_BASE ^ 0x03);
    let dense: Vec<u32> = (0..n as u32).collect();

    for (dist_name, data) in [
        ("uniform", &uniform),
        ("zipf_clustered", &clustered),
        ("dense", &dense),
    ] {
        if !should_run_prefix(&format!("elias_fano/{dist_name}/")) {
            continue;
        }
        trim_allocator_arena();
        let ef = EliasFano::from_sorted(data);
        let pef = PartitionedEliasFano::from_sorted(data);
        let opef = OptimalPartitionedEliasFano::from_sorted(data);
        let cef = ClusteredEliasFano::from_sorted(data);
        let hybrid = HybridPostingList::from_sorted(data);

        let universe = *data.last().unwrap() as u64 + 1;
        let data_u64: Vec<u64> = data.iter().map(|&v| v as u64).collect();
        let mut sucds_builder = SucdsEliasFanoBuilder::new(universe, n).unwrap();
        sucds_builder.extend(data_u64.iter().copied()).unwrap();
        let sucds_ef = sucds_builder.build().enable_rank();

        let vers_ef = VersEliasFanoVec::from_slice(&data_u64);

        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/ef_bits_per_elem"),
            ef.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/pef_bits_per_elem"),
            pef.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/opef_bits_per_elem"),
            opef.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/cef_bits_per_elem"),
            cef.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/hybrid_bits_per_elem"),
            hybrid.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/sucds_ef_bits_per_elem"),
            (sucds_ef.size_in_bytes() * 8) as f64 / n as f64,
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/vers_ef_bits_per_elem"),
            (vers_ef.heap_size() * 8) as f64 / n as f64,
        );

        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x04);
        let targets: Vec<u64> = (0..num_queries)
            .map(|_| rng.random_range(0..universe))
            .collect();

        let stream_len = 1_000.min(num_queries.max(1));
        let num_streams = (num_queries / stream_len).max(1);
        let cursor_streams: Vec<Vec<u64>> = (0..num_streams)
            .map(|_| {
                let mut stream: Vec<u64> = (0..stream_len)
                    .map(|_| rng.random_range(0..universe))
                    .collect();
                stream.sort_unstable();
                stream
            })
            .collect();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/ef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = ef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = pef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = opef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/cef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = cef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut cursor_params = BenchParams::medium(n as u64, (num_streams * stream_len) as u64);
        cursor_params.warmup_calls = 2;
        cursor_params.trials = 7;
        cursor_params.bursts_per_trial = 3;
        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_cursor_advance_to_geq"),
            "ns/op",
            cursor_params,
            || {
                let mut acc = 0u64;
                for stream in &cursor_streams {
                    let mut cursor = pef.cursor();
                    for &t in stream {
                        if cursor.advance_to_geq(black_box(t))
                            && let Some(v) = cursor.current()
                        {
                            acc = acc.wrapping_add(v);
                        }
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_cursor_advance_to_index"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                let mut cursor = opef.cursor();
                for &idx in &indices {
                    if cursor.advance_to_index(black_box(idx))
                        && let Some(v) = cursor.current()
                    {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/ef_select"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = ef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/cef_select"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = cef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        // Competitor: sucds::mii_sequences::EliasFano (1 instance)
        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/sucds_ef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some(v) = sucds_ef.successor(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/sucds_ef_select"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = sucds_ef.select(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        // Competitor: vers_vecs::EliasFanoVec (1 instance)
        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/vers_ef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some(v) = vers_ef.successor(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/vers_ef_select"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = vers_ef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );
    }
}
