#!/usr/bin/env python3
"""Reproducible loop-only profiler for E3 gap benchmarks (S42-R4).

Uses Linux `perf_event_open(2)` (`PERF_TYPE_SOFTWARE`, since hardware PMU events
return `ENOENT` inside this cloud guest VM and the `perf` CLI binary is not
installed) combined with `prctl(PR_TASK_PERF_EVENTS_ENABLE / DISABLE)` inside
`benches/common/harness.rs` (`ZIPORA_PROFILE_LOOP=1`) so counters and 4000 Hz
task-clock IP samples are collected **strictly inside the measured `f(state)`
loop** (excluding process startup, dataset generation, structure construction,
warmup, per-burst `setup(state)`, and `_fini` teardown).

Usage:
  # Profile all 10 E3.1-E3.10 representative gap benchmarks:
  python3 scripts/profile_e3_gaps.py --all-e3 --out-json target/bench/e3_gap_profiles.json

  # Profile a single benchmark by name:
  python3 scripts/profile_e3_gaps.py --cat rank_select --filter rank_select/interleaved256/rank1
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import mmap
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

# Linux x86_64 syscall and perf_event.h constants (/usr/include/linux/perf_event.h)
SYS_PERF_EVENT_OPEN = 298
PERF_TYPE_SOFTWARE = 1

PERF_COUNT_SW_CPU_CLOCK = 0
PERF_COUNT_SW_TASK_CLOCK = 1
PERF_COUNT_SW_PAGE_FAULTS = 2
PERF_COUNT_SW_CONTEXT_SWITCHES = 3
PERF_COUNT_SW_CPU_MIGRATIONS = 4
PERF_COUNT_SW_PAGE_FAULTS_MIN = 5
PERF_COUNT_SW_PAGE_FAULTS_MAJ = 6

PERF_SAMPLE_IP = 1 << 0
PERF_SAMPLE_TID = 1 << 1

PERF_RECORD_MMAP = 1
PERF_RECORD_SAMPLE = 9

# Bitflags in perf_event_attr.flags (little-endian u64 bitfields):
# bit 0: disabled
# bit 1: inherit
# bit 5: exclude_kernel (required when kernel.perf_event_paranoid >= 2)
# bit 6: exclude_hv
# bit 10: mmap (emit PERF_RECORD_MMAP for address-to-symbol resolution)
# bit 11: freq (sample_period is sample_freq in Hz)
ATTR_FLAG_DISABLED = 1 << 0
ATTR_FLAG_INHERIT = 1 << 1
ATTR_FLAG_EXCLUDE_KERNEL = 1 << 5
ATTR_FLAG_EXCLUDE_HV = 1 << 6
ATTR_FLAG_MMAP = 1 << 10
ATTR_FLAG_FREQ = 1 << 11

E3_GAP_TARGETS: list[tuple[str, str, str, str]] = [
    ("E3.1", "rank_select", "rank_select/interleaved256/rank1", "Rank/Select (`RankSelectInterleaved256::rank1`, 1 Gbit)"),
    ("E3.2", "elias_fano", "elias_fano/uniform/ef_select", "Elias-Fano (`EliasFano::get`, 100k uniform)"),
    ("E3.3", "tries", "tries/dat/url_paths/lookup", "Tries (`DoubleArrayTrie::get`, 1M `url_paths`)"),
    ("E3.4", "hash_maps", "hash_maps/zipora_hash_map/get_u64", "Hash Maps (`ZiporaHashMap::get`, 1M `u64`)"),
    ("E3.5", "memory_pools", "memory_pools/lockfree_pool/alloc_free_1t", "Memory Pools (`LockFreeMemoryPool` 256B `alloc`+`free` 1T)"),
    ("E3.6", "entropy", "entropy/huffman_o0/decode_64kib", "Entropy Coding (`HuffmanDecoder::decode` O0, 64 KiB)"),
    ("E3.7", "int_codecs", "int_codecs/stream_vbyte/encode_deltas", "Integer Codecs (`StreamVByte::encode_deltas`, 100k `u32`)"),
    ("E3.8", "radix_sort", "radix_sort/lsd_u64", "Radix Sort (`RadixSort::sort_u64`, 10M `u64`)"),
    ("E3.9", "suffix_array", "suffix_array/construct", "Suffix Array (`SuffixArray::new`, 10M bytes)"),
    ("E3.10", "search_and_scoring", "search_and_scoring/simd_strchr_64kib", "String Search (`SimdStringSearch::sse42_strchr`, 64 KiB)"),
]


class PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period_or_freq", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("__reserved_2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("__reserved_3", ctypes.c_uint32),
        ("sig_data", ctypes.c_uint64),
        ("config3", ctypes.c_uint64),
    ]


libc = ctypes.CDLL("libc.so.6", use_errno=True)


def perf_event_open(attr: PerfEventAttr, pid: int, cpu: int, group_fd: int = -1, flags: int = 0) -> int:
    fd = libc.syscall(
        ctypes.c_long(SYS_PERF_EVENT_OPEN),
        ctypes.byref(attr),
        ctypes.c_int(pid),
        ctypes.c_int(cpu),
        ctypes.c_int(group_fd),
        ctypes.c_ulong(flags),
    )
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def load_elf_symbols(binary_path: str) -> tuple[list[int], list[tuple[int, str]]]:
    proc = subprocess.run(
        ["nm", "-nC", "--defined-only", binary_path],
        capture_output=True,
        text=True,
        check=True,
    )
    addrs: list[int] = []
    entries: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        addr_hex, sym_type, name = parts
        if sym_type not in ("T", "t", "W", "w"):
            continue
        try:
            addr = int(addr_hex, 16)
        except ValueError:
            continue
        addrs.append(addr)
        entries.append((addr, name))
    return addrs, entries


def parse_proc_maps(maps_text: str) -> list[tuple[int, int, int, str]]:
    """Return list of (start, end, elf_base, filename) for executable mappings."""
    file_bases: dict[str, int] = {}
    regions: list[tuple[int, int, int, str]] = []
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        addr_range, perms, offset_hex, _dev, _inode, pathname = parts
        pathname = pathname.strip()
        if not pathname or pathname.startswith("["):
            continue
        start_s, end_s = addr_range.split("-")
        start = int(start_s, 16)
        end = int(end_s, 16)
        pgoff = int(offset_hex, 16)
        if pathname not in file_bases and pgoff == 0:
            file_bases[pathname] = start
        if "x" in perms:
            base = file_bases.get(pathname, start - pgoff)
            regions.append((start, end, base, pathname))
    return regions


def disassemble_hot_ips(
    binary_path: str,
    rel_ip_counts: dict[int, int],
    total_samples: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel_ip, cnt in sorted(rel_ip_counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        proc = subprocess.run(
            [
                "objdump",
                "-d",
                "-C",
                "--no-show-raw-insn",
                f"--start-address=0x{rel_ip:x}",
                f"--stop-address=0x{rel_ip + 16:x}",
                binary_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        insn = ""
        for line in proc.stdout.splitlines():
            s = line.strip()
            if s.startswith(f"{rel_ip:x}:"):
                insn = s.split(":", 1)[1].strip()
                break
        out.append(
            {
                "rel_ip": f"0x{rel_ip:x}",
                "samples": cnt,
                "pct": round((cnt * 100.0) / max(total_samples, 1), 2),
                "instruction": insn,
            }
        )
    return out


def resolve_symbol(
    ip: int,
    mmap_regions: list[tuple[int, int, int, str]],
    bin_path: str,
    sym_addrs: list[int],
    sym_entries: list[tuple[int, str]],
) -> tuple[str, int | None]:
    bin_real = os.path.realpath(bin_path)
    for start, end, elf_base, filename in mmap_regions:
        if start <= ip < end:
            rel = ip - elf_base
            if os.path.realpath(filename) == bin_real or os.path.basename(filename) == os.path.basename(bin_real):
                idx = bisect.bisect_right(sym_addrs, rel) - 1
                if idx >= 0:
                    sym_addr, sym_name = sym_entries[idx]
                    if rel - sym_addr < 0x40000:
                        return sym_name, rel
                return f"{os.path.basename(filename)}+0x{rel:x}", rel
            return f"{os.path.basename(filename)}+0x{rel:x}", None
    return f"0x{ip:x}", None


def parse_perf_ring_buffer(
    mm: mmap.mmap,
    data_pages: int,
) -> tuple[list[tuple[int, int, int, str]], list[int]]:
    page_size = mmap.PAGESIZE
    data_size = data_pages * page_size
    data_offset = page_size
    # Read data_head from perf_event_mmap_page (offset 1024: __u64 data_head)
    data_head = struct.unpack_from("<Q", mm, 1024)[0]
    read_len = min(data_head, data_size)
    raw = bytearray(read_len)
    for i in range(read_len):
        raw[i] = mm[data_offset + (i % data_size)]

    mmap_regions: list[tuple[int, int, int, str]] = []
    samples: list[int] = []
    pos = 0
    while pos + 8 <= len(raw):
        rec_type, _misc, rec_size = struct.unpack_from("<IHH", raw, pos)
        if rec_size < 8 or pos + rec_size > len(raw):
            break
        body = raw[pos + 8 : pos + rec_size]
        if rec_type == PERF_RECORD_MMAP and len(body) >= 32:
            _pid, _tid, addr, length, pgoff = struct.unpack_from("<IIQQQ", body, 0)
            raw_fn = body[32:]
            nul = raw_fn.find(0)
            fn = raw_fn[:nul].decode("utf-8", errors="replace") if nul >= 0 else ""
            mmap_regions.append((addr, addr + length, pgoff, fn))
        elif rec_type == PERF_RECORD_SAMPLE and len(body) >= 8:
            (ip,) = struct.unpack_from("<Q", body, 0)
            samples.append(ip)
        pos += rec_size
    return mmap_regions, samples


def find_competitor_bench_binary(baseline_json: Path) -> str:
    deps = Path("target/release/deps")
    candidates = [
        p for p in deps.glob("competitor_bench-*")
        if p.is_file() and os.access(p, os.X_OK) and "." not in p.name
    ]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())
    if baseline_json.exists():
        data = json.loads(baseline_json.read_text())
        bp = data.get("host_metadata", {}).get("binary_path", "")
        if bp and os.path.isfile(bp) and os.access(bp, os.X_OK):
            return bp
    raise FileNotFoundError("Compiled competitor_bench binary not found; run `cargo bench --bench competitor_bench --no-run` first.")


def profile_single_benchmark(
    binary_path: str,
    cat: str,
    bench_name: str,
    loop_ms: int,
    sym_addrs: list[int],
    sym_entries: list[tuple[int, str]],
) -> dict[str, Any]:
    counter_specs = [
        ("task-clock", PERF_COUNT_SW_TASK_CLOCK),
        ("cpu-clock", PERF_COUNT_SW_CPU_CLOCK),
        ("page-faults", PERF_COUNT_SW_PAGE_FAULTS),
        ("minor-faults", PERF_COUNT_SW_PAGE_FAULTS_MIN),
        ("major-faults", PERF_COUNT_SW_PAGE_FAULTS_MAJ),
        ("context-switches", PERF_COUNT_SW_CONTEXT_SWITCHES),
        ("cpu-migrations", PERF_COUNT_SW_CPU_MIGRATIONS),
    ]

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        fds: list[int] = []
        for _name, cfg in counter_specs:
            attr = PerfEventAttr()
            attr.type = PERF_TYPE_SOFTWARE
            attr.size = ctypes.sizeof(PerfEventAttr)
            attr.config = cfg
            attr.flags = (
                ATTR_FLAG_DISABLED
                | ATTR_FLAG_INHERIT
                | ATTR_FLAG_EXCLUDE_KERNEL
                | ATTR_FLAG_EXCLUDE_HV
            )
            fd = perf_event_open(attr, 0, -1)
            os.set_inheritable(fd, True)
            fds.append(fd)

        sample_attr = PerfEventAttr()
        sample_attr.type = PERF_TYPE_SOFTWARE
        sample_attr.size = ctypes.sizeof(PerfEventAttr)
        sample_attr.config = PERF_COUNT_SW_TASK_CLOCK
        sample_attr.sample_period_or_freq = 4000
        sample_attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID
        sample_attr.flags = (
            ATTR_FLAG_DISABLED
            | ATTR_FLAG_EXCLUDE_KERNEL
            | ATTR_FLAG_EXCLUDE_HV
            | ATTR_FLAG_MMAP
            | ATTR_FLAG_FREQ
        )
        s_fd = perf_event_open(sample_attr, 0, -1)
        os.set_inheritable(s_fd, True)
        fds.append(s_fd)

        socket.send_fds(child_sock, [b"1"], fds)
        # Wait for parent to mmap the sample ring buffer before execve
        child_sock.recv(1)
        child_sock.close()

        env = os.environ.copy()
        env["ZIPORA_BENCH_CAT"] = cat
        env["ZIPORA_PROFILE_LOOP"] = "1"
        env["ZIPORA_PROFILE_LOOP_MS"] = str(loop_ms)
        env["ZIPORA_BENCH_OUT"] = f"target/bench/profile_tmp_{os.getpid()}.json"
        env["ZIPORA_PROFILE_MAPS_OUT"] = f"target/bench/profile_maps_{os.getpid()}.txt"
        os.execve(
            binary_path,
            [binary_path, "--filter", bench_name, "--profile-loop"],
            env,
        )
        os._exit(127)

    child_sock.close()
    _msg, recv_fds, _flags, _addr = socket.recv_fds(parent_sock, 1, len(counter_specs) + 1)
    counter_fds = {
        name: recv_fds[i] for i, (name, _cfg) in enumerate(counter_specs)
    }
    sample_fd = recv_fds[-1]
    data_pages = 64
    mm = mmap.mmap(
        sample_fd,
        (1 + data_pages) * mmap.PAGESIZE,
        mmap.MAP_SHARED,
        mmap.PROT_READ | mmap.PROT_WRITE,
    )
    parent_sock.sendall(b"1")
    parent_sock.close()

    _, status = os.waitpid(pid, 0)
    tmp_json = Path(f"target/bench/profile_tmp_{pid}.json")
    if tmp_json.exists():
        tmp_json.unlink()
    maps_file = Path(f"target/bench/profile_maps_{pid}.txt")
    proc_maps_regions: list[tuple[int, int, int, str]] = []
    if maps_file.exists():
        proc_maps_regions = parse_proc_maps(maps_file.read_text())
        maps_file.unlink()

    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError(f"Profiled child for {bench_name} exited with status {status}")

    counters: dict[str, int] = {}
    for name, fd in counter_fds.items():
        raw_val = os.read(fd, 8)
        counters[name] = struct.unpack("<Q", raw_val)[0]
        os.close(fd)

    mmap_regions, samples = parse_perf_ring_buffer(mm, data_pages)
    if proc_maps_regions:
        mmap_regions = proc_maps_regions
    mm.close()
    os.close(sample_fd)

    sym_counts: dict[str, int] = {}
    rel_ip_counts: dict[int, int] = {}
    for ip in samples:
        sym, rel = resolve_symbol(ip, mmap_regions, binary_path, sym_addrs, sym_entries)
        sym_counts[sym] = sym_counts.get(sym, 0) + 1
        if rel is not None:
            rel_ip_counts[rel] = rel_ip_counts.get(rel, 0) + 1

    total_samples = len(samples)
    top_symbols = [
        {
            "symbol": sym,
            "samples": cnt,
            "pct": round((cnt * 100.0) / max(total_samples, 1), 2),
        }
        for sym, cnt in sorted(sym_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    ]
    top_instructions = disassemble_hot_ips(binary_path, rel_ip_counts, total_samples, limit=5)

    return {
        "benchmark": bench_name,
        "category": cat,
        "loop_ms": loop_ms,
        "sampler": "perf_event_open(PERF_TYPE_SOFTWARE / PERF_COUNT_SW_TASK_CLOCK @ 4000 Hz, gated via prctl(PR_TASK_PERF_EVENTS_ENABLE/DISABLE) around f(state))",
        "counters": counters,
        "total_samples": total_samples,
        "top_symbols": top_symbols,
        "top_instructions": top_instructions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile E3 gap benchmarks strictly inside f(state).")
    parser.add_argument("--binary", default="", help="Path to compiled competitor_bench binary")
    parser.add_argument("--baseline", default="benches/baselines/bindiego/competitor_bench.json")
    parser.add_argument("--cat", default="", help="Category for single-benchmark mode")
    parser.add_argument("--filter", default="", help="Benchmark name for single-benchmark mode")
    parser.add_argument("--loop-ms", type=int, default=400, help="Target loop duration in ms inside f(state)")
    parser.add_argument("--all-e3", action="store_true", help="Profile all 10 E3.1-E3.10 gap benchmarks")
    parser.add_argument("--out-json", default="benches/baselines/bindiego/e3_gap_profiles.json")
    args = parser.parse_args()

    binary_path = args.binary or find_competitor_bench_binary(Path(args.baseline))
    sym_addrs, sym_entries = load_elf_symbols(binary_path)

    targets: list[tuple[str, str, str, str]] = []
    if args.all_e3:
        targets = E3_GAP_TARGETS
    elif args.filter:
        cat = args.cat or args.filter.split("/")[0]
        targets = [("CUSTOM", cat, args.filter, args.filter)]
    else:
        parser.error("Specify either --all-e3 or --filter <bench_name>")

    results: list[dict[str, Any]] = []
    for section_id, cat, bench_name, title in targets:
        prof = profile_single_benchmark(binary_path, cat, bench_name, args.loop_ms, sym_addrs, sym_entries)
        prof["section_id"] = section_id
        prof["title"] = title
        results.append(prof)
        c = prof["counters"]
        print(
            f"[{section_id}] {bench_name}: task-clock={c['task-clock']/1e6:.2f}ms "
            f"page-faults={c['page-faults']} (min={c['minor-faults']}, maj={c['major-faults']}) "
            f"ctx-sw={c['context-switches']} cpu-mig={c['cpu-migrations']} samples={prof['total_symbols_samples'] if 'total_symbols_samples' in prof else prof['total_samples']}"
        )
        for row in prof["top_symbols"][:5]:
            print(f"    sym: {row['pct']:>5.1f}%  ({row['samples']:>4d})  {row['symbol']}")
        for row in prof["top_instructions"][:3]:
            print(f"    ip:  {row['pct']:>5.1f}%  ({row['samples']:>4d})  {row['rel_ip']}  {row['instruction']}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"binary_path": binary_path, "profiles": results}, indent=2) + "\n")
    print(f"Wrote {len(results)} loop-only gap profiles to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
