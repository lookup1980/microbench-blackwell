#!/usr/bin/env python3
"""
Sweep RTX 5090 / sm_120 sparse tensor-core MMA instructions and dense peers.

Each CSV row is a dense-vs-sparse pair at the same semantic M/N/K.  If the
hardware sparse instruction has twice the semantic K of the dense base
instruction, the dense side emits two dense mma.sync instructions per packet.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


M = 16
N = 8


@dataclass(frozen=True)
class Config:
    mode: str
    family: str
    a_type: str
    b_type: str
    k: int
    ordered_metadata: int
    satfinite: int
    iterations: int
    unroll: int
    blocks: int
    warps_per_block: int

    @property
    def sparse_variant(self) -> str:
        return "ordered" if self.ordered_metadata else "sp"

    @property
    def pair_id(self) -> str:
        sat = "sat" if self.satfinite else "nosat"
        return (
            f"{self.mode}:{self.family}:{self.a_type}x{self.b_type}:"
            f"k{self.k}:{self.sparse_variant}:{sat}"
        )


@dataclass
class Result:
    build_status: str
    run_status: str = "not_run"
    stderr_tail: str = ""
    cycles: int | None = None
    cycles_per_packet: float | None = None
    math_ops_per_packet: float | None = None
    ops_per_cycle: float | None = None
    dense_base_k: int | None = None
    dense_instructions_per_packet: int | None = None
    sparse_instructions_per_packet: int | None = None
    warps: int | None = None


def all_sparse_shapes() -> list[tuple[str, str, str, list[int], list[int]]]:
    shapes: list[tuple[str, str, str, list[int], list[int]]] = [
        ("f16", "f16", "f16", [16, 32], [0]),
        ("f16f16", "f16", "f16", [16, 32], [0]),
        ("bf16", "bf16", "bf16", [16, 32], [0]),
        ("tf32", "tf32", "tf32", [8, 16], [0]),
    ]

    for a in ("e4m3", "e5m2"):
        for b in ("e4m3", "e5m2"):
            shapes.append(("fp8", a, b, [64], [0]))

    for a in ("u8", "s8"):
        for b in ("u8", "s8"):
            shapes.append(("int8", a, b, [32, 64], [0, 1]))

    for a in ("u4", "s4"):
        for b in ("u4", "s4"):
            shapes.append(("int4", a, b, [64, 128], [0, 1]))

    return shapes


def smoke_shapes() -> list[tuple[str, str, str, list[int], list[int]]]:
    return [
        ("f16", "f16", "f16", [16, 32], [0]),
        ("f16f16", "f16", "f16", [16], [0]),
        ("bf16", "bf16", "bf16", [16], [0]),
        ("tf32", "tf32", "tf32", [8], [0]),
        ("fp8", "e4m3", "e5m2", [64], [0]),
        ("int8", "s8", "u8", [32, 64], [0, 1]),
        ("int4", "s4", "u4", [64, 128], [0, 1]),
    ]


def make_configs(args: argparse.Namespace) -> list[Config]:
    shape_specs = smoke_shapes() if args.preset == "smoke" else all_sparse_shapes()
    if args.family:
        shape_specs = [s for s in shape_specs if s[0] in args.family]

    modes = ["throughput", "latency"] if args.mode == "both" else [args.mode]
    ordered_values = (
        [0, 1]
        if args.sparse_variant == "both"
        else [1 if args.sparse_variant == "ordered" else 0]
    )

    configs: list[Config] = []
    for mode in modes:
        for family, a_type, b_type, ks, sats in shape_specs:
            for k in ks:
                for sat in sats:
                    for ordered in ordered_values:
                        configs.append(
                            Config(
                                mode=mode,
                                family=family,
                                a_type=a_type,
                                b_type=b_type,
                                k=k,
                                ordered_metadata=ordered,
                                satfinite=sat,
                                iterations=args.iterations_latency
                                if mode == "latency"
                                else args.iterations,
                                unroll=1 if mode == "latency" else args.unroll,
                                blocks=args.blocks,
                                warps_per_block=args.warps_per_block,
                            )
                        )

    if args.max_configs:
        configs = configs[: args.max_configs]
    return configs


def make_vars(config: Config, sparse: int) -> list[str]:
    return [
        f"MODE={config.mode}",
        f"FAMILY={config.family}",
        f"A_TYPE={config.a_type}",
        f"B_TYPE={config.b_type}",
        f"SPARSITY={sparse}",
        f"ORDERED_METADATA={config.ordered_metadata if sparse else 0}",
        f"SATFINITE={config.satfinite}",
        f"MMA_K={config.k}",
        f"ITERATIONS={config.iterations}",
        f"UNROLL={config.unroll}",
        f"BLOCKS={config.blocks}",
        f"WARPS_PER_BLOCK={config.warps_per_block}",
    ]


def run_cmd(cmd: list[str], verbose: bool) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, text=True, capture_output=True)


def tail(text: str, lines: int = 6) -> str:
    return " | ".join(text.strip().splitlines()[-lines:])


def compact_notes(text: str) -> str:
    if "Advisory: Modifier '.sp::ordered_metadata'" in text:
        return "ptxas advisory: .sp::ordered_metadata recommended for sparse performance"
    return tail(text)


def parse_result(stdout: str) -> Result | None:
    for line in stdout.splitlines():
        if not line.startswith("RESULT,"):
            continue
        parts = line.split(",")
        if len(parts) < 18:
            return None
        return Result(
            build_status="ok",
            run_status="ok",
            cycles=int(parts[14]),
            cycles_per_packet=float(parts[15]),
            math_ops_per_packet=float(parts[16]),
            ops_per_cycle=float(parts[17]),
            dense_base_k=int(parts[8]),
            dense_instructions_per_packet=int(parts[9]),
            sparse_instructions_per_packet=int(parts[10]),
            warps=int(parts[13]),
        )
    return None


def build_and_run(config: Config, sparse: int, args: argparse.Namespace) -> Result:
    run_cmd(["make", "clean"], args.verbose)
    build = run_cmd(["make", "sm120_mma_sparse.out", *make_vars(config, sparse)], args.verbose)
    if build.returncode != 0:
        return Result(
            build_status="failed",
            run_status="not_run",
            stderr_tail=compact_notes(build.stderr),
        )

    if args.build_only:
        return Result(build_status="ok", run_status="not_run", stderr_tail=compact_notes(build.stderr))

    result = run_cmd(["./sm120_mma_sparse.out"], args.verbose)
    if result.returncode != 0:
        return Result(
            build_status="ok",
            run_status="failed",
            stderr_tail=compact_notes(result.stderr),
        )

    parsed = parse_result(result.stdout)
    if parsed is None:
        return Result(
            build_status="ok",
            run_status="parse_failed",
            stderr_tail=compact_notes(result.stdout + "\n" + result.stderr),
        )
    parsed.stderr_tail = compact_notes(build.stderr)
    return parsed


CSV_FIELDS = [
    "PairID",
    "Mode",
    "Family",
    "AType",
    "BType",
    "SparseVariant",
    "Satfinite",
    "M",
    "N",
    "SemanticK",
    "DenseBaseK",
    "DenseInstructionsPerPacket",
    "SparseInstructionsPerPacket",
    "Iterations",
    "Unroll",
    "Warps",
    "DenseBuildStatus",
    "DenseRunStatus",
    "SparseBuildStatus",
    "SparseRunStatus",
    "DenseCycles",
    "DenseCyclesPerPacket",
    "DenseOpsPerCycle",
    "SparseCycles",
    "SparseCyclesPerPacket",
    "SparseOpsPerCycle",
    "SparseOverDenseOpsPerCycle",
    "SparseOverDenseCyclesPerPacket",
    "DenseNotes",
    "SparseNotes",
]


def fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def row_from_results(config: Config, dense: Result, sparse: Result) -> dict[str, object]:
    speedup = None
    cycle_ratio = None
    if dense.ops_per_cycle and sparse.ops_per_cycle:
        speedup = sparse.ops_per_cycle / dense.ops_per_cycle
    if dense.cycles_per_packet and sparse.cycles_per_packet:
        cycle_ratio = sparse.cycles_per_packet / dense.cycles_per_packet

    return {
        "PairID": config.pair_id,
        "Mode": config.mode,
        "Family": config.family,
        "AType": config.a_type,
        "BType": config.b_type,
        "SparseVariant": config.sparse_variant,
        "Satfinite": "satfinite" if config.satfinite else "nosat",
        "M": M,
        "N": N,
        "SemanticK": config.k,
        "DenseBaseK": fmt(dense.dense_base_k),
        "DenseInstructionsPerPacket": fmt(dense.dense_instructions_per_packet),
        "SparseInstructionsPerPacket": fmt(sparse.sparse_instructions_per_packet),
        "Iterations": config.iterations,
        "Unroll": config.unroll,
        "Warps": fmt(sparse.warps if sparse.warps is not None else dense.warps),
        "DenseBuildStatus": dense.build_status,
        "DenseRunStatus": dense.run_status,
        "SparseBuildStatus": sparse.build_status,
        "SparseRunStatus": sparse.run_status,
        "DenseCycles": fmt(dense.cycles),
        "DenseCyclesPerPacket": fmt(dense.cycles_per_packet),
        "DenseOpsPerCycle": fmt(dense.ops_per_cycle),
        "SparseCycles": fmt(sparse.cycles),
        "SparseCyclesPerPacket": fmt(sparse.cycles_per_packet),
        "SparseOpsPerCycle": fmt(sparse.ops_per_cycle),
        "SparseOverDenseOpsPerCycle": fmt(speedup),
        "SparseOverDenseCyclesPerPacket": fmt(cycle_ratio),
        "DenseNotes": dense.stderr_tail,
        "SparseNotes": sparse.stderr_tail,
    }


def write_rows(path: str, rows: Iterable[dict[str, object]], overwrite: bool) -> int:
    exists = os.path.exists(path) and not overwrite
    mode = "a" if exists else "w"
    count = 0
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["throughput", "latency", "both"], default="throughput")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sparse-variant", choices=["sp", "ordered", "both"], default="both")
    parser.add_argument("--family", action="append", choices=["f16", "f16f16", "bf16", "tf32", "fp8", "int8", "int4"])
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument("--iterations-latency", type=int, default=2048)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=108)
    parser.add_argument("--warps-per-block", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configs = make_configs(args)
    print(f"Selected {len(configs)} sparse/dense pairs", file=sys.stderr)

    if args.list:
        for cfg in configs:
            print(",".join(make_vars(cfg, 1)))
        return 0

    dense_cache: dict[tuple[object, ...], Result] = {}
    rows: list[dict[str, object]] = []
    for index, cfg in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {cfg.pair_id}", file=sys.stderr)
        dense_key = (
            cfg.mode,
            cfg.family,
            cfg.a_type,
            cfg.b_type,
            cfg.k,
            cfg.satfinite,
            cfg.iterations,
            cfg.unroll,
            cfg.blocks,
            cfg.warps_per_block,
        )
        dense = dense_cache.get(dense_key)
        if dense is None:
            dense = build_and_run(cfg, sparse=0, args=args)
            dense_cache[dense_key] = dense
        sparse = build_and_run(cfg, sparse=1, args=args)
        rows.append(row_from_results(cfg, dense, sparse))

    count = write_rows(args.output, rows, overwrite=args.overwrite)
    print(f"Wrote {count} rows to {args.output}", file=sys.stderr)
    failures = [
        r
        for r in rows
        if r["DenseBuildStatus"] != "ok"
        or r["SparseBuildStatus"] != "ok"
        or (not args.build_only and (r["DenseRunStatus"] != "ok" or r["SparseRunStatus"] != "ok"))
    ]
    if failures:
        print(f"{len(failures)} pairs had build/run failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
