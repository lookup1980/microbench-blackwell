#!/usr/bin/env python3
"""
Run same-workload sparse-vs-dense SM120 tensor-core matrix sweeps.

Each row compares dense and sparse execution for the same synthetic
register-only GEMM workload:

    A[M, K] x B[K, N] -> C[M, N]

The CUDA benchmark uses the same warp-level PTX instruction coverage as the
instruction microbenchmark, but matrix sizes are runtime arguments.  Warmup
kernel launches run before every measured launch so profiler captures see a
stabilized path.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


BASE_M = 16
BASE_N = 16


@dataclass(frozen=True)
class Spec:
    family: str
    a_type: str
    b_type: str
    semantic_k: int
    satfinite: int = 0

    @property
    def format_id(self) -> str:
        sat = "sat" if self.satfinite else "nosat"
        return f"{self.family}:{self.a_type}x{self.b_type}:k{self.semantic_k}:{sat}"


@dataclass(frozen=True)
class Shape:
    scale: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class Config:
    mode: str
    spec: Spec
    sparse_variant: str
    shape: Shape
    iterations: int
    unroll: int
    blocks: int
    warps_per_block: int
    warmup_launches: int

    @property
    def pair_id(self) -> str:
        sat = "sat" if self.spec.satfinite else "nosat"
        return (
            f"{self.mode}:{self.spec.family}:{self.spec.a_type}x{self.spec.b_type}:"
            f"m{self.shape.m}n{self.shape.n}k{self.shape.k}:"
            f"{self.sparse_variant}:{sat}"
        )


@dataclass
class Result:
    build_status: str
    run_status: str = "not_run"
    stderr_tail: str = ""
    executable: str | None = None
    cycles: int | None = None
    cycles_per_packet: float | None = None
    ops_per_cycle: float | None = None
    dense_base_k: int | None = None
    dense_instructions_per_packet: int | None = None
    sparse_instructions_per_packet: int | None = None
    workload_packets: int | None = None
    cycles_per_workload: float | None = None
    workload_ops_per_cycle: float | None = None
    math_ops_per_workload: float | None = None
    warps: int | None = None
    warmup_launches: int | None = None


def representative_specs() -> list[Spec]:
    return [
        Spec("f16", "f16", "f16", 32),
        Spec("f16f16", "f16", "f16", 32),
        Spec("bf16", "bf16", "bf16", 32),
        Spec("tf32", "tf32", "tf32", 16),
        Spec("fp8", "e4m3", "e4m3", 64),
        Spec("int8", "u8", "u8", 64),
        Spec("int4", "u4", "u4", 128),
    ]


def full_specs() -> list[Spec]:
    specs = representative_specs()[:4]
    for a_type in ("e4m3", "e5m2"):
        for b_type in ("e4m3", "e5m2"):
            specs.append(Spec("fp8", a_type, b_type, 64))
    for family, k, types in (
        ("int8", 64, ("u8", "s8")),
        ("int4", 128, ("u4", "s4")),
    ):
        for a_type in types:
            for b_type in types:
                for satfinite in (0, 1):
                    specs.append(Spec(family, a_type, b_type, k, satfinite))
    return specs


def parse_scales(text: str) -> list[int]:
    scales = [int(part) for part in text.split(",") if part]
    if not scales or any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be positive comma-separated integers")
    return scales


def shape_for(spec: Spec, scale: int) -> Shape:
    return Shape(
        scale=scale,
        m=BASE_M * scale,
        n=BASE_N * scale,
        k=spec.semantic_k * scale,
    )


def iterations_for(mode: str, shape: Shape, spec: Spec, unroll: int, target_packets: int) -> int:
    workload_packets = (shape.m // 16) * (shape.n // 8) * (shape.k // spec.semantic_k)
    packets_per_iteration = workload_packets * (unroll if mode == "throughput" else 1)
    return max(1, math.ceil(target_packets / packets_per_iteration))


def selected_specs(args: argparse.Namespace) -> list[Spec]:
    specs = representative_specs() if args.format_preset == "representative" else full_specs()
    if args.family:
        specs = [spec for spec in specs if spec.family in args.family]
    return specs


def selected_scales(args: argparse.Namespace) -> list[int]:
    if args.scales:
        return args.scales
    if args.size_preset == "smoke":
        return [1, 4, 16]
    return [1, 2, 4, 8, 16, 32]


def make_configs(args: argparse.Namespace) -> list[Config]:
    modes = ["throughput", "latency"] if args.mode == "both" else [args.mode]
    sparse_variants = (
        ["sp", "ordered"]
        if args.sparse_variant == "both"
        else [args.sparse_variant]
    )
    specs = selected_specs(args)
    scales = selected_scales(args)
    if args.max_specs:
        specs = specs[: args.max_specs]
    if args.max_shapes:
        scales = scales[: args.max_shapes]

    configs: list[Config] = []
    for mode in modes:
        for spec in specs:
            for sparse_variant in sparse_variants:
                for scale in scales:
                    shape = shape_for(spec, scale)
                    iterations = iterations_for(
                        mode, shape, spec, args.unroll, args.target_packets_per_warp
                    )
                    configs.append(
                        Config(
                            mode=mode,
                            spec=spec,
                            sparse_variant=sparse_variant,
                            shape=shape,
                            iterations=iterations,
                            unroll=1 if mode == "latency" else args.unroll,
                            blocks=args.blocks,
                            warps_per_block=args.warps_per_block,
                            warmup_launches=args.warmup_launches,
                        )
                    )
    return configs


def make_vars(config: Config, sparse: int) -> list[str]:
    ordered = 1 if sparse and config.sparse_variant == "ordered" else 0
    return [
        f"MODE={config.mode}",
        f"FAMILY={config.spec.family}",
        f"A_TYPE={config.spec.a_type}",
        f"B_TYPE={config.spec.b_type}",
        f"SPARSITY={sparse}",
        f"ORDERED_METADATA={ordered}",
        f"SATFINITE={config.spec.satfinite}",
        f"MMA_K={config.spec.semantic_k}",
        f"ITERATIONS={config.iterations}",
        f"UNROLL={config.unroll}",
        f"BLOCKS={config.blocks}",
        f"WARPS_PER_BLOCK={config.warps_per_block}",
        f"WARMUP_LAUNCHES={config.warmup_launches}",
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
        if len(parts) < 26:
            return None
        return Result(
            build_status="ok",
            run_status="ok",
            dense_base_k=int(parts[8]),
            dense_instructions_per_packet=int(parts[9]),
            sparse_instructions_per_packet=int(parts[10]),
            warps=int(parts[13]),
            cycles=int(parts[14]),
            cycles_per_packet=float(parts[15]),
            ops_per_cycle=float(parts[17]),
            workload_packets=int(parts[21]),
            warmup_launches=int(parts[22]),
            cycles_per_workload=float(parts[23]),
            math_ops_per_workload=float(parts[24]),
            workload_ops_per_cycle=float(parts[25]),
        )
    return None


def build(config: Config, sparse: int, args: argparse.Namespace) -> Result:
    key = build_key(config, sparse)
    exe_name = "_".join(str(part).replace("/", "_") for part in key) + ".out"
    executable = os.path.join("build", exe_name)
    build_result = run_cmd(
        ["make", executable, f"OUT={executable}", *make_vars(config, sparse)],
        args.verbose,
    )
    if build_result.returncode != 0:
        return Result(
            build_status="failed",
            run_status="not_run",
            stderr_tail=compact_notes(build_result.stderr),
            executable=executable,
        )
    return Result(
        build_status="ok",
        run_status="not_run",
        stderr_tail=compact_notes(build_result.stderr),
        executable=executable,
    )


def run_benchmark(config: Config, build_result: Result, args: argparse.Namespace) -> Result:
    if build_result.build_status != "ok":
        return build_result
    if args.build_only:
        return build_result
    if not build_result.executable:
        return Result(
            build_status="ok",
            run_status="failed",
            stderr_tail="internal error: missing executable path",
        )

    shape = config.shape
    result = run_cmd(
        [
            build_result.executable,
            str(shape.m),
            str(shape.n),
            str(shape.k),
            str(config.iterations),
            str(config.warmup_launches),
        ],
        args.verbose,
    )
    if result.returncode != 0:
        return Result(
            build_status="ok",
            run_status="failed",
            stderr_tail=compact_notes(result.stdout + "\n" + result.stderr),
        )
    parsed = parse_result(result.stdout)
    if parsed is None:
        return Result(
            build_status="ok",
            run_status="parse_failed",
            stderr_tail=compact_notes(result.stdout + "\n" + result.stderr),
        )
    parsed.stderr_tail = build_result.stderr_tail
    return parsed


CSV_FIELDS = [
    "PairID",
    "Mode",
    "Family",
    "AType",
    "BType",
    "SparseVariant",
    "Satfinite",
    "Scale",
    "M",
    "N",
    "K",
    "AElements",
    "BElements",
    "CElements",
    "SemanticK",
    "DenseBaseK",
    "DenseInstructionsPerPacket",
    "SparseInstructionsPerPacket",
    "WorkloadPackets",
    "Iterations",
    "Unroll",
    "Warps",
    "WarmupLaunches",
    "DenseBuildStatus",
    "DenseRunStatus",
    "SparseBuildStatus",
    "SparseRunStatus",
    "DenseCycles",
    "DenseCyclesPerPacket",
    "DenseCyclesPerWorkload",
    "DenseWorkloadOpsPerCycle",
    "SparseCycles",
    "SparseCyclesPerPacket",
    "SparseCyclesPerWorkload",
    "SparseWorkloadOpsPerCycle",
    "SparseOverDenseWorkloadSpeedup",
    "SparseOverDenseWorkloadCycleRatio",
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
    if dense.workload_ops_per_cycle and sparse.workload_ops_per_cycle:
        speedup = sparse.workload_ops_per_cycle / dense.workload_ops_per_cycle
    if dense.cycles_per_workload and sparse.cycles_per_workload:
        cycle_ratio = sparse.cycles_per_workload / dense.cycles_per_workload

    shape = config.shape
    return {
        "PairID": config.pair_id,
        "Mode": config.mode,
        "Family": config.spec.family,
        "AType": config.spec.a_type,
        "BType": config.spec.b_type,
        "SparseVariant": config.sparse_variant,
        "Satfinite": "satfinite" if config.spec.satfinite else "nosat",
        "Scale": shape.scale,
        "M": shape.m,
        "N": shape.n,
        "K": shape.k,
        "AElements": shape.m * shape.k,
        "BElements": shape.k * shape.n,
        "CElements": shape.m * shape.n,
        "SemanticK": config.spec.semantic_k,
        "DenseBaseK": fmt(dense.dense_base_k),
        "DenseInstructionsPerPacket": fmt(dense.dense_instructions_per_packet),
        "SparseInstructionsPerPacket": fmt(sparse.sparse_instructions_per_packet),
        "WorkloadPackets": fmt(sparse.workload_packets if sparse.workload_packets else dense.workload_packets),
        "Iterations": config.iterations,
        "Unroll": config.unroll,
        "Warps": fmt(sparse.warps if sparse.warps is not None else dense.warps),
        "WarmupLaunches": config.warmup_launches,
        "DenseBuildStatus": dense.build_status,
        "DenseRunStatus": dense.run_status,
        "SparseBuildStatus": sparse.build_status,
        "SparseRunStatus": sparse.run_status,
        "DenseCycles": fmt(dense.cycles),
        "DenseCyclesPerPacket": fmt(dense.cycles_per_packet),
        "DenseCyclesPerWorkload": fmt(dense.cycles_per_workload),
        "DenseWorkloadOpsPerCycle": fmt(dense.workload_ops_per_cycle),
        "SparseCycles": fmt(sparse.cycles),
        "SparseCyclesPerPacket": fmt(sparse.cycles_per_packet),
        "SparseCyclesPerWorkload": fmt(sparse.cycles_per_workload),
        "SparseWorkloadOpsPerCycle": fmt(sparse.workload_ops_per_cycle),
        "SparseOverDenseWorkloadSpeedup": fmt(speedup),
        "SparseOverDenseWorkloadCycleRatio": fmt(cycle_ratio),
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


def build_key(config: Config, sparse: int) -> tuple[object, ...]:
    ordered = 1 if sparse and config.sparse_variant == "ordered" else 0
    return (
        config.mode,
        config.spec.family,
        config.spec.a_type,
        config.spec.b_type,
        config.spec.semantic_k,
        config.spec.satfinite,
        sparse,
        ordered,
        config.unroll,
        config.blocks,
        config.warps_per_block,
        config.warmup_launches,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["throughput", "latency", "both"], default="both")
    parser.add_argument("--format-preset", choices=["representative", "full"], default="representative")
    parser.add_argument("--size-preset", choices=["smoke", "full"], default="full")
    parser.add_argument("--scales", type=parse_scales)
    parser.add_argument("--sparse-variant", choices=["sp", "ordered", "both"], default="ordered")
    parser.add_argument("--family", action="append", choices=["f16", "f16f16", "bf16", "tf32", "fp8", "int8", "int4"])
    parser.add_argument("--output", default="results/sm120_mma_workload_sweep.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-specs", type=int, default=0)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--target-packets-per-warp", type=int, default=4096)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=108)
    parser.add_argument("--warps-per-block", type=int, default=8)
    parser.add_argument("--warmup-launches", type=int, default=3)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.target_packets_per_warp <= 0:
        parser.error("--target-packets-per-warp must be positive")
    if args.warmup_launches <= 0:
        parser.error("--warmup-launches must be positive")

    configs = make_configs(args)
    print(f"Selected {len(configs)} workload sparse/dense rows", file=sys.stderr)

    if args.list:
        for cfg in configs:
            print(",".join(make_vars(cfg, 1)), cfg.shape, f"iterations={cfg.iterations}")
        return 0

    rows: list[dict[str, object]] = []
    build_cache: dict[tuple[object, ...], Result] = {}

    for index, cfg in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {cfg.pair_id}", file=sys.stderr)

        dense_key = build_key(cfg, sparse=0)
        dense_build = build_cache.get(dense_key)
        if dense_build is None:
            dense_build = build(cfg, sparse=0, args=args)
            build_cache[dense_key] = dense_build
        dense = run_benchmark(cfg, dense_build, args)

        sparse_key = build_key(cfg, sparse=1)
        sparse_build = build_cache.get(sparse_key)
        if sparse_build is None:
            sparse_build = build(cfg, sparse=1, args=args)
            build_cache[sparse_key] = sparse_build
        sparse = run_benchmark(cfg, sparse_build, args)

        rows.append(row_from_results(cfg, dense, sparse))

    count = write_rows(args.output, rows, overwrite=args.overwrite)
    print(f"Wrote {count} rows to {args.output}", file=sys.stderr)

    failures = [
        row
        for row in rows
        if row["DenseBuildStatus"] != "ok"
        or row["SparseBuildStatus"] != "ok"
        or (not args.build_only and (row["DenseRunStatus"] != "ok" or row["SparseRunStatus"] != "ok"))
    ]
    if failures:
        print(f"{len(failures)} rows had build/run failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
