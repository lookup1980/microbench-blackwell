#!/usr/bin/env python3
"""
Sweep sparse Blackwell tcgen05 MMA instructions and dense peers.

The driver intentionally rebuilds one compile-time-specialized CUDA translation
unit per configuration, matching the style of the existing umma_* benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


TYPE_INFO = {
    "tf32": {"code": 2, "bits": 32, "family": "tf32"},
    "f16": {"code": 0, "bits": 16, "family": "f16"},
    "bf16": {"code": 1, "bits": 16, "family": "f16"},
    "e4m3": {"code": 0, "bits": 8, "family": "narrow"},
    "e5m2": {"code": 1, "bits": 8, "family": "narrow"},
    "e2m3": {"code": 3, "bits": 6, "family": "narrow"},
    "e3m2": {"code": 4, "bits": 6, "family": "narrow"},
    "e2m1": {"code": 5, "bits": 4, "family": "narrow"},
    "u8": {"code": 0, "bits": 8, "family": "i8"},
    "s8": {"code": 1, "bits": 8, "family": "i8"},
    "mxf4": {"code": 1, "bits": 4, "family": "mxf4"},
    "nvf4": {"code": 1, "bits": 4, "family": "nvf4"},
}

KIND_IDS = {
    "tf32": 0,
    "f16": 1,
    "f8f6f4": 2,
    "i8": 3,
    "mxf8f6f4": 4,
    "mxf4": 5,
    "mxf4nvf4": 6,
}

INSTRUCTION_IDS = {"mma": 0, "ws": 1}
LAYOUT_IDS = {"ss": 0, "ts": 1}


@dataclass(frozen=True)
class FormatVariant:
    kind: str
    a_type: str
    b_type: str
    dense_k: int
    sparse_k: int
    scale_type: str = "ue8m0"
    block_scaled: bool = False


@dataclass(frozen=True)
class Config:
    mode: str
    instruction: str
    variant: FormatVariant
    sparsity: int
    ab_layout: str
    cta_group: int
    m: int
    n: int
    k: int
    depth: int

    @property
    def pair_id(self) -> str:
        return (
            f"{self.instruction}:{self.variant.kind}:"
            f"{self.variant.a_type}x{self.variant.b_type}:"
            f"{self.ab_layout}:cta{self.cta_group}:m{self.m}:n{self.n}"
        )


def native_dense_k(a_type: str) -> int:
    bits = TYPE_INFO[a_type]["bits"]
    if bits == 32:
        return 8
    if bits == 16:
        return 16
    if bits == 8:
        return 32
    if bits == 6:
        return 32
    if bits == 4:
        return 64
    raise ValueError(f"unsupported bit width for {a_type}: {bits}")


def variants() -> list[FormatVariant]:
    out: list[FormatVariant] = []

    out.append(FormatVariant("tf32", "tf32", "tf32", dense_k=8, sparse_k=16))

    for t in ("f16", "bf16"):
        out.append(FormatVariant("f16", t, t, dense_k=16, sparse_k=32))

    for t in ("s8", "u8"):
        out.append(FormatVariant("i8", t, t, dense_k=32, sparse_k=64))

    narrow = ("e4m3", "e5m2", "e2m3", "e3m2", "e2m1")
    for a in narrow:
        for b in narrow:
            dk = native_dense_k(a)
            out.append(FormatVariant("f8f6f4", a, b, dense_k=dk, sparse_k=2 * dk))
            out.append(
                FormatVariant(
                    "mxf8f6f4",
                    a,
                    b,
                    dense_k=dk,
                    sparse_k=2 * dk,
                    block_scaled=True,
                )
            )

    out.append(
        FormatVariant(
            "mxf4", "mxf4", "mxf4", dense_k=64, sparse_k=128, block_scaled=True
        )
    )
    out.append(
        FormatVariant(
            "mxf4nvf4",
            "mxf4",
            "mxf4",
            dense_k=64,
            sparse_k=128,
            scale_type="ue8m0",
            block_scaled=True,
        )
    )
    out.append(
        FormatVariant(
            "mxf4nvf4",
            "nvf4",
            "nvf4",
            dense_k=64,
            sparse_k=128,
            scale_type="ue4m3",
            block_scaled=True,
        )
    )
    return out


def valid_for_instruction(instruction: str, variant: FormatVariant) -> bool:
    if instruction == "mma":
        return True
    if instruction == "ws":
        return not variant.block_scaled
    raise ValueError(instruction)


def depths_for(k: int, mode: str) -> list[int]:
    if mode == "latency":
        return [1]
    if k <= 16:
        return [16, 64, 256]
    if k <= 32:
        return [32, 128, 512]
    if k <= 64:
        return [64, 256, 1024]
    return [128, 512, 1024]


def smoke_variants(all_variants: list[FormatVariant]) -> list[FormatVariant]:
    keep = {
        ("tf32", "tf32", "tf32"),
        ("f16", "f16", "f16"),
        ("f16", "bf16", "bf16"),
        ("f8f6f4", "e4m3", "e4m3"),
        ("f8f6f4", "e2m1", "e2m1"),
        ("i8", "s8", "s8"),
        ("mxf8f6f4", "e4m3", "e4m3"),
        ("mxf4", "mxf4", "mxf4"),
        ("mxf4nvf4", "nvf4", "nvf4"),
    }
    return [
        v
        for v in all_variants
        if (v.kind, v.a_type, v.b_type) in keep
    ]


def make_configs(args: argparse.Namespace) -> list[Config]:
    selected_variants = variants()
    if args.preset == "smoke":
        selected_variants = smoke_variants(selected_variants)

    if args.kind:
        selected_variants = [v for v in selected_variants if v.kind in args.kind]

    instructions = ["mma", "ws"] if args.instruction == "all" else [args.instruction]
    sparsities = [0, 1] if args.sparsity == "both" else [1 if args.sparsity == "sparse" else 0]
    layouts = ["ss", "ts"] if args.ab_layout == "all" else [args.ab_layout]
    cta_groups = [1, 2] if args.cta_group == "all" else [int(args.cta_group)]

    configs: list[Config] = []
    for instruction in instructions:
        for variant in selected_variants:
            if not valid_for_instruction(instruction, variant):
                continue
            for sparsity in sparsities:
                k = variant.sparse_k if sparsity else variant.dense_k
                for ab_layout in layouts:
                    for cta_group in cta_groups:
                        if instruction == "ws" and cta_group != 1:
                            continue
                        m_values = [128] if cta_group == 1 else [256]
                        n_values = [64] if args.preset == "smoke" else [64, 128, 256]
                        for m in m_values:
                            for n in n_values:
                                for depth in depths_for(k, args.mode):
                                    if args.mode == "latency" and depth != 1:
                                        continue
                                    configs.append(
                                        Config(
                                            mode=args.mode,
                                            instruction=instruction,
                                            variant=variant,
                                            sparsity=sparsity,
                                            ab_layout=ab_layout,
                                            cta_group=cta_group,
                                            m=m,
                                            n=n,
                                            k=k,
                                            depth=depth,
                                        )
                                    )

    if args.max_configs:
        configs = configs[: args.max_configs]
    return configs


def make_args(config: Config) -> list[str]:
    return [
        f"MODE={config.mode}",
        f"INSTRUCTION={config.instruction}",
        f"KIND={config.variant.kind}",
        f"A_TYPE={config.variant.a_type}",
        f"B_TYPE={config.variant.b_type}",
        f"SPARSITY={config.sparsity}",
        f"AB_LAYOUT={config.ab_layout}",
        f"CTA_GROUP={config.cta_group}",
        f"MMA_M={config.m}",
        f"MMA_N={config.n}",
        f"MMA_K={config.k}",
        f"MMA_DEPTH={config.depth}",
        f"SCALE_TYPE={config.variant.scale_type}",
    ]


def run_cmd(cmd: list[str], verbose: bool) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, text=True, capture_output=True)


def build_and_run(config: Config, args: argparse.Namespace) -> dict[str, object] | None:
    make_vars = make_args(config)
    run_cmd(["make", "clean"], args.verbose)

    target = "ptx" if args.ptx else "tcgen05_sparse_mma.out"
    build = run_cmd(["make", target, *make_vars], args.verbose)
    if build.returncode != 0:
        print(f"BUILD_FAILED {config.pair_id} sparse={config.sparsity}", file=sys.stderr)
        if args.verbose:
            print(build.stderr, file=sys.stderr)
        return None

    if args.build_only or args.ptx:
        return row_from_config(config, cycles=None, total_mmas=None, cycles_per_mma=None, median=None)

    result = run_cmd(["./tcgen05_sparse_mma.out"], args.verbose)
    if result.returncode != 0:
        print(f"RUN_FAILED {config.pair_id} sparse={config.sparsity}", file=sys.stderr)
        if args.verbose:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        return None

    for line in result.stdout.splitlines():
        if not line.startswith("RESULT,"):
            continue
        parts = line.split(",")
        if config.mode == "throughput" and len(parts) >= 18:
            return row_from_config(
                config,
                cycles=int(parts[12]),
                total_mmas=int(parts[13]),
                cycles_per_mma=float(parts[14]),
                median=None,
            )
        if config.mode == "latency" and len(parts) >= 15:
            return row_from_config(
                config,
                cycles=None,
                total_mmas=None,
                cycles_per_mma=None,
                median=int(parts[11]),
            )

    print(f"PARSE_FAILED {config.pair_id} sparse={config.sparsity}", file=sys.stderr)
    if args.verbose:
        print(result.stdout, file=sys.stderr)
    return None


def row_from_config(
    config: Config,
    cycles: int | None,
    total_mmas: int | None,
    cycles_per_mma: float | None,
    median: int | None,
) -> dict[str, object]:
    math_ops = 2 * config.m * config.n * config.k
    stored_a = config.m * (config.k // 2 if config.sparsity else config.k)
    row = {
        "PairID": config.pair_id,
        "Instruction": config.instruction,
        "Kind": config.variant.kind,
        "AType": config.variant.a_type,
        "BType": config.variant.b_type,
        "Sparsity": "sparse" if config.sparsity else "dense",
        "ABLayout": config.ab_layout.upper(),
        "CTAGroup": config.cta_group,
        "M": config.m,
        "N": config.n,
        "K": config.k,
        "PipelineDepth": "" if config.mode == "latency" else config.depth,
        "Cycles": "" if cycles is None else cycles,
        "TotalMMAs": "" if total_mmas is None else total_mmas,
        "CyclesPerMMA": "" if cycles_per_mma is None else f"{cycles_per_mma:.6f}",
        "MedianCycles": "" if median is None else median,
        "MathOpsPerMMA": math_ops,
        "OpsPerCycle": "" if cycles_per_mma in (None, 0) else f"{math_ops / cycles_per_mma:.3f}",
        "StoredAElements": stored_a,
        "ABytes": (stored_a * TYPE_INFO[config.variant.a_type]["bits"] + 7) // 8,
        "BBytes": (config.k * config.n * TYPE_INFO[config.variant.b_type]["bits"] + 7) // 8,
    }
    return row


CSV_FIELDS = [
    "PairID",
    "Instruction",
    "Kind",
    "AType",
    "BType",
    "Sparsity",
    "ABLayout",
    "CTAGroup",
    "M",
    "N",
    "K",
    "PipelineDepth",
    "Cycles",
    "TotalMMAs",
    "CyclesPerMMA",
    "MedianCycles",
    "MathOpsPerMMA",
    "OpsPerCycle",
    "StoredAElements",
    "ABytes",
    "BBytes",
]


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
    parser.add_argument("--mode", choices=["throughput", "latency"], default="throughput")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sparsity", choices=["dense", "sparse", "both"], default="both")
    parser.add_argument("--instruction", choices=["mma", "ws", "all"], default="all")
    parser.add_argument("--ab-layout", choices=["ss", "ts", "all"], default="ss")
    parser.add_argument("--cta-group", choices=["1", "2", "all"], default="all")
    parser.add_argument("--kind", action="append", choices=sorted(KIND_IDS))
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--ptx", action="store_true", help="Build PTX instead of executable")
    parser.add_argument("--list", action="store_true", help="List configurations without building")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configs = make_configs(args)
    print(f"Selected {len(configs)} configurations", file=sys.stderr)

    if args.list:
        for cfg in configs:
            print(",".join(make_args(cfg)))
        return 0

    rows = []
    for i, cfg in enumerate(configs, 1):
        print(
            f"[{i}/{len(configs)}] {cfg.pair_id} {('sparse' if cfg.sparsity else 'dense')} "
            f"K={cfg.k} depth={cfg.depth}",
            file=sys.stderr,
        )
        row = build_and_run(cfg, args)
        if row:
            rows.append(row)

    count = write_rows(args.output, rows, overwrite=args.overwrite)
    print(f"Wrote {count} rows to {args.output}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
