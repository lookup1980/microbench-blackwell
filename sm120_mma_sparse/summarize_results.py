#!/usr/bin/env python3
"""Summarize sm120_mma_sparse benchmark CSVs into compact comparison tables."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Iterable


FAMILIES = ["f16", "f16f16", "bf16", "tf32", "fp8", "int8", "int4"]
MODES = ["throughput", "latency"]
VARIANTS = ["ordered", "sp"]


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Mode"], row["SparseVariant"], row["Family"])].append(row)

    out: list[dict[str, object]] = []
    for mode in MODES:
        for variant in VARIANTS:
            for family in FAMILIES:
                group = grouped.get((mode, variant, family), [])
                if not group:
                    continue
                speedups = [fnum(r, "SparseOverDenseOpsPerCycle") for r in group]
                cycle_ratios = [fnum(r, "SparseOverDenseCyclesPerPacket") for r in group]
                dense_ops = [fnum(r, "DenseOpsPerCycle") for r in group]
                sparse_ops = [fnum(r, "SparseOpsPerCycle") for r in group]
                out.append(
                    {
                        "Mode": mode,
                        "SparseVariant": variant,
                        "Family": family,
                        "Rows": len(group),
                        "MedianSparseOverDenseOpsPerCycle": fmt(median(speedups)),
                        "MinSparseOverDenseOpsPerCycle": fmt(min(speedups)),
                        "MaxSparseOverDenseOpsPerCycle": fmt(max(speedups)),
                        "MedianSparseOverDenseCyclesPerPacket": fmt(median(cycle_ratios)),
                        "MedianDenseOpsPerCycle": fmt(median(dense_ops)),
                        "MedianSparseOpsPerCycle": fmt(median(sparse_ops)),
                    }
                )
    return out


def build_extremes(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for mode in MODES:
        for variant in VARIANTS:
            group = [r for r in rows if r["Mode"] == mode and r["SparseVariant"] == variant]
            if not group:
                continue
            best = max(group, key=lambda r: fnum(r, "SparseOverDenseOpsPerCycle"))
            worst = min(group, key=lambda r: fnum(r, "SparseOverDenseOpsPerCycle"))
            for label, row in (("best", best), ("worst", worst)):
                out.append(
                    {
                        "Mode": mode,
                        "SparseVariant": variant,
                        "Extreme": label,
                        "PairID": row["PairID"],
                        "SparseOverDenseOpsPerCycle": fmt(fnum(row, "SparseOverDenseOpsPerCycle")),
                        "SparseOverDenseCyclesPerPacket": fmt(fnum(row, "SparseOverDenseCyclesPerPacket")),
                        "DenseOpsPerCycle": fmt(fnum(row, "DenseOpsPerCycle")),
                        "SparseOpsPerCycle": fmt(fnum(row, "SparseOpsPerCycle")),
                        "DenseCyclesPerPacket": fmt(fnum(row, "DenseCyclesPerPacket")),
                        "SparseCyclesPerPacket": fmt(fnum(row, "SparseCyclesPerPacket")),
                    }
                )
    return out


def build_overall(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for mode in MODES:
        for variant in VARIANTS:
            group = [r for r in rows if r["Mode"] == mode and r["SparseVariant"] == variant]
            speedups = [fnum(r, "SparseOverDenseOpsPerCycle") for r in group]
            cycle_ratios = [fnum(r, "SparseOverDenseCyclesPerPacket") for r in group]
            out.append(
                {
                    "Mode": mode,
                    "SparseVariant": variant,
                    "Rows": len(group),
                    "MedianSparseOverDenseOpsPerCycle": fmt(median(speedups)),
                    "MinSparseOverDenseOpsPerCycle": fmt(min(speedups)),
                    "MaxSparseOverDenseOpsPerCycle": fmt(max(speedups)),
                    "MedianSparseOverDenseCyclesPerPacket": fmt(median(cycle_ratios)),
                }
            )
    return out


def mma_suffix(row: dict[str, str]) -> str:
    family = row["Family"]
    a_type = row["AType"]
    b_type = row["BType"]
    satfinite = row["Satfinite"] == "satfinite"

    if family == "f16":
        return "f32.f16.f16.f32"
    if family == "f16f16":
        return "f16.f16.f16.f16"
    if family == "bf16":
        return "f32.bf16.bf16.f32"
    if family == "tf32":
        return "f32.tf32.tf32.f32"
    if family == "fp8":
        return f"f32.{a_type}.{b_type}.f32"
    if family in {"int8", "int4"}:
        prefix = "satfinite." if satfinite else ""
        return f"{prefix}s32.{a_type}.{b_type}.s32"
    raise ValueError(f"unsupported family {family}")


def sparse_instruction(row: dict[str, str]) -> str:
    op = "mma.sp::ordered_metadata" if row["SparseVariant"] == "ordered" else "mma.sp"
    return (
        f"{op}.sync.aligned.m16n8k{row['SemanticK']}.row.col."
        f"{mma_suffix(row)}"
    )


def dense_instruction(row: dict[str, str]) -> str:
    return (
        f"mma.sync.aligned.m16n8k{row['DenseBaseK']}.row.col."
        f"{mma_suffix(row)}"
    )


def input_formats(family: str) -> str:
    if family == "fp8":
        return "e4m3/e5m2 x e4m3/e5m2"
    if family == "int8":
        return "u8/s8 x u8/s8"
    if family == "int4":
        return "u4/s4 x u4/s4"
    if family == "f16f16":
        return "f16 x f16"
    return f"{family} x {family}"


def accumulator(family: str) -> str:
    if family == "f16f16":
        return "f16"
    if family in {"int8", "int4"}:
        return "s32"
    return "f32"


def build_instruction_pairs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["Mode"],
                row["SparseVariant"],
                row["Family"],
                row["Satfinite"],
                row["SemanticK"],
                row["DenseBaseK"],
                row["DenseInstructionsPerPacket"],
            )
        ].append(row)

    out: list[dict[str, object]] = []
    for mode in MODES:
        for variant in VARIANTS:
            for family in FAMILIES:
                keys = [
                    key
                    for key in grouped
                    if key[0] == mode and key[1] == variant and key[2] == family
                ]
                for key in sorted(keys, key=lambda k: (int(k[4]), k[3])):
                    group = grouped[key]
                    sample = group[0]
                    speedups = [fnum(r, "SparseOverDenseOpsPerCycle") for r in group]
                    cycle_ratios = [fnum(r, "SparseOverDenseCyclesPerPacket") for r in group]
                    out.append(
                        {
                            "Mode": mode,
                            "SparseVariant": variant,
                            "Family": family,
                            "Satfinite": sample["Satfinite"],
                            "InputFormats": input_formats(family),
                            "Accumulator": accumulator(family),
                            "SemanticK": sample["SemanticK"],
                            "SparseInstruction": sparse_instruction(sample),
                            "DenseInstruction": dense_instruction(sample),
                            "DenseInstructionsPerPacket": sample["DenseInstructionsPerPacket"],
                            "Rows": len(group),
                            "MedianSparseOverDenseOpsPerCycle": fmt(median(speedups)),
                            "MedianSparseOverDenseCyclesPerPacket": fmt(median(cycle_ratios)),
                        }
                    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="results/sm120_mma_full.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    rows = read_rows(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    failures = [
        r
        for r in rows
        if r["DenseBuildStatus"] != "ok"
        or r["SparseBuildStatus"] != "ok"
        or r["DenseRunStatus"] != "ok"
        or r["SparseRunStatus"] != "ok"
    ]
    if failures:
        raise SystemExit(f"{len(failures)} build/run failures in {args.input}")

    summary = build_summary(rows)
    extremes = build_extremes(rows)
    overall = build_overall(rows)
    instruction_pairs = build_instruction_pairs(rows)

    write_csv(
        os.path.join(args.out_dir, "sm120_mma_summary_by_family.csv"),
        [
            "Mode",
            "SparseVariant",
            "Family",
            "Rows",
            "MedianSparseOverDenseOpsPerCycle",
            "MinSparseOverDenseOpsPerCycle",
            "MaxSparseOverDenseOpsPerCycle",
            "MedianSparseOverDenseCyclesPerPacket",
            "MedianDenseOpsPerCycle",
            "MedianSparseOpsPerCycle",
        ],
        summary,
    )
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_extremes.csv"),
        [
            "Mode",
            "SparseVariant",
            "Extreme",
            "PairID",
            "SparseOverDenseOpsPerCycle",
            "SparseOverDenseCyclesPerPacket",
            "DenseOpsPerCycle",
            "SparseOpsPerCycle",
            "DenseCyclesPerPacket",
            "SparseCyclesPerPacket",
        ],
        extremes,
    )
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_overall.csv"),
        [
            "Mode",
            "SparseVariant",
            "Rows",
            "MedianSparseOverDenseOpsPerCycle",
            "MinSparseOverDenseOpsPerCycle",
            "MaxSparseOverDenseOpsPerCycle",
            "MedianSparseOverDenseCyclesPerPacket",
        ],
        overall,
    )
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_instruction_pairs.csv"),
        [
            "Mode",
            "SparseVariant",
            "Family",
            "Satfinite",
            "InputFormats",
            "Accumulator",
            "SemanticK",
            "SparseInstruction",
            "DenseInstruction",
            "DenseInstructionsPerPacket",
            "Rows",
            "MedianSparseOverDenseOpsPerCycle",
            "MedianSparseOverDenseCyclesPerPacket",
        ],
        instruction_pairs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
