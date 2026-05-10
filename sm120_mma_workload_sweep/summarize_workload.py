#!/usr/bin/env python3
"""Summarize same-workload SM120 sparse-vs-dense matrix sweep CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Iterable


MODES = ["throughput", "latency"]
VARIANTS = ["ordered", "sp"]


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row["DenseRunStatus"] == "ok"
        and row["SparseRunStatus"] == "ok"
        and row["SparseOverDenseWorkloadSpeedup"]
    ]


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sortable_key(values: tuple[str, ...]) -> tuple[object, ...]:
    converted: list[object] = []
    for value in values:
        try:
            converted.append(int(value))
        except ValueError:
            converted.append(value)
    return tuple(converted)


def summarize_group(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: sortable_key(item[0])):
        speedups = [fnum(row, "SparseOverDenseWorkloadSpeedup") for row in group]
        cycle_ratios = [fnum(row, "SparseOverDenseWorkloadCycleRatio") for row in group]
        dense_cycles = [fnum(row, "DenseCyclesPerWorkload") for row in group]
        sparse_cycles = [fnum(row, "SparseCyclesPerWorkload") for row in group]
        dense_ops = [fnum(row, "DenseWorkloadOpsPerCycle") for row in group]
        sparse_ops = [fnum(row, "SparseWorkloadOpsPerCycle") for row in group]
        item: dict[str, object] = dict(zip(keys, key))
        item.update(
            {
                "Rows": len(group),
                "MedianSparseOverDenseWorkloadSpeedup": fmt(median(speedups)),
                "MinSparseOverDenseWorkloadSpeedup": fmt(min(speedups)),
                "MaxSparseOverDenseWorkloadSpeedup": fmt(max(speedups)),
                "MedianSparseOverDenseWorkloadCycleRatio": fmt(median(cycle_ratios)),
                "MedianDenseCyclesPerWorkload": fmt(median(dense_cycles)),
                "MedianSparseCyclesPerWorkload": fmt(median(sparse_cycles)),
                "MedianDenseWorkloadOpsPerCycle": fmt(median(dense_ops)),
                "MedianSparseWorkloadOpsPerCycle": fmt(median(sparse_ops)),
            }
        )
        out.append(item)
    return out


def build_extremes(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Mode"], row["SparseVariant"])].append(row)

    for mode in MODES:
        for variant in VARIANTS:
            group = grouped.get((mode, variant), [])
            if not group:
                continue
            best = max(group, key=lambda row: fnum(row, "SparseOverDenseWorkloadSpeedup"))
            worst = min(group, key=lambda row: fnum(row, "SparseOverDenseWorkloadSpeedup"))
            for label, row in (("best", best), ("worst", worst)):
                out.append(
                    {
                        "Mode": mode,
                        "SparseVariant": variant,
                        "Extreme": label,
                        "PairID": row["PairID"],
                        "Family": row["Family"],
                        "M": row["M"],
                        "N": row["N"],
                        "K": row["K"],
                        "SparseOverDenseWorkloadSpeedup": row["SparseOverDenseWorkloadSpeedup"],
                        "SparseOverDenseWorkloadCycleRatio": row["SparseOverDenseWorkloadCycleRatio"],
                        "DenseCyclesPerWorkload": row["DenseCyclesPerWorkload"],
                        "SparseCyclesPerWorkload": row["SparseCyclesPerWorkload"],
                    }
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    rows = read_rows(args.csv_path)
    if not rows:
        raise SystemExit(f"no successful rows found in {args.csv_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    family_rows = summarize_group(rows, ["Mode", "SparseVariant", "Family"])
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_workload_summary_by_family.csv"),
        [
            "Mode",
            "SparseVariant",
            "Family",
            "Rows",
            "MedianSparseOverDenseWorkloadSpeedup",
            "MinSparseOverDenseWorkloadSpeedup",
            "MaxSparseOverDenseWorkloadSpeedup",
            "MedianSparseOverDenseWorkloadCycleRatio",
            "MedianDenseCyclesPerWorkload",
            "MedianSparseCyclesPerWorkload",
            "MedianDenseWorkloadOpsPerCycle",
            "MedianSparseWorkloadOpsPerCycle",
        ],
        family_rows,
    )

    size_rows = summarize_group(rows, ["Mode", "SparseVariant", "Scale", "M", "N"])
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_workload_summary_by_size.csv"),
        [
            "Mode",
            "SparseVariant",
            "Scale",
            "M",
            "N",
            "Rows",
            "MedianSparseOverDenseWorkloadSpeedup",
            "MinSparseOverDenseWorkloadSpeedup",
            "MaxSparseOverDenseWorkloadSpeedup",
            "MedianSparseOverDenseWorkloadCycleRatio",
            "MedianDenseCyclesPerWorkload",
            "MedianSparseCyclesPerWorkload",
            "MedianDenseWorkloadOpsPerCycle",
            "MedianSparseWorkloadOpsPerCycle",
        ],
        size_rows,
    )

    overall_rows = summarize_group(rows, ["Mode", "SparseVariant"])
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_workload_overall.csv"),
        [
            "Mode",
            "SparseVariant",
            "Rows",
            "MedianSparseOverDenseWorkloadSpeedup",
            "MinSparseOverDenseWorkloadSpeedup",
            "MaxSparseOverDenseWorkloadSpeedup",
            "MedianSparseOverDenseWorkloadCycleRatio",
            "MedianDenseCyclesPerWorkload",
            "MedianSparseCyclesPerWorkload",
            "MedianDenseWorkloadOpsPerCycle",
            "MedianSparseWorkloadOpsPerCycle",
        ],
        overall_rows,
    )

    extremes = build_extremes(rows)
    write_csv(
        os.path.join(args.out_dir, "sm120_mma_workload_extremes.csv"),
        [
            "Mode",
            "SparseVariant",
            "Extreme",
            "PairID",
            "Family",
            "M",
            "N",
            "K",
            "SparseOverDenseWorkloadSpeedup",
            "SparseOverDenseWorkloadCycleRatio",
            "DenseCyclesPerWorkload",
            "SparseCyclesPerWorkload",
        ],
        extremes,
    )

    print(f"summarized {len(rows)} successful rows into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
