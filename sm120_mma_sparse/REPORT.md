# SM120 Sparse MMA Instruction Diagnostic

Measurement date: 2026-05-09

This directory is now a diagnostic appendix.  Do not use this report to state
the sparse-vs-dense speedup for a matrix workload.

The primary performance report is:

```text
../sm120_mma_workload_sweep/REPORT.md
../sm120_mma_workload_sweep/REPORT.pdf
```

That report compares sparse and dense at the same `A[M,K] x B[K,N]` workload.
Those same-workload results are the numbers to quote.

## Why This Is Diagnostic Only

The original benchmark in this directory times individual warp-level MMA packet
forms.  It is useful for checking whether RTX 5090 / `sm_120` accepts and runs a
PTX instruction spelling, but its aggregate medians are not a clean workload
performance metric.

The confusing case was F16/F16 ordered sparse:

- `m16n8k16` sparse compared with one dense `m16n8k16` packet is near parity.
- `m16n8k32` sparse compared with two dense `m16n8k16` packets is near 2x.
- Taking a median across those two packet classes gives about 1.5x, but that is
  not the speedup of a same-size matrix workload.

The same-workload benchmark fixes this by choosing one matrix shape, giving
sparse and dense the same `M`, `N`, and `K`, and timing the tensor-core work for
that equal workload.

## Current Same-Workload Result To Quote

From `sm120_mma_workload_sweep/REPORT.md`:

| Mode | Overall median sparse/dense speedup | Overall median cycle ratio |
|---|---:|---:|
| Throughput | 1.946543x | 0.513732x |
| Latency | 1.962671x | 0.509509x |

For F16 inputs with F16 accumulation:

| Mode | Median sparse/dense speedup | Median cycle ratio |
|---|---:|---:|
| Throughput | 2.039617x | 0.490292x |
| Latency | 2.037302x | 0.490849x |

## What This Diagnostic Still Covers

This diagnostic remains useful for:

- verifying that baseline `sm_120` accepts warp-level `mma.sp` and
  `mma.sp::ordered_metadata` forms;
- identifying that `tcgen05.*` is not accepted for `.target sm_120`;
- checking compiler advisories for plain `.sp`;
- checking whether a sparse PTX spelling runs without launch/runtime failure;
- preserving exact sparse and dense PTX spellings in
  `results/sm120_mma_instruction_pairs.csv`.

## Coverage

The diagnostic sweep completed 176 sparse/dense packet rows with no build or
runtime failures on an NVIDIA GeForce RTX 5090.  It covers:

| Family | Sparse packet shapes | Input formats | Accumulator |
|---|---|---|---|
| F16/F32 | `m16n8k16`, `m16n8k32` | `f16 x f16` | `f32` |
| F16/F16 | `m16n8k16`, `m16n8k32` | `f16 x f16` | `f16` |
| BF16 | `m16n8k16`, `m16n8k32` | `bf16 x bf16` | `f32` |
| TF32 | `m16n8k8`, `m16n8k16` | `tf32 x tf32` | `f32` |
| FP8 | `m16n8k64` | `e4m3/e5m2 x e4m3/e5m2` | `f32` |
| INT8 | `m16n8k32`, `m16n8k64` | `u8/s8 x u8/s8` | `s32` |
| INT4 | `m16n8k64`, `m16n8k128` | `u4/s4 x u4/s4` | `s32` |

For INT8 and INT4, both normal and `.satfinite` accumulation are covered.  For
sparse instructions, both plain `.sp` and ordered metadata are covered.

## Diagnostic Findings

- The useful sparse spelling on RTX 5090 is
  `mma.sp::ordered_metadata.sync.aligned`.
- Plain `mma.sp.sync.aligned` is not a good performance path for this machine.
  `ptxas` emits a performance advisory, and the measured packet path is poor for
  F16/F16 and INT8.
- INT4 sparse packet forms are slower than their dense peers in both the
  instruction diagnostic and the same-workload sweep.
- `tcgen05.*` is not supported by `ptxas` for `.target sm_120`; those forms
  remain outside the RTX 5090 baseline path.

## Artifacts

| Path | Description |
|---|---|
| `sm120_mma_sparse.cu` | Instruction-packet CUDA diagnostic |
| `benchmark.py` | Builds/runs sparse and dense packet pairs |
| `summarize_results.py` | Generates diagnostic CSV summaries |
| `results/sm120_mma_full.csv` | Raw 176-row diagnostic sweep |
| `results/sm120_mma_instruction_pairs.csv` | Exact sparse/dense PTX spelling pairs |
| `results/sm120_mma_overall.csv` | Packet-level aggregate summary |
| `results/sm120_mma_summary_by_family.csv` | Packet-level family summary |
| `results/sm120_mma_extremes.csv` | Packet-level best/worst rows |

## Reproduction

```bash
cd sm120_mma_sparse
python3 benchmark.py --mode both --preset full \
    --sparse-variant both --output results/sm120_mma_full.csv --overwrite
python3 summarize_results.py results/sm120_mma_full.csv --out-dir results
make report
```

Use the same-workload report for performance conclusions:

```bash
cd ../sm120_mma_workload_sweep
python3 workload_benchmark.py \
    --mode both \
    --format-preset full \
    --size-preset full \
    --sparse-variant ordered \
    --target-packets-per-warp 4096 \
    --warmup-launches 3 \
    --output results/sm120_mma_workload_sweep.csv \
    --overwrite
python3 summarize_workload.py results/sm120_mma_workload_sweep.csv --out-dir results
make report
```
