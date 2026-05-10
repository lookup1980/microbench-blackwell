# SM120 Same-Workload Sparse MMA Report

Measurement date: 2026-05-09

This is the primary RTX 5090 sparse tensor-core report in this repo.  It uses
equal semantic matrix workloads:

```text
A[M,K] x B[K,N] -> C[M,N]
```

Sparse and dense rows use the same `M`, `N`, and `K`.  The reported speedups
therefore answer the practical tensor-core question: for the same matrix math,
how many GPU tensor-core cycles does ordered sparse use compared with dense?

## Executive Summary

The full sweep completed 288 same-workload dense/sparse rows with no build or
runtime failures.  The sweep covers F16/F32, F16/F16, BF16, TF32, all FP8
E4M3/E5M2 input combinations, all INT8 signed/unsigned combinations, and all
INT4 signed/unsigned combinations.  INT8 and INT4 include both normal and
`.satfinite` accumulation.

Only the recommended ordered sparse form is used for the performance result:

```text
mma.sp::ordered_metadata.sync.aligned
```

Overall same-workload result:

| Mode | Rows | Median sparse/dense speedup | Median sparse/dense cycle ratio |
|---|---:|---:|---:|
| Throughput | 144 | 1.946543x | 0.513732x |
| Latency | 144 | 1.962671x | 0.509509x |

For most non-INT4 formats, ordered sparse is approximately 2x faster than dense
for the same synthetic matrix workload.  INT4 is the exception and is slower
than dense in this benchmark.

## Direct Answer For F16/F16

For F16 inputs with F16 accumulation, ordered sparse is about 2x faster than
dense for the same workload:

| Mode | Rows | Median speedup | Median cycle ratio |
|---|---:|---:|---:|
| Throughput | 6 | 2.039617x | 0.490292x |
| Latency | 6 | 2.037302x | 0.490849x |

Interpretation:

- Throughput speedup `2.039617x` means sparse performs the same matrix math at
  about 2.04x the dense tensor-core math rate.
- Latency speedup `2.037302x` means the dependent-chain workload also completes
  at about 2.04x the dense rate.
- Cycle ratio near `0.49x` means sparse uses about half the measured GPU
  tensor-core cycles of dense for the same `A[M,K] x B[K,N]` work.

## Timing Boundary

The report counts GPU tensor-core timing only.

- Timing uses `clock64()` inside the CUDA kernel.
- Each row reports the median measured warp among `108 * 8 = 864` warps.
- The timed region includes only the repeated tensor-core MMA work for the
  synthetic workload.
- Python time, compilation time, process launch time, CUDA kernel launch time,
  host synchronization, host-device copies, CSV writing, and report generation
  are excluded.
- Each measured row runs warmup kernel launches first.  The full result used
  `WarmupLaunches = 3`; warmup launches are not included in `DenseCycles` or
  `SparseCycles`.

This is a tensor-core math benchmark, not a full GEMM benchmark.  It does not
include global-memory loads, shared-memory staging, sparse metadata generation,
tile iterators, epilogue stores, or numerical validation.

## Workload Definition

Every row decomposes the matrix workload into warp-level `m16n8k*` MMA packets:

```text
WorkloadPackets = (M / 16) * (N / 8) * (K / SparseSemanticK)
```

Sparse and dense use the same matrix shape.  Dense emits enough related
`mma.sync` work to cover the same semantic K.  For example, F16/F16 ordered
sparse uses sparse semantic `k32`, while dense uses two `k16` `mma.sync`
operations for each same-workload sparse packet.

This is different from the older instruction-packet diagnostic report, which
mixed one-dense-instruction and two-dense-instruction cases.  Those mixed
instruction medians are not used here.

## Matrix Size Sweep

The sweep uses square `M` and `N` sizes and scales `K` by format:

| Scale | M | N |
|---:|---:|---:|
| 1 | 16 | 16 |
| 2 | 32 | 32 |
| 4 | 64 | 64 |
| 8 | 128 | 128 |
| 16 | 256 | 256 |
| 32 | 512 | 512 |

Maximum A/B shapes in the report:

| Family | Max A shape | Max B shape |
|---|---:|---:|
| F16/F32 | `512 x 1024` | `1024 x 512` |
| F16/F16 | `512 x 1024` | `1024 x 512` |
| BF16 | `512 x 1024` | `1024 x 512` |
| TF32 | `512 x 512` | `512 x 512` |
| FP8 | `512 x 2048` | `2048 x 512` |
| INT8 | `512 x 2048` | `2048 x 512` |
| INT4 | `512 x 4096` | `4096 x 512` |

## Throughput Results By Family

| Family | Rows | Median speedup | Min | Max | Median cycle ratio |
|---|---:|---:|---:|---:|---:|
| F16/F32 | 6 | 1.993370x | 1.942656x | 1.999924x | 0.501664x |
| F16/F16 | 6 | 2.039617x | 1.607768x | 2.046796x | 0.490292x |
| BF16 | 6 | 1.992143x | 1.942624x | 1.999932x | 0.501972x |
| TF32 | 6 | 1.992742x | 1.949431x | 2.001675x | 0.501821x |
| FP8 | 24 | 1.995070x | 1.940709x | 2.324557x | 0.501236x |
| INT8 | 48 | 1.989737x | 1.489865x | 2.000026x | 0.502579x |
| INT4 | 48 | 0.670316x | 0.362528x | 0.969383x | 1.506676x |

## Latency Results By Family

| Family | Rows | Median speedup | Min | Max | Median cycle ratio |
|---|---:|---:|---:|---:|---:|
| F16/F32 | 6 | 1.998498x | 1.961920x | 2.001599x | 0.500376x |
| F16/F16 | 6 | 2.037302x | 1.613801x | 2.047014x | 0.490849x |
| BF16 | 6 | 1.998654x | 1.961920x | 2.230839x | 0.500336x |
| TF32 | 6 | 1.999124x | 1.963081x | 2.002195x | 0.500219x |
| FP8 | 24 | 1.999210x | 1.961895x | 2.579650x | 0.500198x |
| INT8 | 48 | 1.992542x | 1.608682x | 2.001129x | 0.501871x |
| INT4 | 48 | 0.613358x | 0.344716x | 0.867011x | 1.650141x |

## Results By Matrix Size

Aggregated across all families, including slower INT4 rows:

| Scale | M=N | Throughput median speedup | Latency median speedup |
|---:|---:|---:|---:|
| 1 | 16 | 1.523300x | 1.611309x |
| 2 | 32 | 1.894440x | 1.928605x |
| 4 | 64 | 1.970939x | 1.988479x |
| 8 | 128 | 1.989851x | 1.996520x |
| 16 | 256 | 1.998695x | 1.998712x |
| 32 | 512 | 1.999824x | 1.999838x |

The small-size rows include more loop and dependency overhead per unit of work.
From `64 x 64` upward, the non-INT4 formats are close to the expected 2x
same-workload tensor-core result.

## Extremes

| Mode | Best row | Speedup | Worst row | Speedup |
|---|---|---:|---|---:|
| Throughput | `fp8:e4m3xe4m3:m512n512k2048` | 2.324557x | `int4:u4xu4:m256n256k2048` | 0.362528x |
| Latency | `fp8:e4m3xe5m2:m512n512k2048` | 2.579650x | `int4:u4xu4:m32n32k256:sat` | 0.344716x |

## Artifacts

| Path | Description |
|---|---|
| `sm120_mma_workload.cu` | CUDA same-workload tensor-core benchmark with warmup launches |
| `workload_benchmark.py` | Builds separate dense/sparse executables and runs matrix-size sweeps |
| `summarize_workload.py` | Generates aggregate CSV summaries |
| `results/sm120_mma_workload_sweep.csv` | Full 288-row raw workload sweep |
| `results/sm120_mma_workload_overall.csv` | Overall throughput/latency summary |
| `results/sm120_mma_workload_summary_by_family.csv` | Per-family summaries |
| `results/sm120_mma_workload_summary_by_size.csv` | Per-size summaries |
| `results/sm120_mma_workload_extremes.csv` | Best/worst rows |
| `REPORT.md`, `REPORT.tex`, `REPORT.pdf` | Human-readable reports |

## Reproduction

```bash
cd sm120_mma_workload_sweep
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

## Validation

- Full run completed 288 rows with no build or runtime failures.
- Every measured row used three warmup launches before the measured launch.
- Timing is from GPU-side `clock64()` inside the measured kernel.
- Python scripts compile with `python3 -m py_compile`.
- TeX report builds to `REPORT.pdf`.
