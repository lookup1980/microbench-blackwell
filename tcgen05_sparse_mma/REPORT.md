# Sparse Blackwell Tensor Core Microbenchmark Plan and Report

Date: 2026-05-09

## Scope

This report defines the benchmark coverage for sparse Blackwell tensor-core MMA
instructions and their dense counterparts. The executable benchmark in this
directory targets the fifth-generation Tensor Core PTX instructions:

- `tcgen05.mma`
- `tcgen05.mma.sp`
- `tcgen05.mma.ws`
- `tcgen05.mma.ws.sp`

The sparse comparison is instruction-pair based:

| Sparse instruction | Dense comparison |
|---|---|
| `tcgen05.mma.sp` | `tcgen05.mma` |
| `tcgen05.mma.ws.sp` | `tcgen05.mma.ws` |

The existing `umma_throughput/` and `umma_latency/` benchmarks already cover a
subset of dense `tcgen05.mma` behavior. This directory keeps the sparse work
isolated while reusing the same measurement model: a pipelined throughput test
and a single-MMA latency test.

Implemented artifacts:

- `tcgen05_sparse_mma.cu`: compile-time-specialized CUDA benchmark
- `benchmark.py`: dense/sparse sweep driver with smoke and full presets
- `Makefile`: named-format build interface and report build targets
- `REPORT.md`, `REPORT.tex`, `REPORT.pdf`: report outputs

## Instruction Families

NVIDIA's PTX ISA lists the following sparse fifth-generation tensor-core
families.

| Family | Sparse PTX kind | Dense peer | Data formats covered here |
|---|---|---|---|
| Standard MMA | `tf32` | `tf32` | TF32 |
| Standard MMA | `f16` | `f16` | FP16, BF16 |
| Standard MMA | `f8f6f4` | `f8f6f4` | E4M3, E5M2, E2M3, E3M2, E2M1 combinations |
| Standard MMA | `i8` | `i8` | S8, U8 |
| Standard MMA, block scaled | `mxf8f6f4.block_scale` | same dense kind | MX E4M3/E5M2/E2M3/E3M2/E2M1 combinations |
| Standard MMA, block scaled | `mxf4.block_scale` | same dense kind | MXFP4 |
| Standard MMA, block scaled | `mxf4nvf4.block_scale` | same dense kind | MXFP4, NVFP4 |
| Weight-stationary MMA | `tf32` | `tf32` | TF32 |
| Weight-stationary MMA | `f16` | `f16` | FP16, BF16 |
| Weight-stationary MMA | `f8f6f4` | `f8f6f4` | E4M3, E5M2, E2M3, E3M2, E2M1 combinations |
| Weight-stationary MMA | `i8` | `i8` | S8, U8 |

`tcgen05.mma.ws.sp` is CTA-group 1 only. Standard `tcgen05.mma.sp` supports CTA
groups 1 and 2. The benchmark driver encodes these constraints in the sweep
matrix so unsupported combinations are skipped rather than failed late.

## Sparse Semantics

Sparse `tcgen05.mma.sp` and `tcgen05.mma.ws.sp` use sparse matrix `A`.

| Kind | Sparse granularity | Metadata pattern used by this microbenchmark |
|---|---:|---|
| `tf32` | 1:2 | `0b0100` nibble repeated |
| `f16` | 2:4 | `0b0100` nibble repeated |
| `f8f6f4` | 2:4 | `0b0100` nibble repeated, selector `0` |
| `mxf8f6f4` | 2:4 | `0b0100` nibble repeated, selector `0` |
| `i8` | 2:4 | `0b0100` nibble repeated, selector `0` |
| `mxf4` | 4:8 in pairs | `0b0100` nibble repeated, selector `0` |
| `mxf4nvf4` | 4:8 in pairs | `0b0100` nibble repeated, selector `0` |

The sparse path sets the PTX instruction descriptor sparsity bit and stores the
sparsity metadata in Tensor Memory. Dense peers use the same shape, data type,
CTA group, operand-placement mode, and timing path without metadata.

## Measurement Method

Throughput mode issues a burst of `MMA_DEPTH` operations in each iteration and
reports cycles per MMA:

```text
cycles_per_mma = total_clock_cycles / (MMA_DEPTH * NUM_ITERS)
```

Latency mode issues one MMA per sample, commits it to an mbarrier, waits for
completion, sorts the samples, and reports the median cycle count.

The benchmark reports both raw instruction throughput and dense-equivalent
operations. Sparse instructions perform half the stored `A` work, but their
mathematical dense-equivalent GEMM shape is still `M x N x K`. The CSV therefore
keeps both:

- `MathOpsPerMMA`: `2*M*N*K`
- `StoredAElements`: dense `M*K`, sparse `M*K/2`

## Expected Outputs

The benchmark driver writes CSVs with a `Sparsity` column so each sparse row can
be paired with its dense peer.

Throughput output fields:

```text
Instruction,Kind,AType,BType,Sparsity,ABLayout,CTAGroup,M,N,K,PipelineDepth,
Cycles,TotalMMAs,CyclesPerMMA,MathOpsPerMMA,OpsPerCycle,StoredAElements
```

Latency output fields:

```text
Instruction,Kind,AType,BType,Sparsity,ABLayout,CTAGroup,M,N,K,MedianCycles,
MathOpsPerMMA,StoredAElements
```

## Usage

Build one configuration:

```bash
cd tcgen05_sparse_mma
make clean
make MODE=throughput INSTRUCTION=mma KIND=f16 A_TYPE=f16 B_TYPE=f16 SPARSITY=1 MMA_M=128 MMA_N=64 MMA_K=32 MMA_DEPTH=256 CTA_GROUP=1 AB_LAYOUT=ss
./tcgen05_sparse_mma.out
```

Run paired dense/sparse sweeps:

```bash
cd tcgen05_sparse_mma
python3 benchmark.py --mode throughput --preset smoke --sparsity both --output sparse_tput.csv --overwrite
python3 benchmark.py --mode latency --preset smoke --sparsity both --output sparse_lat.csv --overwrite
```

Use `--preset full` for the full documented instruction/data-type matrix.

Build without running, useful on machines that have a CUDA toolchain but not an
SM100/SM110 GPU:

```bash
python3 benchmark.py --mode throughput --preset smoke --sparsity both \
    --instruction all --ab-layout ss --cta-group 1 --build-only \
    --output sparse_build_matrix.csv --overwrite
```

The default `CUDA_GENCODE` is:

```text
-gencode arch=compute_100a,code=sm_100a
```

This explicit `a` target is required. On the local CUDA 13.1 install,
`-arch=sm_100a` lowered to a plain `sm_100` target during ptxas, which rejects
`tcgen05.*`.

## Platform Notes

The code targets Blackwell server-family tensor-core instructions. The local RTX
5090 environment documented elsewhere in this repository reports `sm_120`, and
that target rejects `tcgen05.*` instructions. Full runtime validation therefore
requires an SM100/SM110-family target such as `sm_100a` or `sm_110a`.

Local validation completed:

- Python syntax check for `benchmark.py`
- TeX build to `REPORT.pdf`
- Representative `sm_100a` executable builds for:
  - dense `tcgen05.mma.kind::f16`
  - sparse `tcgen05.mma.sp.kind::f16`
  - sparse TS operand form
  - sparse `tcgen05.mma.ws.sp.kind::f16`
  - sparse block-scaled `mxf8f6f4`, `mxf4`, and `mxf4nvf4`
  - sparse CTA-group 2 `i8`
- Local runtime check produces the expected launch error on this host:
  `no kernel image is available for execution on the device`

## Source References

- NVIDIA PTX ISA 9.1, sparse `tcgen05.mma.sp` syntax and semantics:
  <https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-mma-sp>
- NVIDIA PTX ISA 9.1, sparse matrix formats:
  <https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#sparse-matrices>
- NVIDIA PTX ISA 9.1, instruction descriptor:
  <https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#instruction-descriptor>
- NVIDIA CUTLASS Blackwell functionality overview:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>
