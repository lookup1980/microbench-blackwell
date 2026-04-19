# RTX 5090 Runbook

This runbook captures the current status of this repo on an `NVIDIA GeForce RTX 5090`
tested with:

- Driver: `590.44.01`
- CUDA runtime/toolkit: `13.1`
- Compute capability: `sm_120`

The repo was originally written for B200-style Blackwell server parts and included
hardcoded `sm_100`/`sm_100a`, fixed SM counts, and fixed L2 sizing. The commands
below reflect the portability work already applied in this repo.

## Environment

Create and activate the local CUDA development environment:

```bash
./scripts/setup_local_env.sh
source .repo_env/activate.sh
```

Quick sanity checks:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nvcc --version
```

## What Works

These benchmarks were verified end-to-end on the `RTX 5090`:

- `elementwise_throughput`
- `ldgsts_latency`
- `tma2d_latency`
- `sm_l2_distance`
- `tools/gpc_query`

These benchmarks compile and execute, but their throughput collection path is blocked
on this machine by missing NVIDIA performance-counter permission:

- `ldgsts_throughput`
- `tma2d_throughput`
- `tma_throughput`
- `dsmem_throughput`

This benchmark partially works:

- `tma2dmcast_throughput`
  - clustered non-multicast paths worked
  - explicit multicast mode did not complete in the bounded verification run

These paths are currently blocked for reasons unrelated to the basic 5090 port:

- `cutlass_gemm_mainloop`
  - requires an external CUTLASS checkout at `../../cutlass` or `CUTLASS_PATH=/path/to/cutlass`

These benchmarks are not supported on the `RTX 5090` with their current code path:

- `umma_throughput`
- `umma_latency`

Reason: their inline assembly uses `tcgen05.*` and `.cta_group::*` instructions that
`ptxas` rejects for `.target sm_120`.

`umma_saturated_depth/` is plot/data output only in this repo, not a runnable source benchmark.

## Supported Commands

All commands below assume:

```bash
source .repo_env/activate.sh
```

### Elementwise Throughput

```bash
cd elementwise_throughput
make clean
make benchmark.out
./benchmark.out exp 1048576
python benchmark.py --ops exp --sizes 1048576
```

Expected result format:

```text
RESULT op=exp elements=1048576 ...
```

### LDGSTS Latency

```bash
cd ldgsts_latency
make clean
python benchmark.py --ctas 1 --threads 64 --load-types float4
```

Expected result format:

```text
[1/1] CTAs=1, threads=64, load=float4: ...
```

### TMA 2D Latency

```bash
cd tma2d_latency
make clean
python benchmark.py --ctas 1 --heights 2
```

Expected result format:

```text
[1/1] CTAs=1, h=2 ...
```

### SM L2 Distance

```bash
cd sm_l2_distance
make clean
make l2_pointer_chase
mkdir -p results
./l2_pointer_chase 1024 1 > results/distance.csv
```

Expected generated files:

- `results/distance.csv`
- `results/sm_info.csv`
- `results/latency_profile.csv`

### GPC Query Helper

```bash
cd tools
make clean
make gpc_query
./gpc_query
```

Expected output is a list of per-GPC SM counts. On the tested `RTX 5090`, it reported
an 11-group partition:

```text
[8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7]
```

## Throughput Benchmarks Blocked by Counter Permission

On this machine, Nsight Compute-based runs fail with:

```text
ERR_NVGPUCTRPERM
```

That affects the benchmark drivers in:

- `ldgsts_throughput`
- `tma2d_throughput`
- `tma_throughput`
- `dsmem_throughput`
- profiler-backed paths in `tma2dmcast_throughput`

If you enable GPU performance-counter access for your user, rerun the benchmark scripts
directly. The repo no longer hardcodes `sudo ncu`.

## CUTLASS Benchmark

To enable `cutlass_gemm_mainloop`, provide a compatible CUTLASS checkout:

```bash
export CUTLASS_PATH=/absolute/path/to/cutlass
cd cutlass_gemm_mainloop
make clean
make cutlass_gemm_bench.out
./sweep.sh
```

Without CUTLASS, the build now fails fast with a clear message.

## Unsupported UMMA Benchmarks

Both `umma_throughput` and `umma_latency` fail during compilation on the `RTX 5090`.
Representative `ptxas` errors:

```text
Instruction 'tcgen05.alloc' not supported on .target 'sm_120'
Instruction 'tcgen05.mma' not supported on .target 'sm_120'
```

Treat these as B200-specific until there is an `sm_120`-compatible rewrite.

## Notes

- The repo-local environment uses the local `nvcc` from `.repo_env`.
- Several benchmarks now query SM count, L2 size, and shared-memory limits at runtime.
- The current verified `RTX 5090` machine reported:
  - `170` SMs
  - `96.00 MiB` L2
  - `11` GPC groups in the helper/topology tooling
