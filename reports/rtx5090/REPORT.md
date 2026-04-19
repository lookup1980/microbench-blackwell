# RTX 5090 Microbenchmark Report

Date: `2026-04-19`

This report covers two outputs:

1. A measured status report for the local `NVIDIA GeForce RTX 5090`
2. A comparison between this `RTX 5090` and the repo's checked-in Blackwell baselines

## Test Environment

- GPU: `NVIDIA GeForce RTX 5090`
- Compute capability: `sm_120`
- Driver: `590.44.01`
- Max SM clock reported by `nvidia-smi`: `3150 MHz`
- Memory: `32607 MiB`
- Local toolkit used for builds: repo-local CUDA `13.1.115`

Raw outputs are under [reports/rtx5090/results](/home/mgu/github/microbench-blackwell/reports/rtx5090/results).

## 1. Report For 5090

### Quantitative Results

`elementwise_throughput` completed in event-timed mode across all 6 default ops. Output: [elementwise_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/elementwise_5090.csv:1)

| Op | Effective GB/s | Effective GOps/s |
|---|---:|---:|
| `exp` | 1226.5 | 153.3 |
| `tanh` | 1308.8 | 163.6 |
| `rsqrt` | 1287.2 | 160.9 |
| `add` | 1299.4 | 162.4 |
| `multiply` | 1328.8 | 166.1 |
| `relu` | 1312.5 | 164.1 |

Average elementwise throughput was `1293.9 GB/s`, with `multiply` the fastest and `exp` the slowest.

`ldgsts_latency` completed for all 48 configurations. Output: [ldgsts_latency_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/ldgsts_latency_5090.csv:1)

- Best measured median latency: `368.9 ns` at `CTAsPerSM=2`, `ThreadsPerBlock=32`, `LoadType=float`
- Median across the full sweep: `608.3 ns`
- Worst measured median latency: `1434.0 ns` at `CTAsPerSM=3`, `ThreadsPerBlock=256`, `LoadType=float4`

`tma2d_latency` completed for 8 of 9 default heights. Output: [tma2d_latency_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2d_latency_5090.csv:1)

- Best measured median latency: `465.1 ns` at `128x1` (`512 B`)
- Median across valid runs: `1304.8 ns`
- Worst measured median latency: `7869.5 ns` at `128x128` (`64 KiB`)
- `128x256` (`128 KiB`) failed on this GPU with:
  `Assertion 'res == CUDA_SUCCESS' failed`
  Capture: [tma2d_latency_5090_h256_retry.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2d_latency_5090_h256_retry.csv:1)

`sm_l2_distance` completed. Outputs:

- [sm_l2_distance_pairs_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/sm_l2_distance_pairs_5090.csv:1)
- [sm_l2_sm_info_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/sm_l2_sm_info_5090.csv:1)
- [sm_l2_latency_profile_5090.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/sm_l2_latency_profile_5090.csv:1)

The run discovered:

- `170` SMs
- `96.0 MiB` L2
- `11` GPC groups from the pointer-chase discovery path
- GPC sizes: `16,16,16,16,16,16,16,16,14,14,14`

`tools/gpc_query` also ran successfully. Output: [gpc_query_5090.txt](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/gpc_query_5090.txt:1)

### Benchmarks That Ran But Did Not Produce Quantitative Throughput

`tma_throughput` built and ran successfully in a direct binary probe, but it does not emit bandwidth numbers without `ncu`. Probe artifacts:

- [tma_throughput_probe.exit](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma_throughput_probe.exit:1)
- [tma_throughput_probe.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma_throughput_probe.stdout:1)

`dsmem_throughput` ran successfully in a supported direct probe with `LOAD_T=float4`, `ACCESS_MODE=local`, but also does not emit standalone throughput numbers. Probe artifacts:

- [dsmem_probe.exit](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/dsmem_probe.exit:1)
- [dsmem_probe.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/dsmem_probe.stdout:1)

`tma2dmcast_throughput` completed small-shape direct probes for both:

- non-multicast baseline: [tma2dmcast_baseline.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2dmcast_baseline.stdout:1)
- explicit multicast: [tma2dmcast_mcast.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2dmcast_mcast.stdout:1)

This is weaker evidence than a full sweep. It shows the small-shape explicit multicast path can launch on the `5090`, but not that the broader multicast sweep is healthy.

### Blocked Or Unsupported On This Machine

`ldgsts_throughput` and `tma2d_throughput` both hit the same machine-level profiler block:

`ERR_NVGPUCTRPERM`

Captured probes:

- [ldgsts_ncu_direct.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/ldgsts_ncu_direct.stdout:1)
- [tma2d_ncu_direct.stdout](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2d_ncu_direct.stdout:1)

`cutlass_gemm_mainloop` is blocked by a missing external dependency:

- [cutlass_probe.stderr](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/cutlass_probe.stderr:1)

`umma_throughput` and `umma_latency` are not buildable for `sm_120`. `ptxas` rejects the `tcgen05.*` and `.cta_group::*` instruction path. Captures:

- [umma_throughput_probe.stderr](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/umma_throughput_probe.stderr:1)
- [umma_latency_probe.stderr](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/umma_latency_probe.stderr:1)

## 2. Comparison Between 5090 And Blackwell

The Blackwell comparison below uses the repo's checked-in baseline CSVs:

- [compare_mem_latency/ldgsts_lat_results.csv](/home/mgu/github/microbench-blackwell/compare_mem_latency/ldgsts_lat_results.csv:1)
- [compare_mem_latency/tma2d_lat_results.csv](/home/mgu/github/microbench-blackwell/compare_mem_latency/tma2d_lat_results.csv:1)

Derived comparison tables:

- [ldgsts_latency_vs_blackwell.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/ldgsts_latency_vs_blackwell.csv:1)
- [tma2d_latency_vs_blackwell.csv](/home/mgu/github/microbench-blackwell/reports/rtx5090/results/tma2d_latency_vs_blackwell.csv:1)

### LDGSTS Latency

Across all 48 shared `ldgsts_latency` points:

- median `5090 / Blackwell` latency ratio: `0.93x`
- mean `5090 / Blackwell` latency ratio: `0.98x`

Interpretation: the `5090` is roughly on par overall, slightly lower latency on median, but with a wider spread. At very small bytes-in-flight it is often better than the checked-in Blackwell numbers, while at larger bytes-in-flight it often becomes worse.

Examples:

- Best relative win for `5090`: `CTAs=2`, `Threads=32`, `float` at `368.9 ns` vs Blackwell `985.2 ns` (`0.37x`)
- Worst relative loss for `5090`: `CTAs=3`, `Threads=256`, `float2` at `1100.0 ns` vs Blackwell `605.6 ns` (`1.82x`)

By bytes-in-flight median, the crossover is clear:

- `0 KiB`: `5090` better (`0.67x`)
- `1 KiB`: near parity (`1.04x`)
- `4 KiB`: parity (`1.00x`)
- `6 KiB`, `8 KiB`, `12 KiB`, `16 KiB`: `5090` worse (`1.13x` to `1.31x`)

### TMA 2D Latency

Across the 8 shared `tma2d_latency` points:

- median `5090 / Blackwell` latency ratio: `1.30x`
- mean `5090 / Blackwell` latency ratio: `1.30x`

Interpretation: `5090` is better only on the smallest tiles, then degrades relative to the checked-in Blackwell results as tile size grows.

Examples:

- `128x1` (`512 B`): `5090` `465.1 ns` vs Blackwell `1014.2 ns` (`0.46x`)
- `128x2` (`1 KiB`): `5090` `511.4 ns` vs Blackwell `719.1 ns` (`0.71x`)
- `128x8` (`4 KiB`): `5090` `1234.6 ns` vs Blackwell `938.9 ns` (`1.31x`)
- `128x128` (`64 KiB`): `5090` `7869.5 ns` vs Blackwell `2750.6 ns` (`2.86x`)

The `5090` also failed the `128x256` (`128 KiB`) case that exists in the checked-in Blackwell CSV.

### Throughput Comparison

I could not produce a measured 5090 throughput comparison on this machine because the relevant benchmarks require Nsight Compute counters and the host denies them with `ERR_NVGPUCTRPERM`.

For context, the checked-in Blackwell peaks are:

- `ldgsts_throughput`: `6692.2 GB/s`
- `tma2d_throughput`: `7260.0 GB/s`

Those are from the repo baselines only, not from any local 5090 measurement.

### Architectural And Compatibility Differences

- The repo's UMMA path is Blackwell-specific in a stronger sense than the memory benchmarks. It does not compile for `sm_120`.
- The memory and topology microbenchmarks are substantially portable after removing B200-specific constants.
- `5090` reports `170` SMs and `96 MiB` L2, while much of the original repo assumed B200-like `148` SMs and a larger L2.

## Bottom Line

On this `RTX 5090`, the memory latency families and topology probes are usable today and produce meaningful data. Compared with the repo's checked-in Blackwell baselines, `ldgsts` latency is roughly comparable overall, while `tma2d` latency is only better on the very smallest tiles and becomes materially worse on larger transfers. Throughput comparison is still blocked by host profiler permissions, and the UMMA microbenchmarks are architecturally incompatible with `sm_120` in their current form.
