# Blackwell Microbenchmarks

A collection of microbenchmarks for NVIDIA Blackwell (SM 100) GPUs, covering
memory throughput, latency, tensor core (UMMA) performance, and HBM-resident
elementwise throughput.

https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor

## RTX 5090

For the current `RTX 5090` support matrix, setup, exact commands, and known blockers, see
[RTX5090_RUNBOOK.md](RTX5090_RUNBOOK.md).

## Benchmarks

| Path | Purpose |
|---|---|
| `ldgsts_throughput/` | LDGSTS HBM throughput |
| `tma2d_throughput/` | TMA 2D HBM throughput |
| `ldgsts_latency/` | LDGSTS latency |
| `tma2d_latency/` | TMA 2D latency |
| `umma_throughput/` | UMMA tensor-core throughput |
| `umma_latency/` | UMMA tensor-core latency |
| `tcgen05_sparse_mma/` | Sparse vs dense Blackwell tcgen05 tensor-core throughput/latency |
| `sm120_mma_workload_sweep/` | RTX 5090 / sm_120 same-workload sparse vs dense tensor-core throughput/latency |
| `sm120_mma_sparse/` | RTX 5090 / sm_120 sparse MMA instruction diagnostic |
| `elementwise_throughput/` | fp32 HBM-resident activation/elementwise throughput |

<img width="1456" height="1231" alt="image" src="https://github.com/user-attachments/assets/104eabab-7c77-403f-b669-3402cc7a4b86" />


## Acknowledgements

Compute for this project is generously sponsored by **Nebius** and **Verda**.

<p align="center">
  <img src="assets/nebius_logo.jpeg" alt="Nebius" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/verda_cl_logo.jpeg" alt="Verda" height="100">
</p>
