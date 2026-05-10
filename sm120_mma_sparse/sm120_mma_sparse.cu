// RTX 5090 / sm_120 sparse-vs-dense warp-level tensor-core microbenchmark.
//
// This benchmark intentionally targets baseline sm_120 instructions:
// mma.sync.aligned, mma.sp.sync.aligned, and
// mma.sp::ordered_metadata.sync.aligned.  It does not use tcgen05, wgmma, or
// sm_120a-only block-scaled FP4/FP6 forms.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MODE_THROUGHPUT 0
#define MODE_LATENCY 1

#define FAM_F16 0
#define FAM_BF16 1
#define FAM_TF32 2
#define FAM_FP8 3
#define FAM_INT8 4
#define FAM_INT4 5
#define FAM_F16F16 6

#define TYPE_F16 0
#define TYPE_BF16 1
#define TYPE_TF32 2
#define TYPE_E4M3 3
#define TYPE_E5M2 4
#define TYPE_U8 5
#define TYPE_S8 6
#define TYPE_U4 7
#define TYPE_S4 8

#ifndef BENCH_MODE
#error "BENCH_MODE must be defined"
#endif
#ifndef FAMILY_ID
#error "FAMILY_ID must be defined"
#endif
#ifndef A_TYPE_ID
#error "A_TYPE_ID must be defined"
#endif
#ifndef B_TYPE_ID
#error "B_TYPE_ID must be defined"
#endif
#ifndef SPARSITY
#error "SPARSITY must be defined"
#endif
#ifndef ORDERED_METADATA
#error "ORDERED_METADATA must be defined"
#endif
#ifndef SATFINITE
#error "SATFINITE must be defined"
#endif
#ifndef MMA_K
#error "MMA_K must be defined"
#endif
#ifndef ITERATIONS
#error "ITERATIONS must be defined"
#endif
#ifndef THROUGHPUT_UNROLL
#error "THROUGHPUT_UNROLL must be defined"
#endif
#ifndef BLOCKS
#error "BLOCKS must be defined"
#endif
#ifndef WARPS_PER_BLOCK
#error "WARPS_PER_BLOCK must be defined"
#endif

#if BENCH_MODE != MODE_THROUGHPUT && BENCH_MODE != MODE_LATENCY
#error "BENCH_MODE must be 0=throughput or 1=latency"
#endif

#if FAMILY_ID == FAM_F16
#if A_TYPE_ID != TYPE_F16 || B_TYPE_ID != TYPE_F16
#error "F16 family requires A_TYPE=f16 and B_TYPE=f16"
#endif
#if SPARSITY && MMA_K != 16 && MMA_K != 32
#error "Sparse F16 supports semantic K=16 or K=32 in this benchmark"
#endif
#define FAMILY_NAME "f16"
#define A_TYPE_NAME "f16"
#define B_TYPE_NAME "f16"
#define DENSE_BASE_K 16
#define ELEMENT_BITS 16
#define OUTPUT_IS_FLOAT 1
#elif FAMILY_ID == FAM_F16F16
#if A_TYPE_ID != TYPE_F16 || B_TYPE_ID != TYPE_F16
#error "F16F16 family requires A_TYPE=f16 and B_TYPE=f16"
#endif
#if SPARSITY && MMA_K != 16 && MMA_K != 32
#error "Sparse F16F16 supports semantic K=16 or K=32 in this benchmark"
#endif
#define FAMILY_NAME "f16f16"
#define A_TYPE_NAME "f16"
#define B_TYPE_NAME "f16"
#define DENSE_BASE_K 16
#define ELEMENT_BITS 16
#define OUTPUT_IS_FLOAT 0
#elif FAMILY_ID == FAM_BF16
#if A_TYPE_ID != TYPE_BF16 || B_TYPE_ID != TYPE_BF16
#error "BF16 family requires A_TYPE=bf16 and B_TYPE=bf16"
#endif
#if SPARSITY && MMA_K != 16 && MMA_K != 32
#error "Sparse BF16 supports semantic K=16 or K=32 in this benchmark"
#endif
#define FAMILY_NAME "bf16"
#define A_TYPE_NAME "bf16"
#define B_TYPE_NAME "bf16"
#define DENSE_BASE_K 16
#define ELEMENT_BITS 16
#define OUTPUT_IS_FLOAT 1
#elif FAMILY_ID == FAM_TF32
#if A_TYPE_ID != TYPE_TF32 || B_TYPE_ID != TYPE_TF32
#error "TF32 family requires A_TYPE=tf32 and B_TYPE=tf32"
#endif
#if SPARSITY && MMA_K != 8 && MMA_K != 16
#error "Sparse TF32 supports semantic K=8 or K=16 in this benchmark"
#endif
#define FAMILY_NAME "tf32"
#define A_TYPE_NAME "tf32"
#define B_TYPE_NAME "tf32"
#define DENSE_BASE_K 8
#define ELEMENT_BITS 32
#define OUTPUT_IS_FLOAT 1
#elif FAMILY_ID == FAM_FP8
#if A_TYPE_ID != TYPE_E4M3 && A_TYPE_ID != TYPE_E5M2
#error "FP8 family requires A_TYPE=e4m3 or e5m2 on baseline sm_120"
#endif
#if B_TYPE_ID != TYPE_E4M3 && B_TYPE_ID != TYPE_E5M2
#error "FP8 family requires B_TYPE=e4m3 or e5m2 on baseline sm_120"
#endif
#if SPARSITY && MMA_K != 64
#error "Sparse FP8 supports semantic K=64 in this benchmark"
#endif
#define FAMILY_NAME "fp8"
#define DENSE_BASE_K 32
#define ELEMENT_BITS 8
#define OUTPUT_IS_FLOAT 1
#elif FAMILY_ID == FAM_INT8
#if A_TYPE_ID != TYPE_U8 && A_TYPE_ID != TYPE_S8
#error "INT8 family requires A_TYPE=u8 or s8"
#endif
#if B_TYPE_ID != TYPE_U8 && B_TYPE_ID != TYPE_S8
#error "INT8 family requires B_TYPE=u8 or s8"
#endif
#if SPARSITY && MMA_K != 32 && MMA_K != 64
#error "Sparse INT8 supports semantic K=32 or K=64 in this benchmark"
#endif
#define FAMILY_NAME "int8"
#define DENSE_BASE_K 32
#define ELEMENT_BITS 8
#define OUTPUT_IS_FLOAT 0
#elif FAMILY_ID == FAM_INT4
#if A_TYPE_ID != TYPE_U4 && A_TYPE_ID != TYPE_S4
#error "INT4 family requires A_TYPE=u4 or s4"
#endif
#if B_TYPE_ID != TYPE_U4 && B_TYPE_ID != TYPE_S4
#error "INT4 family requires B_TYPE=u4 or s4"
#endif
#if SPARSITY && MMA_K != 64 && MMA_K != 128
#error "Sparse INT4 supports semantic K=64 or K=128 in this benchmark"
#endif
#define FAMILY_NAME "int4"
#define DENSE_BASE_K 64
#define ELEMENT_BITS 4
#define OUTPUT_IS_FLOAT 0
#else
#error "Unsupported FAMILY_ID"
#endif

#if FAMILY_ID == FAM_FP8 || FAMILY_ID == FAM_INT8 || FAMILY_ID == FAM_INT4
#if A_TYPE_ID == TYPE_E4M3
#define A_TYPE_NAME "e4m3"
#elif A_TYPE_ID == TYPE_E5M2
#define A_TYPE_NAME "e5m2"
#elif A_TYPE_ID == TYPE_U8
#define A_TYPE_NAME "u8"
#elif A_TYPE_ID == TYPE_S8
#define A_TYPE_NAME "s8"
#elif A_TYPE_ID == TYPE_U4
#define A_TYPE_NAME "u4"
#elif A_TYPE_ID == TYPE_S4
#define A_TYPE_NAME "s4"
#endif

#if B_TYPE_ID == TYPE_E4M3
#define B_TYPE_NAME "e4m3"
#elif B_TYPE_ID == TYPE_E5M2
#define B_TYPE_NAME "e5m2"
#elif B_TYPE_ID == TYPE_U8
#define B_TYPE_NAME "u8"
#elif B_TYPE_ID == TYPE_S8
#define B_TYPE_NAME "s8"
#elif B_TYPE_ID == TYPE_U4
#define B_TYPE_NAME "u4"
#elif B_TYPE_ID == TYPE_S4
#define B_TYPE_NAME "s4"
#endif
#endif

#if !SPARSITY && MMA_K % DENSE_BASE_K != 0
#error "Dense semantic MMA_K must be a multiple of the dense base instruction K"
#endif

constexpr int kDenseRepeats = SPARSITY ? 1 : (MMA_K / DENSE_BASE_K);
constexpr int kSparseInstructionsPerPacket = SPARSITY ? 1 : 0;
constexpr int kDenseInstructionsPerPacket = SPARSITY ? 0 : kDenseRepeats;
constexpr int kPacketsPerWarp =
    ITERATIONS * (BENCH_MODE == MODE_THROUGHPUT ? THROUGHPUT_UNROLL : 1);
constexpr int kWarpsTotal = BLOCKS * WARPS_PER_BLOCK;

static_assert(WARPS_PER_BLOCK > 0, "WARPS_PER_BLOCK must be positive");
static_assert(THROUGHPUT_UNROLL > 0, "THROUGHPUT_UNROLL must be positive");
static_assert(ITERATIONS > 0, "ITERATIONS must be positive");

__device__ __forceinline__ uint32_t lane_id() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

#if OUTPUT_IS_FLOAT

__device__ __forceinline__ void issue_dense_fp(float& d0, float& d1, float& d2,
                                               float& d3, uint32_t a0,
                                               uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0,
                                               uint32_t b1) {
#if FAMILY_ID == FAM_F16
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif FAMILY_ID == FAM_BF16
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif FAMILY_ID == FAM_TF32
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif FAMILY_ID == FAM_FP8
#if A_TYPE_ID == TYPE_E4M3 && B_TYPE_ID == TYPE_E4M3
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif A_TYPE_ID == TYPE_E4M3 && B_TYPE_ID == TYPE_E5M2
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif A_TYPE_ID == TYPE_E5M2 && B_TYPE_ID == TYPE_E4M3
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
#endif
}

__device__ __forceinline__ void issue_sparse_fp(float& d0, float& d1, float& d2,
                                                float& d3, uint32_t a0,
                                                uint32_t a1, uint32_t a2,
                                                uint32_t a3, uint32_t b0,
                                                uint32_t b1, uint32_t b2,
                                                uint32_t b3, uint32_t meta) {
#if FAMILY_ID == FAM_F16
#if MMA_K == 16
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#elif FAMILY_ID == FAM_BF16
#if MMA_K == 16
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#elif FAMILY_ID == FAM_TF32
#if MMA_K == 8
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#elif FAMILY_ID == FAM_FP8
#if A_TYPE_ID == TYPE_E4M3 && B_TYPE_ID == TYPE_E4M3
#define FP8_SP_SUFFIX "f32.e4m3.e4m3.f32"
#elif A_TYPE_ID == TYPE_E4M3 && B_TYPE_ID == TYPE_E5M2
#define FP8_SP_SUFFIX "f32.e4m3.e5m2.f32"
#elif A_TYPE_ID == TYPE_E5M2 && B_TYPE_ID == TYPE_E4M3
#define FP8_SP_SUFFIX "f32.e5m2.e4m3.f32"
#else
#define FP8_SP_SUFFIX "f32.e5m2.e5m2.f32"
#endif
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col." FP8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col." FP8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
}

#else

__device__ __forceinline__ void issue_dense_int(uint32_t& d0, uint32_t& d1,
                                                uint32_t& d2, uint32_t& d3,
                                                uint32_t a0, uint32_t a1,
                                                uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
#if FAMILY_ID == FAM_F16F16
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif FAMILY_ID == FAM_INT8
#if A_TYPE_ID == TYPE_U8 && B_TYPE_ID == TYPE_U8
#if SATFINITE
#define INT8_DENSE_SUFFIX "satfinite.s32.u8.u8.s32"
#else
#define INT8_DENSE_SUFFIX "s32.u8.u8.s32"
#endif
#elif A_TYPE_ID == TYPE_U8 && B_TYPE_ID == TYPE_S8
#if SATFINITE
#define INT8_DENSE_SUFFIX "satfinite.s32.u8.s8.s32"
#else
#define INT8_DENSE_SUFFIX "s32.u8.s8.s32"
#endif
#elif A_TYPE_ID == TYPE_S8 && B_TYPE_ID == TYPE_U8
#if SATFINITE
#define INT8_DENSE_SUFFIX "satfinite.s32.s8.u8.s32"
#else
#define INT8_DENSE_SUFFIX "s32.s8.u8.s32"
#endif
#else
#if SATFINITE
#define INT8_DENSE_SUFFIX "satfinite.s32.s8.s8.s32"
#else
#define INT8_DENSE_SUFFIX "s32.s8.s8.s32"
#endif
#endif
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col." INT8_DENSE_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif FAMILY_ID == FAM_INT4
#if A_TYPE_ID == TYPE_U4 && B_TYPE_ID == TYPE_U4
#if SATFINITE
#define INT4_DENSE_SUFFIX "satfinite.s32.u4.u4.s32"
#else
#define INT4_DENSE_SUFFIX "s32.u4.u4.s32"
#endif
#elif A_TYPE_ID == TYPE_U4 && B_TYPE_ID == TYPE_S4
#if SATFINITE
#define INT4_DENSE_SUFFIX "satfinite.s32.u4.s4.s32"
#else
#define INT4_DENSE_SUFFIX "s32.u4.s4.s32"
#endif
#elif A_TYPE_ID == TYPE_S4 && B_TYPE_ID == TYPE_U4
#if SATFINITE
#define INT4_DENSE_SUFFIX "satfinite.s32.s4.u4.s32"
#else
#define INT4_DENSE_SUFFIX "s32.s4.u4.s32"
#endif
#else
#if SATFINITE
#define INT4_DENSE_SUFFIX "satfinite.s32.s4.s4.s32"
#else
#define INT4_DENSE_SUFFIX "s32.s4.s4.s32"
#endif
#endif
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col." INT4_DENSE_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void issue_sparse_int(uint32_t& d0, uint32_t& d1,
                                                 uint32_t& d2, uint32_t& d3,
                                                 uint32_t a0, uint32_t a1,
                                                 uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1,
                                                 uint32_t b2, uint32_t b3,
                                                 uint32_t meta) {
#if FAMILY_ID == FAM_F16F16
#if MMA_K == 16
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%0,%1}, %6, 0x0;"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4,%5}, {%0,%1}, %6, 0x0;"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%0,%1}, %10, 0x0;"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%0,%1}, %10, 0x0;"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#elif FAMILY_ID == FAM_INT8
#if A_TYPE_ID == TYPE_U8 && B_TYPE_ID == TYPE_U8
#if SATFINITE
#define INT8_SP_SUFFIX "satfinite.s32.u8.u8.s32"
#else
#define INT8_SP_SUFFIX "s32.u8.u8.s32"
#endif
#elif A_TYPE_ID == TYPE_U8 && B_TYPE_ID == TYPE_S8
#if SATFINITE
#define INT8_SP_SUFFIX "satfinite.s32.u8.s8.s32"
#else
#define INT8_SP_SUFFIX "s32.u8.s8.s32"
#endif
#elif A_TYPE_ID == TYPE_S8 && B_TYPE_ID == TYPE_U8
#if SATFINITE
#define INT8_SP_SUFFIX "satfinite.s32.s8.u8.s32"
#else
#define INT8_SP_SUFFIX "s32.s8.u8.s32"
#endif
#else
#if SATFINITE
#define INT8_SP_SUFFIX "satfinite.s32.s8.s8.s32"
#else
#define INT8_SP_SUFFIX "s32.s8.s8.s32"
#endif
#endif
#if MMA_K == 32
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col." INT8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col." INT8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col." INT8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col." INT8_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#elif FAMILY_ID == FAM_INT4
#if A_TYPE_ID == TYPE_U4 && B_TYPE_ID == TYPE_U4
#if SATFINITE
#define INT4_SP_SUFFIX "satfinite.s32.u4.u4.s32"
#else
#define INT4_SP_SUFFIX "s32.u4.u4.s32"
#endif
#elif A_TYPE_ID == TYPE_U4 && B_TYPE_ID == TYPE_S4
#if SATFINITE
#define INT4_SP_SUFFIX "satfinite.s32.u4.s4.s32"
#else
#define INT4_SP_SUFFIX "s32.u4.s4.s32"
#endif
#elif A_TYPE_ID == TYPE_S4 && B_TYPE_ID == TYPE_U4
#if SATFINITE
#define INT4_SP_SUFFIX "satfinite.s32.s4.u4.s32"
#else
#define INT4_SP_SUFFIX "s32.s4.u4.s32"
#endif
#else
#if SATFINITE
#define INT4_SP_SUFFIX "satfinite.s32.s4.s4.s32"
#else
#define INT4_SP_SUFFIX "s32.s4.s4.s32"
#endif
#endif
#if MMA_K == 64
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col." INT4_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col." INT4_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3}, %8, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta));
#endif
#else
#if ORDERED_METADATA
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col." INT4_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k128.row.col." INT4_SP_SUFFIX
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(b2), "r"(b3), "r"(meta));
#endif
#endif
#endif
}

#endif

__global__ __launch_bounds__(WARPS_PER_BLOCK * 32)
void bench_kernel(unsigned long long* cycles, uint64_t* sink) {
    const int tid = threadIdx.x;
    const int warp_in_block = tid >> 5;
    const int lane = static_cast<int>(lane_id());
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    const uint32_t seed = 0x9e3779b9u ^ (warp_global * 0x10001u) ^ lane;
    uint32_t a0 = seed + 0x101u;
    uint32_t a1 = seed + 0x203u;
    uint32_t a2 = seed + 0x307u;
    uint32_t a3 = seed + 0x40bu;
    uint32_t b0 = seed + 0x503u;
    uint32_t b1 = seed + 0x607u;
    [[maybe_unused]] uint32_t b2 = seed + 0x709u;
    [[maybe_unused]] uint32_t b3 = seed + 0x80fu;
    [[maybe_unused]] uint32_t meta = 0u;

#if BENCH_MODE == MODE_THROUGHPUT
#if OUTPUT_IS_FLOAT
    float d0[THROUGHPUT_UNROLL];
    float d1[THROUGHPUT_UNROLL];
    float d2[THROUGHPUT_UNROLL];
    float d3[THROUGHPUT_UNROLL];
#else
    uint32_t d0[THROUGHPUT_UNROLL];
    uint32_t d1[THROUGHPUT_UNROLL];
    uint32_t d2[THROUGHPUT_UNROLL];
    uint32_t d3[THROUGHPUT_UNROLL];
#endif
    #pragma unroll
    for (int u = 0; u < THROUGHPUT_UNROLL; ++u) {
#if OUTPUT_IS_FLOAT
        d0[u] = static_cast<float>((lane & 7) + 1 + u);
        d1[u] = static_cast<float>((lane & 7) + 2 + u);
        d2[u] = static_cast<float>((lane & 7) + 3 + u);
        d3[u] = static_cast<float>((lane & 7) + 4 + u);
#else
        d0[u] = static_cast<uint32_t>((lane & 7) + 1 + u);
        d1[u] = static_cast<uint32_t>((lane & 7) + 2 + u);
        d2[u] = static_cast<uint32_t>((lane & 7) + 3 + u);
        d3[u] = static_cast<uint32_t>((lane & 7) + 4 + u);
#endif
    }
#else
#if OUTPUT_IS_FLOAT
    float d0 = static_cast<float>((lane & 7) + 1);
    float d1 = static_cast<float>((lane & 7) + 2);
    float d2 = static_cast<float>((lane & 7) + 3);
    float d3 = static_cast<float>((lane & 7) + 4);
#else
    uint32_t d0 = (lane & 7) + 1;
    uint32_t d1 = (lane & 7) + 2;
    uint32_t d2 = (lane & 7) + 3;
    uint32_t d3 = (lane & 7) + 4;
#endif
#endif

    __syncwarp();
    unsigned long long start = clock64();

#if BENCH_MODE == MODE_LATENCY
    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; ++iter) {
#if OUTPUT_IS_FLOAT
#if SPARSITY
        issue_sparse_fp(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, b2, b3, meta);
#else
        #pragma unroll
        for (int rep = 0; rep < kDenseRepeats; ++rep)
            issue_dense_fp(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
#endif
#else
#if SPARSITY
        issue_sparse_int(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, b2, b3, meta);
#else
        #pragma unroll
        for (int rep = 0; rep < kDenseRepeats; ++rep)
            issue_dense_int(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
#endif
#endif
    }
#else
    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        #pragma unroll
        for (int u = 0; u < THROUGHPUT_UNROLL; ++u) {
#if OUTPUT_IS_FLOAT
#if SPARSITY
            issue_sparse_fp(d0[u], d1[u], d2[u], d3[u], a0, a1, a2, a3, b0,
                            b1, b2, b3, meta);
#else
            #pragma unroll
            for (int rep = 0; rep < kDenseRepeats; ++rep)
                issue_dense_fp(d0[u], d1[u], d2[u], d3[u], a0, a1, a2, a3, b0,
                               b1);
#endif
#else
#if SPARSITY
            issue_sparse_int(d0[u], d1[u], d2[u], d3[u], a0, a1, a2, a3, b0,
                             b1, b2, b3, meta);
#else
            #pragma unroll
            for (int rep = 0; rep < kDenseRepeats; ++rep)
                issue_dense_int(d0[u], d1[u], d2[u], d3[u], a0, a1, a2, a3, b0,
                                b1);
#endif
#endif
        }
    }
#endif

    unsigned long long stop = clock64();

    if (lane == 0) {
        cycles[warp_global] = stop - start;
#if BENCH_MODE == MODE_THROUGHPUT
#if OUTPUT_IS_FLOAT
        sink[warp_global] = static_cast<uint64_t>(d0[0]) ^
                            (static_cast<uint64_t>(d1[0]) << 16) ^
                            (static_cast<uint64_t>(d2[0]) << 32) ^
                            (static_cast<uint64_t>(d3[0]) << 48);
#else
        sink[warp_global] = static_cast<uint64_t>(d0[0]) ^
                            (static_cast<uint64_t>(d1[0]) << 16) ^
                            (static_cast<uint64_t>(d2[0]) << 32) ^
                            (static_cast<uint64_t>(d3[0]) << 48);
#endif
#else
#if OUTPUT_IS_FLOAT
        sink[warp_global] = static_cast<uint64_t>(d0) ^
                            (static_cast<uint64_t>(d1) << 16) ^
                            (static_cast<uint64_t>(d2) << 32) ^
                            (static_cast<uint64_t>(d3) << 48);
#else
        sink[warp_global] = static_cast<uint64_t>(d0) ^
                            (static_cast<uint64_t>(d1) << 16) ^
                            (static_cast<uint64_t>(d2) << 32) ^
                            (static_cast<uint64_t>(d3) << 48);
#endif
#endif
    }
}

static const char* cuda_error_string(cudaError_t err) {
    return err == cudaSuccess ? "cudaSuccess" : cudaGetErrorString(err);
}

int main() {
    unsigned long long* d_cycles = nullptr;
    uint64_t* d_sink = nullptr;
    unsigned long long* h_cycles = nullptr;

    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                     cuda_error_string(err));
        return 1;
    }

    if (prop.major != 12 || prop.minor != 0) {
        std::fprintf(stderr,
                     "Warning: expected sm_120, detected sm_%d%d (%s)\n",
                     prop.major, prop.minor, prop.name);
    }

    h_cycles = static_cast<unsigned long long*>(
        std::malloc(sizeof(unsigned long long) * kWarpsTotal));
    if (!h_cycles) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    err = cudaMalloc(&d_cycles, sizeof(unsigned long long) * kWarpsTotal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc cycles failed: %s\n", cuda_error_string(err));
        std::free(h_cycles);
        return 1;
    }
    err = cudaMalloc(&d_sink, sizeof(uint64_t) * kWarpsTotal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc sink failed: %s\n", cuda_error_string(err));
        cudaFree(d_cycles);
        std::free(h_cycles);
        return 1;
    }

    bench_kernel<<<BLOCKS, WARPS_PER_BLOCK * 32>>>(d_cycles, d_sink);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Launch Error: %s\n", cuda_error_string(err));
        cudaFree(d_sink);
        cudaFree(d_cycles);
        std::free(h_cycles);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Sync Error: %s\n", cuda_error_string(err));
        cudaFree(d_sink);
        cudaFree(d_cycles);
        std::free(h_cycles);
        return 1;
    }

    err = cudaMemcpy(h_cycles, d_cycles, sizeof(unsigned long long) * kWarpsTotal,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy cycles failed: %s\n", cuda_error_string(err));
        cudaFree(d_sink);
        cudaFree(d_cycles);
        std::free(h_cycles);
        return 1;
    }

    std::sort(h_cycles, h_cycles + kWarpsTotal);
    const unsigned long long median = h_cycles[kWarpsTotal / 2];
    const double cycles_per_packet =
        static_cast<double>(median) / static_cast<double>(kPacketsPerWarp);
    const double math_ops_per_packet = 2.0 * 16.0 * 8.0 * static_cast<double>(MMA_K);
    const double ops_per_cycle = math_ops_per_packet / cycles_per_packet;

    std::printf(
        "RESULT,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%.6f,%.3f,%.3f\n",
        BENCH_MODE == MODE_THROUGHPUT ? "throughput" : "latency",
        FAMILY_NAME, A_TYPE_NAME, B_TYPE_NAME,
        SPARSITY ? (ORDERED_METADATA ? "sparse_ordered" : "sparse") : "dense",
        SATFINITE ? "satfinite" : "nosat",
        MMA_K, DENSE_BASE_K, kDenseInstructionsPerPacket,
        kSparseInstructionsPerPacket, ITERATIONS, THROUGHPUT_UNROLL,
        kWarpsTotal, median, cycles_per_packet, math_ops_per_packet,
        ops_per_cycle);

    cudaFree(d_sink);
    cudaFree(d_cycles);
    std::free(h_cycles);
    return 0;
}
