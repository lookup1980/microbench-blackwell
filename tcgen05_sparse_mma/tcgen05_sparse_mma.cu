// Sparse/dense Blackwell tcgen05 MMA microbenchmark.
//
// The file is intentionally compile-time specialized. The Python driver sweeps
// the documented sparse tcgen05.mma.sp and tcgen05.mma.ws.sp families and their
// dense peers by rebuilding this TU with different -D values.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define BENCH_THROUGHPUT 0
#define BENCH_LATENCY 1

#define INST_MMA 0
#define INST_WS 1

#define KIND_TF32 0
#define KIND_F16 1
#define KIND_F8F6F4 2
#define KIND_I8 3
#define KIND_MXF8F6F4 4
#define KIND_MXF4 5
#define KIND_MXF4NVF4 6

#define SS_MODE 0
#define TS_MODE 1

#ifndef BENCH_MODE
#error "BENCH_MODE must be defined (0=throughput, 1=latency)"
#endif
#ifndef INSTRUCTION_ID
#error "INSTRUCTION_ID must be defined (0=mma, 1=ws)"
#endif
#ifndef KIND_ID
#error "KIND_ID must be defined"
#endif
#ifndef A_TYPE_CODE
#error "A_TYPE_CODE must be defined"
#endif
#ifndef B_TYPE_CODE
#error "B_TYPE_CODE must be defined"
#endif
#ifndef A_BITS
#error "A_BITS must be defined"
#endif
#ifndef B_BITS
#error "B_BITS must be defined"
#endif
#ifndef SPARSITY
#error "SPARSITY must be defined (0=dense, 1=sparse)"
#endif
#ifndef AB_LAYOUT_ID
#error "AB_LAYOUT_ID must be defined (0=SS, 1=TS)"
#endif
#ifndef CTA_GROUP
#error "CTA_GROUP must be defined"
#endif
#ifndef MMA_M
#error "MMA_M must be defined"
#endif
#ifndef MMA_N
#error "MMA_N must be defined"
#endif
#ifndef MMA_K
#error "MMA_K must be defined"
#endif
#ifndef MMA_DEPTH
#error "MMA_DEPTH must be defined"
#endif
#ifndef SCALE_TYPE_ID
#error "SCALE_TYPE_ID must be defined (1=UE8M0, 0=UE4M3)"
#endif

#define BLOCK_SCALED (KIND_ID == KIND_MXF8F6F4 || KIND_ID == KIND_MXF4 || KIND_ID == KIND_MXF4NVF4)

#if INSTRUCTION_ID == INST_WS && CTA_GROUP != 1
#error "tcgen05.mma.ws(.sp) supports CTA_GROUP=1 only"
#endif

#if INSTRUCTION_ID == INST_WS && BLOCK_SCALED
#error "This benchmark restricts tcgen05.mma.ws(.sp) to non-block-scaled PTX kinds"
#endif

#if CTA_GROUP != 1 && CTA_GROUP != 2
#error "CTA_GROUP must be 1 or 2"
#endif

#if KIND_ID == KIND_TF32
#define KIND_SUFFIX "tf32"
#define KIND_NAME "tf32"
#elif KIND_ID == KIND_F16
#define KIND_SUFFIX "f16"
#define KIND_NAME "f16"
#elif KIND_ID == KIND_F8F6F4
#define KIND_SUFFIX "f8f6f4"
#define KIND_NAME "f8f6f4"
#elif KIND_ID == KIND_I8
#define KIND_SUFFIX "i8"
#define KIND_NAME "i8"
#elif KIND_ID == KIND_MXF8F6F4
#define KIND_SUFFIX "mxf8f6f4.block_scale"
#define KIND_NAME "mxf8f6f4"
#elif KIND_ID == KIND_MXF4
#define KIND_SUFFIX "mxf4.block_scale.block32"
#define KIND_NAME "mxf4"
#elif KIND_ID == KIND_MXF4NVF4
#define KIND_SUFFIX "mxf4nvf4.block_scale.block32"
#define KIND_NAME "mxf4nvf4"
#else
#error "Unsupported KIND_ID"
#endif

__host__ __device__ constexpr int cdiv_constexpr(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ constexpr int align_up_constexpr(int x, int boundary) {
    return (x + boundary - 1) & ~(boundary - 1);
}

__host__ __device__ constexpr int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

constexpr int MMA_M_PER_CTA = MMA_M / CTA_GROUP;
constexpr int A_STORED_K = SPARSITY ? (MMA_K / 2) : MMA_K;
constexpr int A_ROW_BYTES = cdiv_constexpr(A_STORED_K * A_BITS, 8);
constexpr int A_SIZE_RAW = cdiv_constexpr(MMA_M_PER_CTA * A_STORED_K * A_BITS, 8);
constexpr int B_SIZE_RAW = cdiv_constexpr(MMA_K * MMA_N * B_BITS, 8);
constexpr int A_SIZE = align_up_constexpr(A_SIZE_RAW, 128);
constexpr int B_SIZE = align_up_constexpr(B_SIZE_RAW, 128);
[[maybe_unused]] constexpr int SF_SIZE = 128;
[[maybe_unused]] constexpr int SPMETA_COLS = 8;
[[maybe_unused]] constexpr int SF_COLS = 8;
[[maybe_unused]] constexpr int A_TMEM_COLS = 8;

static_assert(MMA_M % CTA_GROUP == 0, "MMA_M must divide CTA_GROUP");
static_assert(!SPARSITY || (MMA_K % 2 == 0), "Sparse semantic K must be even");
static_assert(AB_LAYOUT_ID == SS_MODE || A_ROW_BYTES <= 32,
              "TS mode copies one 32-byte A row into TMEM");

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr, int height) {
    int addr = static_cast<int>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;
    desc |= ((height * 16) >> 4) << 16;
    desc |= (8ULL << 32);
    desc |= (1ULL << 46);
    return desc;
}

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

template <int CtaGroup>
__device__ __forceinline__ void barrier_sync();

template <>
__device__ __forceinline__ void barrier_sync<1>() {
    __syncthreads();
}

template <>
__device__ __forceinline__ void barrier_sync<2>() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

__device__ __forceinline__ void tmem_store_x1(uint32_t tmem_addr, uint32_t value) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                 :
                 : "r"(tmem_addr), "r"(value));
}

__device__ constexpr uint32_t make_i_desc() {
    uint32_t desc = 0;

#if KIND_ID == KIND_MXF8F6F4
    desc |= (SPARSITY ? (1U << 2) : 0U);
    desc |= (0U << 4);                       // B scale factor ID
    desc |= ((uint32_t)A_TYPE_CODE << 7);
    desc |= ((uint32_t)B_TYPE_CODE << 10);
    desc |= ((uint32_t)(MMA_N >> 3) << 17);
    desc |= ((uint32_t)SCALE_TYPE_ID << 23); // UE8M0 in this benchmark
    desc |= ((uint32_t)(MMA_M >> 7) << 27);
    desc |= (0U << 29);                      // A scale factor ID
#elif KIND_ID == KIND_MXF4 || KIND_ID == KIND_MXF4NVF4
    desc |= (SPARSITY ? (1U << 2) : 0U);
    desc |= (0U << 4);                       // B scale factor ID
    desc |= ((uint32_t)A_TYPE_CODE << 7);
    desc |= (((uint32_t)B_TYPE_CODE & 0x3U) << 10);
    desc |= ((uint32_t)(MMA_N >> 3) << 17);
    desc |= ((uint32_t)SCALE_TYPE_ID << 23);
    desc |= ((uint32_t)(MMA_M >> 7) << 27);
    desc |= (0U << 29);                      // A scale factor ID
    desc |= (0U << 31);                      // dense K=64 / sparse K=128
#else
    desc |= (SPARSITY ? (1U << 2) : 0U);     // selector bits 0-1 are zero
#if KIND_ID == KIND_TF32
    desc |= (1U << 4);                       // D=f32
#elif KIND_ID == KIND_F16 || KIND_ID == KIND_F8F6F4
    desc |= (1U << 4);                       // D=f32
#elif KIND_ID == KIND_I8
    desc |= (2U << 4);                       // D=s32
#endif
    desc |= ((uint32_t)A_TYPE_CODE << 7);
    desc |= ((uint32_t)B_TYPE_CODE << 10);
    desc |= ((uint32_t)(MMA_N >> 3) << 17);
    desc |= ((uint32_t)(MMA_M >> 4) << 24);
#endif
    return desc;
}

__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(128)
void tcgen05_sparse_mma_kernel() {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    int cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* A = smem;
    unsigned char* B = smem + A_SIZE;
#if BLOCK_SCALED
    unsigned char* SF_A = B + B_SIZE;
    unsigned char* SF_B = SF_A + SF_SIZE;
#endif

    for (int i = tid; i < A_SIZE_RAW; i += blockDim.x)
        A[i] = static_cast<unsigned char>((i + 17 * cta_rank) * 13 + 1);
    for (int i = tid; i < B_SIZE_RAW; i += blockDim.x)
        B[i] = static_cast<unsigned char>(i * 7 + 3);

#if BLOCK_SCALED
    for (int i = tid; i < SF_SIZE; i += blockDim.x) {
        SF_A[i] = static_cast<unsigned char>(1 + (i & 0x7));
        SF_B[i] = static_cast<unsigned char>(1 + ((i + 3) & 0x7));
    }
#endif

    barrier_sync<CTA_GROUP>();

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbar;
    __shared__ int tmem_addr;

    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar));
    if (warp_id == 0 && elect_sync()) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :
                     : "r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    int tmem_cols = MMA_N;

#if AB_LAYOUT_ID == TS_MODE
    const int tmem_a_offset = tmem_cols;
    tmem_cols += A_TMEM_COLS;
#endif

#if BLOCK_SCALED
    const int tmem_sf_a_offset = tmem_cols;
    const int tmem_sf_b_offset = tmem_cols + 4;
    tmem_cols += SF_COLS;
#endif

#if SPARSITY
    const int tmem_spmeta_offset = tmem_cols;
    tmem_cols += SPMETA_COLS;
#endif

    const int tmem_alloc_cols = next_power_of_2(tmem_cols < 32 ? 32 : tmem_cols);

    if (warp_id == 0) {
        const int tmem_addr_smem = static_cast<int>(__cvta_generic_to_shared(&tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :
                     : "r"(tmem_addr_smem), "r"(tmem_alloc_cols), "n"(CTA_GROUP));
    }
    barrier_sync<CTA_GROUP>();

    const uint32_t tmem_d = tmem_addr;

#if AB_LAYOUT_ID == TS_MODE
    const uint32_t tmem_a = tmem_addr + tmem_a_offset;
    uint32_t a_regs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (tid < MMA_M_PER_CTA) {
        const unsigned char* row = A + tid * A_ROW_BYTES;
        #pragma unroll
        for (int b = 0; b < 32; b++) {
            const uint32_t v = (b < A_ROW_BYTES) ? row[b] : 0U;
            a_regs[b / 4] |= v << ((b & 3) * 8);
        }
    }
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                 :
                 : "r"(tmem_a),
                   "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                   "r"(a_regs[4]), "r"(a_regs[5]), "r"(a_regs[6]), "r"(a_regs[7]));
#endif

#if BLOCK_SCALED
    const uint32_t tmem_sf_a = tmem_addr + tmem_sf_a_offset;
    const uint32_t tmem_sf_b = tmem_addr + tmem_sf_b_offset;
    tmem_store_x1(tmem_sf_a, static_cast<uint32_t>(SF_A[tid & 127]));
    tmem_store_x1(tmem_sf_b, static_cast<uint32_t>(SF_B[tid & 127]));
#endif

#if SPARSITY
    const uint32_t tmem_spmeta = tmem_addr + tmem_spmeta_offset;
    #pragma unroll
    for (int col = 0; col < SPMETA_COLS; col++)
        tmem_store_x1(tmem_spmeta + col, 0x44444444U);
#endif

    barrier_sync<CTA_GROUP>();

    constexpr uint32_t i_desc = make_i_desc();
    uint64_t b_desc = make_smem_desc(B, MMA_N);

#if AB_LAYOUT_ID == SS_MODE
    uint64_t a_desc = make_smem_desc(A, MMA_M_PER_CTA);
#endif

    auto issue_mma = [&](int pred) {
#if INSTRUCTION_ID == INST_WS
#if SPARSITY
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.ws.sp.cta_group::%5.kind::" KIND_SUFFIX " [%0], %1, %2, [%3], %4, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.ws.sp.cta_group::%5.kind::" KIND_SUFFIX " [%0], [%1], %2, [%3], %4, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#endif
#else
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.ws.cta_group::%4.kind::" KIND_SUFFIX " [%0], %1, %2, %3, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.ws.cta_group::%4.kind::" KIND_SUFFIX " [%0], [%1], %2, %3, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#endif
#endif
#elif BLOCK_SCALED
#if SPARSITY
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %8, 0;\n\t"
            "tcgen05.mma.sp.cta_group::%7.kind::" KIND_SUFFIX " [%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %8, 0;\n\t"
            "tcgen05.mma.sp.cta_group::%7.kind::" KIND_SUFFIX " [%0], [%1], %2, [%3], %4, [%5], [%6], p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred));
#endif
#else
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %7, 0;\n\t"
            "tcgen05.mma.cta_group::%6.kind::" KIND_SUFFIX " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc),
              "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %7, 0;\n\t"
            "tcgen05.mma.cta_group::%6.kind::" KIND_SUFFIX " [%0], [%1], %2, %3, [%4], [%5], p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
              "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred));
#endif
#endif
#else
#if SPARSITY
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.sp.cta_group::%5.kind::" KIND_SUFFIX " [%0], %1, %2, [%3], %4, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.sp.cta_group::%5.kind::" KIND_SUFFIX " [%0], [%1], %2, [%3], %4, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(tmem_spmeta), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#endif
#else
#if AB_LAYOUT_ID == SS_MODE
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.cta_group::%4.kind::" KIND_SUFFIX " [%0], %1, %2, %3, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#else
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.cta_group::%4.kind::" KIND_SUFFIX " [%0], [%1], %2, %3, p;\n\t"
            "}"
            :
            : "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
              "n"(CTA_GROUP), "r"(pred));
#endif
#endif
#endif
    };

#if BENCH_MODE == BENCH_THROUGHPUT
    constexpr int NUM_ITERS = 1000;
    uint64_t start_clock, end_clock;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));

    for (int iter = 0, phase = 0; iter < NUM_ITERS; iter++) {
        if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
            issue_mma(0);
            #pragma unroll
            for (int m = 1; m < MMA_DEPTH; m++)
                issue_mma(1);

            const uint16_t cta_mask = (1 << CTA_GROUP) - 1;
            asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                         :
                         : "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP)
                         : "memory");
        }

        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_TPUT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_TPUT;\n\t"
            "bra.uni LAB_WAIT_TPUT;\n\t"
            "DONE_TPUT:\n\t"
            "}"
            :
            : "r"(mbar_addr), "r"(phase));
        phase ^= 1;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));
    asm volatile("tcgen05.fence::after_thread_sync;");

    barrier_sync<CTA_GROUP>();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(tmem_addr), "r"(tmem_alloc_cols), "n"(CTA_GROUP));
    }

    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
        const uint64_t cycles = end_clock - start_clock;
        const uint64_t total_mmas = static_cast<uint64_t>(MMA_DEPTH) * NUM_ITERS;
        printf("RESULT,throughput,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%.6f,%d,%d\n",
               INSTRUCTION_ID, KIND_ID, A_TYPE_CODE, B_TYPE_CODE, SPARSITY,
               AB_LAYOUT_ID, CTA_GROUP, MMA_M, MMA_N, MMA_K, MMA_DEPTH,
               static_cast<unsigned long long>(cycles),
               static_cast<unsigned long long>(total_mmas),
               static_cast<double>(cycles) / static_cast<double>(total_mmas),
               A_SIZE_RAW, B_SIZE_RAW);
    }
#else
    constexpr int NUM_SAMPLES = 100;
    uint64_t samples[NUM_SAMPLES];

    for (int iter = 0, phase = 0; iter < NUM_SAMPLES; iter++) {
        uint64_t t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
            issue_mma(1);
            const uint16_t cta_mask = (1 << CTA_GROUP) - 1;
            asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                         :
                         : "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP)
                         : "memory");
        }

        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_LAT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_LAT;\n\t"
            "bra.uni LAB_WAIT_LAT;\n\t"
            "DONE_LAT:\n\t"
            "}"
            :
            : "r"(mbar_addr), "r"(phase));
        phase ^= 1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        samples[iter] = t1 - t0;
    }

    asm volatile("tcgen05.fence::after_thread_sync;");

    barrier_sync<CTA_GROUP>();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(tmem_addr), "r"(tmem_alloc_cols), "n"(CTA_GROUP));
    }

    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_SAMPLES - 1; i++) {
            for (int j = 0; j < NUM_SAMPLES - 1 - i; j++) {
                if (samples[j] > samples[j + 1]) {
                    uint64_t tmp = samples[j];
                    samples[j] = samples[j + 1];
                    samples[j + 1] = tmp;
                }
            }
        }
        const uint64_t median = (samples[NUM_SAMPLES / 2 - 1] + samples[NUM_SAMPLES / 2]) / 2;
        printf("RESULT,latency,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%d,%d\n",
               INSTRUCTION_ID, KIND_ID, A_TYPE_CODE, B_TYPE_CODE, SPARSITY,
               AB_LAYOUT_ID, CTA_GROUP, MMA_M, MMA_N, MMA_K,
               static_cast<unsigned long long>(median), A_SIZE_RAW, B_SIZE_RAW);
    }
#endif
}

int main() {
    const int smem_size = A_SIZE + B_SIZE
#if BLOCK_SCALED
        + 2 * SF_SIZE
#endif
        ;

    tcgen05_sparse_mma_kernel<<<CTA_GROUP, 128, smem_size>>>();
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("CUDA Launch Error: %s\n", cudaGetErrorString(launch_err));
        return 1;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
