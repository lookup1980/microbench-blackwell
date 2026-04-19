#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}


// Required compile-time configs (pass via: make CTAS_PER_SM=X NUM_STAGES=Y LOAD_SIZE=32768)
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef NUM_STAGES
#error "NUM_STAGES must be defined"
#endif
#ifndef LOAD_SIZE
#error "LOAD_SIZE must be defined"
#endif

constexpr size_t ELEMS_PER_LOAD = LOAD_SIZE / sizeof(float);

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t status__ = (expr);                                       \
        if (status__ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(status__));                           \
            return 1;                                                        \
        }                                                                    \
    } while (0)


__global__ void bulkAsyncCopyKernel(float *arr, size_t N) {
    __shared__ alignas(16) float buff[NUM_STAGES][ELEMS_PER_LOAD];
    __shared__ alignas(8) uint64_t bar[NUM_STAGES];

    if (threadIdx.x == 0) {
        for (int32_t s = 0; s < NUM_STAGES; s++) {
            ptx::mbarrier_init(&bar[s], 1);
        }
    }
    __syncthreads();

    int32_t base = ELEMS_PER_LOAD * blockIdx.x;
    int32_t stride = ELEMS_PER_LOAD * gridDim.x;
    int32_t num_iters = (N - base) / stride;

    for (int32_t i = 0; i < num_iters; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;

        while (i >= NUM_STAGES && !ptx::mbarrier_try_wait_parity(&bar[slot], parity));

        int32_t offset = base + i * stride;
        if (threadIdx.x == 0) {
            cg::invoke_one(cg::coalesced_threads(), [&] () {
                ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buff[slot], arr + offset, static_cast<uint32_t>(LOAD_SIZE), &bar[slot]);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[slot], static_cast<uint32_t>(LOAD_SIZE));
            });
        }
    }

    for (int32_t i = num_iters; i < num_iters + NUM_STAGES; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;
        while (!ptx::mbarrier_try_wait_parity(&bar[slot], parity));
    }
}

int get_device_attribute(cudaDeviceAttr attr, const char* name) {
    int value = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, 0));
    if (value <= 0) {
        fprintf(stderr, "Invalid %s reported by device: %d\n", name, value);
        std::exit(1);
    }
    return value;
}


int main() {
    int32_t num_sms = get_device_attribute(cudaDevAttrMultiProcessorCount, "SM count");
    int32_t l2_size = get_device_attribute(cudaDevAttrL2CacheSize, "L2 size");
    int32_t num_blks = num_sms * CTAS_PER_SM;
    size_t arr_size = alignDataVolume(num_blks * LOAD_SIZE * NUM_STAGES);
    size_t N = arr_size / sizeof(float);

    float *arr = (float*) malloc(arr_size);
    srand((uint32_t) NUM_STAGES * LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand();
    }

    float *d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, arr_size));
    CUDA_CHECK(cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    void *flush_arr;
    CUDA_CHECK(cudaMalloc(&flush_arr, l2_size));
    CUDA_CHECK(cudaMemset(flush_arr, 0xA5, l2_size));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(flush_arr));

    bulkAsyncCopyKernel<<<num_blks, 1>>>(d_arr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_arr));
    free(arr);
    return 0;
}
