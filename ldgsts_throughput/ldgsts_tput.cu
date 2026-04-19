#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}


// Required compile-time configs (pass via: make CTAS_PER_SM=X NUM_STAGES=Y THREADS_PER_BLOCK=Z LOAD_T=float4)
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef NUM_STAGES
#error "NUM_STAGES must be defined"
#endif
#ifndef THREADS_PER_BLOCK
#error "THREADS_PER_BLOCK must be defined"
#endif
#ifndef LOAD_T
#error "LOAD_T must be defined"
#endif

constexpr int32_t VECTOR_WIDTH = sizeof(LOAD_T) / sizeof(float);
constexpr int32_t LOAD_SIZE = sizeof(LOAD_T);

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t status__ = (expr);                                       \
        if (status__ != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(status__));                           \
            return 1;                                                        \
        }                                                                    \
    } while (0)


__global__ void asyncCopyKernel(float *arr, size_t N) {
    __shared__ LOAD_T buff[NUM_STAGES][THREADS_PER_BLOCK];

    int32_t tid = threadIdx.x;
    int32_t base = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = NUM_STAGES * (gridDim.x * blockDim.x);
    int32_t num_iters = ((N / VECTOR_WIDTH) - base) / stride;

    for (int32_t i = 0; i < num_iters; i++) {
        __pipeline_wait_prior(NUM_STAGES - 1);

        int32_t slot = i % NUM_STAGES;
        int32_t offset = (base + stride * i) * VECTOR_WIDTH;

        __pipeline_memcpy_async(buff[slot] + tid, arr + offset, LOAD_SIZE);  // LDGSTS
        __pipeline_commit();                                                 // LDGDEPBAR
    }

    __pipeline_wait_prior(NUM_STAGES - 1);  // DEPBAR.LE SB, NUM_STAGES-1
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
    size_t arr_size = alignDataVolume(num_blks * NUM_STAGES * THREADS_PER_BLOCK * LOAD_SIZE);
    size_t N = arr_size / sizeof(float);

    float *arr = (float*) malloc(arr_size);
    srand((uint32_t) num_blks + LOAD_SIZE);
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

    asyncCopyKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_arr));
    free(arr);
    return 0;
}
