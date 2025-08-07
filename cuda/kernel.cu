#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s failed with error %s\n", file, line, func, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void calculate_distances_kernel(const int* xs, const int* ys, float* distances, int n, int start_x, int start_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int dx = xs[i] - start_x;
    int dy = ys[i] - start_y;
    distances[i] = sqrtf((float)(dx * dx + dy * dy));
}

extern "C" void calculate_distances(const int* h_xs, const int* h_ys, float* h_distances, int n, int start_x, int start_y) {
    int* d_xs = nullptr;
    int* d_ys = nullptr;
    float* d_distances = nullptr;

    size_t int_size = n * sizeof(int);
    size_t float_size = n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xs, int_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ys, int_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_distances, float_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_xs, h_xs, int_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ys, h_ys, int_size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    calculate_distances_kernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_distances, n, start_x, start_y);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_distances, d_distances, float_size, cudaMemcpyDeviceToHost));

    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_distances);
}
