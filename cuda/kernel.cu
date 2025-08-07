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

__global__ void calculate_distances_kernel(const int* xs, const int* ys, float* distances, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int dx = xs[i] - xs[0];
    int dy = ys[i] - ys[0];
    distances[i] = sqrtf((float)(dx * dx + dy * dy));
}

extern "C" void calculate_distances(const int* h_xs, const int* h_ys, float* h_distances, int n) {
    int* d_xs;
    int* d_ys;
    float* d_distances;

    size_t int_size = n * sizeof(int);
    size_t float_size = n * sizeof(float);


    CHECK_CUDA_ERROR(cudaMalloc(&d_xs, int_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ys, int_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, float_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_xs, h_xs, int_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ys, h_ys, int_size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    calculate_distances_kernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_distances, n);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_distances, d_distances, float_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_xs));
    CHECK_CUDA_ERROR(cudaFree(d_ys));
    CHECK_CUDA_ERROR(cudaFree(d_distances));
}