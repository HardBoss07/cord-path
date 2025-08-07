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

// Kernel computes dist(i,j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
__global__ void compute_distance_matrix_kernel(const int* xs, const int* ys, float* dist_matrix, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (i < n && j < n) {
        int dx = xs[i] - xs[j];
        int dy = ys[i] - ys[j];
        dist_matrix[i * n + j] = sqrtf(float(dx * dx + dy * dy));
    }
}

extern "C" void compute_distance_matrix(const int* h_xs, const int* h_ys, float* h_dist_matrix, int n) {
    int* d_xs = nullptr;
    int* d_ys = nullptr;
    float* d_dist_matrix = nullptr;

    size_t int_size = n * sizeof(int);
    size_t matrix_size = n * n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_xs, int_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ys, int_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dist_matrix, matrix_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_xs, h_xs, int_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ys, h_ys, int_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    compute_distance_matrix_kernel<<<blocks, threadsPerBlock>>>(d_xs, d_ys, d_dist_matrix, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_dist_matrix, d_dist_matrix, matrix_size, cudaMemcpyDeviceToHost));

    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_dist_matrix);
}
