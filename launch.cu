#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <cublas_v2.h>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void launch_sgemm(int kernel_idx, int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    switch (kernel_idx) {
        case 1:
            k1::launch_sgemm_naive_coalesced(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 2:
            k2::launch_sgemm_smem_block(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 3:
            k3::launch_sgemm_1d_block_tiling(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 4:
            k4::launch_sgemm_2d_block_tiling(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 5:
            k5::launch_sgemm_2d_block_tiling_vec(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 6:
            k6::launch_sgemm_warp_tiling(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 7:
            k7::launch_sgemm_2d_block_tiling_cute(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        case 8:
            k8::launch_sgemm_2d_block_tiling_cute(M, N, K, alpha, d_A, d_B, beta, d_C);
            break;
        default:
            std::cout << "Invalid kernel index" << std::endl;
            break;
    }
}


void verify_and_benchmark(int kernel_idx, int M, int N, int K, float alpha, float beta, const float* A, const float* B, const float* C) {
    std::vector<float> C_device(M * N);
    std::vector<float> C_cublas(M * N);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // verify correctness with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // For row-major matrices with cuBLAS (column-major): use transpose identity
    // C = A*B (row-major) becomes C^T = B^T * A^T (column-major)
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );
    cudaMemcpy(C_cublas.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    // reset d_C
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // warmup kernel and verify correctness
    launch_sgemm(kernel_idx, M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
    cudaMemcpy(C_device.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // verify correctness
    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_device[i] - C_cublas[i]) > 1e-6) {
            std::cout << "Error at index " << i << ": " << C_device[i] << " != " << C_cublas[i] << std::endl;
            break;
        }
    }

    // benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        launch_sgemm(kernel_idx, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    elapsed_time /= 10.0f;
    std::cout << "Time taken for kernel " << kernel_idx << ": " << elapsed_time << " ms" << std::endl;
    const double flop_count = 2.0 * static_cast<double>(M) * N * K + static_cast<double>(M) * N;
    const double gflops = flop_count / (elapsed_time * 1e6);
    std::cout << "GFLOP/s: " << gflops << std::endl;
    std::cout << "===" << std::endl;
    
    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaCheckError(cudaGetLastError());
}


int main(int argc, char* argv[]) {
    int kernel_idx = -1;
    if (argc == 2) {
        kernel_idx = atoi(argv[1]);
    }
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = (float)rand() / RAND_MAX;
    }
    if (kernel_idx == -1) {
        for (int i = 1; i <= 8; i++) {
            verify_and_benchmark(i, M, N, K, alpha, beta, A.data(), B.data(), C.data());
        }
    } else {
        verify_and_benchmark(kernel_idx, M, N, K, alpha, beta, A.data(), B.data(), C.data());
    }
}

