#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <cublas_v2.h>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void launch_naive(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32, 32);

    auto device_start = std::chrono::high_resolution_clock::now();
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_duration = device_end - device_start;
    std::cout << "Time taken for device: " << device_duration.count() << " seconds" << std::endl;
}

void launch_sgemm_smem_block(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32 * 32);

    auto device_start = std::chrono::high_resolution_clock::now();
    sgemm_smem_block<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_duration = device_end - device_start;
    std::cout << "Time taken for device: " << device_duration.count() << " seconds" << std::endl;
}

void launch_sgemm_1d_warp_tiling(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
    dim3 block(8 * 64);

    auto device_start = std::chrono::high_resolution_clock::now();
    sgemm_1d_warp_tiling<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_duration = device_end - device_start;
    std::cout << "Time taken for device: " << device_duration.count() << " seconds" << std::endl;
}

void launch_sgemm_2d_warp_tiling(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 64));
    dim3 block(8 * 8);

    auto device_start = std::chrono::high_resolution_clock::now();
    sgemm_2d_warp_tiling<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_duration = device_end - device_start;
    std::cout << "Time taken for device: " << device_duration.count() << " seconds" << std::endl;
}

void launch_sgemm_2d_warp_tiling_vec(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block((BM * BN) / (TM * TN));

    auto device_start = std::chrono::high_resolution_clock::now();
    sgemm_2d_warp_tiling_vec<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    auto device_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_duration = device_end - device_start;
    std::cout << "Time taken for device: " << device_duration.count() << " seconds" << std::endl;
}



int main() {
    bool bench_host = false;
    int M = 4096;
    int N = 4096;
    int K = 4096;
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_device(M * N);
    std::vector<float> C_check(M * N);

    for (int i = 0; i < M * K; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        C_device[i] = (float)rand() / RAND_MAX;
        C_check[i] = C_device[i];
    }
    
    // setup and launch device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_device.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // launch
    launch_sgemm_2d_warp_tiling_vec(M, N, K, alpha, d_A, d_B, beta, d_C);

    // Copy kernel result back to host before cuBLAS overwrites it
    cudaMemcpy(C_device.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // verify correctness with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(d_C, C_check.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
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
    cudaMemcpy(C_check.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_check[i] - C_device[i]) > 1e-6) {
            std::cout << "Error at index " << i << ": " << C_check[i] << " != " << C_device[i] << std::endl;
        }
    }

    // cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}


