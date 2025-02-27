#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper function to check cuBLAS errors
#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

double matmul_cublas(int n, int max_run) {
    double *A, *B, *C;
    
    // Allocate memory
    CHECK_CUDA_ERROR(cudaMallocManaged(&A, n * n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&B, n * n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&C, n * n * sizeof(double)));

    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
        C[i] = 0.0f;  // Initialize C to zeros
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    double alpha = 1.0f, beta = 0.0f;
    
    // Make sure data is ready on device before timing
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < max_run; ++run) {
        CHECK_CUBLAS_ERROR(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
                                       &alpha, A, n, B, n, &beta, C, n));
    }
    
    // Ensure all GPU operations are completed before stopping the timer
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Clean up
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(A));
    CHECK_CUDA_ERROR(cudaFree(B));
    CHECK_CUDA_ERROR(cudaFree(C));

    return duration.count() / max_run;
}

int main(int argc, char* argv[]) {
    // Validate command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <max_run>" << std::endl;
        return 1;
    }
    
    int n, max_run;
    try {
        n = std::stoi(argv[1]);
        max_run = std::stoi(argv[2]);
        
        if (n <= 0 || max_run <= 0) {
            std::cerr << "Matrix size and repeat count must be positive integers" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }
    
    double milliseconds = matmul_cublas(n, max_run);
    std::cout << milliseconds << std::endl;
    return 0;
}