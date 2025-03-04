#include <iostream>
#include <chrono>
#include <starpu.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <starpu.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

// CPU版本的矩阵乘法实现
void matrix_multiply_cpu(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    unsigned n = STARPU_MATRIX_GET_NX(buffers[0]);

    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            C[i * n + j] = 0.0;
            for (unsigned k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// CUDA版本的矩阵乘法实现（需要用nvcc编译）
#ifdef STARPU_USE_CUDA
extern "C" void matrix_multiply_cuda(void *buffers[], void *cl_arg)
{
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    unsigned n = STARPU_MATRIX_GET_NX(buffers[0]);

    // 获取CUDA流
    cudaStream_t stream = starpu_cuda_get_local_stream();

    // 这里可以调用自定义的CUDA kernel或cuBLAS
    // 简单示例：使用cuBLAS (需要包含cublas头文件并链接cublas库)
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // 使用cuBLAS的GEMM函数进行矩阵乘法
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                B, n,
                A, n,
                &beta,
                C, n);
                
    cublasDestroy(handle);
}
#endif

// 修改codelet以包括GPU实现
struct starpu_codelet cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {matrix_multiply_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {matrix_multiply_cuda},
#endif
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W}
};

double matmul_starpu(int n, int repeat_count) {
    // 初始化矩阵
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    // 填充随机数据
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
        C[i] = 0.0;
    }

    // 初始化StarPU，启用GPU
    // 初始化StarPU，启用GPU但禁用校准
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncuda = 1;         // 使用1个CUDA设备
    conf.nopencl = 0;       // 不使用OpenCL
    conf.calibrate = 0;     // 禁用性能校准
    starpu_init(&conf);
    
    starpu_data_handle_t A_handle, B_handle, C_handle;

    // 在StarPU中注册矩阵数据
    starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, n, n, n, sizeof(double));
    starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM, (uintptr_t)B, n, n, n, sizeof(double));
    starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM, (uintptr_t)C, n, n, n, sizeof(double));

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 重复执行计算多次
    for (int i = 0; i < repeat_count; ++i) {
        starpu_insert_task(
            &cl,
            STARPU_R, A_handle,
            STARPU_R, B_handle,
            STARPU_W, C_handle,
            0);
    }

    // 等待所有任务完成
    starpu_task_wait_for_all();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // 清理资源
    starpu_data_unregister(A_handle);
    starpu_data_unregister(B_handle);
    starpu_data_unregister(C_handle);
    
    delete[] A;
    delete[] B;
    delete[] C;
    starpu_shutdown();
    
    // 返回平均执行时间（毫秒）
    return duration.count() / repeat_count;
}

int main(int argc, char* argv[]) {
    int n = std::stoi(argv[1]);
    int repeat_count = std::stoi(argv[2]);
    // 重定向, 去除StarPU的输出(没找到StarPU的静默模式)
    int saved_stderr = dup(STDERR_FILENO);
    int dev_null = open("/dev/null", O_WRONLY);
    dup2(dev_null, STDERR_FILENO);
    close(dev_null);
    double milliseconds = matmul_starpu(n, repeat_count);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    std::cout << milliseconds << std::endl;
    return 0;
}
