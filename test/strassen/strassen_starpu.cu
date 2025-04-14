#include <iostream>
#include <chrono>
#include <starpu.h>
#include <fcntl.h>
#include <random>
#include <execution>
#include "algorithm/strassen_problem/cuAdd.h"
#define THRESHOLD 1024

// CUDA kernel implementations for StarPU
void cuda_add(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    //unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    sumMatrix(A, B, C, nx, lda, ldb, ldc, stream);
    cudaStreamSynchronize(stream);
}

// CPU implementation for matrix addition
void cpu_add(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    #pragma omp parallel for
    for(unsigned i = 0; i < ny; i++) {
        for(unsigned j = 0; j < nx; j++) {
            C[i*ldc + j] = A[i*lda + j] + B[i*ldb + j];
        }
    }
}

static struct starpu_codelet add_cl = {
    .cpu_funcs = {cpu_add},
    .cuda_funcs = {cuda_add},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "add_codelet",
};

// CUDA kernel implementations for StarPU
void cuda_sub(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    subMatrix(A, B, C, nx, lda, ldb, ldc, stream);
    cudaStreamSynchronize(stream);
}

// CPU implementation for matrix subtraction
void cpu_sub(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    #pragma omp parallel for
    for(unsigned i = 0; i < ny; i++) {
        for(unsigned j = 0; j < nx; j++) {
            C[i*ldc + j] = A[i*lda + j] - B[i*ldb + j];
        }
    }
}

static struct starpu_codelet sub_cl = {
    .cpu_funcs = {cpu_sub},
    .cuda_funcs = {cuda_sub},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "sub_codelet"
};

// CUDA kernel implementations for StarPU
void cuda_gemm(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    gemm(A, B, C, nx, lda, ldb, ldc, stream);
    cudaStreamSynchronize(stream);
}

// CPU implementation for matrix multiplication
void cpu_gemm(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
    double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
    double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
    
    unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
    unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);
    
    #pragma omp parallel for
    for(unsigned i = 0; i < ny; i++) {
        for(unsigned j = 0; j < nx; j++) {
            double sum = 0.0;
            for(unsigned k = 0; k < nx; k++) {
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] = sum;
        }
    }
}

static struct starpu_codelet mul_cl = {
    .cpu_funcs = {cpu_gemm},
    .cuda_funcs = {cuda_gemm},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "mul_codelet"
};

struct starpu_data_filter horizontal_filter =
{
  .filter_func = starpu_matrix_filter_block,
  .nchildren = 2
};

struct starpu_data_filter vertical_filter = {
    .filter_func = starpu_matrix_filter_vertical_block,
    .nchildren = 2
};

// 创建新矩阵
starpu_data_handle_t create_matrix(unsigned n) {
    double *data;
    starpu_malloc((void**)&data, n * n * sizeof(double));
    
    starpu_data_handle_t handle;
    starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, 
                              (uintptr_t)data, n, n, n, sizeof(double));
    return handle;
}

// Strassen算法的递归实现
void heterogeneous_strassen(starpu_data_handle_t A, starpu_data_handle_t B, 
                       starpu_data_handle_t C, unsigned n) {
    if (n <= THRESHOLD) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &mul_cl;
        task->handles[0] = A;
        task->handles[1] = B;
        task->handles[2] = C;
        
        int ret = starpu_task_submit(task);
        if (ret == -ENODEV) {
            fprintf(stderr, "No worker can execute this task\n");
        }
        return;
    }
    
    // 矩阵A分块
    starpu_data_partition(A, &horizontal_filter);
    starpu_data_handle_t A_top = starpu_data_get_sub_data(A, 1, 0);
    starpu_data_handle_t A_bottom = starpu_data_get_sub_data(A, 1, 1);
    starpu_data_partition(A_top, &vertical_filter);
    starpu_data_partition(A_bottom, &vertical_filter);
    starpu_data_handle_t A11 = starpu_data_get_sub_data(A_top, 1, 0);
    starpu_data_handle_t A12 = starpu_data_get_sub_data(A_top, 1, 1);
    starpu_data_handle_t A21 = starpu_data_get_sub_data(A_bottom, 1, 0);
    starpu_data_handle_t A22 = starpu_data_get_sub_data(A_bottom, 1, 1);
    // 矩阵B分块
    starpu_data_partition(B, &horizontal_filter);
    starpu_data_handle_t B_top = starpu_data_get_sub_data(B, 1, 0);
    starpu_data_handle_t B_bottom = starpu_data_get_sub_data(B, 1, 1);
    starpu_data_partition(B_top, &vertical_filter);
    starpu_data_partition(B_bottom, &vertical_filter);
    starpu_data_handle_t B11 = starpu_data_get_sub_data(B_top, 1, 0);
    starpu_data_handle_t B12 = starpu_data_get_sub_data(B_top, 1, 1);
    starpu_data_handle_t B21 = starpu_data_get_sub_data(B_bottom, 1, 0);
    starpu_data_handle_t B22 = starpu_data_get_sub_data(B_bottom, 1, 1);
    // 矩阵C分块
    starpu_data_partition(C, &horizontal_filter);
    starpu_data_handle_t C_top = starpu_data_get_sub_data(C, 1, 0);
    starpu_data_handle_t C_bottom = starpu_data_get_sub_data(C, 1, 1);
    starpu_data_partition(C_top, &vertical_filter);
    starpu_data_partition(C_bottom, &vertical_filter);
    starpu_data_handle_t C11 = starpu_data_get_sub_data(C_top, 1, 0);
    starpu_data_handle_t C12 = starpu_data_get_sub_data(C_top, 1, 1);
    starpu_data_handle_t C21 = starpu_data_get_sub_data(C_bottom, 1, 0);
    starpu_data_handle_t C22 = starpu_data_get_sub_data(C_bottom, 1, 1);
    
    
    // 创建七个中间结果矩阵 M1-M7
    starpu_data_handle_t M1 = create_matrix(n/2);
    starpu_data_handle_t M2 = create_matrix(n/2);
    starpu_data_handle_t M3 = create_matrix(n/2);
    starpu_data_handle_t M4 = create_matrix(n/2);
    starpu_data_handle_t M5 = create_matrix(n/2);
    starpu_data_handle_t M6 = create_matrix(n/2);
    starpu_data_handle_t M7 = create_matrix(n/2);
    // 辅助矩阵，用于计算 M1-M7
    starpu_data_handle_t temp1 = create_matrix(n/2);
    starpu_data_handle_t temp2 = create_matrix(n/2);
    
    struct starpu_task *task;
    int ret = 0;
    
    // 计算 M1 = (A11 + A22) * (B11 + B22)
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = A11;
    task->handles[1] = A22;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = B11;
    task->handles[1] = B22;
    task->handles[2] = temp2;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(temp1, temp2, M1, n/2);
    
    // 计算 M2 = (A21 + A22) * B11
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = A21;
    task->handles[1] = A22;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(temp1, B11, M2, n/2);
    
    // 计算 M3 = A11 * (B12 - B22)
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = B12;
    task->handles[1] = B22;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(A11, temp1, M3, n/2);
    
    // 计算 M4 = A22 * (B21 - B11)
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = B21;
    task->handles[1] = B11;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(A22, temp1, M4, n/2);
    
    // 计算 M5 = (A11 + A12) * B22
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = A11;
    task->handles[1] = A12;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(temp1, B22, M5, n/2);
    
    // 计算 M6 = (A21 - A11) * (B11 + B12)
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = A21;
    task->handles[1] = A11;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = B11;
    task->handles[1] = B12;
    task->handles[2] = temp2;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(temp1, temp2, M6, n/2);
    
    // 计算 M7 = (A12 - A22) * (B21 + B22)
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = A12;
    task->handles[1] = A22;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = B21;
    task->handles[1] = B22;
    task->handles[2] = temp2;
    ret = starpu_task_submit(task);
    
    heterogeneous_strassen(temp1, temp2, M7, n/2);
    
    // 计算C11 = M1 + M4 - M5 + M7
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = M1;
    task->handles[1] = M7;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = M4;
    task->handles[1] = M5;
    task->handles[2] = temp2;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = temp1;
    task->handles[1] = temp2;
    task->handles[2] = C11;
    ret = starpu_task_submit(task);
    
    // 计算C12 = M3 + M5
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = M3;
    task->handles[1] = M5;
    task->handles[2] = C12;
    ret = starpu_task_submit(task);
    
    // 计算C21 = M2 + M4
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = M2;
    task->handles[1] = M4;
    task->handles[2] = C21;
    ret = starpu_task_submit(task);
    
    // 计算C22 = M1 - M2 + M3 + M6
    task = starpu_task_create();
    task->cl = &sub_cl;
    task->handles[0] = M1;
    task->handles[1] = M2;
    task->handles[2] = temp1;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = M3;
    task->handles[1] = M6;
    task->handles[2] = temp2;
    ret = starpu_task_submit(task);
    
    task = starpu_task_create();
    task->cl = &add_cl;
    task->handles[0] = temp1;
    task->handles[1] = temp2;
    task->handles[2] = C22;
    ret = starpu_task_submit(task);
    
    // 等待所有任务完成
    starpu_task_wait_for_all();
    
    // 释放临时矩阵
    starpu_data_unregister(M1);
    starpu_data_unregister(M2);
    starpu_data_unregister(M3);
    starpu_data_unregister(M4);
    starpu_data_unregister(M5);
    starpu_data_unregister(M6);
    starpu_data_unregister(M7);
    starpu_data_unregister(temp1);
    starpu_data_unregister(temp2);
    
    // 合并分区
    starpu_data_unpartition(A_top, STARPU_MAIN_RAM);
    starpu_data_unpartition(A_bottom, STARPU_MAIN_RAM);
    starpu_data_unpartition(B_top, STARPU_MAIN_RAM);
    starpu_data_unpartition(B_bottom, STARPU_MAIN_RAM);
    starpu_data_unpartition(C_top, STARPU_MAIN_RAM);
    starpu_data_unpartition(C_bottom, STARPU_MAIN_RAM);
    starpu_data_unpartition(A, STARPU_MAIN_RAM);
    starpu_data_unpartition(B, STARPU_MAIN_RAM);
    starpu_data_unpartition(C, STARPU_MAIN_RAM);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " array_size" << std::endl;
        return -1;
    }
    int n = std::stoi(argv[1]);
    int max_run = std::stoi(argv[2]);

    // 重定向 StarPU 输出(隐藏 StarPU 日志)
    int saved_stderr = dup(STDERR_FILENO);
    int dev_null = open("/dev/null", O_WRONLY);
    dup2(dev_null, STDERR_FILENO);
    close(dev_null);

    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncpus = -1;  // 使用所有可用CPU
    conf.ncuda = -1;  // 使用所有可用CUDA设备
    conf.nopencl = 0; // 不使用OpenCL
    conf.sched_policy_name = "dmda";
    int ret = starpu_init(&conf);
    if (ret != 0) {
        fprintf(stderr, "StarPU initialization failed: %s\n", strerror(-ret));
        return 1;
    }
    
    double *A_data, *B_data, *C_data;
    double milliseconds = 0;
    for(int i = 0; i < max_run; i++) {
        // 初始化矩阵
        starpu_malloc((void**)&A_data, n * n * sizeof(_TYPE));
        starpu_malloc((void**)&B_data, n * n * sizeof(_TYPE));
        starpu_malloc((void**)&C_data, n * n * sizeof(_TYPE));
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::for_each(std::execution::par_unseq, A_data, A_data + n * n, [&](double &val) {
            val = dist(rng);
        });
        std::for_each(std::execution::par_unseq, B_data, B_data + n * n, [&](double &val) {
            val = dist(rng);
        });
        // 注册StarPU矩阵
        starpu_data_handle_t A_handle, B_handle, C_handle;
        starpu_matrix_data_register(&A_handle, STARPU_MAIN_RAM,  (uintptr_t)A_data, n, n, n, sizeof(double));
        starpu_matrix_data_register(&B_handle, STARPU_MAIN_RAM,  (uintptr_t)B_data, n, n, n, sizeof(double));
        starpu_matrix_data_register(&C_handle, STARPU_MAIN_RAM,  (uintptr_t)C_data, n, n, n, sizeof(double));
        // 运行并计时
        auto start = std::chrono::high_resolution_clock::now();
        heterogeneous_strassen(A_handle, B_handle, C_handle, n);
        starpu_task_wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();
        milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
        // 资源清理
        starpu_data_unregister(A_handle);
        starpu_data_unregister(B_handle);
        starpu_data_unregister(C_handle);
        starpu_free_noflag(A_data, n*n);
        starpu_free_noflag(B_data, n*n);
        starpu_free_noflag(C_data, n*n);
    }
    milliseconds /= max_run;

    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    std::cout << milliseconds << std::endl;
    starpu_shutdown();
    return 0;
}