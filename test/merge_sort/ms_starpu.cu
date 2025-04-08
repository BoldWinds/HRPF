#include <iostream>
#include <chrono>
#include <starpu.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

// 辅助函数：归并两个有序区间
void merge(double *A, int left, int mid, int right, double* temp) {
    int i = left, j = mid + 1, k = left;
    while(i <= mid && j <= right) {
        if (A[i] <= A[j]) {
            temp[k++] = A[i++];
        } else {
            temp[k++] = A[j++];
        }
    }
    while (i <= mid)
        temp[k++] = A[i++];
    while (j <= right)
        temp[k++] = A[j++];
    // 复制回原数组
    for (int p = left; p <= right; p++)
        A[p] = temp[p];
}

// 递归实现归并排序（CPU 版本）
void merge_sort_recursive(double* A, int left, int right, double* temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort_recursive(A, left, mid, temp);
    merge_sort_recursive(A, mid+1, right, temp);
    merge(A, left, mid, right, temp);
}

// CPU 实现的归并排序 codelet（调用归并排序算法）
void mergesort_cpu(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    double *temp = new double[n];
    merge_sort_recursive(A, 0, n-1, temp);
    delete[] temp;
}

void mergesort_cuda(void *buffers[], void *cl_arg)
{
    double *A = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(A);
    thrust::sort(dev_ptr, dev_ptr + n);
}

// 定义 codelet（同时支持 CPU 和 GPU，其中 GPU 版本仅在启用 STARPU_USE_CUDA 时编译）
struct starpu_codelet cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {mergesort_cpu},
    .cuda_funcs = {mergesort_cuda},
    .nbuffers = 1,
    // 这里注册数据读写权限：排序操作需要读写原始数组
    .modes = {STARPU_RW}
};

double mergesort_starpu(int n) {
    double *A = new double[n];
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // StarPU 初始化
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncuda = 1;
    conf.nopencl = 0;
    conf.calibrate = 0;
    int ret = starpu_init(&conf);
    if(ret != 0)    return -1;
    // 注册数据（作为1行n列的矩阵）
    starpu_data_handle_t A_handle;
    starpu_vector_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, n, sizeof(double));

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 插入排序任务。通过 cl_arg 传入待排序数组长度。
    starpu_insert_task(&cl,
                         STARPU_RW, A_handle,
                         0, &n);
    starpu_task_wait_for_all();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // 清理资源
    starpu_data_unregister(A_handle);
    delete[] A;
    starpu_shutdown();

    return duration.count();
}

int main(int argc, char* argv[]) {
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

    double milliseconds = 0;
    for(int i = 0; i < max_run; i++) {
        milliseconds += mergesort_starpu(n);
    }
    milliseconds /= max_run;

    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    std::cout << milliseconds << std::endl;
    return 0;
}
