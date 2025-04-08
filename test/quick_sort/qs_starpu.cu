#include <iostream>
#include <chrono>
#include <starpu.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>
#include <cstdlib>
#include <random>

#define THRESHOLD 1024

// 辅助函数：交换两个元素
inline void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

void quicksort_recursive(double* A, int low, int high) {
    if (low >= high)
        return;
    // 选择枢纽元素（这里简单采用最后一个元素）
    double pivot = A[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (A[j] < pivot) {
            swap(A[i], A[j]);
            i++;
        }
    }
    swap(A[i], A[high]);
    quicksort_recursive(A, low, i - 1);
    quicksort_recursive(A, i + 1, high);
}

void quicksort_cpu(void *buffers[], void *cl_arg) {
    double *A = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    quicksort_recursive(A, 0, n-1);
}

int getRandomIndex(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(min, max);
    return dist(gen);
}

void gpu_quick_sort(thrust::device_vector<double>& d_vec, int start = 0, int end = -1) {
    if (end == -1) end = d_vec.size() - 1;
    int n = end - start + 1;
    if (n <= 1) return;

    if (n <= THRESHOLD) {
        thrust::sort(d_vec.begin() + start, d_vec.begin() + start + n);
        return;
    }

    double pivot = d_vec[getRandomIndex(start, start + n)];
    auto pivotPtr = thrust::partition(d_vec.begin() + start, d_vec.begin() + start + n, [pivot] __device__ (double x) {
        return x < pivot;
    });
    int pivotIndex = pivotPtr - d_vec.begin();

    if (pivotIndex < end) {
        gpu_quick_sort(d_vec, pivotIndex, end); 
    }

    if (pivotIndex > start) {  // Avoid sorting the same range again
        gpu_quick_sort(d_vec, start, pivotIndex - 1);
    }
}


#ifdef STARPU_USE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
extern "C" void quicksort_cuda(void *buffers[], void *cl_arg)
{
    double *A = (double *)STARPU_VECTOR_GET_PTR(buffers[0]);
    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    thrust::device_vector<double> d_vec(A, A + n);
    gpu_quick_sort(d_vec, 0, n - 1);
}
#endif

// 定义 codelet（支持 CPU 与 GPU 执行）
struct starpu_codelet cl = {
    .where = STARPU_CPU | STARPU_CUDA,
    .cpu_funcs = {quicksort_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs = {quicksort_cuda},
#endif
    .nbuffers = 1,
    .modes = {STARPU_RW}
};

double quicksort_starpu(int n) {
    double *A = new double[n];
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncuda = 1;
    conf.nopencl = 0;
    conf.calibrate = 0;
    int ret = starpu_init(&conf);
    if(ret != 0) {
        std::cerr << "Failed to initialize StarPU" << std::endl;
        return -1;
    }
    
    starpu_data_handle_t A_handle;
    starpu_vector_data_register(&A_handle, STARPU_MAIN_RAM, (uintptr_t)A, n, sizeof(double));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    starpu_insert_task(&cl,
                         STARPU_RW, A_handle,
                         0, &n);
    starpu_task_wait_for_all();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
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
    int max_run =  std::stoi(argv[2]);
    // 重定向输出（隐藏 StarPU 输出）
    int saved_stderr = dup(STDERR_FILENO);
    int dev_null = open("/dev/null", O_WRONLY);
    dup2(dev_null, STDERR_FILENO);
    close(dev_null);

    double milliseconds = 0;
    for(int i = 0; i < max_run; i++) {
        milliseconds += quicksort_starpu(n);
    }
    milliseconds /= max_run;

    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    std::cout << milliseconds << std::endl;
    return 0;
}
