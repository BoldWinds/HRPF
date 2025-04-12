#include <iostream>
#include <chrono>
#include <starpu.h>
#include <fcntl.h>
#include <random>
#include <execution>
#include "algorithm/utils.h"

#define THRESHOLD 1024*256

// CPU上的排序任务实现
void cpu_sort(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    hsort(data, len);
}

// GPU上的排序任务实现
void gpu_sort(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    gsort(data, len, stream);
}

static struct starpu_codelet sort_cl = {
    .cpu_funcs = {cpu_sort},
    .cuda_funcs = {gpu_sort},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .name = "sort_codelet"
};

// CPU上的合并任务实现
void cpu_merge(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector1 = (struct starpu_vector_interface *)buffers[0];
    struct starpu_vector_interface *vector2 = (struct starpu_vector_interface *)buffers[1];
    struct starpu_vector_interface *vector_dst = (struct starpu_vector_interface *)buffers[2];
    
    _TYPE *first = (_TYPE *)STARPU_VECTOR_GET_PTR(vector1);
    _TYPE *second = (_TYPE *)STARPU_VECTOR_GET_PTR(vector2);
    _TYPE *dst = (_TYPE *)STARPU_VECTOR_GET_PTR(vector_dst);
    
    unsigned int lenA = STARPU_VECTOR_GET_NX(vector1);
    unsigned int lenB = STARPU_VECTOR_GET_NX(vector2);
    
    hmerge(first, second, dst, lenA, lenB);
}

// GPU上的合并任务实现
void gpu_merge(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector1 = (struct starpu_vector_interface *)buffers[0];
    struct starpu_vector_interface *vector2 = (struct starpu_vector_interface *)buffers[1];
    struct starpu_vector_interface *vector_dst = (struct starpu_vector_interface *)buffers[2];
    
    _TYPE *first = (_TYPE *)STARPU_VECTOR_GET_PTR(vector1);
    _TYPE *second = (_TYPE *)STARPU_VECTOR_GET_PTR(vector2);
    _TYPE *dst = (_TYPE *)STARPU_VECTOR_GET_PTR(vector_dst);
    
    unsigned int lenA = STARPU_VECTOR_GET_NX(vector1);
    unsigned int lenB = STARPU_VECTOR_GET_NX(vector2);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    gmerge(first, second, dst, lenA, lenB, stream);
}

static struct starpu_codelet merge_cl = {
    .cpu_funcs = {cpu_merge},
    .cuda_funcs = {gpu_merge},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_W},
    .name = "merge_codelet"
};


void heterogeneous_mergesort(starpu_data_handle_t data_handle, unsigned int len)
{
    if (len <= THRESHOLD) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &sort_cl;
        task->handles[0] = data_handle;
        
        int ret = starpu_task_submit(task);
        if (ret == -ENODEV) {
            fprintf(stderr, "No worker can execute this task\n");
            exit(1);
        }
        
        return;
    }
    
    // 分割数据
    unsigned int mid = len / 2;
    _TYPE *left_array, *right_array;
    starpu_data_handle_t left_handle, right_handle;
    starpu_malloc((void**)&left_array, mid * sizeof(_TYPE));
    starpu_malloc((void**)&right_array, (len - mid) * sizeof(_TYPE));
    _TYPE *original_data = (_TYPE *)starpu_data_get_local_ptr(data_handle);
    memcpy(left_array, original_data, mid * sizeof(_TYPE));
    memcpy(right_array, original_data + mid, (len - mid) * sizeof(_TYPE));
    starpu_vector_data_register(&left_handle, STARPU_MAIN_RAM, (uintptr_t)left_array, mid, sizeof(_TYPE));
    starpu_vector_data_register(&right_handle, STARPU_MAIN_RAM, (uintptr_t)right_array, len - mid, sizeof(_TYPE));

    // 递归
    heterogeneous_mergesort(left_handle, mid);
    heterogeneous_mergesort(right_handle, len - mid);

    // 等待左右两部分排序完成
    starpu_data_acquire(left_handle, STARPU_R);
    starpu_data_acquire(right_handle, STARPU_R);
    starpu_data_release(left_handle);
    starpu_data_release(right_handle);
    
    // 提交合并任务
    struct starpu_task *merge_task = starpu_task_create();
    merge_task->cl = &merge_cl;
    merge_task->handles[0] = left_handle;
    merge_task->handles[1] = right_handle;
    merge_task->handles[2] = data_handle;

    int ret = starpu_task_submit(merge_task);
    if (ret == -ENODEV) {
        fprintf(stderr, "No worker can execute this merge task\n");
        exit(1);
    }
    
    // 等待合并完成
    starpu_task_wait_for_all();

    starpu_data_unregister(left_handle);
    starpu_data_unregister(right_handle);
    starpu_free_noflag(left_array, mid);
    starpu_free_noflag(right_array, len - mid);

}


int main(int argc, char **argv)
{

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
    int ret = starpu_init(&conf);
    conf.sched_policy_name = "dmda";
    if (ret != 0) {
        fprintf(stderr, "StarPU initialization failed: %s\n", strerror(-ret));
        return 1;
    }
    
    _TYPE *array;
    

    double milliseconds = 0;
    for(int i = 0; i < max_run; i++) {
        starpu_malloc((void**)&array, n * sizeof(_TYPE));
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::for_each(std::execution::par_unseq, array, array + n, [&](double &val) {
            val = dist(rng);
        });
        starpu_data_handle_t array_handle;
        starpu_vector_data_register(&array_handle, STARPU_MAIN_RAM, (uintptr_t)array, n, sizeof(_TYPE));
        auto start = std::chrono::high_resolution_clock::now();
        heterogeneous_mergesort(array_handle, n);
        starpu_task_wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();
        milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
        starpu_data_unregister(array_handle);
        starpu_free_noflag(array, n);
    }
    milliseconds /= max_run;

    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    std::cout << milliseconds << std::endl;
    starpu_shutdown();
    return 0;
}