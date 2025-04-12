#include <iostream>
#include <chrono>
#include <starpu.h>
#include <fcntl.h>
#include <random>
#include <execution>
#include "algorithm/utils.h"

#define THRESHOLD 1024*256

// CPU上的排序任务实现
void cpu_qsort(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    
    hsort(data, len);
}

// GPU上的排序任务实现
void gpu_qsort(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    gsort(data, len, stream);
}

static struct starpu_codelet qsort_cl = {
    .cpu_funcs = {cpu_qsort},
    .cuda_funcs = {gpu_qsort},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .name = "qsort_codelet"
};

// CPU上的分区任务实现
void cpu_partition(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    
    int pivot_idx = hsplit(data, len);
    
    // 返回分区点位置
    if (cl_arg)
        *(int*)cl_arg = pivot_idx;
}

// GPU上的分区任务实现
void gpu_partition(void *buffers[], void *cl_arg)
{
    struct starpu_vector_interface *vector = (struct starpu_vector_interface *)buffers[0];
    _TYPE *data = (_TYPE *)STARPU_VECTOR_GET_PTR(vector);
    unsigned int len = STARPU_VECTOR_GET_NX(vector);
    
    cudaStream_t stream = starpu_cuda_get_local_stream();
    int pivot_idx = gsplit(data, len, stream);
    
    // 返回分区点位置
    if (cl_arg)
        *(int*)cl_arg = pivot_idx;
}

static struct starpu_codelet partition_cl = {
    .cpu_funcs = {cpu_partition},
    .cuda_funcs = {gpu_partition},
    .nbuffers = 1,
    .modes = {STARPU_RW},
    .name = "partition_codelet"
};

void heterogeneous_quicksort(starpu_data_handle_t data_handle, unsigned int len)
{
    if (len <= THRESHOLD) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &qsort_cl;
        task->handles[0] = data_handle;
        
        int ret = starpu_task_submit(task);
        if (ret == -ENODEV) {
            fprintf(stderr, "No worker can execute this task\n");
            exit(1);
        }
        
        return;
    }
    // 执行分区操作
    int pivot_idx;
    struct starpu_task *partition_task = starpu_task_create();
    partition_task->cl = &partition_cl;
    partition_task->handles[0] = data_handle;
    partition_task->cl_arg = &pivot_idx;
    partition_task->cl_arg_size = sizeof(int);
    partition_task->detach = 0; 
    
    int ret = starpu_task_submit(partition_task);
    if (ret == -ENODEV) {
        fprintf(stderr, "No worker can execute this partition task\n");
        exit(1);
    }

    ret = starpu_task_wait(partition_task);
    if (ret) {
        fprintf(stderr, "Task wait error!\n");
        exit(1);
    }
    
    // 根据分区点创建左右子数组的句柄
    _TYPE *data = (_TYPE *)starpu_data_get_local_ptr(data_handle);
    
    // 如果分区点无效或者分区后一侧没有元素，直接使用排序代码
    if (pivot_idx <= 0 || pivot_idx >= (int)len - 1) {
        struct starpu_task *sort_task = starpu_task_create();
        sort_task->cl = &qsort_cl;
        sort_task->handles[0] = data_handle;
        
        ret = starpu_task_submit(sort_task);
        if (ret == -ENODEV) {
            fprintf(stderr, "No worker can execute this sort task\n");
            exit(1);
        }
        
        return;
    }
    
    starpu_data_handle_t left_handle, right_handle;
    starpu_vector_data_register(&left_handle, STARPU_MAIN_RAM, 
                               (uintptr_t)data, pivot_idx, sizeof(_TYPE));
    starpu_vector_data_register(&right_handle, STARPU_MAIN_RAM, 
                               (uintptr_t)(data + pivot_idx), len - pivot_idx, sizeof(_TYPE));
    
    // 递归排序左右子数组
    heterogeneous_quicksort(left_handle, pivot_idx);
    heterogeneous_quicksort(right_handle, len - pivot_idx);
    
    // 等待所有子任务完成
    starpu_task_wait_for_all();
    
    // 解注册子数组句柄
    starpu_data_unregister(left_handle);
    starpu_data_unregister(right_handle);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " array_size max_run" << std::endl;
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
    conf.sched_policy_name = "dmda";  // 使用数据感知调度器
    
    int ret = starpu_init(&conf);
    if (ret != 0) {
        fprintf(stderr, "StarPU initialization failed: %s\n", strerror(-ret));
        return 1;
    }
    
    _TYPE *array;
    double milliseconds = 0;
    
    for(int i = 0; i < max_run; i++) {
        // 分配并初始化数组
        starpu_malloc((void**)&array, n * sizeof(_TYPE));
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::for_each(std::execution::par_unseq, array, array + n, [&](double &val) {
            val = dist(rng);
        });
        
        starpu_data_handle_t array_handle;
        starpu_vector_data_register(&array_handle, STARPU_MAIN_RAM, (uintptr_t)array, n, sizeof(_TYPE));
        
        auto start = std::chrono::high_resolution_clock::now();
        heterogeneous_quicksort(array_handle, n);
        starpu_task_wait_for_all();
        auto end = std::chrono::high_resolution_clock::now();
        milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
        starpu_data_unregister(array_handle);
        starpu_free_noflag(array, n);
    }
    milliseconds /= max_run;

    // 恢复标准错误输出
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    
    std::cout << milliseconds << std::endl;

    starpu_shutdown();
    return 0;
}