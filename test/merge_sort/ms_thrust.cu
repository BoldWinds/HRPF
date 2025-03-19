#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>


void test_sort(double* data_d, int len, cudaStream_t stream, float& elapsed_time) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录排序开始时间
    cudaEventRecord(start, stream);
    thrust::sort(thrust::cuda::par.on(stream), thrust::device_pointer_cast(data_d), thrust::device_pointer_cast(data_d + len));
    cudaEventRecord(stop, stream);

    // 等待排序完成
    cudaEventSynchronize(stop);

    // 获取排序所用的时间
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv){
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10000.0);

    double milliseconds = 0;
    for (int run = 0; run <= max_run; run++) {
        std::vector<double> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }

        double* data_d;
        cudaMalloc((void**)&data_d, n * sizeof(double));

        cudaMemcpy(data_d, data.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        float elapsed_time = 0.0f;
        test_sort(data_d, n, stream, elapsed_time);
        milliseconds += elapsed_time;

        cudaFree(data_d);
    }
    std::cout << milliseconds << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}
