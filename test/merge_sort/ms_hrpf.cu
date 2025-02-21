#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>

void test_sort(double* data_d, int len, cudaStream_t stream, float& avg_time) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;

    for (int i = 0; i < 100; ++i) {  // 运行 100 次，取平均时间
        cudaEventRecord(start, stream);
        thrust::sort(thrust::cuda::par.on(stream), thrust::device_pointer_cast(data_d), thrust::device_pointer_cast(data_d + len));
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time += elapsed_time;
    }

    avg_time = total_time / 100;  // 计算平均时间

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10000.0);

    for (int n = 10'000'000; n <= 100'000'000; n += 10'000'000) { // 10M 到 100M，步长 10M
        std::vector<double> data(n);
        for (int i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }

        double* data_d;
        cudaMalloc((void**)&data_d, n * sizeof(double));

        cudaMemcpy(data_d, data.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        float avg_time = 0.0f;
        test_sort(data_d, n, stream, avg_time);

        std::cout << "Size: " << n << " elements, Avg Time: " << avg_time << " ms" << std::endl;

        cudaFree(data_d);
    }

    cudaStreamDestroy(stream);
    return 0;
}
