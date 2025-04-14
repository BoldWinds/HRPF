#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <random>
#include <execution>

double test_sort(double *data_d, int len, cudaStream_t stream)
{
    auto start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<double> dev_data(data_d);
    thrust::sort(thrust::cuda::par.on(stream), dev_data, dev_data+ len);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

int main(int argc, char **argv)
{
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamDefault);

    double *data = new double[n];

    double milliseconds = 0;
    for (int run = 0; run <= max_run; run++)
    {
        loadData(data, n);
        double *data_d;
        cudaMalloc((void **)&data_d, n * sizeof(double));
        cudaMemcpy(data_d, data, n * sizeof(double), cudaMemcpyHostToDevice);
        milliseconds += test_sort(data_d, n, stream);
        cudaFree(data_d);
    }
    std::cout << milliseconds << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}