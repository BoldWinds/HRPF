#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/swap.h>
#include <sys/time.h>
#include <random>
#include <algorithm>
#include <execution>

#define THRESHOLD 1024*32

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
    double *h_data = new double[n];
    loadData(h_data, n);
    thrust::device_vector<double> d_vec(h_data, h_data + n);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    gpu_quick_sort(d_vec, 0, n - 1);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    delete h_data;
    exit(EXIT_SUCCESS);
}