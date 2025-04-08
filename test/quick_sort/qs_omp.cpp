#include <iostream>
#include <random>
#include <execution>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

void quickSort(double* data, int low, int high, int depth = 0) {
    if (low >= high) return;

    int i = low, j = high;
    double pivot = data[(low + high) / 2];
    while (i <= j) {
        while (data[i] < pivot) i++;
        while (data[j] > pivot) j--;
        if (i <= j) {
            std::swap(data[i], data[j]);
            i++;
            j--;
        }
    }

    if (depth < 4) {
        #pragma omp task shared(data)
        quickSort(data, low, j, depth + 1);

        #pragma omp task shared(data)
        quickSort(data, i, high, depth + 1);
    } else {
        quickSort(data, low, j, depth + 1);
        quickSort(data, i, high, depth + 1);
    }
}

int main(int argc, char** argv){
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    double* data = new double[n];
    double milliseconds = 0.0;
    for(int i = 0; i < max_run; i++){
        loadData(data, n);
        struct timeval start, end;
        gettimeofday(&start, NULL);
        quickSort(data, 0, n-1);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    delete[] data;
    return 0;
}