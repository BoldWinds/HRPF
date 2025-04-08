#include <iostream>
#include <random>
#include <algorithm>
#include <execution>
#include <sys/time.h>
#include <omp.h>

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

void merge_sort(double* data, int len, int depth = 0) {
    if (len <= 1) return;
    if (len == 2) {
        if (data[0] > data[1]) std::swap(data[0], data[1]);
        return;
    }

    int left = len / 2;
    int right = len - left;

    // 控制并行深度 & 数据大小阈值
    if (depth < 4 && len >= 10000) {
        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort(data, left, depth + 1);
            #pragma omp section
            merge_sort(data + left, right, depth + 1);
        }
    } else {
        merge_sort(data, left, depth + 1);
        merge_sort(data + left, right, depth + 1);
    }

    double* temp = new double[len];
    for (int i = 0, j = 0, k = left; i < len; ++i) {
        if (j >= left) temp[i] = data[k++];
        else if (k >= len) temp[i] = data[j++];
        else temp[i] = data[j] <= data[k] ? data[j++] : data[k++];
    }
    std::copy(temp, temp + len, data);
    delete[] temp;
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
        merge_sort(data, n);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    delete[] data;
    return 0;
}