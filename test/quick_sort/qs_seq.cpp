#include <iostream>
#include <random>
#include <execution>
#include <algorithm>
#include <sys/time.h>

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

void quickSort(double* data, int low, int high){
    if(low >= high){
        return;
    }
    int i = low, j = high;
    double pivot = data[(low + high) / 2];
    while(i <= j){
        while(data[i] < pivot){
            i++;
        }
        while(data[j] > pivot){
            j--;
        }
        if(i <= j){
            double temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }
    quickSort(data, low, j);
    quickSort(data, i, high);
}


int main(int argc, char** argv){
    int n = std::atoi(argv[1]);
    double* data = new double[n];
    loadData(data, n);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    quickSort(data, 0, n-1);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
}