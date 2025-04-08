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

void merge_sort(double* data, int len){
    if(len == 1){
        return;
    }else if(len == 2){
        if(data[0] > data[1]){
            double temp = data[0];
            data[0] = data[1];
            data[1] = temp;
        }
        return;
    }else{
        int left = len/2, right = len - left;
        merge_sort(data, left);
        merge_sort(data + left, right);
        double* temp = new double[len];
        for(int i = 0, j = 0, k = left; i < len; i++){
            if(j >= left){
                temp[i] = data[k++];
            }else if(k >= len){
                temp[i] = data[j++];
            }else{
                temp[i] = data[j] <= data[k] ? data[j++] : data[k++];
            }
        }
        std::copy(temp, temp+len, data);
        delete[] temp;
        return;
    }
}

int main(int argc, char** argv){
    int n = std::atoi(argv[1]);
    double* data = new double[n];
    loadData(data, n);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    merge_sort(data, n);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    
}