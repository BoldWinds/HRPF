#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <fstream>

const int THRESHOLD = 1024*16;

void loadData(double* datar, int length) {
    std::ifstream fin;
    fin.open("./data/datamer.txt");

    if(!fin)
    {
        std::cout<<"can not open the file data.txt"<<std::endl;
        exit(1);
    }

    for(int i = 0; i < length; ++i){
        fin >> datar[i];
    }
}

void gpu_merge_sort(thrust::device_vector<double>& d_vec) {
    int n = d_vec.size();
    if (n <= 1) return;

    if (n <= THRESHOLD) {
        thrust::sort(d_vec.begin(), d_vec.end());
        return;
    }

    int mid = n / 2;
    thrust::device_vector<double> left(d_vec.begin(), d_vec.begin() + mid);
    thrust::device_vector<double> right(d_vec.begin() + mid, d_vec.end());

    gpu_merge_sort(left);
    gpu_merge_sort(right);

    thrust::sort(left.begin(), left.end());
    thrust::sort(right.begin(), right.end());

    thrust::merge(left.begin(), left.end(), right.begin(), right.end(), d_vec.begin());
}

int main(int argc, char** argv) {
    int n = std::atoi(argv[1]);
    double* data = new double[n];
    loadData(data, n);
    thrust::device_vector<double> d_vec(data, data + n);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    gpu_merge_sort(d_vec);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    return 0;
}