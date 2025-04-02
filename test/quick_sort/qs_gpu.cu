#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <sys/time.h>

#define MAX_DEPTH 16
#define THRESHOLD 1024


void gpu_quick_sort(thrust::device_vector<double>& d_vec, int start = 0, int end = -1) {
    if (end == -1) end = d_vec.size();
    int n = end - start;
    if (n <= 1) return;

    // Use thrust sort for small arrays
    if (n <= THRESHOLD) {
        thrust::sort(d_vec.begin() + start, d_vec.begin() + end);
        return;
    }

    // Obtain pivot on host
    double pivot;
    thrust::copy(d_vec.begin() + start + n / 2, d_vec.begin() + start + n / 2 + 1, &pivot);

    // Partitioning in-place
    auto partition_point = thrust::partition(d_vec.begin() + start, d_vec.begin() + end,
        [pivot] __device__ (double x) { return x < pivot; });

    int mid = partition_point - d_vec.begin();

    // Recursive in-place sorting
    gpu_quick_sort(d_vec, start, mid);
    gpu_quick_sort(d_vec, mid, end);
}


void loadData(double *datar, int length)
{
    std::ifstream fin;
    fin.open("./data/datamer.txt");

    if (!fin)
    {
        std::cout << "can not open the file data.txt" << std::endl;
        exit(1);
    }

    for (int i = 0; i < length; ++i)
    {
        fin >> datar[i];
    }
}

int main(int argc, char **argv)
{
    int n = std::atoi(argv[1]);

    double *h_data = new double[n];
    loadData(h_data, n);
    thrust::device_vector<double> d_vec(h_data, h_data + n);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    gpu_quick_sort(d_vec);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    delete h_data;
    exit(EXIT_SUCCESS);
}