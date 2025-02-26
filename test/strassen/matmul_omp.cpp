#include <iostream>
#include <chrono>
#include <omp.h>

double matmul_openmp(int n, int max_run) {
    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];

    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < max_run; ++run) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    C[i * n + j] += A[i * n + k] * B[k * n + j];
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    delete[] A;
    delete[] B;
    delete[] C;

    return duration.count() / max_run;
}

int main(int argc, char* argv[]) {
    int n = std::stoi(argv[1]);
    int max_run = std::stoi(argv[2]);
    double milliseconds = matmul_openmp(n, max_run);
    std::cout <<  milliseconds << std::endl;
    return 0;
}
