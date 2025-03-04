#include <iostream>
#include <chrono>
#include <mkl.h>

double matmul_mkl(int n, int repeat_count) {
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double alpha = 1.0f, beta = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat_count; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, beta, C, n);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    delete[] A;
    delete[] B;
    delete[] C;

    return duration.count() / repeat_count;
}

int main(int argc, char* argv[]) {
    int n = std::stoi(argv[1]);
    int max_run = std::stoi(argv[2]);
    double milliseconds = matmul_mkl(n, max_run);
    std::cout <<  milliseconds << std::endl;
    return 0;
}
