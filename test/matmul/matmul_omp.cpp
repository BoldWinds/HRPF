#include <iostream>
#include <chrono>
#include <omp.h>

double matmul_openmp(int n) {
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];
    double *BT = new double[n * n];

    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            BT[j * n + i] = B[i * n + j];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double Aik = A[i * n + k];  // 预加载 A[i, k]，避免重复读取
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += Aik * BT[j * n + k];  // 访问 BT 而不是 B，提升缓存局部性
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    delete[] A;
    delete[] B;
    delete[] C;

    return duration.count();
}

int main(int argc, char* argv[]) {
    int n = std::stoi(argv[1]);
    double milliseconds = matmul_openmp(n);
    std::cout <<  milliseconds << std::endl;
    return 0;
}
