#include <iostream>
#include <chrono>

double matmul(int n) {
    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
        B[i] = static_cast<double>(rand()) / RAND_MAX;
        C[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
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
    double milliseconds = matmul(n);
    std::cout <<  milliseconds << std::endl;
    return 0;
}
