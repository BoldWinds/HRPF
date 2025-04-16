#include <iostream>
#include <random>
#include <execution>
#include <chrono>

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

void run(double *a, double *b, double *c, int dim) {
    for (int idx = 0; idx < dim * dim; ++idx) {
        c[idx] = a[idx] * b[idx];
    }
}

int main(int argc, char **argv) {
    int dim = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    double *a = new double[dim*dim];
    double *b = new double[dim*dim];
    double *c = new double[dim*dim];

    double milliseconds = 0;
    for(int i = 0; i < max_run; ++i){
        loadData(a, dim*dim);
        loadData(b, dim*dim);
        auto start = std::chrono::high_resolution_clock::now();
        run(a, b, c, dim);
        auto end = std::chrono::high_resolution_clock::now();
        milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    delete [] a;
    delete [] b;
    delete [] c;
}