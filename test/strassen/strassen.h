
#pragma once

#include <iostream>
#include <random>
#include <execution>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "algorithm/strassen_problem/cuAdd.h"

class BasicMatrix{
    protected:
        double* data;
        double* g_data;
        int dim;
    
    public:
        BasicMatrix(int dim) : dim(dim), g_data(nullptr) {
            data = new double[dim * dim];
        }

        ~BasicMatrix() {
            if(data) delete[] data;
            if (g_data) cudaFree(g_data);
        }

        double& operator()(int row, int col) {
            return data[row * dim + col];
        }

        int getDim() const { return dim; }
        double* getData() { return data; }
        const double* getData() const { return data; }
        double* getGPUData() { return g_data; }
        const double* getGPUData() const { return g_data; }

        virtual void matrixAdd(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& result) = 0;
        virtual void matrixSub(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& result) = 0;
        virtual void matrixMul(const BasicMatrix& A, const BasicMatrix& B, BasicMatrix& result) = 0;
        virtual void splitMatrix(const BasicMatrix& A, BasicMatrix& A11, BasicMatrix& A12,
            BasicMatrix& A21, BasicMatrix& A22) = 0;
        virtual void mergeMatrix(BasicMatrix& C, const BasicMatrix& C11, const BasicMatrix& C12,
            const BasicMatrix& C21, const BasicMatrix& C22) = 0;
        
        virtual void generateRandomMatrix() {
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            std::for_each(std::execution::par_unseq, data, data + dim*dim, [&](double &val) {
                val = dist(rng);
            });
        }

        int copyToGPU() {
            if (!g_data) {
                if (cudaMalloc((void**)&g_data, dim * dim * sizeof(double)) != cudaSuccess)
                    return -1;
            }
            if (cudaMemcpy(g_data, data, dim * dim * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
                return -2;
            return 0;
        }

        int copyFromGPU() {
            if (!g_data) return -1;
            if (cudaMemcpy(data, g_data, dim * dim * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
                return -2;
            return 0;
        }
};

class Strassen{
    protected:
        BasicMatrix* A;
        BasicMatrix* B;
        BasicMatrix* C;
        int dim;

    public:
        Strassen(int dim) : A(nullptr), B(nullptr), C(nullptr), dim(dim) {}
 
        ~Strassen() {
            delete A;
            delete B;
            delete C;
        }

        virtual void prepare() = 0;

        virtual void run() = 0;

        double test(int max_run){
            double milliseconds = 0;
            for(int i = 0; i < max_run; i++){
                prepare();
                auto start = std::chrono::high_resolution_clock::now();
                run();
                auto end = std::chrono::high_resolution_clock::now();
                milliseconds += std::chrono::duration<double, std::milli>(end - start).count();
            }
            milliseconds /= max_run;
            return milliseconds;
        }
};
    