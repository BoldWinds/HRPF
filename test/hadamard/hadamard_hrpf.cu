#include <stdio.h>
#include <chrono>
#include <string>
#include <omp.h>
#include "algorithm/parallel_for_zero/parallel_for_zb.h"
#include "tool/initializer.h"
#include "framework/framework.h"
#include "tool/helper.h"

static int dim;

struct UserData_t : public Basedata_t{
public:
    UserData_t(std::vector<Matrix*>m_bf, std::vector<Matrix*> out
        ) : m_buffer(m_bf), result(out){
        }

public:
    std::vector<Matrix*> m_buffer;
    std::vector<Matrix*> result;
};

void cfor_func(Basedata_t* data){
    std::cout << "cfor_func" << std::endl;
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_cdata();
    auto b = ((UserData_t*)(d->buffer))->m_buffer[1]->get_cdata();
    auto c = ((UserData_t*)(d->buffer))->result[0]->get_cdata();

    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    size_t ldb = ((UserData_t*)(d->buffer))->m_buffer[1]->get_ld();
    size_t ldc = ((UserData_t*)(d->buffer))->result[0]->get_ld();

    
    size_t s_i = d->start;
    size_t e_i = d->end;
    int dim = e_i - s_i;
    
    #pragma omp parallel for
    for (int idx = 0; idx < dim * dim; ++idx) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void hadamard_product(const double* a, const double* b, double* c, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim*dim) {
        c[idx] = a[idx] * b[idx];
    }
}


void gfor_func(Basedata_t* data){
    auto start = std::chrono::high_resolution_clock::now();
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_gdata();
    auto b = ((UserData_t*)(d->buffer))->m_buffer[1]->get_gdata();
    auto c = ((UserData_t*)(d->buffer))->result[0]->get_gdata();

    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    size_t ldb = ((UserData_t*)(d->buffer))->m_buffer[1]->get_ld();
    size_t ldc = ((UserData_t*)(d->buffer))->result[0]->get_ld();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    int dim = e_i - s_i;
    int blockSize = 256;
    int gridSize = (dim*dim + blockSize - 1) / blockSize;
    hadamard_product<<<gridSize, blockSize, 0, stream()>>>(a, b, c, dim);
}


int main(int argc, char **argv){
    Framework::init();
    dim = std::atoi(argv[1]);
    //int max_run = std::atoi(argv[2]);
    Matrix* matrix1 = new Matrix(dim, dim);
    Matrix* matrix2 = new Matrix(dim, dim);
    Matrix* result = new Matrix(dim, dim);
    initialize(dim, matrix1);
    initialize(dim, matrix2);
    initialize(dim, result);
    UserData_t* user = new UserData_t({matrix1, matrix2}, {result});
    auto start = std::chrono::high_resolution_clock::now();
    parallel_for(new loopData_t(0, dim, user), cfor_func, gfor_func);
    auto end = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << milliseconds << std::endl;
    delete user;
    delete matrix1;
    delete matrix2;
    delete result;
    return 0;
}