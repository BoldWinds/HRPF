#include <stdio.h>
#include <sys/time.h>
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
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_cdata();
    auto b = ((UserData_t*)(d->buffer))->m_buffer[1]->get_cdata();
    auto c = ((UserData_t*)(d->buffer))->result[0]->get_cdata();

    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    size_t ldb = ((UserData_t*)(d->buffer))->m_buffer[1]->get_ld();
    size_t ldc = ((UserData_t*)(d->buffer))->result[0]->get_ld();

    
    size_t s_i = d->start;
    size_t e_i = d->end;
    
    #pragma omp parallel for num_threads(16)
    for(int i = s_i; i < e_i; ++i){
        for(int j = 0; j < dim; ++j) {
            c[i + j * ldc] = a[i + j * lda] * b[i + j * ldb];
        }
    }
}

__global__ void kernel_hadamard(size_t s_i, size_t e_i, size_t cols,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for(int i = start_i; i < end_i; ++i){
        for(int j = 0; j < cols; ++j) {
            c[i + j * ldc] = a[i + j * lda] * b[i + j * ldb];
        }
    }
}

void gfor_func(Basedata_t* data){
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_gdata();
    auto b = ((UserData_t*)(d->buffer))->m_buffer[1]->get_gdata();
    auto c = ((UserData_t*)(d->buffer))->result[0]->get_gdata();

    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    size_t ldb = ((UserData_t*)(d->buffer))->m_buffer[1]->get_ld();
    size_t ldc = ((UserData_t*)(d->buffer))->result[0]->get_ld();
    
    size_t s_i = d->start;
    size_t e_i = d->end;

    int blocks_required = 1;
    int threads_per_block = 1024;
    int chunk_size = 1;
    int size = e_i - s_i;
    if(size % (threads_per_block * chunk_size)) {
        blocks_required = size / (threads_per_block * chunk_size) + 1;
    }
    else {
        blocks_required = size / (threads_per_block * chunk_size);
    }
    cudaStream_t stream_ = stream();
    kernel_hadamard<<<blocks_required, threads_per_block, 0, stream_>>>(s_i, e_i, dim, 
        lda, ldb, ldc, chunk_size, a, b, c);
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
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, dim, user), cfor_func, gfor_func);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    delete user;
    delete matrix1;
    delete matrix2;
    delete result;
    return 0;
}