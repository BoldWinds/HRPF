// #include <stdio.h>
// #include <sys/time.h>
// #include <string>

// #include "algorithm/parallel_for_zero/parallel_for_zb.h"
// #include "tool/initializer.h"
// #include "framework/framework.h"
// #include "tool/helper.h"
// // #include "parallel_for_harness.hpp"
// #include <omp.h>

// struct UserData_t : public Basedata_t{
// public:
//     UserData_t(std::vector<ArrayList*> buf
//         ) : buffer(buf){
//         }

// public:
//     std::vector<ArrayList*> buffer;
// };

// void cfor_func(Basedata_t* data){
//     // std::cout << "cpu exec" << std::endl;
//     auto d = (loopData_t*)data;
//     auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
//     auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
//     auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();

//     size_t s = d->start;
//     size_t e = d->end;
//     #pragma omp parallel for
//     for(int i = s; i < e; ++i){
//         c[i] = a[i] + b[i];
//     }
// }

// __global__ void kernel(size_t s, size_t e, size_t chunk, double* a, double* b, double* c){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int start = s + tid * chunk;
//     int end = start+chunk < e ? start + chunk : e;

//     for(int i = start; i < end; ++i){
//         c[i] = a[i] + b[i];
//     }
// }

// void gfor_func(Basedata_t* data){
// //    std::cout << "gpu exec" << std::endl;
//     auto d = (loopData_t*)data;
//     auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_gdata();
//     auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_gdata();
//     auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_gdata();

//     size_t s = d->start;
//     size_t e = d->end;

//     int blocks_required = 1;
//     int threads_per_block = 1024;
//     int chunk_size = 1;
//     int size = e - s;
//     if(size % (threads_per_block * chunk_size)) {
//         blocks_required = size / (threads_per_block * chunk_size) + 1;
//     }
//     else {
//         blocks_required = size / (threads_per_block * chunk_size);
//     }
//     cudaStream_t stream_ = stream();
//     kernel<<<blocks_required, threads_per_block, 0, stream_>>>(s, e, chunk_size, a, b, c);
// }

// int main(int argc, char **argv){
//     Framework::init();
//     std::size_t length = std::atoi(argv[1]);
//     ArrayList* data1 = new ArrayList(length);
//     ArrayList* data2 = new ArrayList(length);
//     ArrayList* data3 = new ArrayList(length);
//     initialize(data1, length);
//     initialize(data2, length);
//     initialize(data3, length);

//     UserData_t* user = new UserData_t({data1, data2, data3});
//     // auto loop = new loopData_t(0, length, user);
//     struct timeval start, end;
//     gettimeofday(&start, NULL);
//     parallel_for(new loopData_t(0, length, user), cfor_func, gfor_func);
//     // gfor_func(loop);
//     gettimeofday(&end, NULL);
//     double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
//     std::cout << milliseconds << std::endl;
//     auto res = data3->get_cdata();
//     /*for(int i = 0; i < length; ++i) {
//         // if(i && i % 16 == 0) std::cout << std::endl;
//         std::cout <<(data1->get_cdata())[i] << " " <<(data2->get_cdata())[i] << " "<< res[i] << std::endl;
//     }*/

//     delete user;
//     delete data1;
//     delete data2;
//     delete data3;
//     return 0;
// }

#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include "framework/framework.h"
#include "algorithm/strassen_problem/strassenProblem.h"
#include "algorithm/parallel_for_zero/parallel_for_zb.h"
#include "tool/initializer.h"

int main(int argc, char **argv)
{
    int dim = std::atoi(argv[1]);
    std::string interleaving = argv[2];
    Matrix *matrix1 = new Matrix(dim, dim);
    Matrix *matrix2 = new Matrix(dim, dim);
    Matrix *result = new Matrix(dim, dim);
    initialize(dim, matrix1);
    initialize(dim, matrix2);
    initialize(dim, result);

    Framework::init();
    StrassenProblem *problem = new StrassenProblem(new StrassenData_t(matrix1, matrix2, result), cpu_mul, gpu_mul, nullptr);
    StrassenData_t *user = new StrassenData_t(matrix1, matrix2, result);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Framework::solve(problem, interleaving);
    gettimeofday(&end, NULL);

    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    delete problem;
    delete matrix1;
    delete matrix2;
    delete result;
    return 0;
}