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