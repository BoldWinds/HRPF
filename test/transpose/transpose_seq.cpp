#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

int main(int argc, char **argv) {
    int dim = atoi(argv[1]);
    int max_run = atoi(argv[2]);
    double *a = new double[dim*dim];
    double *b = new double[dim*dim];
    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j) {
            a[j + i*dim] = (double)(1);
        }
    }
        
    double milliseconds = 0;
    for(int run = 0; run < max_run; ++run){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for(int j = 0; j < dim; ++j){
            for(int i = 0; i < dim; ++i){
                b[j+i*dim] = a[i+j *dim];
            }
        }
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    delete [] a;
    delete [] b;
}