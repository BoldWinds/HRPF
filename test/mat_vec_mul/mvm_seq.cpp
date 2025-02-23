#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

int main(int argc, char **argv) {
    int dim = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    
    double *a = new double[dim*dim];
    double *b = new double[dim];
    double *c = new double[dim];

    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j) {
            a[j + i*dim] = (double)(1);
        }
    }

    for(int i = 0; i < dim; ++i){
        b[i] = (double)(rand() % 100);
    }
    double milliseconds = 0;
    for(int run = 0; run <= max_run; ++run){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for(int j = 0; j < dim; ++j){
            double loc = 0;
            for(int i = 0; i < dim; ++i){
                loc += (a[j+i *dim] * b[i]);
            }
            c[j] = loc;
        }
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;

    delete [] a;
    delete [] b;
    delete [] c;
}