#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

static int dim ;

int main(int argc, char **argv) {
    dim = std::atoi(argv[1]);
    double *a = new double[dim*dim];
    double *b = new double[dim*dim];
    double *c = new double[dim*dim];

    // initialize
    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j) {
            a[j + i*dim] = (double)(rand() % 100);
            b[j + i*dim] = (double)(rand() % 100);
        }
    }

    struct timeval start, end;
    int i,j;
    gettimeofday(&start, NULL);
    for( j = 0; j < dim; ++j){
        for( i = 0; i < dim; ++i)
        {
            c[i+j *dim] = a[i+j *dim] * b[i+j *dim];
        }
    }
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    delete [] a;
    delete [] b;
    delete [] c;
}