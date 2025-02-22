#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

int main(int argc, char **argv) {
    int dim = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    double *x1 = new double[dim];
    double *x2 = new double[dim];
    double *x3 = new double[dim];
    double *mass = new double[dim];
    double *v1 = new double[dim];
    double *v2 = new double[dim];
    double *v3 = new double[dim];

    for(int i = 0; i < dim; ++i){
        x1[i] = (double)(rand() % 100);
        x2[i] = (double)(rand() % 100);
        x3[i] = (double)(rand() % 100);
        mass[i] = (double)(rand() % 100);
        v1[i] = (double)(rand() % 100);
        v2[i] = (double)(rand() % 100);
        v3[i] = (double)(rand() % 100);
    }
    double dt = (double)(rand() % 10);
    double milliseconds = 0;
    for(int run = 0; run < max_run; ++run) {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for(int i = 0; i < dim; ++i){
            double Fx = 0; double Fy = 0; double Fz = 0;
            for(int j = 0; j < dim; ++j) {
                double dx = x1[j] - x1[i];
                double dy = x2[j] - x2[i];
                double dz = x3[j] - x3[i];
                double dst = dx*dx + dy*dy + dz*dz + mass[i];
                double invDist = 1.0 / sqrt(dst);
                double invDist3 = pow(invDist, 3);
                Fx += dx*invDist3; Fy += dy*invDist3; Fz += dz*invDist3;
            }
            v1[i] += dt * Fx; 
            v2[i] += dt*Fy; 
            v3[i] += dt*Fz;
        }
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
    delete [] x1;
    delete [] x2;
    delete [] x3;
    delete [] v1;
    delete [] v2;
    delete [] v3;
    delete [] mass;
}