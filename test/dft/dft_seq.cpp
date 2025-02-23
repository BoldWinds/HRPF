#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>

static int length;
#define PI 3.14159

void cfor_func(double* ar, double* ai, double* tr, double* ti){
    size_t s_i = 0;
    size_t e_i = length;
    size_t s_j = 0;
    size_t e_j = length;

    #pragma omp parallel for num_threads(16)
    for(int i = s_i; i < e_i; ++i){
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI * i * j / length );
           wni = sin(2.0 * PI * i * j / length);
           tr[i] += (ar[j] * wnr - ai[j] * wni);
           ti[i] += (ar[j] * wni + ai[j] * wnr);
        }
    }
}

void seq_cfor_func(double* ar, double* ai, double* tr, double* ti){
    size_t s_i = 0;
    size_t e_i = length;
    size_t s_j = 0;
    size_t e_j = length;
    for(int i = s_i; i < e_i; ++i){
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI * i * j / length);
           wni = sin(2.0 * PI * i * j / length);

           tr[i] += (ar[j] * wnr - ai[j] * wni);
           ti[i] += (ar[j] * wni + ai[j] * wnr);
        }
    }
}

void initialize(double* datar, double* datai, int length) {
    std::ifstream fin;
    fin.open("./data/datadft.txt");

	if(!fin)
	{
		std::cout<<"can not open the file data.txt"<<std::endl;
		exit(1);
	}

    for(int i = 0; i < length; ++i){
        fin >> datar[i] >> datai[i];
    }
}


int main(int argc, char **argv){
    std::size_t N = std::atoi(argv[1]);
    length = N;
    int max_run = std::atoi(argv[2]);

    double* datar = new double[length];
    double* datai = new double[length];
    double* tempr = new double[length];
    double* tempi = new double[length];
    initialize(datar, datai, length);

    double milliseconds = 0;
    for(int run = 0; run < max_run; ++run){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        seq_cfor_func(datar, datai, tempr, tempi);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;

    delete []datar;
    delete []datai;
    delete []tempr;
    delete []tempi;
    return 0;
}
