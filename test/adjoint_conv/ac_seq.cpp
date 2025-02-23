#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>

static int length;
#define PI 3.14159

void seq_cfor_func(double* a, double* b, double* c){
    //std::cout << "enter seq ..." << std::endl;
    size_t s_i = 0;
    size_t e_i = length*length;
    size_t s_j = 0;
    size_t e_j = length*length;
    // std::cout << s_i << s_j << e_i << e_j << std::endl;
    // #pragma omp parallel for simd
    std::cout << e_i << std::endl;
    for(int i = s_i; i < e_i; ++i){
        double cur_c = 0;
        for(int j = i; j < e_j; ++j) {
           cur_c += 5.5*b[j]*a[j-i];
        } 
        c[i] = cur_c;
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

void print(double* datar, int length) {
    for(int i = 0; i < length; ++i){
        std::cout << datar[i] << " " ;;//<< datai[i] << std::endl;
        if(i && i % 4 == 0) std::cout << std::endl;
    }
}

int main(int argc, char **argv){
    std::size_t N = std::atoi(argv[1]);
    length = N;
    int max_run = std::atoi(argv[2]);
    // std::cout << "length:" << length << std::endl;
    double* a = new double[length*length];
    double* b = new double[length*length];
    double* c = new double[length*length];

    initialize(a, b, length*length);

    double milliseconds = 0;
    for(int run = 0; run < max_run; ++run){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        seq_cfor_func(a, b, c);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
        
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;

    delete []a;
    delete []b;
    delete []c;

    return 0;
}
