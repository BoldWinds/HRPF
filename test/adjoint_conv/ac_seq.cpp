#include <sys/time.h>
#include <iostream>
#include <fstream>

static int length;
#define PI 3.14159

void conv(double* a, double* b, double* c){
    for(int i = 0; i < length; ++i){
        double cur_c = 0;
        for(int j = i; j < length; ++j) {
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

int main(int argc, char **argv){
    std::size_t N = std::atoi(argv[1]);
    length = N;
    double* a = new double[length];
    double* b = new double[length];
    double* c = new double[length];

    initialize(a, b, length);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    conv(a, b, c);
    gettimeofday(&end, NULL);
    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;

    delete []a;
    delete []b;
    delete []c;

    return 0;
}
