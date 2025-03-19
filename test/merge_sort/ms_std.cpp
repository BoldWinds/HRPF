#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/time.h>

void loadData(double* datar, int length) {
    std::ifstream fin;
    fin.open("./data/datamer.txt");

    if(!fin)
    {
        std::cout<<"can not open the file data.txt"<<std::endl;
        exit(1);
    }

    for(int i = 0; i < length; ++i){
        fin >> datar[i];
    }
}

int main(int argc, char** argv){
    int n = std::atoi(argv[1]);
    int max_run = std::atoi(argv[2]);
    double* data = new double[n];
    loadData(data, n);
    double milliseconds = 0;

    for(int run = 0; run < max_run; run++){
        struct timeval start, end;
        gettimeofday(&start, NULL);
        std::sort(data, data+n);
        gettimeofday(&end, NULL);
        milliseconds += (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    }
    milliseconds /= max_run;
    std::cout << milliseconds << std::endl;
}