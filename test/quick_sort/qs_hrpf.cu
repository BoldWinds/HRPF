#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <execution>
#include "algorithm/utils.h"
#include "framework/framework.h"
#include "algorithm/quick_sort/quickSortProblem.h"
#include "tool/initializer.h"

void loadData(double *datar, int length) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, datar, datar + length, [&](double &val) {
        val = dist(rng);
    });
}

int main(int argc, char **argv){

    std::size_t length = std::atoi(argv[1]);
    std::string interleaving = argv[2];

    ArrayList* data = new ArrayList(length);
    auto& runtime = Runtime::get_instance();
    //auto cpu = runtime.get_cpu();
    auto cpu = runtime.get_gpu();
    (data)->access(cpu, MemAccess::W);
    loadData(data->get_cdata(), length);
    // initialize(data, length);
    Framework::init();
    QuicksortProblem* problem = new QuicksortProblem(new QuickData_t(data), qs_cpu_sort, qs_gpu_sort, nullptr);
    //std::string mask = "10";
    //problem->set_mask(mask);
    //std::cout << "init problem & threads end" << std::endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Framework::solve(problem, interleaving);
    //data->access(Runtime::get_instance().get_cpu(), MemAccess::R);
    gettimeofday(&end, NULL);

    double milliseconds = (end.tv_sec - start.tv_sec) * 1000 + 1.0e-3 * (end.tv_usec - start.tv_usec);
    std::cout << milliseconds << std::endl;
    // data->access(Runtime::get_instance().get_cpu(), MemAccess::R);
    //_TYPE* dd = data->get_cdata();

    // for(int i = 0;  i < length; ++i){
    // 	std::cout << dd[i] <<" ";
    // 	if(i&&i % 16 == 0) std::cout << std::endl;
    // }

    delete problem;
    //std::cout << "delete problem..." << std::endl;
    delete data;
    return 0;
}

