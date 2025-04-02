#pragma once

#include "algorithm/utils.h"
#include "framework/problem.h"
#include "framework/task.h"
#include "common/gpu_device.h"
#include "datastructure/arraylist.h"
#include <string>
#include <algorithm>
#include <bitset>

//void set_mask(std::string mask);
struct QuickData_t : public Basedata_t{
public:
    ArrayList* ha;
    int pivotIndex;
    // ArraySt* m_ha;

    QuickData_t(ArrayList* a) {
        ha = a;
        pivotIndex = 0;
        // m_ha = nullptr;
    }

};

class QuicksortProblem: public Problem {
public:
    //typedef typename std::function<void(ArraySt*, ArraySt*)> Function;
    std::vector<Problem*> split() override;
    void merge(std::vector<Problem*>& subproblems) override;
    bool mustRunBaseCase();
	bool canRunBaseCase(int index);
	
public:
    QuicksortProblem(Basedata_t* m_data, Function _cf, Function _gf, Problem* par);
    ~QuicksortProblem(){
        if(data != nullptr){
            delete data;
            data = nullptr;
        }
    }

    void Input() override;
    void Output() override;
    void IO(Basedata_t* m_data) override;
};

void qs_cpu_sort(Basedata_t* data);
void qs_gpu_sort(Basedata_t* data);
void qs_cpu_partition(Basedata_t* data);
void qs_gpu_partition(Basedata_t* data);
