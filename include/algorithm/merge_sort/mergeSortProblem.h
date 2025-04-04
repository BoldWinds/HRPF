#pragma once

#include "framework/problem.h"
#include "framework/task.h"
#include "common/gpu_device.h"
#include "datastructure/arraylist.h"
#include "algorithm/utils.h"
#include <string>
//void set_mask(std::string mask);
struct MergeData_t : public Basedata_t{
public:
    ArrayList* ha;
    // ArraySt* m_ha;

    MergeData_t(ArrayList* a) {
        ha = a;
        // m_ha = nullptr;
    }

};

class MergesortProblem: public Problem {
public:
    //typedef typename std::function<void(ArraySt*, ArraySt*)> Function;
    std::vector<Problem*> split() override;
    void merge(std::vector<Problem*>& subproblems) override;
    bool mustRunBaseCase();
	bool canRunBaseCase(int index);
	
public:
    MergesortProblem(Basedata_t* m_data, Function _cf, Function _gf, Problem* par);
    ~MergesortProblem(){
        if(data != nullptr){
            delete data;
            data = nullptr;
        }
    }

    void Input() override;
    void Output() override;
    void IO(Basedata_t* m_data) override;
};

#define MergeSort_t MergesortProblem

void ms_cpu_sort(Basedata_t* data);
void ms_gpu_sort(Basedata_t* data);
void ms_merge_cpu(Basedata_t* data);
void ms_merge_gpu(Basedata_t* data);
