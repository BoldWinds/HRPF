#include "algorithm/merge_sort/mergeSortProblem.h"
#include <cstring>
#include <algorithm>
#include <bitset>
#include <chrono>

MergesortProblem::MergesortProblem(Basedata_t* m_data, Function _cf, Function _gf, Problem* par) {
    data = m_data;
    cpu_func = _cf;
    gpu_func = _gf;
    parent = par;        
    m_mask = std::bitset<T_SIZE>("1100"); 
	//set_mask("01");	
}

bool MergesortProblem::canRunBaseCase(int index) {
	//return m_mask[index] == 1;
    return false;
}

/***************************
* end condition
***************************/
bool MergesortProblem::mustRunBaseCase() {
    auto d = (MergeData_t*)data;
    return d->ha->length() <= 1024;
}

bool cmp(_TYPE x, _TYPE y) {
    return x <= y;
}

void MergesortProblem::Input() {
    auto d = (MergeData_t*)data;
    input(d->ha);
}

void MergesortProblem::Output() {
    auto d = (MergeData_t*)data;
    output(d->ha);
}

void MergesortProblem::IO(Basedata_t* m_data) {
    auto d = (MergeData_t*)m_data;

    inputAsc(d->ha);
    outputAsc(d->ha);
}

/***************************
* cpu sort operation
***************************/
void ms_cpu_sort(Basedata_t* data) {
	auto d = (MergeData_t*)data;
    int m_len = d->ha->length();
    auto m_data = d->ha->get_cdata();
    hsort(m_data, m_len);
}

/***************************
* gpu sort operation
***************************/
void ms_gpu_sort(Basedata_t* data) { 
    auto d = (MergeData_t*)data;
    int m_len = d->ha->length();
	auto m_data = d->ha->get_gdata();
	gsort(m_data, m_len, stream());
}

/***************************
* cpu merge operation
***************************/
void ms_merge_cpu(Basedata_t* data) {
    auto d = (MergeData_t*)data;
    auto first = d->ha->get_child(0);
    auto second = d->ha->get_child(1);
    int len = d->ha->length();
    int lenA = first->length();
    auto src_dataA = d->ha->get_cdata();
    auto src_dataB = src_dataA + lenA;
    auto dst_data = new _TYPE[len]; 
    int lenB =  len - lenA;
    hmerge(src_dataA, src_dataB, dst_data, lenA, lenB);
    memcpy(src_dataA, dst_data, sizeof(_TYPE) * len);
    delete dst_data;
}

/***************************
* gpu merge operation
***************************/
void ms_merge_gpu(Basedata_t* data) {
    auto d = (MergeData_t*)data;
    auto first = d->ha->get_child(0);
    auto second = d->ha->get_child(1);
    int len = d->ha->length();
    int lenA = first->length();
    int lenB = len - lenA;
    auto src_dataA = d->ha->get_gdata();
    auto src_dataB = src_dataA + lenA;
    _TYPE* dst_data;
	int size = len*sizeof(_TYPE);
	cudaMalloc((void**)&dst_data, size);
	gmerge(src_dataA, src_dataB, dst_data, lenA, lenB, stream());
	cudaMemcpyAsync(src_dataA, dst_data, size, cudaMemcpyDeviceToDevice, stream());
	cudaFree(dst_data);
}

/***************************
* split operation
***************************/
std::vector<Problem*> MergesortProblem::split() {
    auto d = (MergeData_t*)data;
    d->ha->build_childs();
    auto first = d->ha->get_child(0);
    auto second = d->ha->get_child(1);
    
    std::vector<Problem*> tasks(2);
    tasks[0] = new MergesortProblem(new MergeData_t(first), ms_cpu_sort, ms_gpu_sort, this);
    tasks[1] = new MergesortProblem(new MergeData_t(second), ms_cpu_sort, ms_gpu_sort, this);
    return tasks;
}

/***************************
* merge operation
***************************/
void MergesortProblem::merge(std::vector<Problem*>& subproblems)  {
    auto d = (MergeData_t*)data;
    // run_task(new MergeData_t(d->ha), merge_cpu, merge_gpu);//
    MergesortProblem* merge_p = new MergesortProblem(new MergeData_t(d->ha), ms_merge_cpu, ms_merge_gpu, this);
    //merge_p->record_device('g');
    merge_p->runAsc('g');
}
