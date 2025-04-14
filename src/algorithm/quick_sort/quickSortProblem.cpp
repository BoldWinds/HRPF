#include "algorithm/quick_sort/quickSortProblem.h"


QuicksortProblem::QuicksortProblem(Basedata_t* m_data, Function _cf, Function _gf, Problem* par) {
    data = m_data;
    cpu_func = _cf;
    gpu_func = _gf;
    parent = par;
    m_mask = std::bitset<T_SIZE>("1100");
}

bool QuicksortProblem::canRunBaseCase(int index) {
    return false;
}

bool QuicksortProblem::mustRunBaseCase() {
    auto d = (QuickData_t*)data;
    return d->ha->length() <= 1024;
}

void QuicksortProblem::Input() {
    auto d = (QuickData_t*)data;
    input(d->ha);
}

void QuicksortProblem::Output() {
    auto d = (QuickData_t*)data;
    output(d->ha);
}

void QuicksortProblem::IO(Basedata_t* m_data) {
    auto d = (QuickData_t*)m_data;
    inputAsc(d->ha);
    outputAsc(d->ha);
}

/***************************
* cpu split operation
***************************/
/*void qs_cpu_partition(Basedata_t* data) {
	auto d = (QuickData_t*)data;
    int m_len = d->ha->length();
    auto m_data = d->ha->get_cdata();
    d->pivotIndex = hsplit(m_data, m_len);
}*/

/***************************
* gpu split operation
***************************/
/*void qs_gpu_partition(Basedata_t* data) { 
    auto d = (QuickData_t*)data;
    int m_len = d->ha->length();
	auto m_data = d->ha->get_gdata();
    d->pivotIndex = gsplit(m_data, m_len, stream());
}*/

/***************************
* cpu sort operation
***************************/
void qs_cpu_sort(Basedata_t* data) {
	auto d = (QuickData_t*)data;
    int m_len = d->ha->length();
    auto m_data = d->ha->get_cdata();
    hsort(m_data, m_len);
}

/***************************
* gpu sort operation
***************************/
void qs_gpu_sort(Basedata_t* data) { 
    auto d = (QuickData_t*)data;
    int m_len = d->ha->length();
	auto m_data = d->ha->get_gdata();
	gsort(m_data, m_len, stream());
}

/***************************
* 分割操作
* 快速排序在此阶段基于 pivot 对数组进行分区，
* 假设数据容器提供 partition 接口，返回 pivot 的下标，
* 并根据该下标生成左右两个子区间。
***************************/
std::vector<Problem*> QuicksortProblem::split() {
    auto d = (QuickData_t*)data;
    int len = d->ha->length();
    int pivotIndex = gsplit(d->ha->get_gdata(), len, stream());
    if(pivotIndex <= 1 ) {
        d->ha->build_xchilds(pivotIndex);
        std::vector<Problem*> tasks(1);
        tasks[0] = new QuicksortProblem(new QuickData_t(d->ha->get_child(1)), qs_cpu_sort, qs_gpu_sort, this);
        return tasks;
    }else if(pivotIndex >= len - 2) {
        d->ha->build_xchilds(pivotIndex);
        std::vector<Problem*> tasks(1);
        tasks[0] = new QuicksortProblem(new QuickData_t(d->ha->get_child(0)), qs_cpu_sort, qs_gpu_sort, this);
        return tasks;
    }else{
        d->ha->build_xchilds(pivotIndex);
        auto left = d->ha->get_child(0);
        auto right = d->ha->get_child(1);
        std::vector<Problem*> tasks(2);
        tasks[0] = new QuicksortProblem(new QuickData_t(left), qs_cpu_sort, qs_gpu_sort, this);
        tasks[1] = new QuicksortProblem(new QuickData_t(right), qs_cpu_sort, qs_gpu_sort, this);
        return tasks;
    }
}

/***************************
* 合并操作快速排序为原地排序算法，不需要额外的归并操作
***************************/
void QuicksortProblem::merge(std::vector<Problem*>& subproblems)  {
    return;
}
