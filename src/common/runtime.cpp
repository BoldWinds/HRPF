#include "common/runtime.h"
#include <iostream>

Runtime::Runtime() {
    srand(time(0));
    runtime_state_ = RuntimeState::STOPED;
    
    cpu_ = new CpuDevice();
    gpu_ = new GpuDevice();
}

Runtime::~Runtime() {
    delete cpu_;
    delete gpu_;
    cpu_ = nullptr;
    gpu_ = nullptr;
    // std::cout << "runtime delete..." << std::endl;
}

CpuDevice* Runtime::get_cpu() {
    return cpu_;
}

GpuDevice* Runtime::get_gpu() {
    return gpu_;
}

