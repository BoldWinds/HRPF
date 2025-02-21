#include "tool/helper.h"
#include <thread>
#include "framework/framework.h"

cublasHandle_t handle(){
// #if ASNC
//     std::thread::id id = std::this_thread::get_id();
//     int index = m_map[id]; 
//     // Runtime& instance = Runtime::get_instance();
                   
//     return Runtime::get_instance().get_gpu()->m_handle[index];
// #else
//     return Runtime::get_instance().get_gpu()->m_handle[0];
// #endif
    return Runtime::get_instance().get_gpu()->m_handle[0];
}

cudaStream_t stream(){
    std::thread::id id = std::this_thread::get_id();
    int index = m_map[id]; 
    // Runtime& instance = Runtime::get_instance();
    // if(index < c_num)               
    return Runtime::get_instance().get_gpu()->m_curr_stream[index];
    // return Runtime::get_instance().get_gpu()->m_handle_;
}
