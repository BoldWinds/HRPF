#include "framework/framework.h"
#include "framework/task.h"

void Task::run(Problem* par, char c){
        switch (c)
        {
        case 'c':
            /* code */
            {
                auto cpu = Runtime::get_instance().get_cpu();
                for(int i = 0; i < m_size; ++i){
                    m_problems[i]->device = cpu;
                    m_problems[i]->IO(m_problems[i]->data);
                    m_problems[i]->cpu_func(m_problems[i]->data);
                }
            }    
            break;
        
        case 'g':
            {
                auto gpu = Runtime::get_instance().get_gpu();
#if ASNC
                std::thread::id id = std::this_thread::get_id();
                int t_idx = m_map[id];
                // {
                //     std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
                //     gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});  // �ȴ��п���������
                // }

                // {
                //     std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
                //     gpu->m_curr_stream[t_idx] = gpu->m_idle_streams.back();
                //     gpu->m_idle_streams.pop_back();
                // }
                 while(true)
                {
                    std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
                    {
                        if(!gpu->m_idle_streams.empty()) {
                            gpu->m_curr_stream[t_idx] = gpu->m_idle_streams.back();
                            gpu->m_idle_streams.pop_back();
                            break;
                        } 
                        else{
                            gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});
                        }
                    }
                }

                if(HANDLE_NUM_) cublasSetStream(gpu->m_handle[0], gpu->m_idle_streams[t_idx]);
                // std::cout << "tc:" << par->rc <<std::endl;
                par->rc.fetch_add(m_size);
                for(int i = 0; i < m_size; ++i){
                    m_problems[i]->device = gpu;
                    m_problems[i]->IO(m_problems[i]->data);
                    m_problems[i]->gpu_func(m_problems[i]->data);    
                }
                
                CallBackData *cbd = new CallBackData(gpu, par, m_size, this);
                cudaStreamAddCallback(gpu->m_curr_stream[t_idx], call_back_mer, cbd, 0);
                cudaStreamSynchronize(gpu->m_curr_stream[t_idx]);
                // std::cout << "after tc:" << par->rc <<std::endl;
#else  
                for(int i = 0; i < m_size; ++i){
                    m_problems[i]->device = gpu;
                    m_problems[i]->IO(m_problems[i]->data);
                    m_problems[i]->gpu_func(m_problems[i]->data);    
                }
#endif
            }
            break;
        }
    }