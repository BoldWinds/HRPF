#include "common/cpu_device.h"

void CpuDevice::dev_malloc(_TYPE** ptr, size_t width, size_t height) {
#if PARALLEL_FOR
    cudaHostAlloc(ptr, width * height * sizeof(_TYPE), cudaHostAllocMapped);
#else
    cudaMallocHost(ptr, width * height * sizeof(_TYPE));
#endif
}

void CpuDevice::dev_malloc(_TYPE** ptr, size_t length) {
#if PARALLEL_FOR
    cudaHostAlloc(ptr, length * sizeof(_TYPE), cudaHostAllocMapped);
#else
    cudaMallocHost(ptr, length * sizeof(_TYPE));
#endif
}

void CpuDevice::dev_free(void *ptr) {
    cudaFreeHost(ptr);
}

void CpuDevice::dev_mem_put(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {

    cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                     cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put(void* dst, void* src, size_t length) {
    cudaMemcpy(dst, src, sizeof(_TYPE)*length, cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put_asc(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {

    cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                     cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put_asc(void* dst, void* src, size_t length) {
    cudaMemcpy(dst, src, sizeof(_TYPE)*length, cudaMemcpyDeviceToHost);
}