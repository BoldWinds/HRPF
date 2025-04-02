#include "algorithm/utils.h"

void hsort(_TYPE* data, int len) {
    std::sort(data, data + len);
}

void gsort(_TYPE* data, int len, cudaStream_t stream) {
    thrust::device_ptr<_TYPE> dev_data(data);
    thrust::sort(thrust::cuda::par.on(stream), dev_data, dev_data + len);
}

void gmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int lenA, int lenB, cudaStream_t stream) {
    thrust::device_ptr<_TYPE> dev_first(first);
    thrust::device_ptr<_TYPE> dev_second(second);
    thrust::device_ptr<_TYPE> dev_dst(dst);

    // Thrust 合并
    thrust::merge(thrust::cuda::par.on(stream),
                  dev_first, dev_first + lenA,
                  dev_second, dev_second + lenB,
                  dev_dst);
}

void hmerge(_TYPE* srcA, _TYPE* srcB, _TYPE* dst, int lenA, int lenB) {
    std::merge(srcA, srcA + lenA, srcB, srcB + lenB, dst);
}

int hsplit(_TYPE* data, int len) {
    if (len <= 1) return 0; // 长度为1或0时，直接返回

    _TYPE pivot = data[len - 1];
    int i = -1;

    for (int j = 0; j < len - 1; ++j) {
        if (data[j] < pivot) {
            std::swap(data[++i], data[j]);
        }
    }
    std::swap(data[i + 1], data[len - 1]);

    return i + 1;
}

int gsplit(_TYPE* data, int len,  cudaStream_t stream){
    thrust::device_ptr<_TYPE> d_data(data);
    _TYPE pivot;
    cudaMemcpyAsync(&pivot, &data[len - 1], sizeof(_TYPE), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto end = thrust::partition(thrust::cuda::par.on(stream), d_data, d_data + len, [pivot] __device__ (_TYPE x) {
        return x < pivot;
    });

    return end - d_data;
}