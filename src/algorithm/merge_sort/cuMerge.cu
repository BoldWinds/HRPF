#include "algorithm/merge_sort/cuMerge.h"

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

void hsort(_TYPE* data, int len) {
    std::sort(data, data + len);
}