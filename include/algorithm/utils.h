#pragma once
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#define _TYPE double

void gsort(_TYPE* data, int len,  cudaStream_t stream);
void gmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int lenA, int lenB, cudaStream_t stream);
void hsort(_TYPE* data, int len);
void hmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int lenA, int lenB);
int gsplit(_TYPE* data, int len,  cudaStream_t stream);
int hsplit(_TYPE* data, int len);
