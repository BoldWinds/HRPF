#pragma once

#include <cuda_runtime.h>

#define _TYPE double
void sumMatrixInplace(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, const double p_one);
void sumMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void subMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void gemm(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
