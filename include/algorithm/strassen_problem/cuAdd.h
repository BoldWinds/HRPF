#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define _TYPE double
void sumMatrixInplace(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, const double p_one);
void sumMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void subMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void gemm(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void gemm(double* MatA, double* MatB, double* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream, cublasHandle_t handle);
