#pragma once

#include "common/gpu_device.h"
#include "common/runtime.h"

cublasHandle_t handle();
cudaStream_t stream();
