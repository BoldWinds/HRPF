#pragma once

#include <iostream>


enum class MemState : char { INVALID = 0, SHARED = 1, EXCLUSIVE = 2 };

enum class MemAccess : char { R = 1, W = 2 };

enum class TaskType : char { CPU_ONLY = 1, GPU_ONLY = 2, CPU_GPU = 3 };

enum class DeviceType : char { CPU = 1, GPU = 2 };

enum class DeviceState : char { STOPED = 0, RUNNING = 1 };

enum class RuntimeState : char { STOPED = 0, RUNNING = 1, WAITTING = 3 };

// inline std::ostream opreator << (std::ostream & os, const DeviceType &dt) {
//   switch (dt) {
//   case DeviceType::CPU:
//     os << "CPU";
//     break;
//   case DeviceType::GPU:
//     os << "GPU";
//     break;
//   default:
//     OS_ERROR("unknown device type!!!");
//     break;
//   }
// }
#ifndef _TYPE
#define _TYPE double
#endif
// enum class DeviceState : char { STOPED = 0, RUNNING = 1 };

// enum class RuntimeState : char { STOPED = 0, RUNNING = 1, WAITTING = 3 };