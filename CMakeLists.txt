cmake_minimum_required(VERSION 3.10)
project(HRPA LANGUAGES CXX CUDA)

# -------------------------------
# 设置 C++ 和 CUDA 标准
# -------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)    # 针对4090, 更换GPU需要更改计算能力

# -------------------------------
# 设置构建类型、PIC 和调试标志
# -------------------------------
# 默认 Debug 模式（包含 -O0 和 -g），共享库需要 PIC
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O0 -g)

# -------------------------------
# 添加 OpenMP 支持
# -------------------------------
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -------------------------------
# 设置 CUDA 编译标志
# -------------------------------
# NVCCFLAGS：-I. -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC -std=c++14
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I. -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC")

# -------------------------------
# 添加包含目录（根据 Makefile INCLUDE 变量）
# -------------------------------
include_directories(
        ${CMAKE_SOURCE_DIR}/include    # 项目头文件
        /usr/local/cuda/include        # CUDA 头文件
        /usr/include/c++/11            # C++ 标准库（如有需要）
        /opt/intel/oneapi/mkl/latest/include  # MKL 头文件
)

# -------------------------------
# 添加库搜索路径（根据 LDFLAGS 中的 -L 选项）
# -------------------------------
link_directories(
        /usr/local/cuda/lib64
        /opt/intel/oneapi/mkl/latest/lib
)

# -------------------------------
# 搜索源文件（递归查找 src 下的 .cpp 和 .cu 文件）
# -------------------------------
file(GLOB_RECURSE CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
set(ALL_SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

# -------------------------------
# 生成共享库 libframework.so
# -------------------------------
add_library(framework SHARED ${ALL_SOURCES})
set_target_properties(framework PROPERTIES
        OUTPUT_NAME "framework"
        SUFFIX ".so"
)

# -------------------------------
# 设置链接器标志（如 --no-as-needed）
# -------------------------------
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

# -------------------------------
# 链接外部库（根据 Makefile LDFLAGS）
# -------------------------------
target_link_libraries(framework PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/libcudaMerge.so  # 本项目的自定义 CUDA 库
        cudart                                    # CUDA Runtime
        cublas                                    # cuBLAS 库
        dl                                        # 动态链接库
        iomp5                                     # Intel OpenMP
        mkl_intel_lp64                            # MKL 库
        mkl_core
        mkl_gnu_thread
        pthread                                   # POSIX 线程
        m                                         # 数学库
        OpenMP::OpenMP_CXX                        # OpenMP（如使用 gcc 的 libgomp，则可能不需要额外指定）
)
