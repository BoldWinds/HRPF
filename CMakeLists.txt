cmake_minimum_required(VERSION 3.18)
project(hrpf LANGUAGES CXX CUDA)

# -------------------------------
# 设置 C++ 和 CUDA 标准
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# 设置输出目录
# -------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------------------
# 设置构建类型、PIC 和调试标志
# -------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O3 -DNDEBUG)

# -------------------------------
# 添加 OpenMP 支持
# -------------------------------
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -------------------------------
# 添加 StarPU 支持
# -------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(STARPU REQUIRED starpu-1.3)

if(STARPU_FOUND)
    message(STATUS "Found StarPU: ${STARPU_LIBRARIES}")
    include_directories(${STARPU_INCLUDE_DIRS})
    link_directories(${STARPU_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "StarPU not found")
endif()

# -------------------------------
# 设置 CUDA 编译标志
# -------------------------------
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --extended-lambda")

# -------------------------------
# 添加包含目录（根据 Makefile INCLUDE 变量）
# -------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include    # 项目头文件
    /usr/local/cuda/include        # CUDA 头文件
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
# 生成共享库 libhrpf.so
# -------------------------------
add_library(hrpf SHARED ${ALL_SOURCES})
set_target_properties(hrpf PROPERTIES
    OUTPUT_NAME "hrpf"
    SUFFIX ".so"
)

# -------------------------------
# 设置链接器标志（如 --no-as-needed）
# -------------------------------
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

# -------------------------------
# 链接外部库（根据 Makefile LDFLAGS）
# -------------------------------
target_link_libraries(hrpf PUBLIC
#    ${CMAKE_SOURCE_DIR}/lib/libcudaMerge.so  # 本项目的自定义 CUDA 库
    cudart                                    # CUDA Runtime
    cublas                                    # cuBLAS 库
    iomp5                                     # Intel OpenMP
    mkl_intel_lp64                            # MKL 库
    mkl_core
    mkl_gnu_thread
    pthread                                   # POSIX 线程
    m                                         # 数学库
    OpenMP::OpenMP_CXX                        # OpenMP（如使用 gcc 的 libgomp，则可能不需要额外指定）
    ${STARPU_LIBRARIES}                       # StarPU 库
    dl                                        # 动态链接库
)

# -------------------------------
# 添加测试目标
# -------------------------------
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    # 搜索 test 目录下的所有 .cpp 和 .cu 文件
    file(GLOB_RECURSE TEST_SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp ${CMAKE_SOURCE_DIR}/test/*.cu)

    foreach(test_file ${TEST_SOURCES})
        # 从文件路径中提取子目录和文件名（去掉扩展名）
        get_filename_component(test_name ${test_file} NAME_WE)
        get_filename_component(test_dir ${test_file} DIRECTORY)

        # 计算对应的输出目录（与源文件目录相同）
        string(REPLACE ${CMAKE_SOURCE_DIR}/test ${CMAKE_BINARY_DIR}/bin OUTPUT_DIR ${test_dir})

        # 创建对应的输出目录（如果不存在）
        file(MAKE_DIRECTORY ${OUTPUT_DIR})

        # 为每个测试源文件生成一个可执行文件，并设置输出目录
        add_executable(${test_name} ${test_file})
        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
        )

        # 链接共享库 hrpf
        target_link_libraries(${test_name} PUBLIC hrpf)
    endforeach()
endif()