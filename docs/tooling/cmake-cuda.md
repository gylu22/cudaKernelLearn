# CMake 与 CUDA：语言支持和 Toolkit 库是两件事

本仓库当前使用 PyTorch `CUDAExtension`，不依赖 CMake。本页用于以后编写独立 CUDA
示例或 C++ 可执行文件。

## 1. 启用 CUDA 语言

让 CMake 编译 `.cu` 文件：

```cmake
cmake_minimum_required(VERSION 3.24)
project(cuda_example LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
```

启用 CUDA language 后，CMake 选择 CUDA compiler 并为 CUDA target 处理必要的
device 编译和 runtime 链接。不要全局硬编码 `-gencode`；优先使用
`CMAKE_CUDA_ARCHITECTURES` 或 target 的 `CUDA_ARCHITECTURES`。

## 2. `find_package(CUDAToolkit)`

当需要显式调用 cuBLAS、cuRAND、NVRTC 等 Toolkit 库，或项目本身不编译 CUDA
source 时：

```cmake
find_package(CUDAToolkit REQUIRED)

add_executable(gemm main.cpp)
target_link_libraries(gemm PRIVATE CUDA::cudart CUDA::cublas)
```

`FindCUDAToolkit` 是 CMake 自带的 find module（CMake 3.17 起提供），会创建
`CUDA::cudart`、`CUDA::cublas` 等 imported target。它不要求项目启用 CUDA
language。

原笔记把它描述成搜索 `CUDAToolkitConfig.cmake`，这不准确：标准入口是 CMake 的
`FindCUDAToolkit` 模块，其搜索顺序会考虑已启用 CUDA compiler、
`CUDAToolkit_ROOT`、`CUDA_PATH`、`PATH` 和平台默认位置。

## 3. 为什么 `target_link_libraries` 也影响编译

现代 CMake target 带有 usage requirements。链接 imported target 时，其
`INTERFACE_INCLUDE_DIRECTORIES`、编译定义和传递依赖会应用到消费者，因此既影响
编译也影响链接：

```cmake
add_library(my_kernel STATIC kernel.cu)
target_include_directories(my_kernel PUBLIC include)
target_link_libraries(my_kernel PUBLIC CUDA::cudart)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE my_kernel)
```

`PUBLIC/PRIVATE/INTERFACE` 控制依赖是否继续传播，不要用全局
`include_directories()` 污染所有 target。

## 4. 常用 target 属性

```cmake
set_target_properties(my_kernel PROPERTIES
    CUDA_ARCHITECTURES "80;86"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
```

- `CUDA_SEPARABLE_COMPILATION` 仅在跨 translation unit 的 device symbol/device
  linking 有需要时开启；
- generator expression 确保 CUDA flag 不会传给 host compiler；
- 发布库时明确 PTX/real architecture 策略，避免只为开发机 GPU 构建。

## 5. 配置与诊断

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --verbose
```

多 CUDA 安装时优先显式指定 compiler/toolkit root，不要依赖不稳定的软链接：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
```

## 参考资料

- [CMake CUDA language](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cuda)
- [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)
- [FindCUDAToolkit](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)
