## find_package()底层原理
`find_package` 只是**找到**了 CUDA，创建了目标（如 `CUDA::cudart`），但不会自动应用到任何目标。
必须通过 `target_link_libraries()` 将 CUDA 目标链接到可执行文件/库，才会自动添加：
- 包含路径（头文件目录）
- 链接库
- 编译选项
![[Pasted image 20251223140619.png]]
### 传统方式
当我们调用`find_package(CUDA)`时，CMake会搜索`FindCUDA.cmake`文件。这个文件会在一系列预定义的路径中查找CUDA Toolkit的安装位置，然后设置相关的变量。
```cmake
find_package(CUDA REQUIRED)
# 需要手动传递所有内容
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(myapp myapp.cu)
target_link_libraries(myapp ${CUDA_LIBRARIES})
```
### 现代方式
当我们调用`find_package(CUDAToolkit)`时，CMake会搜索`CUDAToolkitConfig.cmake`文件。这个文件通常安装在CUDA Toolkit的安装路径下，例如`/usr/local/cuda/lib64/cmake/CUDAToolkit/`。这个配置文件会定义一系列的目标，这些目标包含了使用CUDA所需的所有信息。
```cmake
find_package(CUDAToolkit REQUIRED)

# 目标自动传递依赖关系
add_executable(myapp myapp.cu)
target_link_libraries(myapp CUDA::cudart)  # 自动包含头文件和链接库
```
最新的方式cmake3.8+ 
```cmake 
cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)
add_executable(my_app my_app.cu)
# 不需要显式链接 CUDA 库，因为 CMake 已经自动处理了
```

## cmake 作用过程
```text
find_package(CUDAToolkit)
    ↓
创建 CUDA::cudart 目标（包含：头文件路径、库路径、编译选项）
    ↓
add_executable(reduce_v0 ...)  创建你的目标
    ↓
target_link_libraries(reduce_v0 CUDA::cudart)  ← 关键连接！
    ↓
CMake 自动将 CUDA::cudart 的属性传递给 reduce_v0：
    • 包含目录（编译时）
    • 链接库（链接时）
    • 编译标志
```

## target_link_libraries()
在现代 CMake 中，`target_link_libraries` 不仅影响链接，还影响**编译**阶段。
```cmake
target_link_libraries(reduce_v0 CUDA::cudart)
```

这会导致：
1. **编译阶段**：自动添加 `-I/usr/local/cuda/include` 到编译命令
2. **链接阶段**：自动添加 `-lcudart` 到链接命令
3. **依赖传播**：如果 `CUDA::cudart` 有其他依赖，也会自动传递