## 关键步骤
* **1. PyTorch C++ Extension API***
* **2. JIT (just-in-time)编译**
* **3. PyTorch c API 包装**
*  **4. CUDA Runtime API**
### 1. 编写CUDA Kernel
```cuda
// my_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_cuda_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

extern "C" void launch_my_kernel(float* input, float* output, int n) {
    dim3 blocks((n + 255) / 256);
    dim3 threads(256);
    my_cuda_kernel<<<blocks, threads>>>(input, output, n);
    cudaDeviceSynchronize();
}
```
### 2. 创建C++包装层
```cpp
// my_extension.cpp
#include <torch/extension.h>
#include <vector>
void launch_my_kernel(float* input, float* output, int n);

torch::Tensor my_function(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    launch_my_kernel(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_function", &my_function, "My custom CUDA function");
}
```
### 编译过程（两种方式）
#### 方式一：JIT编译（运行时编译）
```python
from torch.utils.cpp_extension import load
# 运行时编译
my_extension = load(
    name='my_extension',
    sources=['my_extension.cpp', 'my_kernel.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=['-O2']
)
```
#### 方式二：预编译（setup.py）
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension(
            name='my_extension',
            sources=['my_extension.cpp', 'my_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```