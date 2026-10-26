# 将 CUDA Kernel 集成到 PyTorch

本仓库采用 AOT `CUDAExtension`。无论 JIT 还是 AOT，工程边界都应保持：

```text
Python API → PyTorch binding → launcher(stream) → CUDA kernel
```

## 1. Kernel 与 launcher

```cpp
// scale_kernel.cu
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

__global__ void scale_kernel(
    const float* input, float* output, int64_t n, float alpha) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = alpha * input[i];
}

cudaError_t launch_scale(
    const float* input,
    float* output,
    int64_t n,
    float alpha,
    cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    if (n < 0) return cudaErrorInvalidValue;
    constexpr int threads = 256;
    const int64_t blocks64 = (n - 1) / threads + 1;
    if (blocks64 > INT_MAX) return cudaErrorInvalidConfiguration;
    const int blocks = static_cast<int>(blocks64);
    scale_kernel<<<blocks, threads, 0, stream>>>(input, output, n, alpha);
    return cudaPeekAtLastError();
}
```

launcher 不调用 `cudaDeviceSynchronize()`；同步由调用者或测试边界决定。

## 2. Binding

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

cudaError_t launch_scale(
    const float*, float*, int64_t, float, cudaStream_t);

torch::Tensor scale(torch::Tensor input, double alpha) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    cudaError_t status = launch_scale(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        static_cast<float>(alpha),
        stream.stream());
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale", &scale);
}
```

必须检查 rank/shape/device/dtype/layout 和整数范围；多输入还要检查同 device。
`CUDAGuard` 与 current stream 让 extension 正确参与 PyTorch 的 device/stream 语义。

## 3. AOT 构建

本仓库的 [setup.py](../../setup.py) 同时创建 `flash_attn_cuda` 和 `sgemm_cuda`：

```python
CUDAExtension(
    name="sgemm_cuda",
    sources=[
        "csrc/bindings/sgemm_api.cpp",
        "csrc/sgemm/kernels/simt/sgemm_fp32.cu",
    ],
    include_dirs=[str(INCLUDE_DIR)],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"],
    },
)
```

```bash
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

`TORCH_CUDA_ARCH_LIST` 应按部署 GPU 设置；架构越多，构建越慢、binary 越大。

## 4. JIT 构建

快速原型可使用：

```python
from torch.utils.cpp_extension import load

module = load(
    name="scale_cuda",
    sources=["scale_binding.cpp", "scale_kernel.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=True,
)
```

JIT 仍需要匹配的 host compiler 和 CUDA Toolkit。它适合实验，不应把编译、数据
生成、benchmark 和 profiling 混成一个难以复现的脚本。

## 5. 测试清单

- 随机输入与 PyTorch reference；
- tile 整数倍与 tail；
- 空/非法 shape、错误 dtype/device/layout；
- non-default stream；
- 多 GPU（若支持）；
- 运行后同步以暴露异步执行错误；
- precision-specific `atol/rtol`。

## 参考资料

- [PyTorch C++/CUDA Extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch CUDA stream utilities](https://docs.pytorch.org/cppdocs/api/cuda/utilities.html)
